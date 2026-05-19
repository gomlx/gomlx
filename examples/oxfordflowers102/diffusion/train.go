// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package diffusion

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/nanlogger"
	"github.com/gomlx/gomlx/core/tensors"
	flowers "github.com/gomlx/gomlx/examples/oxfordflowers102"
	"github.com/gomlx/gomlx/ml/layers/batchnorm"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/metrics"
	optimizers "github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/gomlx/gomlx/support/fsutil"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gomlx/ui/gonb/margaid"
	"github.com/gomlx/gomlx/ui/gonb/plotly"
	stdplots "github.com/gomlx/gomlx/ui/plots"
	"k8s.io/klog/v2"
)

const (
	NoiseSamplesFile       = "noise_samples.tensor"
	FlowerIdsSamplesFile   = "flower_ids_samples.tensor"
	GeneratedSamplesPrefix = "generated_samples_"
)

// AttachCheckpoint loading previous values and using it to save.
//
// It also loads the noise (+flowerIds) samples for this model.
// The idea is that at each evaluation checkpoint we generate the images for these fixed noise samples,
// and one can observe the model quality evolving.
//
// For new models if creates the noise + flowerIds samples used to monitor the model quality evolving.
//
// The returned handler is also set into Config.Checkpoint.
//
// If no path is given (checkpointPath == "") then no checkpoint is created, and it returns nil for all values.
func (c *Config) AttachCheckpoint(checkpointPath string) (
	checkpointHandler *checkpoint.Handler, noise, flowerIDs *tensors.Tensor) {
	if checkpointPath == "" {
		return nil, noise, flowerIDs
	}
	numCheckpointsToKeep := model.GetParamOr(c.Context, "num_checkpoints", 5)
	excludeParams := make([]string, 0, len(c.ParamsSet)+len(ParamsExcludedFromLoading))
	excludeParams = append(excludeParams, c.ParamsSet...)
	excludeParams = append(excludeParams, ParamsExcludedFromLoading...)
	checkpointHandler = check1(checkpoint.Build(c.Context.Store()).
		DirFromBase(checkpointPath, c.DataDir).
		Keep(numCheckpointsToKeep).
		ExcludeParams(excludeParams...).
		Immediate().
		Done())
	c.Checkpoint = checkpointHandler // Save in config.

	// In case the loaded checkpoint has different values, we need to update the config accordingly.
	c.DType = check1(dtypes.DTypeString(
		model.GetParamOr(c.Context, "dtype", "float32")))
	c.ImageSize = model.GetParamOr(c.Context, "image_size", 64)

	// Load/generate sampled noise/flowerIDs.
	noisePath := path.Join(checkpointHandler.Dir(), NoiseSamplesFile)
	flowerIdsPath := path.Join(checkpointHandler.Dir(), FlowerIdsSamplesFile)
	var err error
	noise, err = tensors.Load(noisePath)
	if err == nil {
		flowerIDs, err = tensors.Load(flowerIdsPath)
		if err == nil {
			return checkpointHandler, noise, flowerIDs
		}
	}
	if !os.IsNotExist(err) {
		check(err)
	}

	// Create new noise and flower ids -- and save it for future training.
	numSamples := model.GetParamOr(c.Context, "samples_during_training", 64)
	noise = c.GenerateNoise(numSamples)
	flowerIDs = c.GenerateFlowerIds(numSamples)
	check(noise.Save(noisePath))
	check(flowerIDs.Save(flowerIdsPath))
	return checkpointHandler, noise, flowerIDs
}

// TrainWithStore with hyperparameters given in Store.
// paramsSet enumerate the context parameters that were set and should override values loaded from a checkpointHandler.
func TrainWithStore(store *model.Store, dataDir, checkpointPath string, paramsSet []string, evaluateOnEnd bool, verbosity int) {
	scope := store.RootScope()
	// Backend handles creation of ML computation graphs, accelerator resources, etc.
	backend := compute.MustNew()
	if verbosity >= 1 {
		fmt.Printf("Backend %q:\t%s\n", backend.Name(), backend.Description())
	}
	config := NewConfig(backend, scope, dataDir, paramsSet)

	// Checkpoints saving.
	checkpointHandler, samplesNoise, samplesFlowerIds := config.AttachCheckpoint(checkpointPath)
	if samplesNoise == nil {
		klog.Exitf("A checkpoint directory name with --checkpoint is required, none given")
	}
	if verbosity >= 2 {
		fmt.Println(commandline.SprintSettings(scope))
	}
	if model.GetParamOr(scope, "rng_reset", true) {
		// Reset RNG.
		_ = scope.Store().ResetRNGState()
	}
	if verbosity >= 1 {
		for _, paramsPath := range paramsSet {
			pScope, name := path.Split(paramsPath)
			if pScope != "" && len(pScope) > 1 && strings.HasSuffix(pScope, "/") {
				pScope = pScope[:len(pScope)-1]
			}
			if pScope == "" {
				if value, found := scope.GetParam(name); found {
					fmt.Printf("\t%s=%v\n", name, value)
				}
			} else {
				if value, found := scope.Store().Scope(pScope).GetParam(name); found {
					fmt.Printf("\tscope=%q %s=%v\n", pScope, name, value)
				}
			}
		}
	}

	// Create datasets used for training and evaluation.
	trainInMemoryDS, validationDS := config.CreateInMemoryDatasets()
	trainEvalDS := trainInMemoryDS.Copy()
	trainInMemoryDS.Shuffle().Infinite(true).BatchSize(config.BatchSize, true)
	trainEvalDS.BatchSize(config.EvalBatchSize, false)
	validationDS.BatchSize(config.EvalBatchSize, false)
	var trainDS train.Dataset
	if model.GetParamOr(scope, "diffusion_balanced_dataset", false) {
		fmt.Println("Using balanced datasets.")
		balancedTrainDS := check1(flowers.NewBalancedDataset(config.Backend, config.DataDir, config.ImageSize))
		trainDS = balancedTrainDS
	} else {
		trainDS = trainInMemoryDS
	}

	// Custom loss: model returns scalar loss as the second element of the predictions.
	customLoss := func(labels, predictions []*Node) *Node { return predictions[1] }
	imgMetricFn := func(scope *model.Scope, labels, predictions []*Node) *Node {
		return predictions[2]
	}
	pprintLossFn := func(t *tensors.Tensor) string {
		return fmt.Sprintf("%.3f", t.Value())
	}
	meanImagesLoss := metrics.NewMeanMetric(
		"Images Loss", "img_loss", "img_loss", imgMetricFn, pprintLossFn)
	movingImagesLoss := metrics.NewExponentialMovingAverageMetric(
		"Moving Images Loss", "~img_loss", "img_loss", imgMetricFn, pprintLossFn, 0.05)

	movingNoiseLoss := metrics.NewExponentialMovingAverageMetric(
		"Moving (faster) Noise Loss", "~fast_loss", "loss",
		func(scope *model.Scope, labels, predictions []*Node) *Node {
			return predictions[1]
		}, pprintLossFn, 0.05)

	movingMAE := metrics.NewExponentialMovingAverageMetric(
		"Moving MAE Loss", "~mae", "loss",
		func(scope *model.Scope, labels, predictions []*Node) *Node {
			return predictions[3]
		}, pprintLossFn, 0.05)
	meanMAE := metrics.NewMeanMetric(
		"MAE Loss", "#mae", "loss",
		func(scope *model.Scope, labels, predictions []*Node) *Node {
			return predictions[3]
		}, pprintLossFn)

	useNanLogger := model.GetParamOr(scope, "nan_logger", false)
	if useNanLogger {
		nanLogger = nanlogger.New()
	}

	// Create a train.Trainer: this object will orchestrate running the model, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	trainer := train.NewTrainer(
		backend, scope.Store(), config.BuildTrainingModelGraph(), customLoss,
		optimizers.FromScope(scope),
		[]metrics.Interface{movingImagesLoss, movingNoiseLoss, movingMAE}, // trainMetrics
		[]metrics.Interface{meanImagesLoss, meanMAE})                      // evalMetrics
	if nanLogger != nil {
		trainer.OnExecCreation(func(exec *model.Exec, _ train.GraphType) {
			nanLogger.AttachToExec(exec)
		})
	}

	// Use standard training loop.
	loop := train.NewLoop(trainer)
	if verbosity >= 0 {
		commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.
	}

	// Checkpoint saving: every 3 minutes of training.
	if checkpointHandler != nil {
		period := check1(
			time.ParseDuration(model.GetParamOr(scope, "checkpoint_frequency", "3m")))
		train.PeriodicCallback(loop, period, true, "saving checkpoint", 100,
			func(loop *train.Loop, metrics []*tensors.Tensor) error {
				return checkpointHandler.Save()
			})
	}

	// Attach Plotly plots: plot points at exponential steps.
	// The points generated are saved along the checkpoint directory (if one is given).
	var plotter *plotly.PlotConfig
	if model.GetParamOr(scope, plotly.ParamPlots, false) {
		plotter = plotly.New().
			WithCheckpoint(checkpointHandler).
			Dynamic().
			WithDatasets(trainEvalDS, validationDS).
			WithBatchNormalizationAveragesUpdate(trainEvalDS)
	}

	generator := config.NewImagesGenerator(samplesNoise, samplesFlowerIds, 20)
	var kid *KidGenerator
	if model.GetParamOr(scope, "kid", false) {
		kidDS := validationDS.Copy()
		kidDS.Shuffle().BatchSize(config.EvalBatchSize, true)
		kid = config.NewKidGenerator(kidDS, 5)
	}

	samplesFrequency := model.GetParamOr(scope, "samples_during_training_frequency", 200)
	samplesFrequencyGrowth := model.GetParamOr(scope, "samples_during_training_frequency_growth", 1.2)
	if plotter != nil {
		train.ExponentialCallback(loop, samplesFrequency, samplesFrequencyGrowth, true,
			"Monitor", 0, func(loop *train.Loop, metrics []*tensors.Tensor) error {
				return TrainingMonitor(checkpointHandler, loop, metrics, plotter, plotter.EvalDatasets, generator, kid)
			})
	}

	// Loop for given number of steps.
	numTrainSteps := model.GetParamOr(scope, "train_steps", 0)
	globalStep := int(optimizers.GetGlobalStep(store))
	if globalStep < numTrainSteps {
		_ = check1(loop.RunSteps(trainDS, numTrainSteps-globalStep))
		if verbosity >= 1 {
			fmt.Printf("\t[Step %d] median train step: %d microseconds\n",
				loop.LoopStep, loop.MedianTrainStepDuration().Microseconds())
		}

		// Update batch normalization averages, if they are used.
		if check1(batchnorm.UpdateAverages(trainer, trainEvalDS)) {
			fmt.Println("\tUpdated batch normalization mean/variances averages.")
			if checkpointHandler != nil {
				check(checkpointHandler.Save())
			}
		}

	} else {
		fmt.Printf("\t - target train_steps=%d already reached. To train further, set a number additional "+
			"to current global step.\n", numTrainSteps)
	}

	// Finally, print an evaluation on train and test datasets.
	if evaluateOnEnd {
		if verbosity >= 1 {
			fmt.Println()
		}
		check(commandline.ReportEval(trainer, trainEvalDS, validationDS))
	}
}

// TrainingMonitor is periodically called during training, and is used to report metrics and generate sample images at
// the current training step.
func TrainingMonitor(checkpointHandler *checkpoint.Handler, loop *train.Loop, metrics []*tensors.Tensor,
	plotter stdplots.Plotter, evalDatasets []train.Dataset, generator *ImagesGenerator, kid *KidGenerator) error {
	//fmt.Printf("\n[... evaluating@%d ...] [median train step (ms): %d]\n", loop.LoopStep, loop.MedianTrainStepDuration().Milliseconds())

	// Save checkpoint, just in case.
	if checkpointHandler == nil {
		// Only works if there is a model directory.
		return nil
	}
	check(checkpointHandler.Save())
	check(checkpointHandler.Backup()) // Save backup, so these checkpoint doesn't get automatically collected.

	// Update plotter with metrics.
	check(stdplots.AddTrainAndEvalMetrics(plotter, loop, metrics, evalDatasets, evalDatasets[0]))

	// Kid generator
	if kid != nil {
		kidValue := kid.Eval()
		//fmt.Printf("\nKID=%f\n", kidValue.Value())
		plotter.AddPoint(
			stdplots.Point{
				MetricName: "Kernel Inception Distance",
				Short:      "KID",
				MetricType: "KID",
				Step:       float64(loop.LoopStep),
				Value:      shapes.ConvertTo[float64](kidValue.Value()),
			})
		plotter.DynamicSampleDone(false)
	}

	// Generate intermediary images.
	sampledImages := generator.Generate()
	imagesPath := fmt.Sprintf("%s%07d.tensor", GeneratedSamplesPrefix, loop.LoopStep)
	imagesPath = path.Join(checkpointHandler.Dir(), imagesPath)
	check(sampledImages.Save(imagesPath))
	return nil
}

// DisplayTrainingPlots simply display the training plots of a model, without any training.
//
// paramsSet are hyperparameters overridden, that it should not load from the checkpoint (see commandline.ParseContextSettings).
func DisplayTrainingPlots(scope *model.Scope, dataDir, checkpointPath string, paramsSet []string) {
	backend := compute.MustNew()
	config := NewConfig(backend, scope, dataDir, paramsSet)
	checkpointHandler, _, _ := config.AttachCheckpoint(checkpointPath)
	if checkpointHandler == nil {
		fmt.Printf("You must set --checkpoint='model_sub_dir'!")
		return
	}
	check(plotly.New().WithCheckpoint(checkpointHandler).Plot())
}

// CompareModelPlots display several model metrics on the same plots.
func CompareModelPlots(dataDir string, modelNames ...string) {
	dataDir = fsutil.MustReplaceTildeInDir(dataDir)
	plots := margaid.New(1024, 400).LogScaleX().LogScaleY()
	for _, modelName := range modelNames {
		modelPath := modelName
		if !filepath.IsAbs(modelPath) {
			modelPath = path.Join(dataDir, modelPath)
		}
		modelPath = path.Join(modelPath, stdplots.TrainingPlotFileName)
		_ = check1(plots.PreloadFile(modelPath, func(metricName string) string {
			return fmt.Sprintf("[%s] %s", modelName, metricName)
		}))
	}
	plots.Plot()
}

// check reports and exits on error.
func check(err error) {
	if err == nil {
		return
	}
	klog.Fatalf("Fatal error: %+v", err)
}

// check1 reports and exits on error. Otherwise returns the value passed.
func check1[T any](v T, err error) T {
	check(err)
	return v
}
