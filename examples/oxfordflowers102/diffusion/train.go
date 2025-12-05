package diffusion

import (
	"fmt"
	"os"
	"path"
	"time"

	"github.com/gomlx/gomlx/backends"
	flowers "github.com/gomlx/gomlx/examples/oxfordflowers102"
	"github.com/gomlx/gomlx/internal/must"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/nanlogger"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/checkpoints"
	"github.com/gomlx/gomlx/pkg/ml/layers/batchnorm"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/metrics"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gomlx/ui/gonb/margaid"
	"github.com/gomlx/gomlx/ui/gonb/plotly"
	stdplots "github.com/gomlx/gomlx/ui/plots"
	"github.com/gomlx/gopjrt/dtypes"
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
	checkpoint *checkpoints.Handler, noise, flowerIDs *tensors.Tensor) {
	if checkpointPath == "" {
		return checkpoint, noise, flowerIDs
	}
	numCheckpointsToKeep := context.GetParamOr(c.Context, "num_checkpoints", 5)
	excludeParams := make([]string, 0, len(c.ParamsSet)+len(ParamsExcludedFromLoading))
	excludeParams = append(excludeParams, c.ParamsSet...)
	excludeParams = append(excludeParams, ParamsExcludedFromLoading...)
	checkpoint = must.M1(checkpoints.Build(c.Context).
		DirFromBase(checkpointPath, c.DataDir).
		Keep(numCheckpointsToKeep).
		ExcludeParams(excludeParams...).
		Immediate().
		Done())
	c.Checkpoint = checkpoint // Save in config.

	// In case the loaded checkpoint has different values, we need to update the config accordingly.
	c.DType = must.M1(dtypes.DTypeString(
		context.GetParamOr(c.Context, "dtype", "float32")))
	c.ImageSize = context.GetParamOr(c.Context, "image_size", 64)
	c.BatchSize = context.GetParamOr(c.Context, "batch_size", 64)
	c.EvalBatchSize = context.GetParamOr(c.Context, "eval_batch_size", 128)

	// Load/generate sampled noise/flowerIDs.
	noisePath := path.Join(checkpoint.Dir(), NoiseSamplesFile)
	flowerIdsPath := path.Join(checkpoint.Dir(), FlowerIdsSamplesFile)
	var err error
	noise, err = tensors.Load(noisePath)
	if err == nil {
		flowerIDs, err = tensors.Load(flowerIdsPath)
		if err == nil {
			return checkpoint, noise, flowerIDs
		}
	}
	if !os.IsNotExist(err) {
		must.M(err)
	}

	// Create new noise and flower ids -- and save it for future training.
	numSamples := context.GetParamOr(c.Context, "samples_during_training", 64)
	noise = c.GenerateNoise(numSamples)
	flowerIDs = c.GenerateFlowerIds(numSamples)
	must.M(noise.Save(noisePath))
	must.M(flowerIDs.Save(flowerIdsPath))
	return checkpoint, noise, flowerIDs
}

// TrainModel with hyperparameters given in Context.
// paramsSet enumerate the context parameters that were set and should override values loaded from a checkpoint.
func TrainModel(ctx *context.Context, dataDir, checkpointPath string, paramsSet []string, evaluateOnEnd bool, verbosity int) {
	// Backend handles creation of ML computation graphs, accelerator resources, etc.
	backend := backends.MustNew()
	if verbosity >= 1 {
		fmt.Printf("Backend %q:\t%s\n", backend.Name(), backend.Description())
	}
	config := NewConfig(backend, ctx, dataDir, paramsSet)

	// Checkpoints saving.
	checkpoint, samplesNoise, samplesFlowerIds := config.AttachCheckpoint(checkpointPath)
	if samplesNoise == nil {
		klog.Exitf("A checkpoint directory name with --checkpoint is required, none given")
	}
	if verbosity >= 2 {
		fmt.Println(commandline.SprintContextSettings(ctx))
	}
	if context.GetParamOr(ctx, "rng_reset", true) {
		// Reset RNG.
		ctx.ResetRNGState()
	}
	if verbosity >= 1 {
		for _, paramsPath := range paramsSet {
			scope, name := context.SplitScope(paramsPath)
			if scope == "" {
				if value, found := ctx.GetParam(name); found {
					fmt.Printf("\t%s=%v\n", name, value)
				}
			} else {
				if value, found := ctx.InAbsPath(scope).GetParam(name); found {
					fmt.Printf("\tscope=%q %s=%v\n", scope, name, value)
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
	if context.GetParamOr(ctx, "diffusion_balanced_dataset", false) {
		fmt.Println("Using balanced datasets.")
		balancedTrainDS := must.M1(flowers.NewBalancedDataset(config.Backend, config.DataDir, config.ImageSize))
		trainDS = balancedTrainDS
	} else {
		trainDS = trainInMemoryDS
	}

	// Custom loss: model returns scalar loss as the second element of the predictions.
	customLoss := func(labels, predictions []*Node) *Node { return predictions[1] }
	imgMetricFn := func(ctx *context.Context, labels, predictions []*Node) *Node {
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
		func(ctx *context.Context, labels, predictions []*Node) *Node {
			return predictions[1]
		}, pprintLossFn, 0.05)

	movingMAE := metrics.NewExponentialMovingAverageMetric(
		"Moving MAE Loss", "~mae", "loss",
		func(ctx *context.Context, labels, predictions []*Node) *Node {
			return predictions[3]
		}, pprintLossFn, 0.05)
	meanMAE := metrics.NewMeanMetric(
		"MAE Loss", "#mae", "loss",
		func(ctx *context.Context, labels, predictions []*Node) *Node {
			return predictions[3]
		}, pprintLossFn)

	useNanLogger := context.GetParamOr(ctx, "nan_logger", false)
	if useNanLogger {
		nanLogger = nanlogger.New()
	}

	// Create a train.Trainer: this object will orchestrate running the model, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	trainer := train.NewTrainer(
		backend, ctx, config.BuildTrainingModelGraph(), customLoss,
		optimizers.FromContext(ctx),
		[]metrics.Interface{movingImagesLoss, movingNoiseLoss, movingMAE}, // trainMetrics
		[]metrics.Interface{meanImagesLoss, meanMAE})                      // evalMetrics
	if nanLogger != nil {
		trainer.OnExecCreation(func(exec *context.Exec, _ train.GraphType) {
			nanLogger.AttachToExec(exec)
		})
	}

	// Use standard training loop.
	loop := train.NewLoop(trainer)
	if verbosity >= 0 {
		commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.
	}

	// Checkpoint saving: every 3 minutes of training.
	if checkpoint != nil {
		period := must.M1(
			time.ParseDuration(context.GetParamOr(ctx, "checkpoint_frequency", "3m")))
		train.PeriodicCallback(loop, period, true, "saving checkpoint", 100,
			func(loop *train.Loop, metrics []*tensors.Tensor) error {
				return checkpoint.Save()
			})
	}

	// Attach Plotly plots: plot points at exponential steps.
	// The points generated are saved along the checkpoint directory (if one is given).
	var plotter *plotly.PlotConfig
	if context.GetParamOr(ctx, plotly.ParamPlots, false) {
		plotter = plotly.New().
			WithCheckpoint(checkpoint).
			Dynamic().
			WithDatasets(trainEvalDS, validationDS).
			WithBatchNormalizationAveragesUpdate(trainEvalDS)
	}

	generator := config.NewImagesGenerator(samplesNoise, samplesFlowerIds, 20)
	var kid *KidGenerator
	if context.GetParamOr(ctx, "kid", false) {
		kidDS := validationDS.Copy()
		kidDS.Shuffle().BatchSize(config.EvalBatchSize, true)
		kid = config.NewKidGenerator(kidDS, 5)
	}

	samplesFrequency := context.GetParamOr(ctx, "samples_during_training_frequency", 200)
	samplesFrequencyGrowth := context.GetParamOr(ctx, "samples_during_training_frequency_growth", 1.2)
	if plotter != nil {
		train.ExponentialCallback(loop, samplesFrequency, samplesFrequencyGrowth, true,
			"Monitor", 0, func(loop *train.Loop, metrics []*tensors.Tensor) error {
				return TrainingMonitor(checkpoint, loop, metrics, plotter, plotter.EvalDatasets, generator, kid)
			})
	}

	// Loop for given number of steps.
	numTrainSteps := context.GetParamOr(ctx, "train_steps", 0)
	globalStep := int(optimizers.GetGlobalStep(ctx))
	if globalStep > 0 {
		trainer.SetContext(ctx.Reuse())
	}
	if globalStep < numTrainSteps {
		_ = must.M1(loop.RunSteps(trainDS, numTrainSteps-globalStep))
		if verbosity >= 1 {
			fmt.Printf("\t[Step %d] median train step: %d microseconds\n",
				loop.LoopStep, loop.MedianTrainStepDuration().Microseconds())
		}

		// Update batch normalization averages, if they are used.
		if must.M1(batchnorm.UpdateAverages(trainer, trainEvalDS)) {
			fmt.Println("\tUpdated batch normalization mean/variances averages.")
			if checkpoint != nil {
				must.M(checkpoint.Save())
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
		must.M(commandline.ReportEval(trainer, trainEvalDS, validationDS))
	}
}

// TrainingMonitor is periodically called during training, and is used to report metrics and generate sample images at
// the current training step.
func TrainingMonitor(checkpoint *checkpoints.Handler, loop *train.Loop, metrics []*tensors.Tensor,
	plotter stdplots.Plotter, evalDatasets []train.Dataset, generator *ImagesGenerator, kid *KidGenerator) error {
	//fmt.Printf("\n[... evaluating@%d ...] [median train step (ms): %d]\n", loop.LoopStep, loop.MedianTrainStepDuration().Milliseconds())

	// Save checkpoint, just in case.
	if checkpoint == nil {
		// Only works if there is a model directory.
		return nil
	}
	must.M(checkpoint.Save())
	must.M(checkpoint.Backup()) // Save backup, so these checkpoint doesn't get automatically collected.

	// Update plotter with metrics.
	must.M(stdplots.AddTrainAndEvalMetrics(plotter, loop, metrics, evalDatasets, evalDatasets[0]))

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
	imagesPath = path.Join(checkpoint.Dir(), imagesPath)
	must.M(sampledImages.Save(imagesPath))
	return nil
}

// DisplayTrainingPlots simply display the training plots of a model, without any training.
//
// paramsSet are hyperparameters overridden, that it should not load from the checkpoint (see commandline.ParseContextSettings).
func DisplayTrainingPlots(ctx *context.Context, dataDir, checkpointPath string, paramsSet []string) {
	backend := backends.MustNew()
	config := NewConfig(backend, ctx, dataDir, paramsSet)
	checkpoint, _, _ := config.AttachCheckpoint(checkpointPath)
	if checkpoint == nil {
		fmt.Printf("You must set --checkpoint='model_sub_dir'!")
		return
	}
	must.M(plotly.New().WithCheckpoint(checkpoint).Plot())
}

// CompareModelPlots display several model metrics on the same plots.
func CompareModelPlots(dataDir string, modelNames ...string) {
	dataDir = fsutil.MustReplaceTildeInDir(dataDir)
	plots := margaid.New(1024, 400).LogScaleX().LogScaleY()
	for _, modelName := range modelNames {
		modelPath := modelName
		if !path.IsAbs(modelPath) {
			modelPath = path.Join(dataDir, modelPath)
		}
		modelPath = path.Join(modelPath, stdplots.TrainingPlotFileName)
		_ = must.M1(plots.PreloadFile(modelPath, func(metricName string) string {
			return fmt.Sprintf("[%s] %s", modelName, metricName)
		}))
	}
	plots.Plot()
}
