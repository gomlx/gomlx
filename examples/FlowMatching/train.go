package fm

import (
	"fmt"
	"path"
	"time"

	flowers "github.com/gomlx/gomlx/examples/oxfordflowers102"
	"github.com/gomlx/gomlx/examples/oxfordflowers102/diffusion"
	"github.com/gomlx/gomlx/internal/must"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/checkpoints"
	"github.com/gomlx/gomlx/pkg/ml/layers/batchnorm"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gomlx/pkg/ml/train/metrics"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers/cosineschedule"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gomlx/ui/gonb/plotly"
	stdplots "github.com/gomlx/gomlx/ui/plots"
	"github.com/gomlx/gopjrt/dtypes"
	"k8s.io/klog/v2"
)

// TrainModel with a given config -- it includes the context with hyperparameters.
func TrainModel(config *diffusion.Config, checkpointPath string, evaluateOnEnd bool, verbosity int) {
	ctx := config.Context
	paramsSet := config.ParamsSet
	backend := config.Backend

	// Backend handles creation of ML computation graphs, accelerator resources, etc.
	if verbosity >= 1 {
		fmt.Printf("Backend %q:\t%s\n", backend.Name(), backend.Description())
	}

	// Checkpoints saving.
	checkpoint, samplesNoise, samplesFlowerIds := config.AttachCheckpoint(checkpointPath)
	_ = samplesFlowerIds
	if samplesNoise == nil {
		klog.Exitf("A checkpoint directory name with --checkpoint is required for storing evolution of some samples, none given")
	}
	if verbosity >= 2 {
		fmt.Println(commandline.SprintContextSettings(ctx))
	}
	if context.GetParamOr(ctx, "rng_reset", true) {
		// Reset RNG with some pseudo-random value.
		ctx.ResetRNGState()
	}
	if verbosity >= 1 {
		// Enumerate parameters that were set.
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
		fmt.Println("\t - Using balanced datasets.")
		balancedTrainDS := must.M1(flowers.NewBalancedDataset(config.Backend, config.DataDir, config.ImageSize))
		trainDS = balancedTrainDS
	} else {
		trainDS = trainInMemoryDS
	}

	// Custom loss: model returns scalar loss as the second element of the predictions.
	customLoss := func(labels, predictions []*Node) *Node { return predictions[1] }

	// Create a train.Trainer: this object will orchestrate running the model, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	trainer := train.NewTrainer(
		backend, ctx, BuildTrainComputation(config), customLoss,
		optimizers.FromContext(ctx),
		[]metrics.Interface{}, // trainMetrics
		[]metrics.Interface{}) // evalMetrics
	if config.NanLogger != nil {
		trainer.OnExecCreation(func(exec *context.Exec, _ train.GraphType) {
			config.NanLogger.AttachToExec(exec)
		})
	}

	// Use a standard training loop.
	loop := train.NewLoop(trainer)
	if verbosity >= 0 {
		commandline.AttachProgressBar(loop)
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
	_ = plotter

	// Generate samples from fixed noise to monitor the training.
	generator := NewImagesGenerator(config, samplesNoise, samplesFlowerIds, 20)
	// KID is a InceptionV3 based pretrained model only used to measured similarity of the
	// images between generated flowers and the original. It's a metric.
	var kid *KidGenerator
	if context.GetParamOr(ctx, "kid", false) {
		kidDS := validationDS.Copy()
		kidDS.Shuffle().BatchSize(config.EvalBatchSize, true)
		kid = NewKidGenerator(config, kidDS, 5)
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
		fmt.Println("Starting training stage:")
		_, err := loop.RunSteps(trainDS, numTrainSteps-globalStep)
		if verbosity >= 1 {
			fmt.Printf("\t[Step %d] median train step: %d microseconds\n",
				loop.LoopStep, loop.MedianTrainStepDuration().Microseconds())
		}
		if err != nil {
			if loop.LoopStep > loop.StartStep {
				klog.Infof("Debug checkpoint save before crashing at loop step %d", loop.LoopStep)
				errSave := checkpoint.Save()
				if errSave != nil {
					klog.Errorf("Error while saving checkpoint before crashing: %+v", errSave)
				}
			}
			klog.Fatalf("Error during training: %+v", err)
		}

		// Update batch normalization averages, if they are used.
		bnUpdated, err := batchnorm.UpdateAverages(trainer, trainEvalDS)
		if err != nil {
			klog.Exitf("Error while updating batch normalization averages: %+v", err)
		}
		if bnUpdated {
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

// BuildTrainComputation builds the ModelFn for training and evaluation.
//
// It generates the random noise as the "source distribution" for each example image,
// as well as random values of t -> [0,1), used to train.
func BuildTrainComputation(config *diffusion.Config) train.ModelFn {
	return func(ctx *context.Context, spec any, inputs []*Node) []*Node {
		g := inputs[0].Graph()

		// Prepare the input image and noise.
		images := inputs[0]
		if _, ok := spec.(*flowers.BalancedDataset); ok {
			// For BalancedDataset we need to gather the images from the examples.
			examplesIdx := inputs[1]
			images = Gather(images, InsertAxes(examplesIdx, -1), false)
		}
		flowerIds := inputs[2]
		batchSize := images.Shape().Dimensions[0]
		dtype := config.DType

		// Augment images, if not training.
		images = diffusion.AugmentImages(ctx, images)

		// Convert to the corresponding image size.
		config.NanLogger.TraceFirstNaN(images, "RawImages")
		images = config.PreprocessImages(images, true)
		config.NanLogger.TraceFirstNaN(images, "NormalizedImages")
		images = ConvertDType(images, dtype)

		// Gaussian noise to be transposed to the images.
		noises := ctx.RandomNormal(g, images.Shape())
		config.NanLogger.TraceFirstNaN(noises, "noises")

		// Cosine schedule, if enabled.
		cosineschedule.New(ctx, g, dtype).FromContext().Done()

		// Sample noise at different schedules.
		t := ctx.RandomUniform(g, shapes.Make(dtype, batchSize, 1, 1, 1))
		if ctx.IsTraining(g) {
			// During training, we bias towards the end (larger times t), since it's more detailed shifts.
			t = Sqrt(t)
		} else {
			// During evaluation, we only look for t >= 0.5: for smaller values probably it is going to
			// be pretty random, and the loss gets washed away.
			t = DivScalar(OnePlus(t), 2)
		}
		noisyImages := Add(
			Mul(images, t),
			Mul(noises, OneMinus(t)))
		config.NanLogger.TraceFirstNaN(noisyImages, "noisyImages (A)")
		noisyImages = StopGradient(noisyImages)

		// Target and predicted velocity (aka. u(X,t)).
		targetVelocity := Sub(images, noises)
		predictedVelocity := diffusion.UNetModelGraph(ctx, config.NanLogger, noisyImages, t, flowerIds)
		config.NanLogger.TraceFirstNaN(predictedVelocity, "predictedVelocity")

		// Calculate our loss inside the model: use losses.ParamLoss to define the loss, and if not set,
		// back-off to "diffusion_loss" hyper-param (for backward compatibility).
		// Defaults to "mae" (mean-absolute-error).
		lossName := context.GetParamOr(ctx, losses.ParamLoss,
			context.GetParamOr(ctx, "diffusion_loss", "mse"))
		ctx.SetParam("loss", lossName) // Needed for old models that used "diffusion_loss".
		lossFn := must.M1(losses.LossFromContext(ctx))

		// Large reduce operations lead to overflow for low-precision dtypes. We up-convert in those cases, before calculating the loss.
		if dtype == dtypes.Float16 || dtype == dtypes.BFloat16 {
			targetVelocity = ConvertDType(targetVelocity, dtypes.Float32)
			predictedVelocity = ConvertDType(predictedVelocity, dtypes.Float32)
		}

		loss := lossFn([]*Node{targetVelocity}, []*Node{predictedVelocity})
		if !loss.IsScalar() {
			loss = ReduceAllMean(loss)
		}
		return []*Node{predictedVelocity, loss}
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
	imagesPath := fmt.Sprintf("%s%07d.tensor", diffusion.GeneratedSamplesPrefix, loop.LoopStep)
	imagesPath = path.Join(checkpoint.Dir(), imagesPath)
	must.M(sampledImages.Save(imagesPath))

	return nil
}
