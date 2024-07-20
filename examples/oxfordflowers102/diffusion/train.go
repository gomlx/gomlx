package diffusion

import (
	"flag"
	"fmt"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/examples/notebook/gonb/margaid"
	stdplots "github.com/gomlx/gomlx/examples/notebook/gonb/plots"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/checkpoints"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/commandline"
	"github.com/gomlx/gomlx/ml/train/metrics"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"k8s.io/klog/v2"
	"os"
	"path"
	"strings"
	"time"
)

var (
	flagCheckpoint         = flag.String("checkpoint", "", "Directory save and load checkpoints from. If left empty, no checkpoints are created.")
	flagCheckpointKeep     = flag.Int("checkpoint_keep", 20, "Number of checkpoints to keep, if --checkpoint is set.")
	flagCheckpointPeriod   = flag.Int("checkpoint_period", 60, "Period of time, in seconds, between checkpoints are saved.")
	flagCheckpointTakeMean = flag.Int("checkpoint_mean", 1, "If != 1, take the mean of the latest checkpoints. This is disabled (set to 1) if training.")
)

const (
	NoiseSamplesFile       = "noise_samples.tensor"
	FlowerIdsSamplesFile   = "flower_ids_samples.tensor"
	GeneratedSamplesPrefix = "generated_samples_"
)

// LoadCheckpointToContext and attaches to it, so that it gets saved.
//
// It also loads the noise (+flowerIds) samples for this model.
// The idea is that at each evaluation checkpoint we generate the images for these fixed noise samples,
// and one can observe the model quality evolving.
//
// For new models -- whose directory didn't previously exist, it does 2 things:
//
//   - It creates the noise + flowerIds samples used to monitor the model quality evolving.
//   - It creates the file `args.txt` with a copy of the arguments used to create the model.
//     Later, if the same model is used, it checks that the arguments match (with some exceptions),
//     and warns about mismatches.
func LoadCheckpointToContext(backend backends.Backend, ctx *context.Context) (checkpoint *checkpoints.Handler, noise, flowerIds *tensors.Tensor) {
	Init()
	if *flagCheckpoint == "" {
		return
	}

	checkpointPath := data.ReplaceTildeInDir(*flagCheckpoint)
	if !path.IsAbs(checkpointPath) {
		checkpointPath = path.Join(DataDir, checkpointPath)
	}
	var err error
	checkpoint, err = checkpoints.Build(ctx).Dir(checkpointPath).
		Keep(*flagCheckpointKeep).TakeMean(*flagCheckpointTakeMean, backend).
		Done()
	AssertNoError(err)

	// Check if args file exists, if not create it.
	argsPath := path.Join(checkpoint.Dir(), "args.txt")
	argsBytes, err := os.ReadFile(argsPath)
	if err != nil && os.IsNotExist(err) {
		// Doesn't exist yet, so let's create it.
		AssertNoError(os.WriteFile(argsPath, []byte(strings.Join(os.Args[1:], "\n")), 0664))
	} else if err == nil {
		// Read original args, print out diff:
		originalArgs := types.MakeSet[string]()
		originalArgs.Insert(strings.Split(string(argsBytes), "\n")...)
		currentArgs := types.MakeSet[string]()
		currentArgs.Insert(os.Args[1:]...)
		for arg := range originalArgs.Sub(currentArgs) {
			if !isArgIrrelevant(arg) {
				fmt.Printf("* Warning: missing argument %q used when model was originally created.\n", arg)
			}
		}
		for arg := range currentArgs.Sub(originalArgs) {
			if !isArgIrrelevant(arg) {
				fmt.Printf("* Warning: argument %q not used when model was originally created.\n", arg)
			}
		}
	} else {
		AssertNoError(err)
	}

	// Load/generate sampled noise/flowerIds.
	noisePath, flowerIdsPath := path.Join(checkpointPath, NoiseSamplesFile), path.Join(checkpointPath, FlowerIdsSamplesFile)
	noise, err = tensors.Load(noisePath)
	if err == nil {
		flowerIds, err = tensors.Load(flowerIdsPath)
		if err == nil {
			return
		}
	}
	if !os.IsNotExist(err) {
		AssertNoError(err)
	}

	// Create new noise and flower ids -- and save it for future training.
	noise = GenerateNoise(*flagTrainGeneratedSamples)
	flowerIds = GenerateFlowerIds(*flagTrainGeneratedSamples)
	AssertNoError(noise.Save(noisePath))
	AssertNoError(flowerIds.Save(flowerIdsPath))
	return
}

var irrelevantArgs = types.Set[string]{"": {}, "--plots": {}}

func isArgIrrelevant(arg string) bool {
	if irrelevantArgs.Has(arg) {
		return true
	}
	for _, prefix := range []string{"-steps", "--steps", "--batch", "--eval_batch", "--checkpoint_mean", "--platform"} {
		if strings.HasPrefix(arg, prefix) {
			return true
		}
	}
	return false
}

var (
	flagNumSteps = flag.Int("steps", 2000, "Number of gradient descent steps to perform in total "+
		"-- this includes the steps already trained, if restarting training a model.")
	flagPlots = flag.Bool("plots", true, "Plots during training: perform periodic evaluations, "+
		"save results if --checkpoint is set and draw plots, if in a Jupyter notebook.")
	flagKid = flag.Bool("kid", true, "If true, calculate Kernel Inception Distance (KID) on evaluation "+
		"-- it is quite expensive.")

	// Training hyperparameters:
	flagLearningRate     = flag.Float64("learning_rate", 0.001, "Initial learning rate.")
	flagL2Regularization = flag.Float64("l2_reg", 0, "L2 regularization on kernels. It doesn't interact well with --batch_norm.")
	flagReport           = flag.Bool("report", true, "If true generate evaluation report at end of training.")
	flagRngReset         = flag.Bool("rng_reset", true, "If true will reset the random number generator state with a new random value -- useful when continuing training.")

	// Sample generate images to monitor progress on;
	flagTrainMonitorStartFrequency  = flag.Int("monitor_start", 100, "Training step to monitor, exponentially increasing.")
	flagTrainMonitorFrequencyFactor = flag.Float64("monitor_factor", 1.2, "Training step to monitor, exponentially increasing.")
	flagTrainGeneratedSamples       = flag.Int("train_samples", 64, "Number of images to monitor progress on training.")
)

func TrainModel() {
	Init()
	*flagCheckpointTakeMean = 1 // Disable mean of checkpoints if training.
	trainDS, validationDS := CreateInMemoryDatasets()
	trainEvalDS := trainDS.Copy()

	trainDS.Shuffle().Infinite(true).BatchSize(BatchSize, true)
	trainEvalDS.BatchSize(EvalBatchSize, false)
	validationDS.BatchSize(EvalBatchSize, false)

	// Context holds the variables and hyperparameters for the model.
	ctx := context.NewContext()
	ctx.SetParam(optimizers.ParamLearningRate, *flagLearningRate)
	ctx.SetParam(layers.ParamL2Regularization, *flagL2Regularization)

	// Checkpoints saving.
	checkpoint, noise, flowerIds := LoadCheckpointToContext(backend, ctx)
	if noise == nil {
		klog.Exitf("A checkpoint directory name with --checkpoint is required, none given")
	}
	if *flagRngReset {
		ctx.RngStateReset()
	}
	globalStep := int(optimizers.GetGlobalStep(ctx))
	if globalStep != 0 {
		fmt.Printf("Restarting training from global_step=%d\n", globalStep)
	}
	if globalStep >= *flagNumSteps {
		klog.Exitf("Current global step %d >= target --steps=%d, exiting.", globalStep, *flagNumSteps)
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
		"Moving Images Loss", "~img_loss", "img_loss", imgMetricFn, pprintLossFn, 0.01)

	// Create a train.Trainer: this object will orchestrate running the model, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	trainer := train.NewTrainer(
		backend, ctx, TrainingModelGraph, customLoss,
		optimizers.Adam().WeightDecay(1e-4).Done(),
		[]metrics.Interface{movingImagesLoss}, // trainMetrics
		[]metrics.Interface{meanImagesLoss})   // evalMetrics
	if *flagNanLogger {
		trainer.OnExecCreation(func(exec *context.Exec, _ train.GraphType) {
			nanLogger.AttachToExec(exec)
		})
	}
	// Use standard training loop.
	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.

	// Attach a checkpoint.
	if checkpoint != nil {
		period := time.Second * time.Duration(*flagCheckpointPeriod)
		train.PeriodicCallback(loop, period, true, "saving checkpoint", 100,
			func(loop *train.Loop, metrics []*tensors.Tensor) error {
				fmt.Printf("\n[saving checkpoint@%d] [median train step (ms): %d]\n", loop.LoopStep, loop.MedianTrainStepDuration().Milliseconds())
				return checkpoint.Save()
			})
	}

	// Monitoring training: plotter, generator of images, kid evaluator.
	var plotter *margaid.Plots
	if *flagPlots {
		// No need to store the returned `plot.Plots` object, it attaches itself to the loop, and will
		// generate plot points and update the plot until the `loop` ends.
		plotter = margaid.New(1024, 400, trainEvalDS, validationDS).LogScaleX().LogScaleY()
		if checkpoint != nil {
			_, err := plotter.WithFile(path.Join(checkpoint.Dir(), "training_plot_points.json"))
			AssertNoError(err)
		}
		plotter.DynamicUpdates()
	}

	generator := NewImagesGenerator(ctx, noise, flowerIds, 20)

	var kid *KidGenerator
	if *flagKid {
		kidDS := validationDS.Copy()
		kidDS.Shuffle().BatchSize(EvalBatchSize, true)
		kid = NewKidGenerator(ctx, kidDS, 5)
	}

	train.ExponentialCallback(loop, *flagTrainMonitorStartFrequency, *flagTrainMonitorFrequencyFactor, true,
		"Monitor", 0, func(loop *train.Loop, metrics []*tensors.Tensor) error {
			return TrainingMonitor(checkpoint, loop, metrics, plotter, plotter.EvalDatasets, generator, kid)
		})

	// Loop for given number of steps.
	_, err := loop.RunSteps(trainDS, *flagNumSteps-globalStep)
	if err != nil {
		fmt.Printf("\nFailed: %v\n\n", err)
		return
	}
	// AssertNoError(err)
	fmt.Printf("\tMedian train step duration: %d ms\n\t(not counting evaluations)\n",
		loop.MedianTrainStepDuration().Milliseconds())

	// Finally, print an evaluation on train and test datasets.
	if *flagReport {
		fmt.Println()
		err = commandline.ReportEval(trainer, trainEvalDS, validationDS)
		AssertNoError(err)
		fmt.Println()
	}
}

// TrainingMonitor is periodically called during training, and is used to report metrics and generate sample images at
// the current training step.
func TrainingMonitor(checkpoint *checkpoints.Handler, loop *train.Loop, metrics []*tensors.Tensor,
	plotter stdplots.Plotter, evalDatasets []train.Dataset, generator *ImagesGenerator, kid *KidGenerator) error {

	fmt.Printf("\n[... evaluating@%d ...] [median train step (ms): %d]\n", loop.LoopStep, loop.MedianTrainStepDuration().Milliseconds())
	// Save checkpoint, just in case.
	if checkpoint == nil {
		// Only works if there is a model directory.
		return nil
	}
	AssertNoError(checkpoint.Save())

	// Update plotter with metrics.
	AssertNoError(stdplots.AddTrainAndEvalMetrics(plotter, loop, metrics, evalDatasets))

	// Kid generator
	if kid != nil {
		kidValue := kid.Eval()
		//fmt.Printf("\nKID=%f\n", kidValue.Value())
		plotter.AddPoint(stdplots.Point{MetricName: "KID", MetricType: "KID", Step: float64(loop.LoopStep), Value: shapes.ConvertTo[float64](kidValue.Value())})
	}
	plotter.DynamicSampleDone(false)

	// Generate intermediary images.
	images := generator.Generate()
	imagesPath := fmt.Sprintf("%s%07d.tensor", GeneratedSamplesPrefix, loop.LoopStep)
	imagesPath = path.Join(checkpoint.Dir(), imagesPath)
	AssertNoError(images.Save(imagesPath))
	return nil
}

// DisplayTrainingPlots simply display the training plots of a model, without any training.
func DisplayTrainingPlots() {
	Init()
	ctx := context.NewContext()
	checkpoint, _, _ := LoadCheckpointToContext(backend, ctx)
	if checkpoint == nil {
		fmt.Printf("You must set --checkpoint='model_sub_dir'!")
		return
	}

	plotter := margaid.New(1024, 400).LogScaleX().LogScaleY()
	_, err := plotter.WithFile(path.Join(checkpoint.Dir(), "training_plot_points.json"))
	AssertNoError(err)
	plotter.Plot()
}

// CompareModelPlots display several model metrics on the same plots.
func CompareModelPlots(modelNames ...string) {
	Init()
	plots := margaid.New(1024, 400).LogScaleX().LogScaleY()
	for _, modelName := range modelNames {
		modelPath := path.Join(DataDir, modelName, "training_plot_points.json")
		_, err := plots.PreloadFile(modelPath, func(metricName string) string {
			return fmt.Sprintf("[%s] %s", modelName, metricName)
		})
		AssertNoError(err)
	}
	plots.Plot()
}
