package cifar

import (
	"fmt"
	"os"
	"slices"
	"time"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/checkpoints"
	"github.com/gomlx/gomlx/pkg/ml/layers/batchnorm"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gomlx/pkg/ml/train/metrics"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gomlx/ui/gonb/plotly"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
)

var (
	// DType used in the mode.
	DType = dtypes.Float32

	// C10ValidModels is the list of model types supported.
	C10ValidModels = []string{"fnn", "kan", "cnn"}

	// ParamsExcludedFromSaving is the list of parameters (see CreateDefaultContext) that shouldn't be saved
	// along on the models checkpoints, and may be overwritten in further training sessions.
	ParamsExcludedFromSaving = []string{
		"data_dir", "train_steps", "num_checkpoints", "plots",
	}
)

// Backend is created once and reused if train is called multiple times.
var Backend backends.Backend

// TrainCifar10Model with hyperparameters given in ctx.
func TrainCifar10Model(ctx *context.Context, dataDir, checkpointPath string, evaluateOnEnd bool, verbosity int, paramsSet []string) {
	// Data directory: datasets and top-level directory holding checkpoints for different models.
	dataDir = fsutil.MustReplaceTildeInDir(dataDir)
	if !fsutil.MustFileExists(dataDir) {
		must.M(os.MkdirAll(dataDir, 0777))
	}
	must.M(DownloadCifar10(dataDir))
	//must.M(DownloadCifar100(dataDir))

	// Backend handles creation of ML computation graphs, accelerator resources, etc.
	if Backend == nil {
		Backend = backends.MustNew()
	}
	if verbosity >= 1 {
		fmt.Printf("Backend %q:\t%s\n", Backend.Name(), Backend.Description())
	}

	// Create datasets used for training and evaluation.
	batchSize := context.GetParamOr(ctx, "batch_size", int(0))
	if batchSize <= 0 {
		exceptions.Panicf("Batch size must be > 0 (maybe it was not set?): %d", batchSize)
	}
	evalBatchSize := context.GetParamOr(ctx, "eval_batch_size", int(0))
	if evalBatchSize <= 0 {
		evalBatchSize = batchSize
	}
	trainDS, trainEvalDS, testEvalDS := CreateDatasets(Backend, dataDir, batchSize, evalBatchSize)

	// Checkpoints saving.
	var checkpoint *checkpoints.Handler
	if checkpointPath != "" {
		numCheckpointsToKeep := context.GetParamOr(ctx, "num_checkpoints", 3)
		checkpoint = must.M1(checkpoints.Build(ctx).
			DirFromBase(checkpointPath, dataDir).
			Keep(numCheckpointsToKeep).
			ExcludeParams(append(paramsSet, ParamsExcludedFromSaving...)...).
			Done())
		fmt.Printf("Checkpointing model to %q\n", checkpoint.Dir())
	}
	if verbosity >= 2 {
		fmt.Println(commandline.SprintContextSettings(ctx))
	}

	// Select model graph building function.
	modelFn, err := SelectModelFn(ctx)
	if err != nil {
		panic(err)
	}

	// Metrics we are interested.
	meanAccuracyMetric := metrics.NewSparseCategoricalAccuracy("Mean Accuracy", "#acc")
	movingAccuracyMetric := metrics.NewMovingAverageSparseCategoricalAccuracy("Moving Average Accuracy", "~acc", 0.01)

	// Create a train.Trainer: this object will orchestrate running the model, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	ctx = ctx.In("model") // Convention scope used for model creation.
	trainer := train.NewTrainer(Backend, ctx, modelFn,
		losses.SparseCategoricalCrossEntropyLogits,
		optimizers.FromContext(ctx),
		[]metrics.Interface{movingAccuracyMetric}, // trainMetrics
		[]metrics.Interface{meanAccuracyMetric})   // evalMetrics

	// Use standard training loop.
	loop := train.NewLoop(trainer)
	if verbosity >= 0 {
		commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.
	}

	// Checkpoint saving: every 3 minutes of training.
	if checkpoint != nil {
		period := time.Minute * 3
		train.PeriodicCallback(loop, period, true, "saving checkpoint", 100,
			func(loop *train.Loop, metrics []*tensors.Tensor) error {
				return checkpoint.Save()
			})
	}

	// Attach Plotly plots: plot points at exponential steps.
	// The points generated are saved along the checkpoint directory (if one is given).
	if context.GetParamOr(ctx, plotly.ParamPlots, false) {
		_ = plotly.New().
			WithCheckpoint(checkpoint).
			Dynamic().
			WithDatasets(trainEvalDS, testEvalDS).
			ScheduleExponential(loop, 200, 1.2).
			WithBatchNormalizationAveragesUpdate(trainEvalDS)
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
			if verbosity >= 1 {
				fmt.Println("\tUpdated batch normalization mean/variances averages.")
			}
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
		must.M(commandline.ReportEval(trainer, testEvalDS, trainEvalDS))
	}
}

// SelectModelFn based on hyperparameter "model" in Context.
func SelectModelFn(ctx *context.Context) (modelFn train.ModelFn, err error) {
	modelFn = C10PlainModelGraph // Handles all models except CNN.
	modelType := context.GetParamOr(ctx, "model", C10ValidModels[0])
	if slices.Index(C10ValidModels, modelType) == -1 {
		return nil, errors.Errorf("Parameter \"model\" must take one value from %v, got %q", C10ValidModels, modelType)
	}
	if modelType == "cnn" {
		modelFn = C10ConvolutionModelGraph
	}
	return modelFn, nil
}

func CreateDatasets(backend backends.Backend, dataDir string, batchSize, evalBatchSize int) (trainDS, trainEvalDS, validationEvalDS train.Dataset) {
	baseTrain := NewDataset(backend, "Training", dataDir, C10, DType, Train)
	baseTest := NewDataset(backend, "Validation", dataDir, C10, DType, Test)
	trainDS = baseTrain.Copy().BatchSize(batchSize, true).Shuffle().Infinite(true)
	trainEvalDS = baseTrain.BatchSize(evalBatchSize, false)
	validationEvalDS = baseTest.BatchSize(evalBatchSize, false)
	return
}
