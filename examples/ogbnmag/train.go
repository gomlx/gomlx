package ogbnmag

import (
	"fmt"
	"github.com/gomlx/gomlx/examples/notebook/gonb/margaid"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/checkpoints"
	mldata "github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/commandline"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/metrics"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/pkg/errors"
	"path"
	"time"
)

var (
	ParamCheckpointPath = "checkpoint"

	// ParamNumCheckpoints is the number of past checkpoints to keep.
	// The default is 10.
	ParamNumCheckpoints = "num_checkpoints"

	// ParamReuseKernels context parameter configs whether the kernels for similar sampling rules will be reused.
	ParamReuseKernels = "mag_reuse_kernels"
)

// Train GNN model based on configuration in `ctx`.
func Train(ctx *context.Context, baseDir string) error {
	baseDir = mldata.ReplaceTildeInDir(baseDir)
	ReuseShareableKernels = context.GetParamOr(ctx, ParamReuseKernels, true)

	trainDS, trainEvalDS, validEvalDS, testEvalDS, err := MakeDatasets(baseDir)
	_ = testEvalDS
	if err != nil {
		return err
	}
	UploadOgbnMagVariables(ctx)

	// Context values (both parameters and variables) are reloaded from checkpoint,
	// any values that we don't want overwritten need to be read before the checkpointing.
	trainSteps := context.GetParamOr(ctx, "train_steps", 100)

	// Checkpoint: it loads if already exists, and it will save as we train.
	checkpointPath := context.GetParamOr(ctx, ParamCheckpointPath, "")
	numCheckpointsToKeep := context.GetParamOr(ctx, ParamNumCheckpoints, 5)
	var checkpoint *checkpoints.Handler
	if checkpointPath != "" {
		checkpointPath = mldata.ReplaceTildeInDir(checkpointPath) // If the path starts with "~", it is replaced.
		if !path.IsAbs(checkpointPath) {
			checkpointPath = path.Join(baseDir, checkpointPath)
		}
		var err error

		// Exclude from saving all the variables created by the `mag` package -- specially the frozen papers embeddings,
		// which take most space.
		var varsToExclude []*context.Variable
		ctx.InAbsPath(OgbnMagVariablesScope).EnumerateVariablesInScope(func(v *context.Variable) {
			varsToExclude = append(varsToExclude, v)
		})

		if numCheckpointsToKeep <= 1 {
			// Only limit the amount of checkpoints kept if >= 2.
			numCheckpointsToKeep = -1
		}
		if numCheckpointsToKeep > 0 {
			checkpoint, err = checkpoints.Build(ctx).Dir(checkpointPath).Keep(numCheckpointsToKeep).
				ExcludeVarsFromSaving(varsToExclude...).Done()
		} else {
			checkpoint, err = checkpoints.Build(ctx).Dir(checkpointPath).
				ExcludeVarsFromSaving(varsToExclude...).Done()
		}
		if err != nil {
			return errors.WithMessagef(err, "while setting up checkpoint to %q (keep=%d)",
				checkpointPath, numCheckpointsToKeep)
		}
		globalStep := optimizers.GetGlobalStep(ctx)
		if globalStep != 0 {
			fmt.Printf("> restarting training from global_step=%d (training until %d)\n", globalStep, trainSteps)
		}
		if trainSteps <= int(globalStep) {
			fmt.Printf("> training already reached target train_steps=%d. To train further, set a number additional "+
				"to current global step. Use Eval to get reading on current performance.\n", trainSteps)
			return nil
		}
		trainSteps -= int(globalStep)
	}

	// Create trainer and loop.
	trainer := newTrainer(ctx)
	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.

	// Attach a checkpoint: checkpoint every 1 minute of training.
	if checkpoint != nil && numCheckpointsToKeep > 1 {
		period := time.Minute * 3
		train.PeriodicCallback(loop, period, true, "saving checkpoint", 100,
			func(loop *train.Loop, metrics []tensor.Tensor) error {
				return checkpoint.Save()
			})
	}

	// Attach a margaid plots: plot points at exponential steps.
	// The points generated are saved along the checkpoint directory (if one is given).
	var plots *margaid.Plots
	usePlots := context.GetParamOr(ctx, margaid.ParamPlots, false)
	if usePlots {
		plots = margaid.NewDefault(loop, checkpoint.Dir(), 200, 1.2, trainEvalDS, validEvalDS).
			WithEvalLossType("eval-loss")
		stepsPerEpoch := TrainSplit.Shape().Size()/BatchSize + 1
		plots.PlotEveryNSteps(loop, stepsPerEpoch)
	}

	// Loop for given number of steps
	_, err = loop.RunSteps(trainDS, trainSteps)
	if err != nil {
		return errors.WithMessage(err, "while running steps")
	}
	fmt.Printf("\t[Step %d] median train step: %d microseconds\n",
		loop.LoopStep, loop.MedianTrainStepDuration().Microseconds())
	if checkpoint != nil && numCheckpointsToKeep <= 1 {
		// Save checkpoint at end of training.
		err = checkpoint.Save()
		if err != nil {
			klog.Errorf("Failed to save final checkpoint in %q: %+v", checkpointPath, err)
		}
	}
	fmt.Printf("Median training step duration: %s\n", loop.MedianTrainStepDuration())

	// Finally, print an evaluation on train and test datasets.
	fmt.Println()
	err = commandline.ReportEval(trainer, validEvalDS, trainEvalDS)
	if err != nil {
		return errors.WithMessage(err, "while reporting eval")
	}
	return nil
}

func newTrainer(ctx *context.Context) *train.Trainer {
	// Loss: multi-class classification problem.
	lossFn := losses.SparseCategoricalCrossEntropyLogits

	// Metrics we are interested.
	meanAccuracyMetric := metrics.NewSparseCategoricalAccuracy("Mean Accuracy", "#acc")
	movingAccuracyMetric := metrics.NewMovingAverageSparseCategoricalAccuracy("Moving Average Accuracy", "~acc", 0.01)

	// Create a train.Trainer: this object will orchestrate running the model, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	trainer := train.NewTrainer(ctx.Manager(), ctx, MagModelGraph,
		lossFn,
		optimizers.FromContext(ctx),               // Based on `ctx.GetParam("optimizer")`.
		[]metrics.Interface{movingAccuracyMetric}, // trainMetrics
		[]metrics.Interface{meanAccuracyMetric}) // evalMetrics
	return trainer
}

// Eval GNN model based on configuration in `ctx`.
func Eval(ctx *context.Context, baseDir string, datasets ...train.Dataset) error {
	baseDir = mldata.ReplaceTildeInDir(baseDir)
	UploadOgbnMagVariables(ctx)

	// Load checkpoint.
	checkpointPath := context.GetParamOr(ctx, ParamCheckpointPath, "")
	numCheckpointsToKeep := context.GetParamOr(ctx, ParamNumCheckpoints, 10)
	if checkpointPath == "" {
		return errors.Errorf("no checkpoint defined in Context.GetParam(%q), please configure it to the checkpoint directory",
			ParamCheckpointPath)
	}
	checkpointPath = mldata.ReplaceTildeInDir(checkpointPath) // If the path starts with "~", it is replaced.
	if !path.IsAbs(checkpointPath) {
		checkpointPath = path.Join(baseDir, checkpointPath)
	}
	if numCheckpointsToKeep <= 1 {
		// Only limit the amount of checkpoints kept if >= 2.
		numCheckpointsToKeep = -1
	}
	var err error
	if numCheckpointsToKeep > 0 {
		_, err = checkpoints.Build(ctx).Dir(checkpointPath).Keep(numCheckpointsToKeep).Done()
	} else {
		_, err = checkpoints.Build(ctx).Dir(checkpointPath).Done()
	}
	if err != nil {
		return errors.WithMessagef(err, "while loading checkpoint from %q (%s=%d)",
			checkpointPath, ParamNumCheckpoints, numCheckpointsToKeep)
	}

	// Model stats:
	globalStep := optimizers.GetGlobalStep(ctx)
	fmt.Printf("Model in %q trained for %d steps.\n", checkpointPath, globalStep)

	// Evaluation on the various eval datasets.
	trainer := newTrainer(ctx)
	for _, ds := range datasets {
		start := time.Now()
		err := commandline.ReportEval(trainer, ds)
		if err != nil {
			return errors.WithMessagef(err, "while reporting eval on %q: %+v", ds.Name(), err)
		}
		elapsed := time.Since(start)
		fmt.Printf("\telapsed %s (%s)\n", elapsed, ds.Name())
	}
	return nil
}
