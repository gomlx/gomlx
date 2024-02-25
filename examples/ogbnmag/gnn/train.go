package gnn

import (
	"fmt"
	"github.com/gomlx/gomlx/examples/notebook/gonb/margaid"
	mag "github.com/gomlx/gomlx/examples/ogbnmag"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/checkpoints"
	mldata "github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/commandline"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/metrics"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"path"
	"time"
)

var (
	ParamCheckpointPath = "checkpoint"

	// ParamNumCheckpoints is the number of past checkpoints to keep.
	// The default is 10.
	ParamNumCheckpoints = "num_checkpointss"
)

// Train FNN model based on configuration in `ctx`.
func Train(ctx *context.Context, baseDir string) error {
	baseDir = mldata.ReplaceTildeInDir(baseDir)
	layers.ParamDropoutRate
	trainDS, trainEvalDS, validEvalDS, testEvalDS, err := MakeDatasets(baseDir)
	if err != nil {
		return err
	}
	mag.UploadOgbnMagVariables(ctx)

	// Context values (both parameters and variables) are reloaded from checkpoint,
	// any values that we don't want overwritten need to be read before the checkpointing.
	trainSteps := context.GetParamOr(ctx, "train_steps", 100)

	// Checkpoint: it loads if already exists, and it will save as we train.
	checkpointPath := context.GetParamOr(ctx, ParamCheckpointPath, "")
	numCheckpointsToKeep := context.GetParamOr(ctx, ParamNumCheckpoints, 10)
	var checkpoint *checkpoints.Handler
	if checkpointPath != "" {
		checkpointPath = mldata.ReplaceTildeInDir(checkpointPath) // If the path starts with "~", it is replaced.
		if !path.IsAbs(checkpointPath) {
			checkpointPath = path.Join(baseDir, checkpointPath)
		}
		var err error
		if numCheckpointsToKeep <= 1 {
			// Only limit the amount of checkpoints kept if >= 2.
			numCheckpointsToKeep = -1
		}
		if numCheckpointsToKeep > 0 {
			checkpoint, err = checkpoints.Build(ctx).Dir(checkpointPath).Keep(numCheckpointsToKeep).TakeMean(3).Done()
		} else {
			checkpoint, err = checkpoints.Build(ctx).Dir(checkpointPath).Done()
		}
		if err != nil {
			return errors.WithMessagef(err, "while setting up checkpoint to %q (keep=%d)",
				checkpointPath, numCheckpointsToKeep)
		}
		globalStep := optimizers.GetGlobalStep(ctx)
		if globalStep != 0 {
			fmt.Printf("> restarting training from global_step=%d\n", globalStep)
		}
	}

	// Create trainer and loop.
	trainer := newTrainer(ctx)
	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.

	// Attach a checkpoint: checkpoint every 1 minute of training.
	if checkpoint != nil && numCheckpointsToKeep > 1 {
		period := time.Minute * 1
		train.PeriodicCallback(loop, period, true, "saving checkpoint", 100,
			func(loop *train.Loop, metrics []tensor.Tensor) error {
				return checkpoint.Save()
			})
	}

	// Attach a margaid plots: plot points at exponential steps.
	// The points generated are saved along the checkpoint directory (if one is given).
	var plots *margaid.Plots
	usePlots := context.GetParamOr(ctx, "plots", false)
	if usePlots {
		plots = margaid.NewDefault(loop, checkpoint.Dir(), 100, 1.1, validEvalDS, testEvalDS, trainEvalDS).
			WithEvalLossType("eval-loss")
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
	err = commandline.ReportEval(trainer, trainEvalDS, validEvalDS, testEvalDS)
	if err != nil {
		return errors.WithMessage(err, "while reporting eval")
	}
	if plots != nil {
		// Save plot points.
		plots.Done()
	}
	fmt.Println()
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
	optimizer := optimizers.MustOptimizerByName(context.GetParamOr(ctx, "optimizer", "adamw"))
	optimizers.CosineAnnealingSchedule()
	trainer := train.NewTrainer(ctx.Manager(), ctx, MagModelGraph,
		lossFn,
		optimizer,
		[]metrics.Interface{movingAccuracyMetric}, // trainMetrics
		[]metrics.Interface{meanAccuracyMetric})   // evalMetrics
	return trainer
}

// Eval FNN model based on configuration in `ctx`.
func Eval(ctx *context.Context, baseDir string, datasets ...train.Dataset) error {
	baseDir = mldata.ReplaceTildeInDir(baseDir)
	mag.UploadOgbnMagVariables(ctx)

	// Load checkpoint.
	checkpointPath := context.GetParamOr(ctx, ParamCheckpointPath, "")
	numCheckpointsToKeep := context.GetParamOr(ctx, ParamNumCheckpoints, 10)
	if checkpointPath == "" {
		return errors.Errorf("No checkpoint defined in Context.GetParam(%q), please configure it to the checkpoint name")
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
		_, err = checkpoints.Build(ctx).Dir(checkpointPath).Keep(numCheckpointsToKeep).TakeMean(3).Done()
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
			return errors.WithMessagef(err, "while reporting eval on %q", ds.Name())
		}
		elapsed := time.Since(start)
		fmt.Printf("\telapsed %s (%s)\n", elapsed, ds.Name())
	}
	return nil
}
