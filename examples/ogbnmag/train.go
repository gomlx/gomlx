package ogbnmag

// Train and Eval functions.

import (
	"fmt"
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/examples/notebook/gonb/margaid"
	"github.com/gomlx/gomlx/examples/notebook/gonb/plotly"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/checkpoints"
	mldata "github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/commandline"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/metrics"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/janpfeifer/must"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
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

	// ParamIdentitySubSeeds controls whether to use an IdentitySubSeed, to allow more sharing of the kernel.
	ParamIdentitySubSeeds = "mag_identity_sub_seeds"

	// ParamDType controls the dtype to be used: either "float32" or "float16".
	ParamDType = "mag_dtype"
)

// Train GNN model based on configuration in `ctx`.
func Train(ctx *context.Context, baseDir string, layerWiseEval, report bool) error {
	baseDir = mldata.ReplaceTildeInDir(baseDir)
	ReuseShareableKernels = context.GetParamOr(ctx, ParamReuseKernels, true)
	IdentitySubSeeds = context.GetParamOr(ctx, ParamIdentitySubSeeds, true)

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

		if numCheckpointsToKeep <= 1 {
			// Only limit the amount of checkpoints kept if >= 2.
			numCheckpointsToKeep = -1
		}
		if numCheckpointsToKeep > 0 {
			checkpoint, err = checkpoints.Build(ctx).Dir(checkpointPath).Keep(numCheckpointsToKeep).Done()
		} else {
			checkpoint, err = checkpoints.Build(ctx).Dir(checkpointPath).Done()
		}
		ExcludeOgbnMagVariablesFromSave(ctx, checkpoint)

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
			func(loop *train.Loop, metrics []tensors.Tensor) error {
				return checkpoint.Save()
			})
	}

	// Attach Plotly plots: plot points at exponential steps.
	// The points generated are saved along the checkpoint directory (if one is given).
	var plots *plotly.PlotConfig
	usePlots := context.GetParamOr(ctx, margaid.ParamPlots, false)
	if usePlots {
		stepsPerEpoch := TrainSplit.Shape().Size()/BatchSize + 1
		plots = plotly.New().Dynamic().
			ScheduleExponential(loop, 200, 1.2).
			ScheduleEveryNSteps(loop, stepsPerEpoch)
		if layerWiseEval {
			magSampler := must.M1(NewSampler(baseDir))
			layerWiseStrategy := NewSamplerStrategy(magSampler, 1, nil)
			plots = plots.WithCustomMetricFn(BuildLayerWiseCustomMetricFn(ctx, layerWiseStrategy))
		} else {
			plots = plots.WithDatasets(trainEvalDS, validEvalDS)
		}
		if checkpoint != nil {
			plots = plots.WithCheckpoint(checkpoint.Dir())
		}
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
	if report {
		fmt.Println()
		err = evalWithContext(ctx, baseDir, layerWiseEval, false)
		if err != nil {
			return errors.WithMessage(err, "while reporting eval")
		}
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
		optimizers.FromContext(ctx), // Based on `ctx.GetParam("optimizer")`.
		[]metrics.Interface{movingAccuracyMetric}, // trainMetrics
		[]metrics.Interface{meanAccuracyMetric})   // evalMetrics
	return trainer
}

func Eval(ctx *context.Context, baseDir string, layerWise, skipTrain bool) error {
	if err := loadCheckpointToContext(ctx, baseDir); err != nil {
		return err
	}
	return evalWithContext(ctx, baseDir, layerWise, skipTrain)
}

func evalWithContext(ctx *context.Context, baseDir string, layerWise, skipTrain bool) error {
	if layerWise {
		return evalLayerWise(ctx, baseDir)
	}

	// Evaluate on various datasets.
	_, trainEvalDS, validEvalDS, testEvalDS := must.M4(MakeDatasets(baseDir))
	if skipTrain {
		return evalSampled(ctx, validEvalDS, testEvalDS)
	} else {
		return evalSampled(ctx, trainEvalDS, validEvalDS, testEvalDS)
	}
}

// evalSampled evaluates GNN model based on configuration in `ctx` using sampled sub-graphs.
func evalSampled(ctx *context.Context, datasets ...train.Dataset) error {
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

// evalLayerWise evaluates GNN model based on configuration in `ctx` using layer-wise inference.
func evalLayerWise(ctx *context.Context, baseDir string) error {
	// Create the OGBN-MAG strategy, used by the layer-wise inference: batch-size is irrelevant.
	magSampler, err := NewSampler(baseDir)
	if err != nil {
		return err
	}
	magStrategy := NewSamplerStrategy(magSampler, 1, nil)
	trainAcc, validationAcc, testAcc := LayerWiseEvaluation(ctx, magStrategy)
	fmt.Printf("Train Accuracy:     \t%.2f%%\n", 100*trainAcc)
	fmt.Printf("Validation Accuracy:\t%.2f%%\n", 100*validationAcc)
	fmt.Printf("Test Accuracy:      \t%.2f%%\n", 100*testAcc)
	fmt.Printf("Copy&paste version: \t%.2f%%,%.2f%%,%.2f%%", 100*trainAcc, 100*validationAcc, 100*testAcc)

	// Evaluation on the various eval datasets.
	return nil
}

func loadCheckpointToContext(ctx *context.Context, baseDir string) error {
	baseDir = mldata.ReplaceTildeInDir(baseDir)
	checkpointPath := context.GetParamOr(ctx, ParamCheckpointPath, "")
	if checkpointPath == "" {
		return errors.Errorf("no checkpoint defined in Context.GetParam(%q), please configure it to the checkpoint directory",
			ParamCheckpointPath)
	}
	checkpointPath = mldata.ReplaceTildeInDir(checkpointPath) // If the path starts with "~", it is replaced.
	if !path.IsAbs(checkpointPath) {
		checkpointPath = path.Join(baseDir, checkpointPath)
	}
	_, err := checkpoints.Build(ctx).Dir(checkpointPath).Done()
	if err != nil {
		return errors.WithMessagef(err, "while loading checkpoint from %q", checkpointPath)
	}

	// Model stats:
	globalStep := optimizers.GetGlobalStep(ctx)
	fmt.Printf("Model in %q trained for %d steps.\n", checkpointPath, globalStep)

	// Upload OGBN-MAG variables -- and possibly convert them.
	_ = UploadOgbnMagVariables(ctx)
	return nil
}

// getDType returns the dtype selected in the context hyperparameters.
func getDType(ctx *context.Context) dtypes.DType {
	dtypeStr := context.GetParamOr(ctx, ParamDType, "float32")
	switch dtypeStr {
	case "float32":
		return dtypes.Float32
	case "float16":
		return shapes.F16
	case "float64":
		return dtypes.Float64
	default:
		Panicf("Invalid DType %q given to parameters %q", dtypeStr, ParamDType)
	}
	return shapes.InvalidDType
}

// convertPaperEmbeddings converts the "PapersEmbeddings" variable to the selected dtype, if needed.
//
// One should be careful not to save the converted values -- ideally, the values are saved in the original Float32.
func convertPapersEmbeddings(ctx *context.Context) {
	dtype := getDType(ctx)
	dtypeEmbed := dtype
	if dtype == dtypes.Float16 {
		// See comment on model.go, in function FeaturePreprocessing.
		dtypeEmbed = dtypes.Float32
	}

	papersVar := ctx.InspectVariable(OgbnMagVariablesScope, "PapersEmbeddings")
	if papersVar == nil || papersVar.Value() == nil {
		Panicf("Cannot convert papers embeddings if variable \"PapersEmbeddings\" is not set yet")
		panic(nil) // Clear lint warning.
	}
	if papersVar.Value().DType() == dtypeEmbed {
		// Nothing to convert.
		return
	}

	e := context.NewExec(ctx.Manager(), ctx, func(ctx *context.Context, g *Graph) *Node {
		return ConvertDType(papersVar.ValueGraph(g), dtype)
	})
	converted := e.Call()[0]
	papersVar.SetValuePreservingOld(converted) // We don't want to destroy the unconverted values, in case we need it again (it happens in tests).
	klog.V(1).Infof("Converted papers embeddings to %s: new shape is %s", dtype, papersVar.Shape())
}
