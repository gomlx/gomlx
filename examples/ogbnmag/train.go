package ogbnmag

// Train and Eval functions.

import (
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/internal/must"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/nanlogger"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/checkpoints"
	"github.com/gomlx/gomlx/pkg/ml/layers/kan"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gomlx/pkg/ml/train/metrics"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gomlx/ui/gonb/margaid"
	"github.com/gomlx/gomlx/ui/gonb/plotly"
	"github.com/gomlx/gomlx/ui/notebooks"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
	"github.com/schollz/progressbar/v3"
	"k8s.io/klog/v2"
)

var (
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

// progressBarSpacing prints enough line breaks that whatever is printed in the output is not overwritten
// by the progressbar. This should be used if something is printed in the middle of the training.
func progressBarSpacing() {
	lineBreaks := 9
	if notebooks.IsNotebook() {
		lineBreaks = 1
	}
	fmt.Printf("%s", strings.Repeat("\n", lineBreaks))
}

// InitTrainingSchedule initializes custom scheduled training.
// It's enabled with the hyperparameter "scheduled_training".
func InitTrainingSchedule(ctx *context.Context) {
	// Start with split points frozen.
	ctx.SetParam(kan.ParamDiscreteSplitPointsFrozen, true)
}

// TrainingSchedule is used to control hyperparameters during training.
// The parameters fromStep and toStep are the starting and final global_steps of training.
// It's enabled with the hyperparameter "scheduled_training".
func TrainingSchedule(ctx *context.Context, fromStep, toStep int) train.OnStepFn {
	percentageToStep := func(pct int) int {
		return pct * (fromStep + toStep) / 100
	}
	splitPointsStart, splitPointsEnd := percentageToStep(10), percentageToStep(20)

	return func(loop *train.Loop, _ []*tensors.Tensor) error {
		if loop.LoopStep == splitPointsStart {
			time.Sleep(100 * time.Millisecond)
			fmt.Printf("\nResetting training computation graphs @ step=%d.\n", loop.LoopStep)
			loop.Trainer.ResetComputationGraphs()

			fmt.Println("\tTrain split_points, smoothness_schedule=none")
			ctx.SetParam(kan.ParamDiscreteSplitPointsFrozen, false)
			ctx.SetParam(kan.ParamDiscreteSoftnessSchedule, "none")
			ctx.EnumerateVariables(func(v *context.Variable) {
				if v.Name() == "kan_discrete_split_points" {
					v.Trainable = true
				} else if slices.Index([]string{"kan_discrete_control_points", "embeddings", "weights", "biases"}, v.Name()) != -1 {
					v.Trainable = true
				} else if v.Trainable {
					fmt.Printf("\t\t%q trainable\n", v.ScopeAndName())
				}
			})
			progressBarSpacing()

		} else if loop.LoopStep == splitPointsEnd {
			time.Sleep(100 * time.Millisecond)
			fmt.Printf("\nResetting training computation graphs @ step=%d.\n", loop.LoopStep)
			loop.Trainer.ResetComputationGraphs()

			fmt.Println("\tTrain control_points, freeze split_points, , smoothness_schedule=exponential")
			ctx.SetParam(kan.ParamDiscreteSoftnessSchedule, "exponential")
			ctx.EnumerateVariables(func(v *context.Variable) {
				if v.Name() == "kan_discrete_split_points" {
					v.Trainable = false
				} else if slices.Index([]string{"kan_discrete_control_points", "embeddings", "weights", "biases"}, v.Name()) != -1 {
					v.Trainable = true
				}
			})
			progressBarSpacing()

		}
		return nil
	}
}

// Train GNN model based on configuration in `ctx`.
func Train(
	backend backends.Backend,
	ctx *context.Context,
	dataDir, checkpointPath string,
	layerWiseEval, report bool,
	paramsSet []string,
) error {
	dataDir = fsutil.MustReplaceTildeInDir(dataDir)
	ReuseShareableKernels = context.GetParamOr(ctx, ParamReuseKernels, true)
	IdentitySubSeeds = context.GetParamOr(ctx, ParamIdentitySubSeeds, true)

	trainDS, trainEvalDS, validEvalDS, testEvalDS, err := MakeDatasets(dataDir)
	_ = testEvalDS
	if err != nil {
		return err
	}
	UploadOgbnMagVariables(backend, ctx)

	// Checkpoint: it loads if already exists, and it will save as we train.
	var checkpoint *checkpoints.Handler
	if checkpointPath != "" {
		numCheckpointsToKeep := context.GetParamOr(ctx, ParamNumCheckpoints, 5)
		var err error
		checkpoint, err = checkpoints.Build(ctx).
			DirFromBase(checkpointPath, dataDir).
			Keep(numCheckpointsToKeep).
			ExcludeParams(paramsSet...).
			Done()
		if err != nil {
			return errors.WithMessagef(err, "while setting up checkpoint to %q (keep=%d)",
				checkpointPath, numCheckpointsToKeep)
		}
		ExcludeOgbnMagVariablesFromSave(ctx, checkpoint)
	}

	// Create trainer and loop.
	trainer := newTrainer(backend, ctx)
	loop := train.NewLoop(trainer)
	commandline.ProgressbarStyle = progressbar.ThemeUnicode
	commandline.AttachProgressBar(loop)

	// Attach a checkpoint: checkpoint every 1 minute of training.
	if checkpoint != nil {
		period := time.Minute * 3
		train.PeriodicCallback(loop, period, true, "saving checkpoint", 100,
			func(loop *train.Loop, metrics []*tensors.Tensor) error {
				return checkpoint.Save()
			})
	}

	// Attach Plotly plots: plot points at exponential steps.
	// The points generated are saved along the checkpoint directory (if one is given).
	var plots *plotly.PlotConfig
	usePlots := context.GetParamOr(ctx, margaid.ParamPlots, false)
	if usePlots {
		stepsPerEpoch := TrainSplit.Shape().Size()/BatchSize + 1
		plots = plotly.New().WithCheckpoint(checkpoint).Dynamic().
			ScheduleExponential(loop, 200, 1.2).
			ScheduleEveryNSteps(loop, stepsPerEpoch)
		if layerWiseEval {
			magSampler := must.M1(NewSampler(dataDir))
			layerWiseStrategy := NewSamplerStrategy(magSampler, 1, nil)
			plots = plots.WithCustomMetricFn(BuildLayerWiseCustomMetricFn(backend, ctx, layerWiseStrategy))
		} else {
			plots = plots.WithDatasets(trainEvalDS, validEvalDS)
		}
	}

	// Loop for given number of steps
	trainSteps := context.GetParamOr(ctx, "train_steps", 100)
	globalStep := int(optimizers.GetGlobalStep(ctx))
	if trainSteps <= globalStep {
		fmt.Printf("> training already reached target train_steps=%d. To train further, set a number additional "+
			"to current global step. Use Eval to get reading on current performance.\n", trainSteps)
		return nil
	}
	if globalStep != 0 {
		fmt.Printf("> restarting training from global_step=%d (training until %d)\n", globalStep, trainSteps)
		ctx = ctx.Reuse()
	}

	// Set up scheduled training.
	if context.GetParamOr(ctx, "scheduled_training", false) {
		InitTrainingSchedule(ctx)
		loop.OnStep(
			"TrainingSchedule",
			0,
			TrainingSchedule(ctx, globalStep, trainSteps),
		) // register custom TrainingSchedule
	}

	// Run training loop.
	fmt.Println("Compiling graph... (once it's done, training immediately starts)")
	_, err = loop.RunSteps(trainDS, trainSteps-globalStep)
	// Save checkpoint at end of training (even if training failed)
	err2 := checkpoint.Save()
	if err2 != nil {
		klog.Errorf("Failed to save final checkpoint in %q: %+v", checkpointPath, err2)
	}
	fmt.Printf("\t[Step %d] median train step: %s\n", loop.LoopStep, loop.MedianTrainStepDuration())

	// Check whether training failed.
	if err != nil {
		return errors.WithMessage(err, "while running steps")
	}

	// Finally, print an evaluation on train and test datasets.
	if report {
		fmt.Println()
		err = evalWithContext(backend, ctx, dataDir, layerWiseEval, false)
		if err != nil {
			return errors.WithMessage(err, "while reporting eval")
		}
	}
	return nil
}

var NanLogger *nanlogger.NanLogger

func newTrainer(backend backends.Backend, ctx *context.Context) *train.Trainer {
	// Loss: multi-class classification problem.
	lossFn := losses.SparseCategoricalCrossEntropyLogits

	// Metrics we are interested.
	meanAccuracyMetric := metrics.NewSparseCategoricalAccuracy("Mean Accuracy", "#acc")
	movingAccuracyMetric := metrics.NewMovingAverageSparseCategoricalAccuracy("Moving Average Accuracy", "~acc", 0.01)

	// Create a train.Trainer: this object will orchestrate running the model, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	//NanLogger = nanlogger.New()
	trainer := train.NewTrainer(backend, ctx, MagModelGraph,
		lossFn,
		optimizers.FromContext(ctx), // Based on `ctx.GetParam("optimizer")`.
		[]metrics.Interface{movingAccuracyMetric}, // trainMetrics
		[]metrics.Interface{meanAccuracyMetric})   // evalMetrics
	NanLogger.AttachToTrainer(trainer)
	return trainer
}

func Eval(
	backend backends.Backend,
	ctx *context.Context,
	dataDir, checkpointPath string,
	layerWise, skipTrain bool,
) error {
	_, err := checkpoints.Build(ctx).DirFromBase(checkpointPath, dataDir).Done()
	if err != nil {
		return errors.WithMessagef(err, "while loading checkpoint from %q", checkpointPath)
	}

	// Model stats:
	globalStep := optimizers.GetGlobalStep(ctx)
	fmt.Printf("Model in %q trained for %d steps.\n", checkpointPath, globalStep)

	// Upload OGBN-MAG variables -- and possibly convert them.
	_ = UploadOgbnMagVariables(backend, ctx)

	return evalWithContext(backend, ctx, dataDir, layerWise, skipTrain)
}

func evalWithContext(backend backends.Backend, ctx *context.Context, baseDir string, layerWise, skipTrain bool) error {
	if layerWise {
		return evalLayerWise(backend, ctx, baseDir)
	}

	// Evaluate on various datasets.
	_, trainEvalDS, validEvalDS, testEvalDS := must.M4(MakeDatasets(baseDir))
	if skipTrain {
		return evalSampled(backend, ctx, validEvalDS, testEvalDS)
	} else {
		return evalSampled(backend, ctx, trainEvalDS, validEvalDS, testEvalDS)
	}
}

// evalSampled evaluates GNN model based on configuration in `ctx` using sampled sub-graphs.
func evalSampled(backend backends.Backend, ctx *context.Context, datasets ...train.Dataset) error {
	// Evaluation on the various eval datasets.
	trainer := newTrainer(backend, ctx)
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
func evalLayerWise(backend backends.Backend, ctx *context.Context, baseDir string) error {
	// Create the OGBN-MAG strategy, used by the layer-wise inference: batch-size is irrelevant.
	magSampler, err := NewSampler(baseDir)
	if err != nil {
		return err
	}
	magStrategy := NewSamplerStrategy(magSampler, 1, nil)
	trainAcc, validationAcc, testAcc := LayerWiseEvaluation(backend, ctx, magStrategy)
	fmt.Printf("Train Accuracy:     \t%.2f%%\n", 100*trainAcc)
	fmt.Printf("Validation Accuracy:\t%.2f%%\n", 100*validationAcc)
	fmt.Printf("Test Accuracy:      \t%.2f%%\n", 100*testAcc)
	fmt.Printf("Copy&paste version: \t%.2f%%,%.2f%%,%.2f%%", 100*trainAcc, 100*validationAcc, 100*testAcc)

	// Evaluation on the various eval datasets.
	return nil
}

// getDType returns the dtype selected in the context hyperparameters.
func getDType(ctx *context.Context) dtypes.DType {
	dtypeStr := context.GetParamOr(ctx, ParamDType, "float32")
	switch dtypeStr {
	case "float32":
		return dtypes.Float32
	case "float16":
		return dtypes.Float16
	case "bfloat16":
		return dtypes.BFloat16
	case "float64":
		return dtypes.Float64
	default:
		Panicf("Invalid DType %q given to parameters %q", dtypeStr, ParamDType)
	}
	return dtypes.InvalidDType
}

// convertPaperEmbeddings converts the "PapersEmbeddings" variable to the selected dtype, if needed.
//
// One should be careful not to save the converted values -- ideally, the values are saved in the original Float32.
func convertPapersEmbeddings(backend backends.Backend, ctx *context.Context) {
	dtype := getDType(ctx)
	dtypeEmbed := dtype
	if dtype == dtypes.Float16 || dtype == dtypes.BFloat16 {
		// See comment on model.go, in function FeaturePreprocessing.
		dtypeEmbed = dtypes.Float32
	}

	papersVar := ctx.GetVariableByScopeAndName(OgbnMagVariablesScope, "PapersEmbeddings")
	if papersVar == nil || papersVar.MustValue() == nil {
		Panicf("Cannot convert papers embeddings if variable \"PapersEmbeddings\" is not set yet")
		panic(nil) // Clear lint warning.
	}
	if papersVar.MustValue().DType() == dtypeEmbed {
		// Nothing to convert.
		return
	}

	e := context.MustNewExec(backend, ctx, func(ctx *context.Context, g *Graph) *Node {
		return ConvertDType(papersVar.ValueGraph(g), dtype)
	})
	converted := e.MustExec()[0]
	// We don't want to destroy the unconverted values in case we need them again (it happens in tests).
	must.M(papersVar.SetValuePreservingOld(converted))
	klog.V(1).Infof("Converted papers embeddings to %s: new shape is %s", dtype, papersVar.Shape())
}
