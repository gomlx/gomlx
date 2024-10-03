package main

import (
	"flag"
	"fmt"
	"github.com/gomlx/gomlx/backends"
	mag "github.com/gomlx/gomlx/examples/ogbnmag"
	"github.com/gomlx/gomlx/examples/ogbnmag/gnn"
	"github.com/gomlx/gomlx/ml/context"
	mldata "github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activations"
	"github.com/gomlx/gomlx/ml/layers/kan"
	"github.com/gomlx/gomlx/ml/layers/regularizers"
	"github.com/gomlx/gomlx/ml/train/commandline"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"
	"time"

	_ "github.com/gomlx/gomlx/backends/xla"
)

var (
	flagEval          = flag.Bool("eval", false, "Set to true to run evaluation instead of training.")
	flagSkipReport    = flag.Bool("skip_report", false, "Set to true to skip report of quality after training.")
	flagSkipTrainEval = flag.Bool("skip_train_eval", false, "Set to true to skip evaluation on training data, which takes longer.")
	flagDataDir       = flag.String("data", "~/work/ogbnmag", "Directory to cache downloaded and generated dataset files.")
	flagCheckpoint    = flag.String("checkpoint", "", "Checkpoint subdirectory under --data directory. If empty does not use checkpoints.")
	flagLayerWise     = flag.Bool("layerwise", true, "Whether to use Layer-Wise inference for evaluation -- default is true.")
)

const paramWithReplacement = "mag_with_replacement"

func createDefaultContext() *context.Context {
	ctx := context.New()
	ctx.RngStateReset()
	ctx.SetParams(map[string]any{
		"checkpoint":         "",
		"num_checkpoints":    3,
		"train_steps":        0,
		"batch_size":         128,
		"plots":              true,
		paramWithReplacement: false,

		// KAN network parameters:
		// They are used only for readout layers -- they have worked very poorly for deeper networks.
		// Also, one should decrease the learning rate to use them.
		"kan":                                 false, // Enable kan
		kan.ParamNumControlPoints:             6,     // Number of control points for B-Spline (default) KAN.
		kan.ParamDiscrete:                     false,
		kan.ParamDiscretePerturbation:         "triangular",
		kan.ParamDiscreteNumControlPoints:     20,
		kan.ParamDiscreteSplitPointsTrainable: true, // Discrete-KAN trainable split-points.
		kan.ParamDiscreteSplitsMargin:         0.01, // Discrete-KAN trainable split-points margin.
		kan.ParamDiscreteSoftness:             0.1,  // Discrete-KAN softness
		kan.ParamDiscreteSoftnessSchedule:     kan.SoftnessScheduleNone.String(),
		kan.ParamResidual:                     true,
		kan.ParamConstantRegularizationL1:     0.0,

		// Experimental GR-KAN version, using KAN with rational functions as univariate learnable functions.
		"grkan":                  false, // Enable GR-Kan
		"grkan_num_input_groups": 4,     // Number of input groups, set to 0 to disable.

		optimizers.ParamOptimizer:           "adam",
		optimizers.ParamLearningRate:        0.001,
		optimizers.ParamCosineScheduleSteps: 0,
		optimizers.ParamClipStepByValue:     0.0,
		optimizers.ParamAdamEpsilon:         1e-7,
		optimizers.ParamAdamDType:           "float32",
		optimizers.ParamClipNaN:             false,

		regularizers.ParamL2:        1e-5,
		layers.ParamDropoutRate:     0.2,
		activations.ParamActivation: "swish",

		gnn.ParamEdgeDropoutRate:       0.0,
		gnn.ParamNumGraphUpdates:       6, // gnn_num_messages
		gnn.ParamPoolingType:           "mean|logsum",
		gnn.ParamUpdateStateType:       "residual",
		gnn.ParamUsePathToRootStates:   false,
		gnn.ParamGraphUpdateType:       "simultaneous",
		gnn.ParamUpdateNumHiddenLayers: 0,
		gnn.ParamMessageDim:            32, // 128 or 256 will work better, but takes way more time
		gnn.ParamStateDim:              32, // 128 or 256 will work better, but takes way more time
		gnn.ParamUseRootAsContext:      false,

		mag.ParamEmbedDropoutRate:     0.0,
		mag.ParamSplitEmbedTablesSize: 1,
		mag.ParamReuseKernels:         true,
		mag.ParamIdentitySubSeeds:     true,
		mag.ParamDType:                "float32",
	})
	ctx.In("readout").SetParam(gnn.ParamUpdateNumHiddenLayers, 2)
	return ctx
}

func SetTrainSteps(ctx *context.Context) {
	numTrainSteps := context.GetParamOr(ctx, "train_steps", 0)
	if numTrainSteps <= 0 {
		stepsPerEpoch := mag.TrainSplit.Shape().Size()/mag.BatchSize + 1
		numEpochs := 10 // Taken from TF-GNN OGBN-MAG notebook.
		numTrainSteps = numEpochs * stepsPerEpoch
		ctx.SetParam("train_steps", numTrainSteps)
	}
	cosineScheduleSteps := context.GetParamOr(ctx, optimizers.ParamCosineScheduleSteps, 0)
	if cosineScheduleSteps == 0 {
		ctx.SetParam(optimizers.ParamCosineScheduleSteps, numTrainSteps)
	}
}

func main() {
	// Init GoMLX manager and default context.
	backend := backends.New()
	ctx := createDefaultContext()

	// Flags with context settings.
	settings := commandline.CreateContextSettingsFlag(ctx, "")
	klog.InitFlags(nil)
	flag.Parse()
	paramsSet := must.M1(commandline.ParseContextSettings(ctx, *settings))

	// Set checkpoint accordingly.
	*flagDataDir = mldata.ReplaceTildeInDir(*flagDataDir)
	if *flagCheckpoint == "" && *flagEval {
		klog.Fatal("To run eval (--eval) you need to specify a checkpoint (--checkpoint).")
	}

	mag.BatchSize = context.GetParamOr(ctx, "batch_size", 128)

	// Load data from OGBN-MAG.
	fmt.Printf("Loading data ... ")
	start := time.Now()
	must.M(mag.Download(*flagDataDir))
	fmt.Printf("elapsed: %s\n", time.Since(start))
	SetTrainSteps(ctx) // Can only be set after mag data is loaded.

	// RunWithMap train / eval.
	mag.WithReplacement = context.GetParamOr(ctx, paramWithReplacement, false)
	var err error
	if *flagEval {
		err = mag.Eval(backend, ctx, *flagDataDir, *flagCheckpoint, *flagLayerWise, *flagSkipTrainEval)
	} else {
		if mag.WithReplacement {
			fmt.Println("Training dataset with replacement")
		}

		// Train.
		err = mag.Train(backend, ctx, *flagDataDir, *flagCheckpoint, *flagLayerWise, !*flagSkipReport, paramsSet)
	}
	if err != nil {
		fmt.Printf("%+v\n", err)
	}
}
