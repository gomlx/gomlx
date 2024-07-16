package main

import (
	"flag"
	"fmt"
	"github.com/gomlx/gomlx/backends"
	mag "github.com/gomlx/gomlx/examples/ogbnmag"
	"github.com/gomlx/gomlx/examples/ogbnmag/gnn"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	mldata "github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/train/commandline"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"
	"path"
	"time"
)

var (
	flagEval             = flag.Bool("eval", false, "Set to true to run evaluation instead of training.")
	flagSkipReport       = flag.Bool("skip_report", false, "Set to true to skip report of quality after training.")
	flagSkipTrainEval    = flag.Bool("skip_train_eval", false, "Set to true to skip evaluation on training data, which takes longer.")
	flagDataDir          = flag.String("data", "~/work/ogbnmag", "Directory to cache downloaded and generated dataset files.")
	flagCheckpointSubdir = flag.String("checkpoint", "", "Checkpoint subdirectory under --data directory. If empty does not use checkpoints.")
	flagLayerWise        = flag.Bool("layerwise", true, "Whether to use Layer-Wise inference for evaluation -- default is true.")
)

const paramWithReplacement = "mag_with_replacement"

func createDefaultContext(manager *Manager) *context.Context {
	ctx := context.NewContext(manager)
	ctx.RngStateReset()
	ctx.SetParams(map[string]any{
		"checkpoint":         "",
		"num_checkpoints":    3,
		"train_steps":        0,
		"plots":              true,
		paramWithReplacement: false,

		optimizers.ParamOptimizer:           "adam",
		optimizers.ParamLearningRate:        0.001,
		optimizers.ParamCosineScheduleSteps: 0,
		optimizers.ParamClipStepByValue:     0.0,
		optimizers.ParamAdamEpsilon:         1e-7,
		optimizers.ParamAdamDType:           "",

		layers.ParamL2Regularization: 1e-5,
		layers.ParamDropoutRate:      0.2,
		layers.ParamActivation:       "swish",

		gnn.ParamEdgeDropoutRate:       0.0,
		gnn.ParamNumGraphUpdates:       6, // gnn_num_messages
		gnn.ParamReadoutHiddenLayers:   2,
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
	must.M(commandline.ParseContextSettings(ctx, *settings))

	// Set checkpoint accordingly.
	*flagDataDir = mldata.ReplaceTildeInDir(*flagDataDir)
	checkpointPath := mldata.ReplaceTildeInDir(*flagCheckpointSubdir)
	if checkpointPath != "" && !path.IsAbs(checkpointPath) {
		checkpointPath = path.Join(*flagDataDir, checkpointPath)
	}
	if checkpointPath != "" {
		ctx.SetParam("checkpoint", checkpointPath)
	} else {
		checkpointPath = context.GetParamOr(ctx, "checkpoint", "")
	}
	if checkpointPath != "" {
		fmt.Printf("Model checkpoints in %s\n", checkpointPath)
	} else if *flagEval {
		klog.Fatal("To run eval (--eval) you need to specify a checkpoint (--checkpoint).")
	}

	// Load data from OGBN-MAG.
	fmt.Printf("Loading data ... ")
	start := time.Now()
	must.M(mag.Download(*flagDataDir))
	fmt.Printf("elapsed: %s\n", time.Since(start))
	SetTrainSteps(ctx) // Can only be set after mag data is loaded.

	// Run train / eval.
	mag.WithReplacement = context.GetParamOr(ctx, paramWithReplacement, false)
	var err error
	if *flagEval {
		err = mag.Eval(ctx, *flagDataDir, *flagLayerWise, *flagSkipTrainEval)
	} else {
		if mag.WithReplacement {
			fmt.Println("Training dataset with replacement")
		}

		// Train.
		err = mag.Train(ctx, *flagDataDir, *flagLayerWise, !*flagSkipReport)
	}
	if err != nil {
		fmt.Printf("%+v\n", err)
	}
}
