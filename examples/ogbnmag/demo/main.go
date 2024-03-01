package main

import (
	"flag"
	"fmt"
	mag "github.com/gomlx/gomlx/examples/ogbnmag"
	"github.com/gomlx/gomlx/examples/ogbnmag/gnn"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	mldata "github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"
	"path"
	"time"
)

var (
	flagEval             = flag.Bool("eval", false, "Set to true to run evaluation instead of training.")
	flagDataDir          = flag.String("data", "~/work/ogbnmag", "Directory to cache downloaded and generated dataset files.")
	flagCheckpointSubdir = flag.String("checkpoint", "", "Checkpoint subdirectory under --data directory. If empty does not use checkpoints.")
	manager              = NewManager()
)

func configGnn(baseDir string) *context.Context {
	ctx := context.NewContext(manager)
	ctx.RngStateReset()

	stepsPerEpoch := mag.TrainSplit.Shape().Size()/mag.BatchSize + 1
	numEpochs := 10 // Taken from TF-GNN OGBN-MAG notebook.
	numTrainSteps := numEpochs * stepsPerEpoch
	checkpointPath := mldata.ReplaceTildeInDir(*flagCheckpointSubdir)
	if checkpointPath != "" && !path.IsAbs(checkpointPath) {
		checkpointPath = path.Join(baseDir, checkpointPath)
	}

	fmt.Printf("checkpoint=%s\n", checkpointPath)

	ctx.SetParams(map[string]any{
		"checkpoint":      checkpointPath,
		"num_checkpoints": 3,

		"train_steps":                       numTrainSteps,
		optimizers.ParamOptimizer:           "adam",
		optimizers.ParamLearningRate:        0.001,
		optimizers.ParamCosineScheduleSteps: numTrainSteps,
		layers.ParamL2Regularization:        1e-5,
		layers.ParamDropoutRate:             0.2,
		gnn.ParamEdgeDropoutRate:            0.0,
		gnn.ParamNumMessages:                4,
		gnn.ParamReadoutHiddenLayers:        2,
		mag.ParamEmbedDropoutRate:           0.0,
		gnn.ParamPoolingType:                "mean|sum",
		gnn.ParamUsePathToRootStates:        false,
		gnn.ParamGraphUpdateType:            "simultaneous",
		"plots":                             true,
	})
	return ctx
}
func main() {
	flag.Parse()
	*flagDataDir = mldata.ReplaceTildeInDir(*flagDataDir)

	fmt.Printf("Loading data ... ")
	start := time.Now()
	must.M(mag.Download(*flagDataDir))
	fmt.Printf("elapsed: %s\n", time.Since(start))

	ctx := configGnn(*flagDataDir)
	checkpointPath := context.GetParamOr(ctx, "checkpoint", "")
	if checkpointPath != "" {
		fmt.Printf("Model checkpoints in %s\n", checkpointPath)
	} else if *flagEval {
		klog.Fatal("To run eval (--eval) you need to specify a checkpoint (--checkpoint).")
	}

	var err error
	if *flagEval {
		// Evaluate on various datasets.
		_, trainEvalDS, validEvalDS, testEvalDS := must.M4(mag.MakeDatasets(*flagDataDir))
		_, _, _ = trainEvalDS, validEvalDS, testEvalDS
		err = mag.Eval(ctx, *flagDataDir, trainEvalDS, validEvalDS, testEvalDS)
	} else {
		// Train.
		err = mag.Train(ctx, *flagDataDir)
	}
	if err != nil {
		fmt.Printf("%+v\n", err)
	}
}
