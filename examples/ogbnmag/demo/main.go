// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/gomlx/compute"
	mag "github.com/gomlx/gomlx/examples/ogbnmag"
	"github.com/gomlx/gomlx/examples/ogbnmag/gnn"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/layers/kan"
	"github.com/gomlx/gomlx/ml/layers/regularizer"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/gomlx/gomlx/ml/train/optimizer/cosineschedule"
	"github.com/gomlx/gomlx/support/fsutil"
	"github.com/gomlx/gomlx/ui/commandline"
	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/default"
)

var (
	flagVerbosity     = flag.Int("verbosity", 1, "Level of verbosity, the higher the more verbose.")
	flagEval          = flag.Bool("eval", false, "Set to true to run evaluation instead of training.")
	flagSkipReport    = flag.Bool("skip_report", false, "Set to true to skip report of quality after training.")
	flagSkipTrainEval = flag.Bool("skip_train_eval", false, "Set to true to skip evaluation on training data, which takes longer.")
	flagDataDir       = flag.String("data", "~/work/ogbnmag", "Directory to cache downloaded and generated dataset files.")
	flagCheckpoint    = flag.String("checkpoint", "", "Checkpoint subdirectory under --data directory. If empty does not use checkpoint.")
	flagLayerWise     = flag.Bool("layerwise", true, "Whether to use Layer-Wise inference for evaluation -- default is true.")
)

const paramWithReplacement = "mag_with_replacement"

func createModelStore() *model.Store {
	store := model.NewStore()
	store.ResetRNGState()
	store.SetParams(map[string]any{
		"checkpoint":         "",
		"num_checkpoints":    3,
		"train_steps":        1000,
		"batch_size":         128,
		"plots":              true,
		paramWithReplacement: false,
		"scheduled_training": false,

		// KAN network parameters:
		// They are used only for readout layers -- they have worked very poorly for deeper networks.
		// Also, one should decrease the learning rate to use them.
		"kan":                             false, // Enable kan
		kan.ParamNumControlPoints:         10,    // Number of control points for KAN.
		kan.ParamInputGroupSize:           1,
		kan.ParamResidual:                 true,
		kan.ParamConstantRegularizationL1: 0.0,

		// Discrete-KAN exclusive parameters.
		kan.ParamDiscrete:                     false,
		kan.ParamDiscretePerturbation:         "normal",
		kan.ParamDiscreteSplitPointsTrainable: false, // Discrete-KAN trainable split-points.
		kan.ParamDiscreteSplitPointsFrozen:    false, // Discrete-KAN trainable split-points should be frozen.
		kan.ParamDiscreteSplitsMargin:         0.1,   // Discrete-KAN trainable split-points margin.
		kan.ParamDiscreteSoftness:             0.03,  // Discrete-KAN softness
		kan.ParamDiscreteSoftnessSchedule:     kan.SoftnessScheduleExponential.String(),

		// GR-KAN (rational functions KAN) exclusive parameters.
		kan.ParamRational:                     false, // Enable GR-Kan
		kan.ParamRationalNumeratorDegree:      5,
		kan.ParamRationalDenominatorDegree:    4,
		kan.ParamRationalInitialApproximation: "identity",

		// PWL-KAN (Piecewise-Linear KAN) exclusive parameters.
		kan.ParamPiecewiseLinear:         false,
		kan.ParamPWLExtrapolate:          false,
		kan.ParamPWLSplitPointsTrainable: false,

		// Optimizer parameters.
		optimizer.ParamOptimizer:            "adamw",
		optimizer.ParamLearningRate:         0.001,
		cosineschedule.ParamPeriodSteps:     0, // If 0, it is disabled. Mutually exclusive with "cosine_schedule_cycles".
		cosineschedule.ParamCycles:          0, // If 0, it is disabled. Mutually exclusive with "cosine_schedule_steps".
		cosineschedule.ParamMinLearningRate: 0.0,
		optimizer.ParamClipStepByValue:      0.0,
		optimizer.ParamAdamEpsilon:          1e-7,
		optimizer.ParamAdamDType:            "float32",
		optimizer.ParamClipNaN:              false,

		regularizer.ParamL2:        1e-5,
		layers.ParamDropoutRate:    0.2,
		activation.ParamActivation: "swish",

		gnn.ParamEdgeDropoutRate:       0.0,
		gnn.ParamNumGraphUpdates:       6, // gnn_num_messages
		gnn.ParamPoolingType:           "mean|logsum",
		gnn.ParamUpdateStateType:       "residual",
		gnn.ParamUsePathToRootStates:   false,
		gnn.ParamGraphUpdateType:       "simultaneous",
		gnn.ParamUpdateNumHiddenLayers: 0,
		gnn.ParamMessageDim:            32, // 128 or 256 will work better, but takes way more time
		gnn.ParamStateDim:              32, // 128 or 256 will work better, but takes way more time
		gnn.ParamUseRootAsScope:        false,
		gnn.ParamNoKanForLayers:        "",

		mag.ParamEmbedDropoutRate:     0.0,
		mag.ParamSplitEmbedTablesSize: 1,
		mag.ParamReuseKernels:         true,
		mag.ParamIdentitySubSeeds:     true,
		mag.ParamDType:                "float32",
	})
	store.Scope("readout").SetParam(gnn.ParamUpdateNumHiddenLayers, 2)
	return store
}

func SetTrainSteps(store *model.Store) {
	numTrainSteps := model.GetParamOr(store.RootScope(), "train_steps", 0)
	if numTrainSteps <= 0 {
		stepsPerEpoch := mag.TrainSplit.Shape().Size()/mag.BatchSize + 1
		numEpochs := 10 // Taken from TF-GNN OGBN-MAG notebook.
		numTrainSteps = numEpochs * stepsPerEpoch
		store.RootScope().SetParam("train_steps", numTrainSteps)
	}
	//cosineScheduleSteps := model.GetParamOr(scope, cosineschedule.ParamPeriodSteps, 0)
	//if cosineScheduleSteps < 0 {
	//	scope.SetParam(cosineschedule.ParamPeriodSteps, numTrainSteps/(-cosineScheduleSteps))
	//}
}

func main() {
	// Init GoMLX manager and default model.
	backend := compute.MustNew()
	store := createModelStore()

	// Flags with scope settings.
	settings := commandline.CreateSettingsFlag(store, "")
	klog.InitFlags(nil)
	flag.Parse()

	// Change current directory to data directory.
	*flagDataDir = fsutil.MustReplaceTildeInDir(*flagDataDir)
	if err := os.Chdir(*flagDataDir); err != nil {
		klog.Fatalf("Failed to change to current directory to %q: %v", *flagDataDir, err)
	}

	// Parse hyperparameter settings.
	paramsSet := check1(commandline.ParseSettings(store, *settings))
	if *flagVerbosity >= 1 {
		fmt.Printf("Backend: %s\n\t%s\n", backend.Name(), backend.Description())
		fmt.Println(commandline.SprintSettings(store))
	}
	mag.BatchSize = model.GetParamOr(store.RootScope(), "batch_size", 128)

	//Early sanity checks.
	if *flagCheckpoint == "" && *flagEval {
		klog.Fatal("To run eval (--eval) you need to specify a checkpoint (--checkpoint).")
	}

	// Load data from OGBN-MAG.
	fmt.Printf("Loading data ... ")
	start := time.Now()
	check(mag.Download(*flagDataDir))
	fmt.Printf("elapsed: %s\n", time.Since(start))
	SetTrainSteps(store) // Can only be set after mag data is loaded.

	// RunWithMap train / eval.
	mag.WithReplacement = model.GetParamOr(store.RootScope(), paramWithReplacement, false)
	var err error
	if *flagEval {
		err = mag.EvalWithStore(backend, store, *flagDataDir, *flagCheckpoint, *flagLayerWise, *flagSkipTrainEval)
	} else {
		if mag.WithReplacement {
			fmt.Println("Training dataset with replacement")
		}

		// Train.
		err = mag.TrainWithStore(backend, store, *flagDataDir, *flagCheckpoint, *flagLayerWise, !*flagSkipReport, paramsSet)
	}
	if err != nil {
		fmt.Printf("%+v\n", err)
	}
}

// check reports and exits on error.
func check(err error) {
	if err == nil {
		return
	}
	klog.Fatalf("Fatal error: %+v", err)
}

// check1 reports and exits on error. Otherwise returns the value passed.
func check1[T any](v T, err error) T {
	check(err)
	return v
}
