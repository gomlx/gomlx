// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/examples/adult"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/layers/fnn"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/metric"
	"github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gomlx/ui/gonb/plotly"
	"github.com/schollz/progressbar/v3"

	_ "github.com/gomlx/gomlx/backends/default"
)

var (
	flagDataDir   = flag.String("data", "~/work/uci-adult", "Directory to save dataset and checkpoint files.")
	flagCheckpoint = flag.String("checkpoint", "checkpoint", "Checkpoint subdirectory name.")
)

func main() {
	flag.Parse()

	// 1. Initialize backend and model store
	//md_start:trainer_creation
	backend := compute.MustNew()
	store := model.NewStore()

	// Configure hyperparameters in the model store
	store.SetParams(map[string]any{
		"batch_size":                128,
		"train_steps":                1000,
		optimizer.ParamOptimizer:    "adam",
		optimizer.ParamLearningRate: 0.001,
		activation.ParamActivation:  "relu",
		fnn.ParamNumHiddenLayers:    2,
		fnn.ParamNumHiddenNodes:     64,
	})
	scope := store.RootScope()
	//md_end:trainer_creation

	// 2. Load preprocessed Adult dataset
	adult.LoadAndPreprocessData(*flagDataDir, 100, false, 0)
	inMemoryDS := adult.NewDataset(backend, adult.Data.Train, "train")
	trainDS := inMemoryDS.BatchSize(128, true).Shuffle().Infinite(true)
	trainEvalDS := inMemoryDS.Copy().BatchSize(128, false)
	testEvalDS := adult.NewDataset(backend, adult.Data.Test, "test").BatchSize(128, false)

	// 3. Define the model function
	//md_start:trainer_creation
	modelFn := func(scope *model.Scope, spec any, inputs []*Node) []*Node {
		// In a real example, inputs would contain calibrated continuous and categorical features.
		// For simplicity, we just run a Feed-Forward Neural Network on the continuous features.
		continuousFeatures := inputs[1]
		logits := fnn.New(scope, continuousFeatures, 1).Done()
		return []*Node{logits}
	}

	// 4. Configure metrics
	// Moving average is recommended for training since the model changes at each step.
	movingAccuracy := metric.NewMovingAverageBinaryLogitsAccuracy("Moving Average Accuracy", "~acc", 0.01)
	// Mean accuracy is recommended for evaluation because the model is frozen.
	meanAccuracy := metric.NewMeanBinaryLogitsAccuracy("Mean Accuracy", "#acc")

	// 5. Create the Trainer
	trainer := train.NewTrainer(
		backend,
		store,
		modelFn,
		loss.BinaryCrossentropyLogits,
		optimizer.Adam().LearningRate(0.001).Done(),
		[]metric.Interface{movingAccuracy}, // metrics evaluated at train time
		[]metric.Interface{meanAccuracy},   // metrics evaluated at eval time
	)
	//md_end:trainer_creation

	// 6. Create the Training Loop
	//md_start:loop_creation
	loop := train.NewLoop(trainer)
	//md_end:loop_creation

	// 7. Attach periodic callbacks to the Loop
	//md_start:loop_callbacks
	// A. Attach a progress bar to monitor progress in the terminal
	commandline.ProgressbarStyle = progressbar.ThemeUnicode
	commandline.AttachProgressBar(loop)

	// B. Setup checkpoint saving (every minute or every N steps)
	var checkpointHandler *checkpoint.Handler
	if *flagCheckpoint != "" {
		checkpointHandler, _ = checkpoint.Build(store).
			DirFromBase(*flagCheckpoint, *flagDataDir).
			Keep(3).
			Done()
		
		// Save checkpoints periodically every minute
		train.PeriodicCallback(loop, time.Minute, true, "checkpoint", 100, checkpointHandler.SaveOnStepFn)
	}

	// C. Attach Plotly to automatically generate training plots
	plotly.New().
		WithCheckpoint(checkpointHandler).
		Dynamic().
		WithDatasets(trainEvalDS, testEvalDS).
		ScheduleExponential(loop, 100, 1.2)

	// D. Register a custom callback to print moving average metrics
	train.EveryNSteps(loop, 500, "log_metrics", 0, func(l *train.Loop, metrics []*tensors.Tensor) error {
		// metrics[0] is the batch loss, metrics[1] is the moving average loss, etc.
		fmt.Printf("\nStep %d: Loss = %.4f, Moving Acc = %.4f\n", l.LoopStep, metrics[0].Value(), metrics[2].Value())
		return nil
	})
	//md_end:loop_callbacks

	// 8. Run the loop to train
	//md_start:loop_execution
	trainSteps := model.GetParamOr(scope, "train_steps", 1000)
	globalStep := int(optimizer.GetGlobalStep(scope))
	if globalStep != 0 {
		fmt.Printf("Restarting training from global step %d\n", globalStep)
	}

	_, err := loop.RunToGlobalStep(trainDS, trainSteps)
	if err != nil {
		panic(err)
	}
	fmt.Println("Training complete!")
	//md_end:loop_execution
}
