// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package fnn implements a feed-forward neural network for the OGBN-MAG problem.
package fnn

import (
	"fmt"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	mag "github.com/gomlx/gomlx/examples/ogbnmag"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activations"
	"github.com/gomlx/gomlx/ml/layers/kan"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/metrics"
	optimizers "github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/gomlx/gomlx/support/exceptions"
	"github.com/gomlx/gomlx/support/fsutil"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gomlx/ui/gonb/plotly"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// FnnModelGraph builds a FnnModel for the OGBN-MAP dataset.
func FnnModelGraph(scope *model.Scope, spec any, inputs []*Node) []*Node {
	seeds := inputs[0]
	g := seeds.Graph()
	getMagVar := func(name string) *Node {
		magVar := scope.Store().GetVariable(model.JoinPath(mag.OgbnMagVariablesScope, name))
		if magVar == nil {
			exceptions.Panicf("Missing OGBN-MAG dataset variables (%q), pls call UploadOgbnMagVariables() on context first.", name)
		}
		return magVar.NodeValue(g)
	}
	log1pMagVar := func(name string) *Node {
		return Log1p(ConvertDType(getMagVar(name), dtypes.Float32))
	}

	// Gather and concatenate all features from the seeds (indices of papers).
	logits := Concatenate([]*Node{
		Gather(getMagVar("PapersEmbeddings"), seeds),
		//Gather(log1pMagVar("CountPapersCites"), seeds),
		//Gather(log1pMagVar("CountPapersIsCited"), seeds),
		//Gather(log1pMagVar("CountPapersFieldsOfStudy"), seeds),
		Gather(log1pMagVar("CountPapersAuthors"), seeds),
	}, 1)

	// Build FNN.
	numLayers := model.GetParamOr(scope, "hidden_layers", 2)
	numNodes := model.GetParamOr(scope, "num_nodes", 128)
	useKan := model.GetParamOr(scope, "kan", false)
	if useKan {
		logits = kan.New(scope, logits, mag.NumLabels).NumHiddenLayers(numLayers, numNodes).Done()
	} else {
		// Normal FNN
		for layerNum := range numLayers {
			logits = layers.DenseWithBias(scope.In("layer-%d", layerNum), logits, numNodes)
			logits = activations.LeakyRelu(logits)
			dropoutRate := model.GetParamOr(scope, "dropout_rate", 0.0)
			if dropoutRate > 0 {
				dropoutRateNode := Scalar(g, dtypes.Float32, dropoutRate)
				logits = layers.Dropout(scope, logits, dropoutRateNode)
			}
		}
		logits = layers.DenseWithBias(scope.In("readout"), logits, mag.NumLabels)
	}

	return []*Node{logits} // Return only the logits.
}

var ModelFn = FnnModelGraph

// Train FNN model based on configuration in `scope`.
func Train(backend compute.Backend, scope *model.Scope) error {
	trainDS, validDS, testDS, err := mag.PapersSeedDatasets(backend)
	mag.UploadOgbnMagVariables(backend, scope.Store())
	//scope.EnumerateVariables(func(v *model.Variable) {
	//	fmt.Printf("%s :: %s:\t%s\n", v.Scope(), v.Name(), v.Value().Shape())
	//})

	if err != nil {
		return err
	}

	batchSize := model.GetParamOr(scope, "batch_size", 128)
	trainEvalDS := trainDS.Copy()
	trainDS = trainDS.Shuffle().BatchSize(batchSize, true).Infinite(true)

	// Evaluation datasets.
	evalBatchSize := model.GetParamOr(scope, "eval_batch_size", 1024)
	trainEvalDS = trainEvalDS.BatchSize(evalBatchSize, false).Infinite(false)
	validDS = validDS.BatchSize(evalBatchSize, false).Infinite(false)
	testDS = testDS.BatchSize(evalBatchSize, false).Infinite(false)

	// Get trainSteps before a checkpoint is loaded -- in which case it will be overwritten.
	trainSteps := model.GetParamOr(scope, "train_steps", 100)

	// Checkpoint: it loads if already exists, and it will save as we train.
	checkpointPath := model.GetParamOr(scope, "checkpoint", "")
	numCheckpointsToKeep := model.GetParamOr(scope, "num_checkpoints", 10)
	var checkpointHandler *checkpoint.Handler
	var globalStep int64
	if checkpointPath != "" {
		checkpointPath = fsutil.MustReplaceTildeInDir(checkpointPath) // If the path starts with "~", it is replaced.
		var err error
		if numCheckpointsToKeep <= 1 {
			// Only limit the amount of checkpoints kept if >= 2.
			numCheckpointsToKeep = -1
		}
		if numCheckpointsToKeep > 0 {
			checkpointHandler, err = checkpoint.Build(scope.Store()).Dir(checkpointPath).Keep(numCheckpointsToKeep).TakeMean(3, backend).Done()
		} else {
			checkpointHandler, err = checkpoint.Build(scope.Store()).Dir(checkpointPath).Done()
		}
		if err != nil {
			return errors.WithMessagef(err, "while setting up checkpoint to %q (keep=%d)",
				checkpointPath, numCheckpointsToKeep)
		}
		globalStep = optimizer.GetGlobalStep(scope.Store())
		if globalStep != 0 {
			fmt.Printf("> restarting training from global_step=%d\n", globalStep)
		}
		scope.SetParam("train_steps", trainSteps)
	}

	// Loss: multi-class classification problem.
	lossFn := losses.SparseCategoricalCrossEntropyLogits

	// Metrics we are interested.
	meanAccuracyMetric := metrics.NewSparseCategoricalAccuracy("Mean Accuracy", "#acc")
	movingAccuracyMetric := metrics.NewMovingAverageSparseCategoricalAccuracy("Moving Average Accuracy", "~acc", 0.01)

	// Create a train.Trainer: this object will orchestrate running the model, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	theOptimizer := optimizer.ByName(scope, model.GetParamOr(scope, "optimizer", "adamw"))
	trainer := train.NewTrainer(backend, scope.Store(), ModelFn,
		lossFn,
		theOptimizer,
		[]metrics.Interface{movingAccuracyMetric}, // trainMetrics
		[]metrics.Interface{meanAccuracyMetric})   // evalMetrics

	// Use standard training loop.
	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.

	// Attach a checkpoint: checkpoint every 1 minute of training.
	if checkpointHandler != nil && numCheckpointsToKeep > 1 {
		period := time.Minute * 1
		train.PeriodicCallback(loop, period, true, "saving checkpoint", 100,
			func(loop *train.Loop, metrics []*tensors.Tensor) error {
				return checkpointHandler.Save()
			})
	}

	// Attach a margaid plots: plot points at exponential steps.
	// The points generated are saved along the checkpoint directory (if one is given).
	usePlots := model.GetParamOr(scope, "plots", false)
	if usePlots {
		_ = plotly.New().
			WithCheckpoint(checkpointHandler).
			Dynamic().
			WithDatasets(validDS, testDS, trainEvalDS).
			ScheduleExponential(loop, 200, 1.2).
			WithBatchNormalizationAveragesUpdate(trainEvalDS)
	}

	// Loop for given number of steps
	if int(globalStep) < trainSteps {
		_, err = loop.RunSteps(trainDS, trainSteps-int(globalStep))
		if err != nil {
			return errors.WithMessage(err, "while running steps")
		}
		fmt.Printf("\t[Step %d] median train step: %d microseconds\n",
			loop.LoopStep, loop.MedianTrainStepDuration().Microseconds())
		if checkpointHandler != nil && numCheckpointsToKeep <= 1 {
			// Save checkpoint at end of training.
			err = checkpointHandler.Save()
			if err != nil {
				klog.Errorf("Failed to save final checkpoint in %q: %+v", checkpointPath, err)
			}
		}
	}

	// Finally, print an evaluation on train and test datasets.
	fmt.Println()
	err = commandline.ReportEval(trainer, trainEvalDS, validDS, testDS)
	if err != nil {
		return errors.WithMessage(err, "while reporting eval")
	}
	fmt.Println()
	return nil
}
