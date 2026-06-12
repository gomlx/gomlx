// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

/*
 *	Copyright 2025 Rener Castro
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package mnist

import (
	"fmt"
	"os"
	"time"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/nanlogger"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/layers/norm"
	"github.com/gomlx/gomlx/ml/layers/regularizer"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/metric"
	"github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/gomlx/gomlx/ml/train/optimizer/cosineschedule"
	"github.com/gomlx/gomlx/support/fsutil"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gomlx/ui/gonb/plotly"
)

var ModelList = []string{"linear", "cnn"}

var excludeParams = []string{"data_dir", "train_steps", "num_checkpoints", "plots"}

type ScopeFn func(scope *model.Scope) *model.Scope

func CreateStore() *model.Store {
	store := model.NewStore()
	_ = store.ResetRNGState()
	store.SetParams(map[string]any{
		// Model type to use
		"model":           "linear",
		"loss":            "sparse_cross_logits",
		"num_checkpoints": 3,
		"train_steps":     4000,

		// batch_size for training.
		"batch_size": 600,

		// eval_batch_size can be larger than training, it's more efficient.
		"eval_batch_size": 1000,

		// Debug parameters.
		"nan_logger": false, // Trigger nan error as soon as it happens -- expensive, but helps debugging.

		// "plots" trigger generating intermediary eval data for plotting, and if running in GoNB, to actually
		// draw the plot with Plotly.
		//
		// From the command-line, an easy way to monitor the metrics being generated during the training of a model
		// is using the gomlx_checkpointss tool:
		//
		//	$ gomlx_checkpointss --metrics --metrics_labels --metrics_types=accuracy  --metrics_names='E(Tra)/#loss,E(Val)/#loss' --loop=3s "<checkpoint_path>"
		plotly.ParamPlots: false,

		optimizer.ParamOptimizer:            "adamw",
		optimizer.ParamLearningRate:         1e-4,
		optimizer.ParamAdamEpsilon:          1e-7,
		optimizer.ParamAdamDType:            "",
		cosineschedule.ParamPeriodSteps:     0, // If 0, it is disabled. Mutually exclusive with "cosine_schedule_cycles".
		cosineschedule.ParamCycles:          0, // If 0, it is disabled. Mutually exclusive with "cosine_schedule_steps".
		cosineschedule.ParamMinLearningRate: 0,
		activation.ParamActivation:          "relu",
		layers.ParamDropoutRate:             0.5,
		regularizer.ParamL2:                 0.0,
		regularizer.ParamL1:                 0.0,

		// CNN
		"cnn_dropout_rate":  0.5,
		"cnn_normalization": "layer", // "layer" or "batch".

		// Triplet
		loss.ParamTripletLossPairwiseDistanceMetric: "L2",
		loss.ParamTripletLossMiningStrategy:         "Hard",
		loss.ParamTripletLossMargin:                 0.5,
	})
	return store
}

// NewDatasetsConfigurationFromScope create a preprocessing configuration based on hyperparameters
// set in the model.
func NewDatasetsConfigurationFromScope(scope *model.Scope, dataDir string) *DatasetsConfiguration {
	dataDir = fsutil.MustReplaceTildeInDir(dataDir)
	config := &DatasetsConfiguration{}
	config.DataDir = dataDir
	config.BatchSize = model.GetParamOr(scope, "batch_size", 0)
	config.EvalBatchSize = model.GetParamOr(scope, "eval_batch_size", 0)
	config.UseParallelism = true
	config.BufferSize = 100
	config.Dtype = dtypes.Float32
	return config
}

// Train based on configuration and flags.
func Train(store *model.Store, dataDir, checkpointPath string, paramsSet []string) error {
	scope := store.RootScope()
	dataDir = fsutil.MustReplaceTildeInDir(dataDir)
	if !fsutil.MustFileExists(dataDir) {
		if err := os.MkdirAll(dataDir, 0777); err != nil {
			return err
		}
	}

	modelType := model.GetParamOr(scope, "model", "")
	var modelFn func(*model.Scope, *graph.Node) *graph.Node
	switch modelType {
	case "linear":
		modelFn = LinearModelGraph
	case "cnn":
		modelFn = CnnModelGraph

	default:
		return errors.Errorf("Can't find model %q, available models: %q\n", modelType, ModelList)
	}

	fmt.Printf("Training %s model:\n", modelType)
	backend := compute.MustNew()
	fmt.Printf("Backend %s: %s\n", backend.Name(), backend.Description())

	lossFn, err := loss.LossFromScope(scope)
	if err != nil {
		return err
	}

	if err := Download(dataDir); err != nil {
		return err
	}

	dsConfig := NewDatasetsConfigurationFromScope(scope, dataDir)
	trainDS, trainEvalDS, validationEvalDS := CreateDatasets(backend, dsConfig)

	// Metrics we are interested in.
	meanAccuracyMetric := metric.NewSparseCategoricalAccuracy("Mean Accuracy", "#acc")
	movingAccuracyMetric := metric.NewMovingAverageSparseCategoricalAccuracy("Moving Average Accuracy", "~acc", 0.01)

	// Create a train.Trainer: this object will orchestrate running the modelType, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	var trainer *train.Trainer
	trainer = train.NewTrainer(backend, store,
		modelFn,
		lossFn,
		optimizer.FromStore(store),
		[]metric.Interface{movingAccuracyMetric}, // trainMetrics
		[]metric.Interface{meanAccuracyMetric})   // evalMetrics

	// Debugging.
	if model.GetParamOr(scope, "nan_logger", false) {
		nanlogger.New().AttachToTrainer(trainer)
	}

	// Use a standard training loop.
	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.

	// Checkpoints saving.
	var checkpointHandler *checkpoint.Handler
	if checkpointPath != "" {
		numCheckpointsToKeep := model.GetParamOr(scope, "num_checkpoints", 3)
		checkpointHandler, err = checkpoint.Build(store).
			DirFromBase(checkpointPath, dataDir).
			Keep(numCheckpointsToKeep).
			ExcludeParams(append(paramsSet, excludeParams...)...).
			Done()
		if err != nil {
			return errors.WithMessagef(err, "failed to create/load checkpoint %s, from data directory %s",
				checkpointPath, dataDir)
		}
		fmt.Printf("\t- checkpoint in %s\n", checkpointHandler.Dir())
		period := time.Minute * 1
		train.PeriodicCallback(loop, period, true, "saving checkpoint", 100,
			func(loop *train.Loop, metrics []*tensors.Tensor) error {
				fmt.Printf("\n\t- saving checkpoint@%d\n", loop.LoopStep)
				return checkpointHandler.Save()
			})
	}

	// Attach Plotly plots: plot points at exponential steps.
	// The points generated are saved along the checkpoint directory (if one is given).
	if model.GetParamOr(scope, plotly.ParamPlots, false) {
		_ = plotly.New().
			WithCheckpoint(checkpointHandler).
			Dynamic().
			WithDatasets(trainEvalDS, validationEvalDS).
			ScheduleExponential(loop, 10, 1.2).
			WithBatchNormalizationAveragesUpdate(trainEvalDS)
	}

	// Loop for a given number of steps.
	numTrainSteps := model.GetParamOr(scope, "train_steps", 0)
	globalStep := int(optimizer.GetGlobalStep(scope))
	if globalStep > 0 {
		fmt.Printf("\t- restarting from global step %d\n", globalStep)
	}
	if globalStep < numTrainSteps {
		_, err = loop.RunSteps(trainDS, numTrainSteps-globalStep)
		if err != nil {
			return errors.WithMessagef(err, "failed to train model %d steps", numTrainSteps-globalStep)
		}
		fmt.Printf("\t- trained to step %d, median train step: %d microseconds\n",
			loop.LoopStep, loop.MedianTrainStepDuration().Microseconds())

		// Update batch normalization averages, if they are used.
		if check1(norm.UpdateBatchNormAverages(trainer, trainEvalDS)) {
			fmt.Println("\tUpdated batch normalization mean/variances averages.")
			if checkpointHandler != nil {
				if err := checkpointHandler.Save(); err != nil {
					return errors.WithMessage(err, "save checkpoint")
				}
			}
		}
	} else {
		fmt.Printf("\t - target train_steps=%d already reached. To train further, set a number additional "+
			"to current global step.\n", numTrainSteps)
	}
	// Finally, print an evaluation on train and test datasets.
	fmt.Println()
	if err = commandline.ReportEval(trainer, trainEvalDS, validationEvalDS); err != nil {
		return errors.WithMessagef(err, "while generating report with evaluation of trained %s", modelType)
	}
	fmt.Println()
	return nil
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
