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

	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
	"github.com/pkg/errors"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/pkg/core/graph/nanlogger"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/checkpoints"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/layers/batchnorm"
	"github.com/gomlx/gomlx/pkg/ml/layers/regularizers"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gomlx/pkg/ml/train/metrics"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers/cosineschedule"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gomlx/ui/gonb/plotly"
	"github.com/gomlx/gopjrt/dtypes"
)

var ModelList = []string{"linear", "cnn"}

var excludeParams = []string{"data_dir", "train_steps", "num_checkpoints", "plots"}

type ContextFn func(ctx *context.Context) *context.Context

func CreateDefaultContext() *context.Context {
	ctx := context.New()
	ctx.ResetRNGState()
	ctx.SetParams(map[string]any{
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
		// is using the gomlx_checkpoints tool:
		//
		//	$ gomlx_checkpoints --metrics --metrics_labels --metrics_types=accuracy  --metrics_names='E(Tra)/#loss,E(Val)/#loss' --loop=3s "<checkpoint_path>"
		plotly.ParamPlots: false,

		optimizers.ParamOptimizer:       "adamw",
		optimizers.ParamLearningRate:    1e-4,
		optimizers.ParamAdamEpsilon:     1e-7,
		optimizers.ParamAdamDType:       "",
		cosineschedule.ParamPeriodSteps: 0,
		activations.ParamActivation:     "relu",
		layers.ParamDropoutRate:         0.5,
		regularizers.ParamL2:            0.0,
		regularizers.ParamL1:            0.0,

		// CNN
		"cnn_dropout_rate":  0.5,
		"cnn_normalization": "layer", // "layer" or "batch".

		// Triplet
		losses.ParamTripletLossPairwiseDistanceMetric: "L2",
		losses.ParamTripletLossMiningStrategy:         "Hard",
		losses.ParamTripletLossMargin:                 0.5,
	})
	return ctx
}

// NewDatasetsConfigurationFromContext create a preprocessing configuration based on hyperparameters
// set in the context.
func NewDatasetsConfigurationFromContext(ctx *context.Context, dataDir string) *DatasetsConfiguration {
	dataDir = fsutil.MustReplaceTildeInDir(dataDir)
	config := &DatasetsConfiguration{}
	config.DataDir = dataDir
	config.BatchSize = context.GetParamOr(ctx, "batch_size", 0)
	config.EvalBatchSize = context.GetParamOr(ctx, "eval_batch_size", 0)
	config.UseParallelism = true
	config.BufferSize = 100
	config.Dtype = dtypes.Float32
	return config
}

// TrainModel based on configuration and flags.
func TrainModel(ctx *context.Context, dataDir, checkpointPath string, paramsSet []string) error {
	dataDir = fsutil.MustReplaceTildeInDir(dataDir)
	if !fsutil.MustFileExists(dataDir) {
		must.M(os.MkdirAll(dataDir, 0777))
	}

	modelType := context.GetParamOr(ctx, "model", "")
	var modelFn train.ModelFn
	switch modelType {
	case "linear":
		modelFn = LinearModelGraph
	case "cnn":
		modelFn = CnnModelGraph

	default:
		return errors.Errorf("Can't find model %q, available models: %q\n", modelType, ModelList)
	}

	fmt.Printf("Training %s model:\n", modelType)
	backend := backends.MustNew()
	fmt.Printf("Backend %s: %s\n", backend.Name(), backend.Description())

	lossFn, err := losses.LossFromContext(ctx)
	if err != nil {
		return err
	}

	if err := Download(dataDir); err != nil {
		return err
	}

	dsConfig := NewDatasetsConfigurationFromContext(ctx, dataDir)
	trainDS, trainEvalDS, validationEvalDS := CreateDatasets(backend, dsConfig)

	// Metrics we are interested.
	meanAccuracyMetric := metrics.NewSparseCategoricalAccuracy("Mean Accuracy", "#acc")
	movingAccuracyMetric := metrics.NewMovingAverageSparseCategoricalAccuracy("Moving Average Accuracy", "~acc", 0.01)

	// Create a train.Trainer: this object will orchestrate running the modelType, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	var trainer *train.Trainer
	optimizer := optimizers.FromContext(ctx)
	trainer = train.NewTrainer(backend, ctx,
		modelFn,
		lossFn,
		optimizer,
		[]metrics.Interface{movingAccuracyMetric}, // trainMetrics
		[]metrics.Interface{meanAccuracyMetric})   // evalMetrics

	// Debugging.
	if context.GetParamOr(ctx, "nan_logger", false) {
		nanlogger.New().AttachToTrainer(trainer)
	}

	// Use standard training loop.
	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.

	// Checkpoints saving.
	var checkpoint *checkpoints.Handler
	if checkpointPath != "" {
		numCheckpointsToKeep := context.GetParamOr(ctx, "num_checkpoints", 3)
		checkpoint, err = checkpoints.Build(ctx).
			DirFromBase(checkpointPath, dataDir).
			Keep(numCheckpointsToKeep).
			ExcludeParams(append(paramsSet, excludeParams...)...).
			Done()
		if err != nil {
			return errors.WithMessagef(err, "failed to create/load checkpoint %s, from data directory %s",
				checkpointPath, dataDir)
		}
		fmt.Printf("\t- checkpoint in %s\n", checkpoint.Dir())
		period := time.Minute * 1
		train.PeriodicCallback(loop, period, true, "saving checkpoint", 100,
			func(loop *train.Loop, metrics []*tensors.Tensor) error {
				fmt.Printf("\n\t- saving checkpoint@%d\n", loop.LoopStep)
				return checkpoint.Save()
			})
	}

	// Attach Plotly plots: plot points at exponential steps.
	// The points generated are saved along the checkpoint directory (if one is given).
	if context.GetParamOr(ctx, plotly.ParamPlots, false) {
		_ = plotly.New().
			WithCheckpoint(checkpoint).
			Dynamic().
			WithDatasets(trainEvalDS, validationEvalDS).
			ScheduleExponential(loop, 10, 1.2).
			WithBatchNormalizationAveragesUpdate(trainEvalDS)
	}

	// Loop for given number of steps.
	numTrainSteps := context.GetParamOr(ctx, "train_steps", 0)
	globalStep := int(optimizers.GetGlobalStep(ctx))
	if globalStep > 0 {
		fmt.Printf("\t- restarting from global step %d\n", globalStep)
		trainer.SetContext(ctx.Reuse())
	}
	if globalStep < numTrainSteps {
		_, err = loop.RunSteps(trainDS, numTrainSteps-globalStep)
		if err != nil {
			return errors.WithMessagef(err, "failed to train model %d steps", numTrainSteps-globalStep)
		}
		fmt.Printf("\t- trained to step %d, median train step: %d microseconds\n",
			loop.LoopStep, loop.MedianTrainStepDuration().Microseconds())

		// Update batch normalization averages, if they are used.
		if must.M1(batchnorm.UpdateAverages(trainer, trainEvalDS)) {
			fmt.Println("\tUpdated batch normalization mean/variances averages.")
			if checkpoint != nil {
				must.M(checkpoint.Save())
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
