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

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/nanlogger"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/checkpoints"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activations"
	"github.com/gomlx/gomlx/ml/layers/batchnorm"
	"github.com/gomlx/gomlx/ml/layers/regularizers"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/metrics"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/ml/train/optimizers/cosineschedule"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gomlx/ui/gonb/plotly"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/janpfeifer/must"
)

var excludeParams = []string{"data_dir", "train_steps", "num_checkpoints", "plots"}

type ContextFn func(ctx *context.Context) *context.Context

var Models = map[string]struct {
	ctx   ContextFn
	model train.ModelFn
}{
	"logistic": {CreateSoftMaxModelContext, LinearModelGraph},
	"cnn":      {CreateCnnModelContext, CnnModelGraph},
}

var Losses = map[string]losses.LossFn{
	"cross-entropy": losses.BinaryCrossentropyLogits,

	"triplet": func(labels, predictions []*Node) (loss *Node) {
		return losses.TripletLoss(labels, predictions, losses.TripletMiningStrategyAll, 1.0, losses.PairwiseDistanceMetricL2)
	},
	"triplet softmax": func(labels, predictions []*Node) (loss *Node) {
		return losses.TripletLoss(labels, predictions, losses.TripletMiningStrategyAll, -1.0, losses.PairwiseDistanceMetricL2)
	},
	"triplet softmax with cosine distance": func(labels, predictions []*Node) (loss *Node) {
		return losses.TripletLoss(labels, predictions, losses.TripletMiningStrategyAll, -1.0, losses.PairwiseDistanceMetricCosine)
	},
	"hard triple softmax": func(labels, predictions []*Node) (loss *Node) {
		return losses.TripletLoss(labels, predictions, losses.TripletMiningStrategyHard, -1.0, losses.PairwiseDistanceMetricL2)
	},
	"semi-hard triplet softmax": func(labels, predictions []*Node) (loss *Node) {
		return losses.TripletLoss(labels, predictions, losses.TripletMiningStrategySemiHard, -1.0, losses.PairwiseDistanceMetricL2)
	},
	"semi-hard triplet softmax with cosine distance": func(labels, predictions []*Node) (loss *Node) {
		return losses.TripletLoss(labels, predictions, losses.TripletMiningStrategySemiHard, -1.0, losses.PairwiseDistanceMetricCosine)
	},
}

// CreateCnnModelContext sets the context with default hyperparameters to use with TrainModel.
func CreateSoftMaxModelContext(ctx *context.Context) *context.Context {
	ctx.RngStateReset()
	ctx.SetParams(map[string]any{
		// Model type to use
		"model":           "logistic",
		"num_classes":     10,
		"num_checkpoints": 3,
		"train_steps":     20000,

		// loss
		"loss": "cross-entropy",

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

		optimizers.ParamOptimizer:       "sgd",
		optimizers.ParamLearningRate:    1e-4,
		cosineschedule.ParamPeriodSteps: 0,
		regularizers.ParamL2:            0.0,
		regularizers.ParamL1:            0.0,
	})
	return ctx
}

// CreateCnnModelContext sets the context with default hyperparameters to use with TrainModel.
func CreateCnnModelContext(ctx *context.Context) *context.Context {
	ctx.RngStateReset()
	ctx.SetParams(map[string]any{
		// Model type to use
		"model":           "cnn",
		"num_classes":     10,
		"num_checkpoints": 3,
		"train_steps":     20000,

		// loss
		"loss": "cross-entropy",

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

		// "normalization" is overridden by "fnn_normalization" and "cnn_normalization", if they are set.
		layers.ParamNormalization: "none",

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
		"cnn_num_layers":      2.0,
		"cnn_dropout_rate":    0.5,
		"cnn_embeddings_size": 128,
	})
	return ctx
}

// NewDatasetsConfigurationFromContext create a preprocessing configuration based on hyperparameters
// set in the context.
func NewDatasetsConfigurationFromContext(ctx *context.Context, dataDir string) *DatasetsConfiguration {
	dataDir = data.ReplaceTildeInDir(dataDir)
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
func TrainModel(ctx *context.Context, dataDir string, model, loss string) {
	dataDir = data.ReplaceTildeInDir(dataDir)
	if !data.FileExists(dataDir) {
		must.M(os.MkdirAll(dataDir, 0777))
	}

	modelFn, ok := Models[model]
	if !ok {
		fmt.Printf("can't find model %s, using logistic", model)
		modelFn = Models["logistic"]
	}
	ctx = modelFn.ctx(ctx)
	backend := backends.New()

	lossFn, ok := Losses[loss]
	if !ok {
		fmt.Printf("can't find loss %s, using cross-entropy", loss)
		lossFn = Losses["cross-entropy"]
	}

	must.M(Download(dataDir))

	dsConfig := NewDatasetsConfigurationFromContext(ctx, dataDir)
	trainDS, trainEvalDS, validationEvalDS := CreateDatasets(backend, dsConfig)

	// Metrics we are interested.
	meanAccuracyMetric := metrics.NewMeanBinaryLogitsAccuracy("Mean Accuracy", "#acc")
	movingAccuracyMetric := metrics.NewMovingAverageBinaryLogitsAccuracy("Moving Average Accuracy", "~acc", 0.01)

	// Create a train.Trainer: this object will orchestrate running the model, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	var trainer *train.Trainer
	optimizer := optimizers.FromContext(ctx)

	trainer = train.NewTrainer(backend, ctx,
		modelFn.model,
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
	if dataDir != "" {
		numCheckpointsToKeep := context.GetParamOr(ctx, "num_checkpoints", 3)
		checkpoint = must.M1(checkpoints.Build(ctx).
			DirFromBase(dataDir, dataDir).
			Keep(numCheckpointsToKeep).
			ExcludeParams(excludeParams...).
			Done())
		fmt.Printf("Checkpointing model to %q\n", checkpoint.Dir())
	}

	// Attach Plotly plots: plot points at exponential steps.
	// The points generated are saved along the checkpoint directory (if one is given).
	if context.GetParamOr(ctx, plotly.ParamPlots, false) {
		_ = plotly.New().
			WithCheckpoint(checkpoint).
			Dynamic().
			WithDatasets(trainEvalDS, validationEvalDS).
			ScheduleExponential(loop, 200, 1.2).
			WithBatchNormalizationAveragesUpdate(trainEvalDS)
	}

	// Loop for given number of steps.
	numTrainSteps := context.GetParamOr(ctx, "train_steps", 0)
	globalStep := int(optimizers.GetGlobalStep(ctx))
	if globalStep > 0 {
		trainer.SetContext(ctx.Reuse())
	}
	if globalStep < numTrainSteps {
		_ = must.M1(loop.RunSteps(trainDS, numTrainSteps-globalStep))
		fmt.Printf("\t[Step %d] median train step: %d microseconds\n",
			loop.LoopStep, loop.MedianTrainStepDuration().Microseconds())

		// Update batch normalization averages, if they are used.
		if batchnorm.UpdateAverages(trainer, trainEvalDS) {
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
	must.M(commandline.ReportEval(trainer, trainEvalDS, validationEvalDS))
	fmt.Println()
}
