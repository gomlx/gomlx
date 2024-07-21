/*
 *	Copyright 2023 Jan Pfeifer
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

// Linear generates random synthetic data, based on some linear mode + noise. Then it tries
// to learn the weights used to generate the data.
package main

import (
	"flag"
	"fmt"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/examples/adult"
	"github.com/gomlx/gomlx/examples/notebook/gonb/margaid"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/checkpoints"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/kan"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/commandline"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/metrics"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	"log"
	"path"
	"time"

	_ "github.com/gomlx/gomlx/backends/xla"
)

var (
	// ModelDType used for the model. AssertNoError match RawData Go types.
	ModelDType = dtypes.Float32
)

var (
	flagDataDir        = flag.String("data", "~/tmp/uci-adult", "Directory to save and load downloaded and generated dataset files.")
	flagCheckpoint     = flag.String("checkpoint", "", "Directory save and load checkpoints from (relative to --data). If left empty, no checkpoints are created.")
	flagCheckpointKeep = flag.Int("checkpoint_keep", 10, "Number of checkpoints to keep, if --checkpoint is set.")
	flagBatchSize      = flag.Int("batch", 128, "Dataset size for training")
	flagNumSteps       = flag.Int("steps", 5000, "Number of gradient descent steps to perform")
	flagForceDownload  = flag.Bool("force_download", false, "Force re-download of Adult dataset files.")

	flagOptimizer       = flag.String("optimizer", "adam", "Type of optimizer to use: 'sgd' or 'adam'")
	flagLearningRate    = flag.Float64("learning_rate", 0.001, "Initial learning rate.")
	flagNumQuantiles    = flag.Int("quantiles", 100, "Max number of quantiles to use for numeric features, used during piece-wise linear calibration. It will only use unique values, so if there are fewer variability, fewer quantiles are used.")
	flagEmbeddingDim    = flag.Int("embedding_dim", 8, "Default embedding dimension for categorical values.")
	flagVerbosity       = flag.Int("verbosity", 0, "Level of verbosity, the higher the more verbose.")
	flagNumHiddenLayers = flag.Int("num_hidden_layers", 8, "Number of hidden layers, stacked with residual connection.")
	flagNumHiddenNodes  = flag.Int("num_hidden_nodes", 32, "Number of nodes in hidden layers.")
	flagUseKAN          = flag.Bool("kan", false, "Use KAN - Kolmogorov–Arnold Networks")
	flagDropoutRate     = flag.Float64("dropout", 0, "Dropout rate")

	flagUseCategorical       = flag.Bool("use_categorical", true, "Use categorical features.")
	flagUseContinuous        = flag.Bool("use_continuous", true, "Use continuous features.")
	flagTrainableCalibration = flag.Bool("trainable_calibration", true, "Allow piece-wise linear calibration to adjust outputs.")
	flagPlots                = flag.Bool("plots", true, "Plots during training: perform periodic evaluations, "+
		"save results if --checkpoint is set and draw plots, if in a Jupyter notebook.")
)

// AssertNoError logs err and panics, if it is not nil.
func AssertNoError(err error) {
	if err != nil {
		log.Fatalf("Failed: %+v", err)
	}
}

func main() {
	flag.Parse()

	// Fixes directories.
	*flagDataDir = data.ReplaceTildeInDir(*flagDataDir)
	*flagCheckpoint = data.ReplaceTildeInDir(*flagCheckpoint)
	if *flagCheckpoint != "" && !path.IsAbs(*flagCheckpoint) {
		*flagCheckpoint = path.Join(*flagDataDir, *flagCheckpoint)
	}

	// Check variables validity.
	optimizerFn, found := optimizers.KnownOptimizers[*flagOptimizer]
	if !found {
		log.Fatalf("Unknown optimizer %q, please use one of %v",
			*flagOptimizer, xslices.Keys(optimizers.KnownOptimizers))
	}

	// Load training data and initialize statistics (vocabularies and quantiles).
	adult.LoadAndPreprocessData(*flagDataDir, *flagNumQuantiles, *flagForceDownload, *flagVerbosity)

	// Crate Backend and upload data to device tensors.
	backend := backends.New()
	if *flagVerbosity >= 1 {
		fmt.Printf("Backend: %s\n", backend.Name())
	}
	if *flagVerbosity >= 2 {
		adult.PrintBatchSamples(backend, adult.Data.Train)
	}

	// Create datasets for training and evaluation.
	trainDS := adult.NewDataset(backend, adult.Data.Train, "batched train")
	trainEvalDS := trainDS.Copy().BatchSize(*flagBatchSize, false)
	testEvalDS := adult.NewDataset(backend, adult.Data.Test, "test").
		BatchSize(*flagBatchSize, false)
	// For training, we shuffle and loop indefinitely.
	trainDS.BatchSize(*flagBatchSize, true).Shuffle().Infinite(true)

	// Metrics we are interested.
	meanAccuracyMetric := metrics.NewMeanBinaryLogitsAccuracy("Mean Accuracy", "#acc")
	movingAccuracyMetric := metrics.NewMovingAverageBinaryLogitsAccuracy("Moving Average Accuracy", "~acc", 0.01)

	// Context holds the variables and hyperparameters for the model.
	ctx := context.NewContext()
	ctx.SetParam(optimizers.ParamLearningRate, *flagLearningRate)

	// Checkpoints saving.
	var checkpoint *checkpoints.Handler
	if *flagCheckpoint != "" {
		var err error
		checkpoint, err = checkpoints.Build(ctx).Dir(*flagCheckpoint).Keep(*flagCheckpointKeep).Done()
		AssertNoError(err)
	}

	// Create a train.Trainer: this object will orchestrate running the model, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	trainer := train.NewTrainer(backend, ctx, ModelGraph, losses.BinaryCrossentropyLogits,
		optimizerFn(ctx),
		[]metrics.Interface{movingAccuracyMetric}, // trainMetrics
		[]metrics.Interface{meanAccuracyMetric})   // evalMetrics

	// Use standard training loop.
	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.

	// Attach a checkpoint.
	if checkpoint != nil {
		period := time.Minute * 1
		train.PeriodicCallback(loop, period, true, "saving checkpoint", 100,
			func(loop *train.Loop, metrics []*tensors.Tensor) error {
				fmt.Printf("\n[saving checkpoint@%d] [median train step (ms): %d]\n", loop.LoopStep, loop.MedianTrainStepDuration().Milliseconds())
				return checkpoint.Save()
			})
	}

	// Attach a margaid plots: plot points at exponential steps,
	// that are saved along the checkpoint directory (if one is given).
	if *flagPlots {
		_ = margaid.NewDefault(loop, checkpoint.Dir(), 100, 1.1, trainEvalDS, testEvalDS)
	}

	// Train for the selected *flagNumSteps
	_, err := loop.RunSteps(trainDS, *flagNumSteps)
	AssertNoError(err)
	fmt.Printf("\t[Step %d] median train step: %d microseconds\n", loop.LoopStep, loop.MedianTrainStepDuration().Microseconds())

	// Finally, print an evaluation on train and test datasets.
	fmt.Println()
	AssertNoError(commandline.ReportEval(trainer, trainEvalDS, testEvalDS))
}

// ModelGraph outputs the logits (not the probabilities). The parameter inputs should contain 3 tensors:
//
// - categorical inputs, shaped  `(int64)[batch_size, len(VocabulariesFeatures)]`
// - continuous inputs, shaped `(float32)[batch_size, len(Quantiles)]`
// - weights: not currently used, but shaped `(float32)[batch_size, 1]`.
func ModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	_ = spec // Not used, since the dataset is always the same.
	g := inputs[0].Graph()

	// Use Cosine schedule of the learning rate.
	optimizers.CosineAnnealingSchedule(ctx, g, ModelDType).
		PeriodInSteps(*flagNumSteps / 3).Done()

	categorical, continuous := inputs[0], inputs[1]
	batchSize := categorical.Shape().Dimensions[0]

	var allEmbeddings []*Node

	if *flagUseCategorical {
		// Embedding of categorical values, each with its own vocabulary.
		numCategorical := categorical.Shape().Dimensions[1]
		for catIdx := 0; catIdx < numCategorical; catIdx++ {
			// Take one column at a time of the categorical values.
			split := Slice(categorical, AxisRange(), AxisRange(catIdx, catIdx+1))
			// Embed it accordingly.
			embedCtx := ctx.In(fmt.Sprintf("categorical_%d_%s", catIdx, adult.Data.VocabulariesFeatures[catIdx]))
			vocab := adult.Data.Vocabularies[catIdx]
			vocabSize := len(vocab)
			embedding := layers.Embedding(embedCtx, split, ModelDType, vocabSize, *flagEmbeddingDim)
			embedding.AssertDims(batchSize, *flagEmbeddingDim) // 2-dim tensor, with batch size as the leading dimension.
			allEmbeddings = append(allEmbeddings, embedding)
		}
	}

	if *flagUseContinuous {
		// Piecewise-linear calibration of the continuous values. Each feature has its own number of quantiles.
		numContinuous := continuous.Shape().Dimensions[1]
		for contIdx := 0; contIdx < numContinuous; contIdx++ {
			// Take one column at a time of the continuous values.
			split := Slice(continuous, AxisRange(), AxisRange(contIdx, contIdx+1))
			featureName := adult.Data.QuantilesFeatures[contIdx]
			calibrationCtx := ctx.In(fmt.Sprintf("continuous_%d_%s", contIdx, featureName))
			quantiles := adult.Data.Quantiles[contIdx]
			layers.AssertQuantilesForPWLCalibrationValid(quantiles)
			calibrated := layers.PieceWiseLinearCalibration(calibrationCtx, split, Const(g, quantiles),
				*flagTrainableCalibration)
			calibrated.AssertDims(batchSize, 1) // 2-dim tensor, with batch size as the leading dimension.
			allEmbeddings = append(allEmbeddings, calibrated)
		}
	}

	layer := Concatenate(allEmbeddings, -1)
	layer.AssertDims(batchSize, -1) // 2-dim tensor, with batch size as the leading dimension (-1 means it is not checked).

	var logits *Node
	if *flagUseKAN {
		logits = kan.New(ctx.In("kan_layers"), layer, 1).
			NumHiddenLayers(*flagNumHiddenLayers, *flagNumHiddenNodes).
			Done()
	} else {
		// Normal FNN
		layer = layers.DenseWithBias(ctx.In(fmt.Sprintf("DenseLayer_%d", 0)), layer, *flagNumHiddenNodes)
		for ii := 1; ii < *flagNumHiddenLayers; ii++ {
			ctx := ctx.In(fmt.Sprintf("DenseLayer_%d", ii))
			// Add layer with residual connection.
			tmp := Sigmoid(layer)
			if *flagDropoutRate > 0 {
				tmp = layers.Dropout(ctx, tmp, Scalar(g, ModelDType, *flagDropoutRate))
			}
			tmp = layers.DenseWithBias(ctx, tmp, *flagNumHiddenNodes)
			layer = Add(layer, tmp) // Residual connections
		}
		layer = Sigmoid(layer)
		logits = layers.DenseWithBias(ctx.In("DenseFinal"), layer, 1)
	}
	logits.AssertDims(batchSize, 1) // 2-dim tensor, with batch size as the leading dimension.
	return []*Node{logits}
}
