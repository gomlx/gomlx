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
	"github.com/gomlx/gomlx/examples/adult"
	"github.com/gomlx/gomlx/examples/notebook/bashkernel"
	"github.com/gomlx/gomlx/examples/notebook/bashkernel/chartjs"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/checkpoints"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/commandline"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/metrics"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/pkg/errors"
	"log"
	"path"
)

var (
	// ModelDType used for the model. AssertNoError match RawData Go types.
	ModelDType = shapes.Float32
)

var (
	flagDataDir        = flag.String("data", "~/tmp/uci-adult", "Directory to save and load downloaded and generated dataset files.")
	flagCheckpoint     = flag.String("checkpoint", "", "Directory save and load checkpoints from (relative to --data). If left empty, no checkpoints are created.")
	flagCheckpointKeep = flag.Int("checkpoint_keep", 10, "Number of checkpoints to keep, if --checkpoint is set.")
	flagBatchSize      = flag.Int("batch", 128, "Dataset size for training")
	flagNumSteps       = flag.Int("steps", 5000, "Number of gradient descent steps to perform")
	flagNumThreads     = flag.Int("num_threads", -1, "Number of threads for XLA. Leave as -1 to use as many as there are cores.")
	flagNumReplicas    = flag.Int("num_replicas", 1, "Number of replicas for XLA. Leave as 1 for now.")
	flagPlatform       = flag.String("platform", "", "Platform to use, if empty uses the default one.")
	flagForceDownload  = flag.Bool("force_download", false, "Force re-download of Adult dataset files.")

	flagOptimizer       = flag.String("optimizer", "adam", "Type of optimizer to use: 'sgd' or 'adam'")
	flagLearningRate    = flag.Float64("learning_rate", 0.001, "Initial learning rate.")
	flagNumQuantiles    = flag.Int("quantiles", 100, "Max number of quantiles to use for numeric features, used during piece-wise linear calibration. It will only use unique values, so if there are fewer variability, fewer quantiles are used.")
	flagEmbeddingDim    = flag.Int("embedding_dim", 8, "Default embedding dimension for categorical values.")
	flagVerbosity       = flag.Int("verbosity", 0, "Level of verbosity, the higher the more verbose.")
	flagNumHiddenLayers = flag.Int("hidden_layers", 8, "Number of hidden layers, stacked with residual connection.")
	flagNumNodes        = flag.Int("num_nodes", 32, "Number of nodes in hidden layers.")
	flagDropoutRate     = flag.Float64("dropout", 0, "Dropout rate")

	flagUseCategorical       = flag.Bool("use_categorical", true, "Use categorical features.")
	flagUseContinuous        = flag.Bool("use_continuous", true, "Use continuous features.")
	flagTrainableCalibration = flag.Bool("trainable_calibration", true, "Allow piece-wise linear calibration to adjust outputs.")
	flagNumPlotPoints        = flag.Int("plot_points", 0, "Number points to plot using Chart.JS.")
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
			*flagOptimizer, slices.Keys(optimizers.KnownOptimizers))
	}

	// Load training data and initialize statistics (vocabularies and quantiles).
	adult.LoadAndPreprocessData(*flagDataDir, *flagNumQuantiles, *flagForceDownload, *flagVerbosity)

	// Crate Manager and upload data to device tensors.
	manager := BuildManager().NumThreads(*flagNumThreads).NumReplicas(*flagNumReplicas).Platform(*flagPlatform).MustDone()
	adult.Data.Train.CreateTensors(manager)
	adult.Data.Test.CreateTensors(manager)
	if *flagVerbosity >= 2 {
		adult.PrintBatchSamples(adult.Data.Train, manager)
	}

	// Create datasets for training and evaluation.
	trainingDS := adult.NewDataset("batched train", adult.Data.Train, manager, *flagBatchSize)
	trainEvalDS := adult.NewDatasetForEval("train", adult.Data.Train, manager)
	testEvalDS := adult.NewDatasetForEval("test", adult.Data.Test, manager)

	// Metrics we are interested.
	meanAccuracyMetric := metrics.NewMeanBinaryLogitsAccuracy("Mean Accuracy", "#acc")
	movingAccuracyMetric := metrics.NewMovingAverageBinaryLogitsAccuracy("Moving Average Accuracy", "~acc", 0.01)

	// Context holds the variables and hyperparameters for the model.
	ctx := context.NewContext(manager)
	ctx.SetParam(optimizers.LearningRateKey, *flagLearningRate)

	// Checkpoints saving.
	var checkpoint *checkpoints.Handler
	if *flagCheckpoint != "" {
		var err error
		checkpoint, err = checkpoints.Build(ctx).Dir(*flagCheckpoint).Keep(*flagCheckpointKeep).Done()
		AssertNoError(err)
	}

	// Create a train.Trainer: this object will orchestrate running the model, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	trainer := train.NewTrainer(manager, ctx, ModelGraph, losses.BinaryCrossentropyLogits,
		optimizerFn(),
		[]metrics.Interface{movingAccuracyMetric}, // trainMetrics
		[]metrics.Interface{meanAccuracyMetric})   // evalMetrics
	AssertNoError(ctx.Error())

	// Use standard training loop.
	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.
	if checkpoint != nil {
		train.NTimesDuringLoop(loop, *flagNumPlotPoints, "checkpointing", 100, func(_ *train.Loop, _ []tensor.Tensor) error {
			return checkpoint.Save()
		})
	}

	// Plot metrics, if inside a Bash notebook.
	if bashkernel.IsBashNotebook() {
		attachPlot(loop, trainEvalDS)
	}

	_, err := loop.RunSteps(trainingDS, *flagNumSteps)
	AssertNoError(err)

	// Finally print an evaluation on train and test datasets.
	fmt.Println()
	err = commandline.ReportEval(trainer, trainEvalDS, testEvalDS)
	AssertNoError(err)
}

// attachPlot attaches a plot to the end of the loop execution, collecting evaluation points during the loop.
func attachPlot(loop *train.Loop, sampler *adult.Dataset) {
	numPlotPoints := *flagNumPlotPoints
	var lossData *chartjs.LinePlotData
	lossData = &chartjs.LinePlotData{
		Lines: []*chartjs.LineData{
			{Label: "Moving Average Loss (train)", Points: make([]chartjs.XY, 0, numPlotPoints+2)},
			{Label: "Moving Average Accuracy (train)", IsSecondAxis: true,
				Points: make([]chartjs.XY, 0, numPlotPoints+2)},
			{Label: "Mean Loss (test)", Points: make([]chartjs.XY, 0, numPlotPoints+2)},
			{Label: "Mean Accuracy (test)", IsSecondAxis: true,
				Points: make([]chartjs.XY, 0, numPlotPoints+2)},
		},
		HasTwoAxis: true,
		XTitle:     "Global Step",
		YTitle:     "Loss",
		Y2Title:    "Accuracy",
	}
	// Attach itself to `loop`, it will run the function passed numPlotPoints times distributed evenly in the training loop.
	train.NTimesDuringLoop(loop, numPlotPoints, "Flat for plotting", 0, func(loop *train.Loop, metrics []tensor.Tensor) error {
		x := float64(loop.LoopStep)
		lossData.AddPoint(0, x, metrics[1].Value().(float64))
		lossData.AddPoint(1, x, metrics[2].Value().(float64))
		// Run evaluation on test dataset.
		evalMetrics, err := loop.Trainer.Eval(sampler)
		if err != nil {
			return errors.WithMessagef(err, "Evaluating on test for global_step=%d", loop.LoopStep)
		}
		lossData.AddPoint(2, x, evalMetrics[0].Value().(float64))
		lossData.AddPoint(3, x, evalMetrics[1].Value().(float64))
		return nil
	})
	loop.OnEnd("Plotting", 100, func(loop *train.Loop, metrics []tensor.Tensor) error {
		plot := chartjs.NewLinePlot(lossData)
		return plot.Plot()
	})
}

// ModelGraph outputs the logits (not the probabilities). The parameter inputs should contain 3 tensors:
//
// - categorical inputs, shaped  `(int64)[batch_size, len(VocabulariesFeatures)]`
// - continuous inputs, shaped `(float32)[batch_size, len(Quantiles)]`
// - weights: not currently used, but shaped `(float32)[batch_size, 1]`.
func ModelGraph(ctx *context.Context, spec any, inputs []*Node) (logits []*Node) {
	_ = spec // Not used, we know exactly what the inputs are.
	categorical, continuous := inputs[0], inputs[1]
	// batchSize := categorical.Shape().Dimensions[0]
	var allEmbeddings []*Node
	graph := categorical.Graph()

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
			if err := layers.ValidateQuantilesForPWLCalibration(quantiles); err != nil {
				graph.SetError(errors.Wrapf(err, "quantile for features %q invalid", featureName))
				return nil
			}
			calibrated := layers.PieceWiseLinearCalibration(calibrationCtx, split, Const(graph, quantiles), *flagTrainableCalibration)
			allEmbeddings = append(allEmbeddings, calibrated)
		}
	}

	layer := Concatenate(allEmbeddings, -1)
	layer = layers.DenseWithBias(ctx.In(fmt.Sprintf("Dense_%d", 0)), layer, *flagNumNodes)
	for ii := 1; ii < *flagNumHiddenLayers; ii++ {
		// Add layer with residual connection.
		tmp := Sigmoid(layer)
		if *flagDropoutRate > 0 {
			tmp = layers.Dropout(ctx, tmp, Const(graph, shapes.CastAsDType(*flagDropoutRate, ModelDType)))
		}
		tmp = layers.DenseWithBias(ctx.In(fmt.Sprintf("Dense_%d", ii)), tmp, *flagNumNodes)
		layer = Add(layer, tmp)
	}
	layer = Sigmoid(layer)
	logits0 := layers.DenseWithBias(ctx.In("denseFinal"), layer, 1)
	return []*Node{logits0}
}
