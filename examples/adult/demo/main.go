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

// UCI-Adult demo trainer.
// It supports FNNs, KAN and DiscreteKAN models, with many different options.
// All input features are calibrated with piecewise linear functions.
package main

import (
	"flag"
	"fmt"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/examples/adult"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/checkpoints"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activations"
	"github.com/gomlx/gomlx/ml/layers/batchnorm"
	"github.com/gomlx/gomlx/ml/layers/fnn"
	"github.com/gomlx/gomlx/ml/layers/kan"
	"github.com/gomlx/gomlx/ml/layers/regularizers"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/metrics"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/ml/train/optimizers/cosineschedule"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gomlx/ui/gonb/margaid"
	"github.com/gomlx/gomlx/ui/gonb/plotly"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/janpfeifer/must"
	"github.com/schollz/progressbar/v3"
	"k8s.io/klog/v2"
	"time"

	_ "github.com/gomlx/gomlx/backends/default"
)

var (
	// ModelDType used for the model.
	ModelDType = dtypes.Float32
)

func createDefaultContext() *context.Context {
	ctx := context.New()
	ctx.RngStateReset()
	ctx.SetParams(map[string]any{
		// Number of steps to take during training.
		"train_steps": 5000,
		"batch_size":  128,

		// "plots" trigger generating intermediary eval data for plotting, and if running in GoNB, to actually
		// draw the plot with Plotly.
		//
		// From the command-line, an easy way to monitor the metrics being generated during the training of a model
		// is using the gomlx_checkpoints tool:
		//
		//	$ gomlx_checkpoints --metrics --metrics_labels --metrics_types=accuracy  --metrics_names='E(bat)/#loss,E(tes)/#loss' --loop=3s fnn
		"plots": true,

		optimizers.ParamOptimizer:       "adam",
		optimizers.ParamLearningRate:    0.001,
		optimizers.ParamAdamEpsilon:     1e-7,
		optimizers.ParamAdamDType:       "",
		cosineschedule.ParamPeriodSteps: 0,
		activations.ParamActivation:     "sigmoid",
		layers.ParamDropoutRate:         0.0,
		regularizers.ParamL2:            1e-5,
		regularizers.ParamL1:            1e-5,

		// FNN network parameters:
		fnn.ParamNumHiddenLayers: 1,
		fnn.ParamNumHiddenNodes:  4,
		fnn.ParamResidual:        true,
		fnn.ParamNormalization:   "layer",

		// KAN network parameters:
		"kan":                                 false, // Enable kan
		kan.ParamNumControlPoints:             20,    // Number of control points
		kan.ParamNumHiddenNodes:               4,
		kan.ParamNumHiddenLayers:              1,
		kan.ParamBSplineDegree:                2,
		kan.ParamBSplineMagnitudeL1:           1e-5,
		kan.ParamBSplineMagnitudeL2:           0.0,
		kan.ParamDiscrete:                     false,
		kan.ParamDiscretePerturbation:         "triangular",
		kan.ParamDiscreteSoftness:             0.1,
		kan.ParamDiscreteSoftnessSchedule:     kan.SoftnessScheduleNone.String(),
		kan.ParamDiscreteSplitPointsTrainable: true,
		kan.ParamResidual:                     true,
	})
	return ctx
}

var (
	flagDataDir    = flag.String("data", "~/work/uci-adult", "Directory to save and load downloaded and generated dataset files.")
	flagCheckpoint = flag.String("checkpoint", "", "Checkpoint subdirectory under the --data directory. "+
		"If empty does not use checkpoints. If absolute path, use that instead.")
	flagForceDownload = flag.Bool("force_download", false, "Force re-download of Adult dataset files.")

	flagNumQuantiles = flag.Int("quantiles", 100, "Max number of quantiles to use for numeric features, used during piece-wise linear calibration. It will only use unique values, so if there are fewer variability, fewer quantiles are used.")
	flagEmbeddingDim = flag.Int("embedding_dim", 8, "Default embedding dimension for categorical values.")
	flagVerbosity    = flag.Int("verbosity", 1, "Level of verbosity, the higher the more verbose.")

	flagUseCategorical       = flag.Bool("use_categorical", true, "Use categorical features.")
	flagUseContinuous        = flag.Bool("use_continuous", true, "Use continuous features.")
	flagTrainableCalibration = flag.Bool("trainable_calibration", true, "Allow piece-wise linear calibration to adjust outputs.")
)

func main() {
	// Flags with context settings.
	ctx := createDefaultContext()
	settings := commandline.CreateContextSettingsFlag(ctx, "")
	klog.InitFlags(nil)
	flag.Parse()
	paramsSet := must.M1(commandline.ParseContextSettings(ctx, *settings))
	err := mainWithContext(ctx, *flagDataDir, *flagCheckpoint, paramsSet)
	if err != nil {
		klog.Fatalf("Failed with error: %+v", err)
	}
}

func mainWithContext(ctx *context.Context, dataDir, checkpointPath string, paramsSet []string) error {
	backend := backends.New()
	dataDir = data.ReplaceTildeInDir(dataDir)
	if *flagVerbosity >= 1 {
		fmt.Printf("Backend: %s\n\t%s\n", backend.Name(), backend.Description())
		fmt.Println(commandline.SprintContextSettings(ctx))
	}

	// Checkpoints loading (and saving)
	var checkpoint *checkpoints.Handler
	if checkpointPath != "" {
		numCheckpointsToKeep := context.GetParamOr(ctx, "num_checkpoints", 3)
		checkpoint = must.M1(checkpoints.Build(ctx).
			DirFromBase(checkpointPath, dataDir).
			Keep(numCheckpointsToKeep).
			ExcludeParams(append(paramsSet, "train_steps", "plots", "num_checkpoints")...).
			Done())
	}

	// Load training data and initialize statistics (vocabularies and quantiles).
	adult.LoadAndPreprocessData(dataDir, *flagNumQuantiles, *flagForceDownload, *flagVerbosity)

	// Crate Backend and upload data to device tensors.
	if *flagVerbosity >= 1 {
		fmt.Printf("Backend: %s\n", backend.Name())
	}
	if *flagVerbosity >= 2 {
		adult.PrintBatchSamples(backend, adult.Data.Train)
	}

	// Create datasets for training and evaluation.
	batchSize := context.GetParamOr(ctx, "batch_size", 128)
	trainDS := adult.NewDataset(backend, adult.Data.Train, "batched train")
	trainEvalDS := trainDS.Copy().BatchSize(batchSize, false)
	testEvalDS := adult.NewDataset(backend, adult.Data.Test, "test").
		BatchSize(batchSize, false)
	// For training, we shuffle and loop indefinitely.
	trainDS.BatchSize(batchSize, true).Shuffle().Infinite(true)

	// Metrics we are interested.
	meanAccuracyMetric := metrics.NewMeanBinaryLogitsAccuracy("Mean Accuracy", "#acc")
	movingAccuracyMetric := metrics.NewMovingAverageBinaryLogitsAccuracy("Moving Average Accuracy", "~acc", 0.01)

	// Create a train.Trainer: this object will orchestrate running the model, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	trainer := train.NewTrainer(backend, ctx, ModelGraph, losses.BinaryCrossentropyLogits,
		optimizers.FromContext(ctx),
		[]metrics.Interface{movingAccuracyMetric}, // trainMetrics
		[]metrics.Interface{meanAccuracyMetric})   // evalMetrics

	// Use standard training loop.
	loop := train.NewLoop(trainer)
	commandline.ProgressbarStyle = progressbar.ThemeUnicode
	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.

	// Attach a checkpoint saver.
	if checkpoint != nil {
		period := time.Minute * 1
		train.PeriodicCallback(loop, period, true, "saving checkpoint", 100,
			func(loop *train.Loop, metrics []*tensors.Tensor) error {
				fmt.Printf("\n[saving checkpoint@%d] [median train step (ms): %d]\n", loop.LoopStep, loop.MedianTrainStepDuration().Milliseconds())
				return checkpoint.Save()
			})
	}

	// Attach Plotly plots: plot points at exponential steps.
	// The points generated are saved along the checkpoint directory (if one is given).
	if context.GetParamOr(ctx, margaid.ParamPlots, false) {
		_ = plotly.New().
			WithCheckpoint(checkpoint).
			Dynamic().
			WithDatasets(trainEvalDS, testEvalDS).
			ScheduleExponential(loop, 200, 1.2).
			WithBatchNormalizationAveragesUpdate(trainEvalDS)
	}

	// Train up to "train_steps".
	globalStep := int(optimizers.GetGlobalStep(ctx))
	trainSteps := context.GetParamOr(ctx, "train_steps", 0)
	if globalStep < trainSteps {
		if globalStep != 0 {
			fmt.Printf("\t- restarting training from global_step=%d\n", globalStep)
			trainer.SetContext(ctx.Reuse())
		}
		_, err := loop.RunSteps(trainDS, trainSteps-globalStep)
		if err != nil {
			return err
		}
		fmt.Printf("\t[Step %d] median train step: %d microseconds\n", loop.LoopStep, loop.MedianTrainStepDuration().Microseconds())
		fmt.Println()
		// Update batch normalization averages, if they are used.
		if batchnorm.UpdateAverages(trainer, trainEvalDS) {
			fmt.Println("\tUpdated batch normalization mean/variances averages.")
			must.M(checkpoint.Save())
		}

	} else {
		fmt.Printf("\t - target train_steps=%d already reached. To train further, set a number larger than "+
			"current global step.\n", trainSteps)
	}

	if *flagVerbosity >= 2 {
		fmt.Println("\nVariables:")
		ctx.EnumerateVariables(func(v *context.Variable) {
			if !v.Trainable {
				return
			}
			fmt.Printf("\t%s : %s -> %s\n", v.Scope(), v.Name(), v.Shape())
		})
	}

	// Finally, print an evaluation on train and test datasets.
	return commandline.ReportEval(trainer, trainEvalDS, testEvalDS)
}

// ModelGraph outputs the logits (not the probabilities). The parameter inputs should contain 3 tensors:
//
// - categorical inputs, shaped  `(int64)[batch_size, len(VocabulariesFeatures)]`
// - continuous inputs, shaped `(float32)[batch_size, len(Quantiles)]`
// - weights: not currently used, but shaped `(float32)[batch_size, 1]`.
func ModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	_ = spec // Not used, since the dataset is always the same.
	g := inputs[0].Graph()
	dtype := inputs[1].DType() // From continuous features.
	ctx = ctx.In("model")

	// Use Cosine schedule of the learning rate, if hyperparameter is set to a value > 0.
	cosineschedule.New(ctx, g, dtype).FromContext().Done()

	categorical, continuous := inputs[0], inputs[1]
	batchSize := categorical.Shape().Dimensions[0]

	// Feature preprocessing:
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
			embedding := layers.Embedding(embedCtx, split, ModelDType, vocabSize, *flagEmbeddingDim, false)
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
	logits := Concatenate(allEmbeddings, -1)
	logits.AssertDims(batchSize, -1) // 2-dim tensor, with batch size as the leading dimension (-1 means it is not checked).

	// Model itself is an FNN or a KAN.
	if context.GetParamOr(ctx, "kan", false) {
		// Use KAN, all configured by context hyperparameters. See createDefaultContext for defaults.
		logits = kan.New(ctx.In("kan"), logits, 1).Done()
	} else {
		// Normal FNN, all configured by context hyperparameters. See createDefaultContext for defaults.
		logits = fnn.New(ctx.In("fnn"), logits, 1).Done()
	}
	logits.AssertDims(batchSize, 1) // 2-dim tensor, with batch size as the leading dimension.
	return []*Node{logits}
}
