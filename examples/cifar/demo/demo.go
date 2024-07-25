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

// Demo for cifar library: it implements 2 models, a FNN and a CNN.
package main

import (
	"flag"
	"fmt"
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/examples/cifar"
	"github.com/gomlx/gomlx/examples/notebook/gonb/margaid"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/checkpoints"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activations"
	"github.com/gomlx/gomlx/ml/layers/fnn"
	"github.com/gomlx/gomlx/ml/layers/kan"
	"github.com/gomlx/gomlx/ml/layers/regularizers"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/commandline"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/metrics"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"
	"log"
	"os"
	"time"

	_ "github.com/gomlx/gomlx/backends/xla"
)

var (
	flagDataDir = flag.String("data", "~/work/cifar", "Directory to cache downloaded and generated dataset files.")

	// Training hyperparameters:
	validModels = []string{"fnn", "kan", "cnn"}
	flagModel   = flag.String("model", validModels[0], fmt.Sprintf("Valid model types: %v", validModels))
	flagEval    = flag.Bool("eval", true, "Whether to evaluate the model on the validation data in the end.")

	// UI
	flagPlots = flag.Bool("plots", true, "Plots during training: perform periodic evaluations, "+
		"save results if --checkpoint is set and draw plots, if in a Jupyter notebook.")

	// Checkpointing.
	flagCheckpoint     = flag.String("checkpoint", "", "Directory save and load checkpoints from. If left empty, no checkpoints are created.")
	flagCheckpointKeep = flag.Int("checkpoint_keep", 3, "Number of checkpoints to keep, if --checkpoint is set.")
)

// createDefaultContext sets the context with default hyperparameters
func createDefaultContext() *context.Context {
	ctx := context.NewContext()
	ctx.RngStateReset()
	ctx.SetParams(map[string]any{
		"checkpoint":      "",
		"num_checkpoints": 3,
		"train_steps":     2000,
		"batch_size":      50,
		"plots":           true,

		optimizers.ParamOptimizer:           "adamw",
		optimizers.ParamLearningRate:        1e-4,
		optimizers.ParamAdamEpsilon:         1e-7,
		optimizers.ParamAdamDType:           "",
		optimizers.ParamCosineScheduleSteps: 0,
		activations.ParamActivation:         "sigmoid",
		layers.ParamDropoutRate:             0.0,
		regularizers.ParamL2:                1e-5,
		regularizers.ParamL1:                1e-5,

		// FNN network parameters:
		fnn.ParamNumHiddenLayers: 8,
		fnn.ParamNumHiddenNodes:  128,
		fnn.ParamResidual:        true,
		fnn.ParamNormalization:   "layer",

		// KAN network parameters:
		"kan":                       false, // Enable kan
		kan.ParamNumControlPoints:   20,    // Number of control points
		kan.ParamNumHiddenNodes:     4,
		kan.ParamNumHiddenLayers:    1,
		kan.ParamBSplineDegree:      2,
		kan.ParamBSplineMagnitudeL1: 1e-5,
		kan.ParamBSplineMagnitudeL2: 0.0,
	})
	return ctx
}

// DType used in the mode.
var DType = dtypes.Float32

// EvalBatchSize can be larger than training, it's more efficient.
const EvalBatchSize = 2000

func main() {
	// Flags with context settings.
	ctx := createDefaultContext()
	settings := commandline.CreateContextSettingsFlag(ctx, "")
	klog.InitFlags(nil)
	flag.Parse()

	*flagDataDir = data.ReplaceTildeInDir(*flagDataDir)
	if !data.FileExists(*flagDataDir) {
		must.M(os.MkdirAll(*flagDataDir, 0777))
	}
	must.M(commandline.ParseContextSettings(ctx, *settings))

	// Training:
	if *flagModel != "fnn" && *flagModel != "cnn" {
		log.Fatalf("Flag --model can only take values fnn or cnn, %q given.", *flagModel)
	}
	trainModel(ctx)
}

func trainModel(ctx *context.Context) {
	must.M(cifar.DownloadCifar10(*flagDataDir))
	must.M(cifar.DownloadCifar100(*flagDataDir))

	// Backend handles creation of ML computation graphs, accelerator resources, etc.
	backend := backends.New()
	fmt.Printf("Backend %q:\t%s\n", backend.Name(), backend.Description())

	// Create datasets used for training and evaluation.
	trainDS, evalOnTrainDS, evalOnTestDS := CreateDatasets(backend, *flagDataDir)

	// Create closure for model graph building function, that uses statically the dataset
	// used for its Dataset.GatherImage, to convert image indices to the actual images.
	// This is the signature of model function that the train.Trainer accepts.
	modelFn := PlainModelGraph
	if *flagModel == "cnn" {
		modelFn = ConvolutionModelGraph
	}

	// Metrics we are interested.
	meanAccuracyMetric := metrics.NewSparseCategoricalAccuracy("Mean Accuracy", "#acc")
	movingAccuracyMetric := metrics.NewMovingAverageSparseCategoricalAccuracy("Moving Average Accuracy", "~acc", 0.01)

	// Context holds the variables and hyperparameters for the model.
	ctx := createDefaultContext()

	// Checkpoints saving.
	var checkpoint *checkpoints.Handler
	var globalStep int
	if *flagCheckpoint != "" {
		checkpointPath := data.ReplaceTildeInDir(*flagCheckpoint)
		checkpoint = must.M1(checkpoints.Build(ctx).
			DirFromBase(checkpointPath, *flagDataDir).Keep(*flagCheckpointKeep).Done())
		fmt.Printf("Checkpointing model to %q\n", checkpoint.Dir())
		globalStep = int(optimizers.GetGlobalStep(ctx))
		if globalStep != 0 {
			fmt.Printf("Restarting training from global_step=%d\n", globalStep)
			ctx = ctx.Reuse()
		}
	}

	// Create a train.Trainer: this object will orchestrate running the model, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	trainer := train.NewTrainer(backend, ctx, modelFn,
		losses.SparseCategoricalCrossEntropyLogits,
		optimizers.FromContext(ctx),
		[]metrics.Interface{movingAccuracyMetric}, // trainMetrics
		[]metrics.Interface{meanAccuracyMetric})   // evalMetrics

	// Use standard training loop.
	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.

	// Attach a checkpoint: checkpoint every 1 minute of training.
	if checkpoint != nil {
		period := time.Minute * 1
		train.PeriodicCallback(loop, period, true, "saving checkpoint", 100,
			func(loop *train.Loop, metrics []*tensors.Tensor) error {
				fmt.Printf("\n[saving checkpoint@%d] [median train step (ms): %d]\n", loop.LoopStep, loop.MedianTrainStepDuration().Milliseconds())
				return checkpoint.Save()
			})
	}

	// Attach a margaid plots: plot points at exponential steps.
	// The points generated are saved along the checkpoint directory (if one is given).
	if *flagPlots {
		_ = margaid.NewDefault(loop, checkpoint.Dir(), 100, 1.1, evalOnTrainDS, evalOnTestDS)
	}

	// Loop for given number of steps.
	_, err := loop.RunSteps(trainDS, *flagNumSteps)
	must.M(err)
	fmt.Printf("\t[Step %d] median train step: %d microseconds\n",
		loop.LoopStep, loop.MedianTrainStepDuration().Microseconds())

	// Finally, print an evaluation on train and test datasets.
	if *flagEval {
		fmt.Println()
		err = commandline.ReportEval(trainer, evalOnTestDS, evalOnTrainDS)
		must.M(err)
	}

	// Release memory -- not really needed since we are exiting, just for the example.
	cifar.ResetCache()
}

func CreateDatasets(manager backends.Backend, dataDir string) (trainDS, trainEvalDS, validationEvalDS train.Dataset) {
	baseTrain := cifar.NewDataset(manager, "Training", dataDir, cifar.C10, DType, cifar.Train)
	baseTest := cifar.NewDataset(manager, "Validation", dataDir, cifar.C10, DType, cifar.Test)
	trainDS = baseTrain.Copy().BatchSize(*flagBatchSize, true).Shuffle().Infinite(true)
	trainEvalDS = baseTrain.BatchSize(EvalBatchSize, false)
	validationEvalDS = baseTest.BatchSize(EvalBatchSize, false)
	return
}

func normalizeImage(ctx *context.Context, x *Node) *Node {
	x.AssertRank(4) // [batch_size, width, height, depth]
	switch *flagNormalization {
	case "layer":
		return layers.LayerNormalization(ctx, x, 1, 2).ScaleNormalization(false).Done()
	case "batch":
		return layers.BatchNormalization(ctx, x, -1).Done()
	case "none":
		return x
	}
	Panicf("invalid normalization selected %q -- valid values are batch, layer, none", *flagNormalization)
	return nil
}

func normalizeFeatures(ctx *context.Context, x *Node) *Node {
	x.AssertRank(2) // [batch_size, embedding_dim]
	switch *flagNormalization {
	case "layer":
		return layers.LayerNormalization(ctx, x, -1).Done()
	case "batch":
		return layers.BatchNormalization(ctx, x, -1).Done()
	case "none":
		return x
	}
	Panicf("invalid normalization selected %q -- valid values are batch, layer, none", *flagNormalization)
	return nil
}

// PlainModelGraph implements train.ModelFn, and returns the logit Node, given the input image.
// It's a basic FNN (Feedforward Neural Network), so no convolutions. It is meant only as an example.
//
// If dataset is not nil, assumes batchImage contain instead indices, and that the images need to be
// gathered from the dataset table (cifar.Dataset.GatherImagesGraph).
//
// Notice this cannot be used as a ModelFn for train.Trainer because of the dataset static parameter: one need
// to create a small closure for that, see above for an example.
func PlainModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	g := inputs[0].Graph()
	batchedImages := inputs[0]
	batchSize := batchedImages.Shape().Dimensions[0]
	logits := Reshape(batchedImages, batchSize, -1)
	{
		ctx := ctx.In("Dense_0")
		logits = layers.DenseWithBias(ctx, logits, *flagNumNodes)
		logits = normalizeFeatures(ctx, logits)
	}
	for ii := 1; ii < *flagNumHiddenLayers; ii++ {
		ctx := ctx.In(fmt.Sprintf("Dense_%d", ii))
		// Add layer with residual connection.
		tmp := Sigmoid(logits)
		if *flagDropoutRate > 0 {
			tmp = layers.Dropout(ctx, tmp, Const(g, shapes.CastAsDType(*flagDropoutRate, tmp.DType())))
		}
		tmp = layers.DenseWithBias(ctx, tmp, *flagNumNodes)
		tmp = normalizeFeatures(ctx, tmp)
		logits = Add(logits, tmp)
	}
	logits = Sigmoid(logits)
	logits = layers.DenseWithBias(ctx.In("denseFinal"), logits, len(cifar.C10Labels))
	return []*Node{logits}
}

// ConvolutionModelGraph implements train.ModelFn and returns the logit Node, given the input image.
// It's a straight forward CNN (Convolution Neural Network) model.
//
// If dataset is not nil, assumes batchImage contain instead indices, and that the images need to be
// gathered from the dataset table (cifar.Dataset.GatherImagesGraph).
//
// This is modeled after the TensorFlow/Keras example in https://www.tensorflow.org/tutorials/images/cnn.
//
// Notice this cannot be used as a ModelFn for train.Trainer because of the dataset static parameter: one need
// to create a small closure for that, see above for an example.
func ConvolutionModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	g := inputs[0].Graph()
	batchedImages := inputs[0]
	batchSize := batchedImages.Shape().Dimensions[0]
	logits := batchedImages
	{
		ctx := ctx.In("conv_0")
		logits = layers.Convolution(ctx, logits).Filters(32).KernelSize(3).Done()
		logits = normalizeImage(ctx, logits)
		logits = activations.Relu(logits)
		logits = MaxPool(logits).Window(2).Done()
	}
	{
		ctx := ctx.In("conv_1")
		logits = layers.Convolution(ctx, logits).Filters(64).KernelSize(3).Done()
		logits = normalizeImage(ctx, logits)
		logits = activations.Relu(logits)
		logits = MaxPool(logits).Window(2).Done()
	}
	{
		ctx := ctx.In("conv_2")
		logits = layers.Convolution(ctx, logits).Filters(64).KernelSize(3).Done()
		logits = normalizeImage(ctx, logits)
		logits = Reshape(logits, batchSize, -1)
		logits = activations.Relu(logits)
	}
	{
		ctx := ctx.In("dense_0")
		logits = layers.DenseWithBias(ctx, logits, *flagNumNodes)
		logits = normalizeFeatures(ctx, logits)
	}
	for ii := 1; ii < *flagNumHiddenLayers; ii++ {
		ctx := ctx.In(fmt.Sprintf("dense_%d", ii))
		// Add layer with residual connection.
		tmp := activations.Relu(logits)
		if *flagDropoutRate > 0 {
			tmp = layers.Dropout(ctx, tmp, Const(g, shapes.CastAsDType(*flagDropoutRate, tmp.DType())))
		}
		tmp = layers.DenseWithBias(ctx, tmp, *flagNumNodes)
		tmp = normalizeFeatures(ctx, tmp)
		logits = Add(logits, tmp)
	}
	logits = activations.Relu(logits)
	logits = layers.DenseWithBias(ctx.In("denseFinal"), logits, len(cifar.C10Labels))
	return []*Node{logits}
}
