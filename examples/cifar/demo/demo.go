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
	"github.com/gomlx/gomlx/examples/cifar"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/commandline"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/metrics"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
	"log"
	"os"
)

func AssertNoError(err error) {
	if err != nil {
		log.Fatalf("Failed: %+v", err)
	}
}

var (
	flagDataDir = flag.String("data", "~/tmp/cifar", "Directory to cache downloaded and generated dataset files.")

	// ML Manager creation:
	flagNumThreads  = flag.Int("num_threads", -1, "Number of threads. Leave as -1 to use as many as there are cores.")
	flagNumReplicas = flag.Int("num_replicas", 1, "Number of replicas.")
	flagPlatform    = flag.String("platform", "", "Platform to use, if empty uses the default one.")

	// Training hyperparameters:
	flagModel        = flag.String("model", "fnn", "Model type: fnn or cnn.")
	flagOptimizer    = flag.String("optimizer", "adam", "Optimizer, options: adam or sgd.")
	flagNumSteps     = flag.Int("steps", 2000, "Number of gradient descent steps to perform")
	flagBatchSize    = flag.Int("batch", 100, "Batch size for training")
	flagLearningRate = flag.Float64("learning_rate", 0.0001, "Initial learning rate.")

	// Model hyperparameters:
	flagNumHiddenLayers  = flag.Int("hidden_layers", 8, "Number of hidden layers, stacked with residual connection.")
	flagNumNodes         = flag.Int("num_nodes", 128, "Number of nodes in hidden layers.")
	flagDropoutRate      = flag.Float64("dropout", 0, "Dropout rate")
	flagL2Regularization = flag.Float64("l2_reg", 0, "L2 regularization on kernels. It doesn't interact well with --batch_norm.")
	flagNormalization    = flag.String("norm", "batch", "Type of normalization to use. Valid values are \"none\", \"batch\", \"layer\".")

	// UI
	flagUseProgressBar = flag.Bool("bar", true, "If to display a progress bar during training")

	// All data, kept in memory since it usually fits, and makes things much faster and easier.
	images10, labels10, images100, labels100 tensor.Tensor
)

// Data type used for the demo.
var dtype = shapes.Float32

const EvalBatchSize = 2000 // Can be larger than training, more efficient.

func main() {
	flag.Parse()
	*flagDataDir = data.ReplaceTildeInDir(*flagDataDir)
	if !data.FileExists(*flagDataDir) {
		AssertNoError(os.MkdirAll(*flagDataDir, 0777))
	}
	AssertNoError(cifar.DownloadCifar10(*flagDataDir))
	AssertNoError(cifar.DownloadCifar100(*flagDataDir))

	// Training:
	if *flagModel != "fnn" && *flagModel != "cnn" {
		log.Fatalf("Flag --model can only take values fnn or cnn, %q given.", *flagModel)
	}
	trainModel()
}

func trainModel() {
	// Manager handles creation of ML computation graphs, accelerator resources, etc.
	manager := BuildManager().NumThreads(*flagNumThreads).NumReplicas(*flagNumReplicas).Platform(*flagPlatform).MustDone()
	fmt.Printf("Platform: %s\n", manager.Platform())

	// Create datasets used for training and evaluation.
	trainDS, err := cifar.NewDataset("Cifar-10 Batched Train", *flagDataDir, cifar.C10, dtype, cifar.Train, *flagBatchSize, false) // loops forever.
	AssertNoError(err)
	evalOnTestDS, err := cifar.NewDataset("Cifar-10 Test", *flagDataDir, cifar.C10, dtype, cifar.Test, EvalBatchSize, true) // 1 epoch.
	AssertNoError(err)
	evalOnTrainDS, err := cifar.NewDataset("Cifar-10 Train", *flagDataDir, cifar.C10, dtype, cifar.Train, EvalBatchSize, true) // 1 epoch.
	AssertNoError(err)

	// Create closure for model graph building function, that uses statically the dataset
	// used for its Dataset.GatherImage, to convert image indices to the actual images.
	// This is the signature of model function that the train.Trainer accepts.
	modelFn := FNNModelGraph
	if *flagModel == "cnn" {
		modelFn = CNNModelGraph
	}

	// Metrics we are interested.
	meanAccuracyMetric := metrics.NewSparseCategoricalAccuracy("Mean Accuracy", "#acc")
	movingAccuracyMetric := metrics.NewMovingAverageSparseCategoricalAccuracy("Moving Average Accuracy", "~acc", 0.01)

	// Context holds the variables and hyperparameters for the model.
	ctx := context.NewContext(manager)
	ctx.SetParam(optimizers.LearningRateKey, *flagLearningRate)
	ctx.SetParam(layers.L2RegularizationKey, *flagL2Regularization)

	// Create a train.Trainer: this object will orchestrate running the model, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	trainer := train.NewTrainer(manager, ctx, modelFn,
		losses.SparseCategoricalCrossEntropyLogits,
		optimizers.MustOptimizerByName(*flagOptimizer),
		[]metrics.Interface{movingAccuracyMetric}, // trainMetrics
		[]metrics.Interface{meanAccuracyMetric})   // evalMetrics

	// Use standard training loop.
	loop := train.NewLoop(trainer)
	if *flagUseProgressBar {
		commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.
	}

	// Loop for given number of steps.
	_, err = loop.RunSteps(trainDS, *flagNumSteps)
	AssertNoError(err)

	// Finally print an evaluation on train and test datasets.
	fmt.Println()
	err = commandline.ReportEval(trainer, evalOnTestDS, evalOnTrainDS)
	AssertNoError(err)

	// Release memory -- not really needed since we are exiting, just for the example.
	cifar.ResetCache()
}

func normalizeImage(ctx *context.Context, x *Node) *Node {
	switch *flagNormalization {
	case "layer":
		return layers.LayerNormalization(ctx, x, 1, 2).ScaleNormalization(false).Done()
	case "batch":
		return layers.BatchNormalization(ctx, x, -1).Done()
	case "none":
		return x
	}
	g := x.Graph()
	g.SetErrorf("invalid normalization selected %q -- valid values are batch, layer, none", *flagNormalization)
	return g.InvalidNode()
}

func normalizeFeatures(ctx *context.Context, x *Node) *Node {
	switch *flagNormalization {
	case "layer":
		return layers.LayerNormalization(ctx, x, -1).Done()
	case "batch":
		return layers.BatchNormalization(ctx, x, -1).Done()
	case "none":
		return x
	}
	g := x.Graph()
	g.SetErrorf("invalid normalization selected %q -- valid values are batch, layer, none", *flagNormalization)
	return g.InvalidNode()
}

func getBatchedImages(ctx *context.Context, spec any, inputs []*Node) (batchedImages *Node) {
	g := inputs[0].Graph()
	if !g.Ok() {
		return nil
	}

	if spec != nil {
		// spec should hold the dataset that converts the image indices to the actual
		// images, by gathering from a large variable that holds all the images.
		dataset, ok := spec.(*cifar.Dataset)
		if !ok {
			g.SetErrorf("spec given to FNNModelGraph is not a *cifarDataset, instead got %T", spec)
			return nil
		}
		// We assume that batchedImages passed is actually a list of indices, and we need to gather
		// the actual images.
		batchedImages = dataset.GatherImagesGraph(ctx, inputs[0])
	} else {
		// If a spec was not given, we assume the raw images are being fed for inference.
		batchedImages = inputs[0]
	}
	if !batchedImages.Ok() {
		g.SetErrorf("failed to load batch of images")
		return nil
	}
	return
}

// FNNModelGraph implements train.ModelFn, and returns the logit Node, given the input image.
// It's a very simple FNN (Feedforward Neural Network), so no convolutions. It is meant only as an example.
//
// If dataset is not nil, assumes batchImage contain instead indices, and that the images need to be
// gathered from the dataset table (cifar.Dataset.GatherImagesGraph).
//
// Notice this cannot be used as a ModelFn for train.Trainer because of the dataset static parameter: one need
// to create a small closure for that, see above for an example.
func FNNModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	g := inputs[0].Graph()
	if !g.Ok() {
		return nil
	}
	batchedImages := getBatchedImages(ctx, spec, inputs)
	if !g.Ok() {
		return nil
	}
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

// CNNModelGraph implements train.ModelFn and returns the logit Node, given the input image.
// It's a straight forward CNN (Convolution Neural Network) model.
//
// If dataset is not nil, assumes batchImage contain instead indices, and that the images need to be
// gathered from the dataset table (cifar.Dataset.GatherImagesGraph).
//
// This is modeled after the TensorFlow/Keras example in https://www.tensorflow.org/tutorials/images/cnn.
//
// Notice this cannot be used as a ModelFn for train.Trainer because of the dataset static parameter: one need
// to create a small closure for that, see above for an example.
func CNNModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	g := inputs[0].Graph()
	if !g.Ok() {
		return nil
	}
	batchedImages := getBatchedImages(ctx, spec, inputs)
	if !g.Ok() {
		return nil
	}
	batchSize := batchedImages.Shape().Dimensions[0]
	logits := batchedImages
	{
		ctx := ctx.In("conv_0")
		logits = layers.Convolution(ctx, logits).Filters(32).KernelSize(3).Done()
		logits = normalizeImage(ctx, logits)
		logits = layers.Relu(logits)
		logits = MaxPool(logits).Window(2).Done()
	}
	{
		ctx := ctx.In("conv_1")
		logits = layers.Convolution(ctx, logits).Filters(64).KernelSize(3).Done()
		logits = normalizeImage(ctx, logits)
		logits = layers.Relu(logits)
		logits = MaxPool(logits).Window(2).Done()
	}
	{
		ctx := ctx.In("conv_2")
		logits = layers.Convolution(ctx, logits).Filters(64).KernelSize(3).Done()
		logits = normalizeImage(ctx, logits)
		logits = Reshape(logits, batchSize, -1)
		logits = layers.Relu(logits)
	}
	{
		ctx := ctx.In("dense_0")
		logits = layers.DenseWithBias(ctx, logits, *flagNumNodes)
		logits = normalizeFeatures(ctx, logits)
	}
	for ii := 1; ii < *flagNumHiddenLayers; ii++ {
		ctx := ctx.In(fmt.Sprintf("dense_%d", ii))
		// Add layer with residual connection.
		tmp := layers.Relu(logits)
		if *flagDropoutRate > 0 {
			tmp = layers.Dropout(ctx, tmp, Const(g, shapes.CastAsDType(*flagDropoutRate, tmp.DType())))
		}
		tmp = layers.DenseWithBias(ctx, tmp, *flagNumNodes)
		tmp = normalizeFeatures(ctx, tmp)
		logits = Add(logits, tmp)
	}
	logits = layers.Relu(logits)
	logits = layers.DenseWithBias(ctx.In("denseFinal"), logits, len(cifar.C10Labels))
	return []*Node{logits}
}
