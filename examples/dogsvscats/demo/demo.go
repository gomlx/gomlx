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

// demo for Dogs vs Cats library: you can run this program in 3 different ways:
//
//  1. With `demo --download`: it will simply download and unpack Kaggle Cats and Dogs dataset.
//  2. With `demo --pre`: It will pre-generate augmented data for subsequent training: since it spends more time
//     augmenting data than training, this is handy and accelerates training. But uses up lots of space (~13Gb with
//     the default number of generated epochs).
//  3. With `demo --train`: trains a CNN (convolutional neural network) model for "Dogs vs Cats".
package main

import (
	"flag"
	"fmt"
	"github.com/gomlx/gomlx/examples/dogsvscats"
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
	"github.com/gomlx/gomlx/models/inceptionv3"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	timage "github.com/gomlx/gomlx/types/tensor/image"
	"log"
	"os"
	"path"
)

var (
	flagDataDir     = flag.String("data", "~/tmp/dogs_vs_cats", "Directory to cache downloaded dataset and save checkpoints.")
	flagPreGenerate = flag.Bool("pre", false, "Pre-generate preprocessed image data to speed up training.")

	// ML Manager creation:
	flagNumThreads     = flag.Int("num_threads", -1, "Number of threads. Leave as -1 to use as many as there are cores.")
	flagNumReplicas    = flag.Int("num_replicas", 1, "Number of replicas.")
	flagPlatform       = flag.String("platform", "", "Platform to use, if empty uses the default one.")
	flagCheckpoint     = flag.String("checkpoint", "", "Directory save and load checkpoints from. If left empty, no checkpoints are created.")
	flagCheckpointKeep = flag.Int("checkpoint_keep", 10, "Number of checkpoints to keep, if --checkpoint is set.")

	// Training hyperparameters:
	flagModelType           = flag.String("model", "cnn", "Model types: \"cnn\" or \"inception\"")
	flagOptimizer           = flag.String("optimizer", "adamw", fmt.Sprintf("Optimizer, options: %q", slices.SortedKeys(optimizers.KnownOptimizers)))
	flagNumSteps            = flag.Int("steps", 2000, "Number of gradient descent steps to perform")
	flagBatchSize           = flag.Int("batch", dogsvscats.DefaultConfig.BatchSize, "Batch size for training")
	flagLearningRate        = flag.Float64("learning_rate", 0.0001, "Initial learning rate.")
	flagL2Regularization    = flag.Float64("l2_reg", 0, "L2 regularization on kernels. It doesn't interact well with --batch_norm.")
	flagNormalization       = flag.String("norm", "layer", fmt.Sprintf("Type of layer normalization to use. Valid values: %q.", slices.SortedKeys(layers.KnownNormalizers)))
	flagEval                = flag.Bool("eval", true, "Whether to evaluate trained model on test data in the end.")
	flagInceptionPreTrained = flag.Bool("pretrained", true, "If using inception model, whether to use the pre-trained weights to transfer learn")
	flagInceptionFineTuning = flag.Bool("finetuning", true, "If using inception model, whether to fine-tune the inception model")

	// Flat part of model, after convolutions and models being flattened:
	flagNumHiddenLayers = flag.Int("hidden_layers", 3, "Number of hidden layers, stacked with residual connection.")
	flagNumNodes        = flag.Int("num_nodes", 128, "Number of nodes in hidden layers.")

	// Flat augmentation hyperparameters:
	flagAngleStdDev  = flag.Float64("angle", 5.0, "Standard deviation of noise used to rotate the image. Disabled if --augment=false.")
	flagFlipRandomly = flag.Bool("flip", true, "Randomly flip the image horizontally. Disabled if --augment=false.")

	// Pre-Generation parameters:
	flagPreGenEpochs = flag.Int("pregen_epochs", 40, "Number of epochs to pre-generate for the training data. Each epoch will take ~310Mb")

	// UI
	flagUseProgressBar = flag.Bool("bar", true, "If to display a progress bar during training")
)

func AssertNoError(err error) {
	if err != nil {
		log.Fatalf("Failed: %+v", err)
	}
}

func main() {
	flag.Parse()
	*flagDataDir = data.ReplaceTildeInDir(*flagDataDir)
	if !data.FileExists(*flagDataDir) {
		AssertNoError(os.MkdirAll(*flagDataDir, 0777))
	}

	AssertNoError(dogsvscats.Download(*flagDataDir))
	AssertNoError(dogsvscats.FilterValidImages(*flagDataDir))

	config := &dogsvscats.Configuration{}
	*config = *dogsvscats.DefaultConfig
	config.DataDir = *flagDataDir
	config.AngleStdDev = *flagAngleStdDev
	config.FlipRandomly = *flagFlipRandomly
	config.BatchSize = *flagBatchSize
	if *flagModelType == "inception" {
		config.ModelImageSize = inceptionv3.MinimumImageSize
	}
	if *flagPreGenerate {
		dogsvscats.PreGenerate(config, *flagPreGenEpochs)
		return
	}

	trainModel(config)
	fmt.Printf("Finished!\n")
}

func trainModel(config *dogsvscats.Configuration) {
	trainDS, trainEvalDS, validationEvalDS := dogsvscats.CreateDatasets(config)

	// Manager handles creation of ML computation graphs, accelerator resources, etc.
	manager := BuildManager().NumThreads(*flagNumThreads).NumReplicas(*flagNumReplicas).Platform(*flagPlatform).MustDone()

	// Metrics we are interested.
	meanAccuracyMetric := metrics.NewMeanBinaryLogitsAccuracy("Mean Accuracy", "#acc")
	movingAccuracyMetric := metrics.NewMovingAverageBinaryLogitsAccuracy("Moving Average Accuracy", "~acc", 0.01)

	// Context holds the variables and hyperparameters for the model.
	ctx := context.NewContext(manager)
	ctx.SetParam(optimizers.LearningRateKey, *flagLearningRate)
	ctx.SetParam(layers.L2RegularizationKey, *flagL2Regularization)

	// Checkpoints saving.
	var checkpoint *checkpoints.Handler
	if *flagCheckpoint != "" {
		checkpointPath := data.ReplaceTildeInDir(*flagCheckpoint)
		if !path.IsAbs(checkpointPath) {
			checkpointPath = path.Join(config.DataDir, checkpointPath)
		}
		var err error
		checkpoint, err = checkpoints.Build(ctx).Dir(checkpointPath).Keep(*flagCheckpointKeep).Done()
		AssertNoError(err)
	}

	// Select the model type we are using:
	var modelFn func(ctx *context.Context, spec any, inputs []*Node) []*Node
	switch *flagModelType {
	case "cnn":
		modelFn = CnnModelGraph
	case "inception":
		modelFn = InceptionV3ModelGraph
		if *flagInceptionPreTrained {
			AssertNoError(inceptionv3.DownloadAndUnpackWeights(*flagDataDir))
		}
	}

	// Create a train.Trainer: this object will orchestrate running the model, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	trainer := train.NewTrainer(manager, ctx, modelFn,
		losses.BinaryCrossentropyLogits,
		optimizers.MustOptimizerByName(*flagOptimizer),
		[]metrics.Interface{movingAccuracyMetric}, // trainMetrics
		[]metrics.Interface{meanAccuracyMetric})   // evalMetrics

	// Use standard training loop.
	loop := train.NewLoop(trainer)
	if *flagUseProgressBar {
		commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.
	}

	// Attach a checkpoint.
	if checkpoint != nil {
		train.NTimesDuringLoop(loop, *flagCheckpointKeep, "checkpointing", 100, func(_ *train.Loop, _ []tensor.Tensor) error {
			return checkpoint.Save()
		})
	}

	// Loop for given number of steps.
	_, err := loop.RunSteps(trainDS, *flagNumSteps)
	AssertNoError(err)

	// Finally, print an evaluation on train and test datasets.
	if *flagEval {
		fmt.Println()
		err = commandline.ReportEval(trainer, trainEvalDS, validationEvalDS)
		AssertNoError(err)
	}
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

// CnnModelGraph builds the CNN model for our demo. It returns the logits, not the predictions, which works with
// most losses.
func CnnModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	_ = spec // Not needed.
	x := inputs[0]
	g := x.Graph()
	const filterSize = 32
	batchSize := x.Shape().Dimensions[0]
	logits := x
	for convIdx, imgSize := 0, 128; imgSize > 16; convIdx, imgSize = convIdx+1, imgSize/2 {
		ctx := ctx.In(fmt.Sprintf("conv_%d", convIdx))
		residual := logits
		if convIdx > 0 {
			logits = layers.Relu(logits)
		}
		logits = layers.Convolution(ctx, logits).Filters(filterSize).KernelSize(3).PadSame().Done()
		logits = layers.Relu(logits)
		logits = normalizeImage(ctx, logits)

		if convIdx > 0 {
			logits = Add(logits, residual)
		}
		logits = MaxPool(logits).Window(2).Done()
		AssertNoError(g.Error())
	}

	// Flatten the resulting image, and treat the convolved logits as flattened.
	logits = Reshape(logits, batchSize, -1)
	logits = FnnOnTop(ctx, logits)
	return []*Node{logits}
}

// FnnOnTop adds a feedforward neural network on top of the CNN layer.
func FnnOnTop(ctx *context.Context, logits *Node) *Node {
	for ii := 0; ii < *flagNumHiddenLayers; ii++ {
		ctx := ctx.In(fmt.Sprintf("dense_%d", ii))
		residual := logits
		// Add layer with residual connection.
		logits = layers.Relu(logits)
		logits = layers.DenseWithBias(ctx, logits, *flagNumNodes)
		logits = normalizeFeatures(ctx, logits)
		if ii >= 1 {
			logits = Add(logits, residual)
		}
	}
	logits = layers.Relu(logits)
	logits = layers.DenseWithBias(ctx.In("readout"), logits, 1)
	return logits
}

// InceptionV3ModelGraph uses an optionally pre-trained inception model.
//
// Results on validation after training 1000 steps, inception pre-trained weights used:
// - no scaling (from 0.0 to 1.0): 90.4% accuracy
// - with Keras scale (from -1.0 to 1.0): 90.6% accuracy
//
// Results if we don't use the pre-trained weights (it can probably get much better with more training):
// - no scaling (from 0.0 to 1.0): 62.5% accuracy
// - with Keras scale (from -1.0 to 1.0): 61.8% accuracy
func InceptionV3ModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	_ = spec           // Not needed.
	image := inputs[0] // Images scaled from 0.0 to 1.0
	channelsConfig := timage.ChannelsLast
	image = inceptionv3.PreprocessImage(image, 1.0, channelsConfig)

	var preTrainedPath string
	if *flagInceptionPreTrained {
		preTrainedPath = *flagDataDir
	}
	logits := inceptionv3.BuildGraph(ctx, image).PreTrained(preTrainedPath).
		SetPooling(inceptionv3.MaxPooling).Trainable(*flagInceptionFineTuning).Done()
	if !*flagInceptionFineTuning {
		logits = StopGradient(logits) // We don't want to train the inception model.
	}
	logits = FnnOnTop(ctx, logits)
	return []*Node{logits}
}
