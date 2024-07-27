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
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"
	"log"
	"os"
	"slices"
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
		kan.ParamNumControlPoints:   20, // Number of control points
		kan.ParamNumHiddenNodes:     32,
		kan.ParamNumHiddenLayers:    4,
		kan.ParamBSplineDegree:      2,
		kan.ParamBSplineMagnitudeL1: 1e-5,
		kan.ParamBSplineMagnitudeL2: 0.0,
		kan.ParamDiscrete:           false,
		kan.ParamDiscreteSoftness:   0.1,

		// CNN
		layers.ParamNormalizationType: "layer",
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
	fmt.Println(commandline.SprintContextSettings(ctx))

	// Training:
	if slices.Index(validModels, *flagModel) == -1 {
		log.Fatalf("Flag --model must take one value from %v, got %q", validModels, *flagModel)
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
	batchSize := context.GetParamOr(ctx, "batch_size", int(0))
	if batchSize <= 0 {
		Panicf("Batch size must be > 0 (maybe it was not set?): %d", batchSize)
	}
	trainDS, evalOnTrainDS, evalOnTestDS := CreateDatasets(backend, *flagDataDir, batchSize)

	// Create closure for model graph building function, that uses statically the dataset
	// used for its Dataset.GatherImage, to convert image indices to the actual images.
	// This is the signature of model function that the train.Trainer accepts.
	modelFn := PlainModelGraph
	if *flagModel == "cnn" {
		modelFn = ConvolutionModelGraph
	}
	fmt.Printf("Model: %s\n", *flagModel)

	// Metrics we are interested.
	meanAccuracyMetric := metrics.NewSparseCategoricalAccuracy("Mean Accuracy", "#acc")
	movingAccuracyMetric := metrics.NewMovingAverageSparseCategoricalAccuracy("Moving Average Accuracy", "~acc", 0.01)

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
	numTrainSteps := context.GetParamOr(ctx, "train_steps", 0)
	if globalStep < numTrainSteps {
		_ = must.M1(loop.RunSteps(trainDS, numTrainSteps-globalStep))
		fmt.Printf("\t[Step %d] median train step: %d microseconds\n", loop.LoopStep, loop.MedianTrainStepDuration().Microseconds())
		fmt.Println()
	} else {
		fmt.Printf("\t - target train_steps=%d already reached. To train further, set a number additional "+
			"to current global step.\n", numTrainSteps)
	}
	fmt.Printf("\t[Step %d] median train step: %d microseconds\n",
		loop.LoopStep, loop.MedianTrainStepDuration().Microseconds())

	// Finally, print an evaluation on train and test datasets.
	if *flagEval {
		fmt.Println()
		must.M(commandline.ReportEval(trainer, evalOnTestDS, evalOnTrainDS))
	}

	// Release memory -- not really needed since we are exiting, just for the example.
	cifar.ResetCache()
}

func CreateDatasets(backend backends.Backend, dataDir string, batchSize int) (trainDS, trainEvalDS, validationEvalDS train.Dataset) {
	baseTrain := cifar.NewDataset(backend, "Training", dataDir, cifar.C10, DType, cifar.Train)
	baseTest := cifar.NewDataset(backend, "Validation", dataDir, cifar.C10, DType, cifar.Test)
	trainDS = baseTrain.Copy().BatchSize(batchSize, true).Shuffle().Infinite(true)
	trainEvalDS = baseTrain.BatchSize(EvalBatchSize, false)
	validationEvalDS = baseTest.BatchSize(EvalBatchSize, false)
	return
}

func normalizeImage(ctx *context.Context, x *Node) *Node {
	x.AssertRank(4) // [batch_size, width, height, depth]
	normalizationType := context.GetParamOr(ctx, layers.ParamNormalizationType, "none")
	switch normalizationType {
	case "layer":
		return layers.LayerNormalization(ctx, x, 1, 2).ScaleNormalization(false).Done()
	case "batch":
		return layers.BatchNormalization(ctx, x, -1).Done()
	case "none", "":
		return x
	}
	Panicf("invalid normalization type selected %q (hyperparameter %q) -- valid values are batch, layer, none", normalizationType, layers.ParamNormalizationType)
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
	batchedImages := inputs[0]
	batchSize := batchedImages.Shape().Dimensions[0]
	logits := Reshape(batchedImages, batchSize, -1)
	numClasses := len(cifar.C10Labels)
	if *flagModel == "kan" {
		// Configuration of the KAN layer(s) use the context hyperparameters.
		logits = kan.New(ctx, logits, numClasses).Done()
	} else {
		// Configuration of the FNN layer(s) use the context hyperparameters.
		logits = fnn.New(ctx, logits, numClasses).Done()
	}
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

	// Here logits are flat, and we can use the usual FNN/KAN.
	numClasses := len(cifar.C10Labels)
	if *flagModel == "kan" {
		// Configuration of the KAN layer(s) use the context hyperparameters.
		logits = kan.New(ctx, logits, numClasses).Done()
	} else {
		// Configuration of the FNN layer(s) use the context hyperparameters.
		logits = fnn.New(ctx, logits, numClasses).Done()
	}
	return []*Node{logits}
}
