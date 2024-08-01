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

// CIFAR-10 demo trainer.
// It supports CNNs, FNNs, KAN and DiscreteKAN models, with many different options.
package main

import (
	"flag"
	"fmt"
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/examples/cifar"
	"github.com/gomlx/gomlx/examples/notebook/gonb/plotly"
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
	"os"
	"slices"
	"strings"
	"time"

	_ "github.com/gomlx/gomlx/backends/xla"
)

var (
	flagDataDir = flag.String("data", "~/work/cifar", "Directory to cache downloaded and generated dataset files.")

	// Training hyperparameters:
	validModels   = []string{"fnn", "kan", "cnn", "cnn-kan"}
	flagModel     = flag.String("model", validModels[0], fmt.Sprintf("Valid model types: %v", validModels))
	flagEval      = flag.Bool("eval", true, "Whether to evaluate the model on the validation data in the end.")
	flagVerbosity = flag.Int("verbosity", 1, "Level of verbosity, the higher the more verbose.")

	// Checkpointing.
	flagCheckpoint     = flag.String("checkpoint", "", "Directory save and load checkpoints from. If left empty, no checkpoints are created.")
	flagCheckpointKeep = flag.Int("checkpoint_keep", 3, "Number of checkpoints to keep, if --checkpoint is set.")
)

// createDefaultContext sets the context with default hyperparameters
func createDefaultContext() *context.Context {
	ctx := context.New()
	ctx.RngStateReset()
	ctx.SetParams(map[string]any{
		"checkpoint":      "",
		"num_checkpoints": 3,
		"train_steps":     3000,

		// batch_size for training.
		"batch_size": 64,

		// eval_batch_size can be larger than training, it's more efficient.
		"eval_batch_size": 200,

		// "plots" trigger generating intermediary eval data for plotting, and if running in GoNB, to actually
		// draw the plot with Plotly.
		plotly.ParamPlots: true,

		optimizers.ParamOptimizer:           "adamw",
		optimizers.ParamLearningRate:        1e-4,
		optimizers.ParamAdamEpsilon:         1e-7,
		optimizers.ParamAdamDType:           "",
		optimizers.ParamCosineScheduleSteps: 0,
		activations.ParamActivation:         "swish",
		layers.ParamDropoutRate:             0.0,
		regularizers.ParamL2:                1e-5,
		regularizers.ParamL1:                1e-5,

		// FNN network parameters:
		fnn.ParamNumHiddenLayers: 8,
		fnn.ParamNumHiddenNodes:  128,
		fnn.ParamResidual:        true,
		fnn.ParamNormalization:   "none",

		// KAN network parameters:
		kan.ParamNumControlPoints:   10, // Number of control points
		kan.ParamNumHiddenNodes:     64,
		kan.ParamNumHiddenLayers:    4,
		kan.ParamBSplineDegree:      2,
		kan.ParamBSplineMagnitudeL1: 1e-5,
		kan.ParamBSplineMagnitudeL2: 0.0,
		kan.ParamDiscrete:           false,
		kan.ParamDiscreteSoftness:   0.1,

		// CNN
		"cnn_normalization": "layer",
		"cnn_num_layers":    4,
		"cnn_num_filters":   32,
	})
	return ctx
}

// DType used in the mode.
var DType = dtypes.Float32

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
	trainModel(ctx)
}

func trainModel(ctx *context.Context) {
	must.M(cifar.DownloadCifar10(*flagDataDir))
	must.M(cifar.DownloadCifar100(*flagDataDir))

	// Backend handles creation of ML computation graphs, accelerator resources, etc.
	backend := backends.New()
	if *flagVerbosity >= 1 {
		fmt.Printf("Backend %q:\t%s\n", backend.Name(), backend.Description())
	}

	// Create datasets used for training and evaluation.
	batchSize := context.GetParamOr(ctx, "batch_size", int(0))
	if batchSize <= 0 {
		Panicf("Batch size must be > 0 (maybe it was not set?): %d", batchSize)
	}
	evalBatchSize := context.GetParamOr(ctx, "eval_batch_size", int(0))
	if evalBatchSize <= 0 {
		evalBatchSize = batchSize
	}
	trainDS, evalOnTrainDS, evalOnTestDS := CreateDatasets(backend, *flagDataDir, batchSize, evalBatchSize)

	// Create closure for model graph building function, that uses statically the dataset
	// used for its Dataset.GatherImage, to convert image indices to the actual images.
	// This is the signature of model function that the train.Trainer accepts.
	modelFn := PlainModelGraph
	if slices.Index(validModels, *flagModel) == -1 {
		Panicf("Flag --model must take one value from %v, got %q", validModels, *flagModel)
	}
	if strings.HasPrefix(*flagModel, "cnn") {
		modelFn = ConvolutionModelGraph
	}
	fmt.Printf("Model: %s\n", *flagModel)

	// Metrics we are interested.
	meanAccuracyMetric := metrics.NewSparseCategoricalAccuracy("Mean Accuracy", "#acc")
	movingAccuracyMetric := metrics.NewMovingAverageSparseCategoricalAccuracy("Moving Average Accuracy", "~acc", 0.01)

	// Read hyperparameters from context that we don't want overwritten by loading fo the context from a checkpoint.
	numTrainSteps := context.GetParamOr(ctx, "train_steps", 0)
	usePlots := context.GetParamOr(ctx, plotly.ParamPlots, false)

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
	if *flagVerbosity >= 2 {
		fmt.Println(commandline.SprintContextSettings(ctx))
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
	if *flagVerbosity >= 0 {
		commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.
	}

	// Attach a checkpoint: checkpoint every 1 minute of training.
	if checkpoint != nil {
		period := time.Minute * 1
		train.PeriodicCallback(loop, period, true, "saving checkpoint", 100,
			func(loop *train.Loop, metrics []*tensors.Tensor) error {
				return checkpoint.Save()
			})
	}

	// Attach Plotly plots: plot points at exponential steps.
	// The points generated are saved along the checkpoint directory (if one is given).
	if usePlots {
		plots := plotly.New().Dynamic().
			ScheduleExponential(loop, 200, 1.2).
			WithDatasets(evalOnTrainDS, evalOnTestDS)
		if checkpoint != nil {
			plots.WithCheckpoint(checkpoint.Dir())
		}
	}

	// Loop for given number of steps.
	if globalStep < numTrainSteps {
		_ = must.M1(loop.RunSteps(trainDS, numTrainSteps-globalStep))
		if *flagVerbosity >= 1 {
			fmt.Printf("\t[Step %d] median train step: %d microseconds\n",
				loop.LoopStep, loop.MedianTrainStepDuration().Microseconds())
		}

		if context.GetParamOr(ctx, "cnn_normalization", "none") == "batch" ||
			context.GetParamOr(ctx, fnn.ParamNormalization, "none") == "batch" {
			layers.BatchNormalizationResetWeights(ctx)
			trainer.BatchNormAveragesUpdate(evalOnTrainDS)
			if checkpoint != nil {
				must.M(checkpoint.Save())
			}
		}

	} else {
		fmt.Printf("\t - target train_steps=%d already reached. To train further, set a number additional "+
			"to current global step.\n", numTrainSteps)
	}

	// Finally, print an evaluation on train and test datasets.
	if *flagEval {
		if *flagVerbosity >= 1 {
			fmt.Println()
		}
		must.M(commandline.ReportEval(trainer, evalOnTestDS, evalOnTrainDS))
	}
}

func CreateDatasets(backend backends.Backend, dataDir string, batchSize, evalBatchSize int) (trainDS, trainEvalDS, validationEvalDS train.Dataset) {
	baseTrain := cifar.NewDataset(backend, "Training", dataDir, cifar.C10, DType, cifar.Train)
	baseTest := cifar.NewDataset(backend, "Validation", dataDir, cifar.C10, DType, cifar.Test)
	trainDS = baseTrain.Copy().BatchSize(batchSize, true).Shuffle().Infinite(true)
	trainEvalDS = baseTrain.BatchSize(evalBatchSize, false)
	validationEvalDS = baseTest.BatchSize(evalBatchSize, false)
	return
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
	ctx = ctx.In("model")
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
	logits.AssertDims(batchSize, numClasses)
	return []*Node{logits}
}

// normalizeImage to be used in between convolutions.
func normalizeImage(ctx *context.Context, x *Node) *Node {
	x.AssertRank(4) // [batch_size, width, height, depth]
	normalizationType := context.GetParamOr(ctx, "cnn_normalization", "none")

	/*
		wasTraining := ctx.IsTraining(x.Graph())
		ctx.SetTraining(x.Graph(), true)
		defer func() { ctx.SetTraining(x.Graph(), wasTraining) }()
	*/

	switch normalizationType {
	case "layer":
		return layers.LayerNormalization(ctx, x, 1, 2).ScaleNormalization(false).Done()
	case "batch":
		return layers.BatchNormalization(ctx, x, -1).Epsilon(1e-03).UseBackendInference(false).Done()
	case "none", "":
		return x
	}
	Panicf("invalid normalization type selected %q (hyperparameter %q) -- valid values are batch, layer, none", normalizationType, layers.ParamNormalization)
	return nil
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
	ctx = ctx.In("model")
	batchedImages := inputs[0]
	g := batchedImages.Graph()
	dtype := batchedImages.DType()
	batchSize := batchedImages.Shape().Dimensions[0]
	logits := batchedImages
	numFilters := context.GetParamOr(ctx, "cnn_num_filters", 32)
	numLayers := context.GetParamOr(ctx, "cnn_num_layers", 4)
	var dropoutRate *Node
	dropoutRateConfig := context.GetParamOr(ctx, layers.ParamDropoutRate, 0.0)
	if dropoutRateConfig > 0 {
		dropoutRate = Scalar(g, dtype, dropoutRateConfig)
	}
	//fmt.Printf("logits[0].shape=%s\n", logits.Shape())
	for ii := range numLayers + 3 {
		ctx := ctx.Inf("conv_layer_%d", ii)
		residual := logits
		conv := layers.Convolution(ctx, logits).Filters(numFilters).KernelSize(3)
		if ii < numLayers {
			conv = conv.PadSame()
		}
		logits = conv.Done()
		logits = normalizeImage(ctx, logits)
		logits = activations.ApplyFromContext(ctx, logits)
		if dropoutRate != nil {
			logits = layers.DropoutNormalize(ctx, logits, dropoutRate, true)
		}
		if ii >= numLayers {
			// Shrink image, while increasing number of filters.
			logits = MaxPool(logits).Window(2).Strides(2).Done()
			numFilters *= 2 // Could be 4, since we are reducing the image by a factor of 4.
		}
		if logits.Shape().Equal(residual.Shape()) {
			logits = Add(residual, logits)
		}
		//fmt.Printf("logits[%d].shape=%s\n", ii+1, logits.Shape())
	}
	//fmt.Println()
	//fmt.Println()
	//fmt.Println()

	// Flatten logits, and we can use the usual FNN/KAN.
	logits = Reshape(logits, batchSize, -1)
	numClasses := len(cifar.C10Labels)
	if *flagModel == "cnn-kan" {
		// Configuration of the KAN layer(s) use the context hyperparameters.
		logits = kan.New(ctx, logits, numClasses).Done()
	} else {
		// Configuration of the FNN layer(s) use the context hyperparameters.
		logits = fnn.New(ctx, logits, numClasses).Done()
	}
	return []*Node{logits}
}
