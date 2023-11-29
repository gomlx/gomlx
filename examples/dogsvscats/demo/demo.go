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
	"github.com/gomlx/gomlx/examples/notebook/gonb/margaid"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/nanlogger"
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
	. "github.com/gomlx/gomlx/types/exceptions"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	"log"
	"os"
	"time"
)

var (
	flagDataDir       = flag.String("data", "~/tmp/dogs_vs_cats", "Directory to cache downloaded dataset and save checkpoints.")
	flagPreGenerate   = flag.Bool("pre", false, "Pre-generate preprocessed image data to speed up training.")
	flagForceOriginal = flag.Bool("force_original", false, "Set to true to use original images and dynamically read and augment images.")

	// ML Manager creation:
	flagNumThreads     = flag.Int("num_threads", -1, "Number of threads. Leave as -1 to use as many as there are cores.")
	flagNumReplicas    = flag.Int("num_replicas", 1, "Number of replicas.")
	flagPlatform       = flag.String("platform", "", "Platform to use, if empty uses the default one.")
	flagCheckpoint     = flag.String("checkpoint", "", "Directory save and load checkpoints from. If left empty, no checkpoints are created.")
	flagCheckpointKeep = flag.Int("checkpoint_keep", 2, "Number of checkpoints to keep, if --checkpoint is set.")
	flagNanLogger      = flag.Bool("nanlogger", false, "Set to enable logging of NaN values, as soon as they happen.")

	// Training hyperparameters:
	flagModelType        = flag.String("model", "cnn", "Model types: \"cnn\" or \"inception\"")
	flagOptimizer        = flag.String("optimizer", "adamw", fmt.Sprintf("Optimizer, options: %q", slices.SortedKeys(optimizers.KnownOptimizers)))
	flagNumSteps         = flag.Int("steps", 2000, "Number of gradient descent steps to perform")
	flagBatchSize        = flag.Int("batch", dogsvscats.DefaultConfig.BatchSize, "Batch size for training")
	flagLearningRate     = flag.Float64("learning_rate", 0.0001, "Initial learning rate.")
	flagL2Regularization = flag.Float64("l2_reg", 0, "L2 regularization on kernels. It doesn't interact well with --batch_norm.")
	flagNormalization    = flag.String("norm", "layer", fmt.Sprintf("Type of layer normalization to use. Valid values: %q.", slices.SortedKeys(layers.KnownNormalizers)))
	flagEval             = flag.Bool("eval", true, "Whether to evaluate trained model on test data in the end.")

	// Convolution
	flagNumConvolutions = flag.Int("num_convolutions", 5, "Number of convolutions -- there will be at least as many to reduce the image to 16x16")
	flagConvDropout     = flag.Float64("conv_dropout", 0.1, "Amount of dropout in the convolution layers. 0 means no dropout.")

	// Flat part of model, after convolutions and models being flattened:
	flagDropout         = flag.Float64("dropout", 0.1, "Amount of dropout in the convolution layers. 0 means no dropout.")
	flagNumHiddenLayers = flag.Int("hidden_layers", 3, "Number of hidden layers, stacked with residual connection.")
	flagNumNodes        = flag.Int("num_nodes", 128, "Number of nodes in hidden layers.")

	// InceptionV3 model parameters:
	flagInceptionPreTrained = flag.Bool("pretrained", true, "If using inception model, whether to use the pre-trained weights to transfer learn")
	flagInceptionFineTuning = flag.Bool("finetuning", true, "If using inception model, whether to fine-tune the inception model")

	// Flat augmentation hyperparameters:
	flagAngleStdDev  = flag.Float64("angle", 20.0, "Standard deviation of noise used to rotate the image. Disabled if --augment=false.")
	flagFlipRandomly = flag.Bool("flip", true, "Randomly flip the image horizontally. Disabled if --augment=false.")

	// Pre-Generation parameters:
	flagPreGenEpochs = flag.Int("pregen_epochs", 40, "Number of epochs to pre-generate for the training data. Each epoch will take ~310Mb")

	// UI
	flagPlots = flag.Bool("plots", true, "Plots during training: perform periodic evaluations, "+
		"save results if --checkpoint is set and draw plots, if in a Jupyter notebook.")
)

func AssertNoError(err error) {
	if err != nil {
		log.Fatalf("Failed: %+v", err)
	}
}

// nanLogger is used for debugging, enabled with --nanlogger in the command line.
// See `nanlogger` package for details.
var nanLogger *nanlogger.NanLogger

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
		dogsvscats.PreGenerate(config, *flagPreGenEpochs, true)
		return
	}
	config.ForceOriginal = *flagForceOriginal
	config.UseParallelism = true
	config.BufferSize = 100
	config.YieldImagePairs = *flagModelType == "byol"
	trainModel(config)
}

// NewContext returns a new context with the parameters set from the flags values.
func NewContext(manager *Manager) *context.Context {
	ctx := context.NewContext(manager)
	ctx.SetParam("optimizer", *flagOptimizer) // Just so it is saved along with the context.
	ctx.SetParam(optimizers.LearningRateKey, *flagLearningRate)
	ctx.SetParam(layers.L2RegularizationKey, *flagL2Regularization)
	ctx.SetParam("normalization", *flagNormalization)
	ctx.SetParam("num_convolutions", *flagNumConvolutions)
	ctx.SetParam("conv_dropout", *flagConvDropout)
	ctx.SetParam("dropout", *flagDropout)
	ctx.SetParam("hidden_layers", *flagNumHiddenLayers)
	ctx.SetParam("num_nodes", *flagNumNodes)

	// BYOL model parameters.
	if *flagModelType == "byol" {
		ctx.SetParam("byol_hidden_layers", *flagByolProjectionNumLayers)
		ctx.SetParam("byol_num_nodes", *flagByolProjectionNumNodes)
		ctx.SetParam("byol_target_update_ratio", *flagByolTargetUpdateRatio)
		ctx.SetParam("byol_regularization_rate", *flagByolRegularizationRate)
		ctx.SetParam("byol_inception", *flagByolInception)
	}

	if *flagNanLogger {
		nanLogger = nanlogger.New()
	}
	return ctx
}

// trainModel based on configuration and flags.
func trainModel(config *dogsvscats.Configuration) {
	trainDS, trainEvalDS, validationEvalDS := dogsvscats.CreateDatasets(config)

	// Manager handles creation of ML computation graphs, accelerator resources, etc.
	manager := BuildManager().NumThreads(*flagNumThreads).NumReplicas(*flagNumReplicas).Platform(*flagPlatform).Done()

	// Context holds the variables and hyperparameters for the model.
	ctx := NewContext(manager)

	// Metrics we are interested.
	meanAccuracyMetric := metrics.NewMeanBinaryLogitsAccuracy("Mean Accuracy", "#acc")
	movingAccuracyMetric := metrics.NewMovingAverageBinaryLogitsAccuracy("Moving Average Accuracy", "~acc", 0.01)

	// Checkpoint: it loads if already exists, and it will save as we train.
	var checkpoint *checkpoints.Handler
	if *flagCheckpoint != "" {
		var err error
		checkpoint, err = checkpoints.Build(ctx).
			DirFromBase(*flagCheckpoint, config.DataDir).
			Keep(*flagCheckpointKeep).Done()
		AssertNoError(err)
		globalStep := optimizers.GetGlobalStep(ctx)
		if globalStep != 0 {
			fmt.Printf("> restarting training from global_step=%d\n", globalStep)
		}
	}

	// Select the model type we are using:
	var (
		modelFn     train.ModelFn
		preTraining = false
	)
	switch *flagModelType {
	case "cnn":
		modelFn = CnnModelGraph
	case "inception":
		modelFn = InceptionV3ModelGraph
		if *flagInceptionPreTrained {
			AssertNoError(inceptionv3.DownloadAndUnpackWeights(*flagDataDir))
		}
	case "byol":
		modelFn = ByolCnnModelGraph
		preTraining = *flagByolPretraining
	default:
		Panicf("Unknown model %q: valid values are \"cnn\", \"inception\" or \"byol\"", *flagModelType)
	}
	if preTraining && checkpoint == nil {
		fmt.Println("*** WARNING: pre-training model but not saving the pretrained weights is only useful for debugging")
	}

	// Create a train.Trainer: this object will orchestrate running the model, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	var trainer *train.Trainer
	optimizer := optimizers.MustOptimizerByName(*flagOptimizer)
	if !preTraining {
		trainer = train.NewTrainer(manager, ctx, modelFn,
			losses.BinaryCrossentropyLogits,
			optimizer,
			[]metrics.Interface{movingAccuracyMetric}, // trainMetrics
			[]metrics.Interface{meanAccuracyMetric})   // evalMetrics
	} else {
		// Pre-training: no loss, no metrics.
		trainer = train.NewTrainer(manager, ctx, modelFn,
			nil,
			optimizers.MustOptimizerByName(*flagOptimizer),
			nil, // trainMetrics
			nil) // evalMetrics
	}
	nanLogger.AttachToTrainer(trainer) // It's a no-op if nanLogger is nil.

	// Use standard training loop.
	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.

	// Attach a checkpoint: checkpoint every 1 minute of training.
	if checkpoint != nil {
		period := time.Minute * 1
		train.PeriodicCallback(loop, period, true, "saving checkpoint", 100,
			func(loop *train.Loop, metrics []tensor.Tensor) error {
				return checkpoint.Save()
			})
	}

	// Attach a margaid plots: plot points at exponential steps.
	// The points generated are saved along the checkpoint directory (if one is given).
	var plots *margaid.Plots
	if *flagPlots {
		if !preTraining {
			plots = margaid.NewDefault(loop, checkpoint.Dir(), 100, 1.1, trainEvalDS, validationEvalDS).
				WithEvalLossType("eval-loss")
		} else {
			// Pre-training: no evaluation.
			plots = margaid.NewDefault(loop, checkpoint.Dir(), 100, 1.1)
		}
	}

	// Loop for given number of steps.
	_, err := loop.RunSteps(trainDS, *flagNumSteps)
	AssertNoError(err)
	fmt.Printf("\t[Step %d] median train step: %d microseconds\n",
		loop.LoopStep, loop.MedianTrainStepDuration().Microseconds())

	// If pre-training, skip evaluation, and clear out optimizer variables.
	if preTraining {
		fmt.Println("Pre-training only, no evaluation.")
		optimizer.Clear(ctx)
		optimizers.DeleteGlobalStep(ctx)
		if checkpoint != nil {
			fmt.Println("- Saving cleared checkpoint.")
			checkpoint.Save()
		}
		return
	}

	// Finally, print an evaluation on train and test datasets.
	fmt.Println()
	err = commandline.ReportEval(trainer, trainEvalDS, validationEvalDS)
	if plots != nil {
		// Save plot points.
		plots.Done()
	}
	AssertNoError(err)
	fmt.Println()
}

func normalizeImage(ctx *context.Context, x *Node) *Node {
	x.AssertRank(4) // [batch_size, width, height, depth]
	norm := context.GetParamOr(ctx, "normalization", "none")
	switch norm {
	case "layer":
		return layers.LayerNormalization(ctx, x, 1, 2).ScaleNormalization(false).Done()
	case "batch":
		return layers.BatchNormalization(ctx, x, -1).Done()
	case "none":
		return x
	}
	Panicf("invalid normalization selected %q -- valid values are batch, layer, none", norm)
	return nil
}

func normalizeFeatures(ctx *context.Context, x *Node) *Node {
	x.AssertRank(2) // [batch_size, embedding_dim]
	norm := context.GetParamOr(ctx, "normalization", "none")
	switch norm {
	case "layer":
		return layers.LayerNormalization(ctx, x, -1).Done()
	case "batch":
		return layers.BatchNormalization(ctx, x, -1).Done()
	case "none":
		return x
	}
	Panicf("invalid normalization selected %q -- valid values are batch, layer, none", norm)
	return nil
}
