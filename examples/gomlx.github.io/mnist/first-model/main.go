// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// first-model is the runnable companion to the "Your First Model: MNIST" tutorial
// on gomlx.github.io. Code snippets in that page are synced from this file via
// tagged comments -- see cmd/sync_code in the gomlx.github.io repository.
package main

import ( //md:imports,full
	"fmt"  //md:imports,full
	"log"  //md:imports,full
	"time" //md:checkpoint_imports,full

	"github.com/gomlx/compute"                    //md:imports,full
	"github.com/gomlx/compute/dtypes"             //md:imports,full
	. "github.com/gomlx/gomlx/core/graph"         //md:imports,full
	"github.com/gomlx/gomlx/examples/mnist"       //md:imports,full
	"github.com/gomlx/gomlx/ml/layers"            //md:imports,full
	"github.com/gomlx/gomlx/ml/layers/activation" //md:imports,full
	"github.com/gomlx/gomlx/ml/model"             //md:imports,full
	"github.com/gomlx/gomlx/ml/model/checkpoint"  //md:checkpoint_imports,full
	"github.com/gomlx/gomlx/ml/train"             //md:imports,full
	"github.com/gomlx/gomlx/ml/train/loss"        //md:imports,full
	"github.com/gomlx/gomlx/ml/train/metric"      //md:imports,full
	"github.com/gomlx/gomlx/ml/train/optimizer"   //md:imports,full
	"github.com/gomlx/gomlx/support/fsutil"       //md:imports,full
	"github.com/gomlx/gomlx/ui/commandline"       //md:imports,full

	_ "github.com/gomlx/gomlx/backends/default" //md:imports,full
) //md:imports,full

const dataDir = "~/mnist_data"

//md_start:dataset,full

const batchSize = 128

// prepareDatasets ensures MNIST is downloaded, then returns batched train/test datasets.
func prepareDatasets(backend compute.Backend, dataDir string) (trainDS, testDS train.Dataset) {
	dataDir = fsutil.MustReplaceTildeInDir(dataDir)

	// mnist.Download is idempotent: it verifies checksums and skips files already on disk.
	if err := mnist.Download(dataDir); err != nil {
		log.Fatalf("Failed to download MNIST dataset: %+v", err)
	}

	// mnist.NewDataset returns the *whole* split as a single in-memory, unbatched dataset --
	// we still need to batch (and, for training, shuffle) it ourselves.
	rawTrain, err := mnist.NewDataset(backend, "MNIST Train", dataDir, "train", dtypes.Float32)
	if err != nil {
		log.Fatalf("Failed to load training dataset: %+v", err)
	}
	rawTest, err := mnist.NewDataset(backend, "MNIST Test", dataDir, "test", dtypes.Float32)
	if err != nil {
		log.Fatalf("Failed to load test dataset: %+v", err)
	}

	// dropIncompleteBatch=true for training keeps every batch a fixed size, which XLA prefers
	// (it recompiles the graph whenever it sees a new shape). Shuffle() reorders examples each epoch.
	trainDS = rawTrain.Shuffle().BatchSize(batchSize, true)
	// For evaluation we want to see every example, so we don't drop the last, short batch.
	testDS = rawTest.BatchSize(batchSize, false)
	return
}

//md_end:dataset

//md_start:model,full

// ConvModel defines a simple CNN architecture for MNIST digit classification.
// This is a "model function": GoMLX calls it once, while building the computation graph,
// not once per example -- the returned Nodes are symbolic, not actual numbers yet.
func ConvModel(scope *model.Scope, spec any, inputs []*Node) []*Node {
	// inputs[0] shape: [BatchSize, 28, 28, 1] (grayscale).
	images := inputs[0]
	batchSize := images.Shape().Dimensions[0]

	// Block 1: Conv2D (16 filters, 3x3 kernel, "same" padding keeps 28x28) -> ReLU -> MaxPool 2x2 -> 14x14.
	x := layers.Convolution(scope.In("conv1"), images).Filters(16).KernelSize(3).PadSame().Done()
	x = activation.Relu(x)
	x = MaxPool(x).Window(2).Done()

	// Block 2: Conv2D (32 filters) -> ReLU -> MaxPool 2x2 -> 7x7.
	x = layers.Convolution(scope.In("conv2"), x).Filters(32).KernelSize(3).PadSame().Done()
	x = activation.Relu(x)
	x = MaxPool(x).Window(2).Done()

	// Flatten: [BatchSize, 7, 7, 32] -> [BatchSize, 7*7*32]. GoMLX doesn't have a separate
	// Flatten layer -- a Reshape with -1 for the last dimension does the job.
	x = Reshape(x, batchSize, -1)

	// Fully-connected hidden layer (128 units) + ReLU.
	x = layers.Dense(scope.In("dense1"), x, true, 128)
	x = activation.Relu(x)

	// Output layer: logits for the 10 classes. No activation here -- the loss function
	// (SparseCategoricalCrossEntropyLogits) applies softmax internally, on the logits, which is
	// more numerically stable than applying softmax yourself and feeding probabilities to the loss.
	logits := layers.Dense(scope.In("logits"), x, true, mnist.NumClasses)

	return []*Node{logits}
}

//md_end:model

func main() {
	//md_start:training,full
	backend := compute.MustNew()
	defer backend.Finalize()
	fmt.Printf("Backend: %s (%s)\n", backend.Name(), backend.Description())

	store := model.NewStore()
	trainDS, testDS := prepareDatasets(backend, dataDir)

	accuracyMetric := metric.NewSparseCategoricalAccuracy("Accuracy", "acc")
	trainer := train.NewTrainer(
		backend,
		store,
		ConvModel,
		loss.SparseCategoricalCrossEntropyLogits, // cross-entropy loss for integer labels
		optimizer.Adam().LearningRate(1e-3).Done(),
		[]metric.Interface{accuracyMetric}, // metrics reported during training
		[]metric.Interface{accuracyMetric}, // metrics reported during evaluation
	)

	const epochs = 5
	loop := train.NewLoop(trainer)
	//md_end:training

	//md_start:checkpoint_setup,full
	// Save a checkpoint every minute, keeping the 3 most recent ones. checkpoint.Build
	// also *loads* an existing checkpoint from checkpointDir if one is already there, so
	// re-running this program resumes training instead of starting over.
	checkpointHandler, err := checkpoint.Build(store).
		DirFromBase("checkpoint", dataDir).
		Keep(3).
		Done()
	if err != nil {
		log.Fatalf("Failed to create checkpoint handler: %+v", err)
	}
	train.PeriodicCallback(loop, time.Minute, true, "saving checkpoint", 100, checkpointHandler.SaveOnStepFn)
	//md_end:checkpoint_setup

	//md_start:training,full
	// AttachProgressBar gives you a live progress bar with loss/metric values -- there's no
	// need to print them by hand.
	commandline.AttachProgressBar(loop)

	fmt.Printf("Starting training for %d epochs...\n", epochs)
	if _, err := loop.RunEpochs(trainDS, epochs); err != nil {
		log.Fatalf("Training loop failed: %+v", err)
	}

	fmt.Println("\nEvaluating model performance on test set...")
	if err := commandline.ReportEval(trainer, testDS); err != nil {
		log.Fatalf("Evaluation failed: %+v", err)
	}
	//md_end:training
}
