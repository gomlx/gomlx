// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// plot_fixture is a throwaway helper for the Vizb integration work
// (dev/vizb-integration branch): it runs a short MNIST training session with
// plotly.PlotConfig wired in, purely to produce a real training_plot_points.json
// fixture on disk -- see vizb-implementation-guide.md, Milestone 1.
//
// Not part of any documentation sync; not meant to be merged upstream.
package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/examples/mnist"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/metric"
	"github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/gomlx/gomlx/support/fsutil"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gomlx/ui/gonb/plotly"

	_ "github.com/gomlx/gomlx/backends/default"
)

const batchSize = 128

var (
	flagDataDir       = flag.String("data", "~/mnist_data", "Directory to cache the MNIST dataset.")
	flagCheckpointDir = flag.String("checkpoint", "plot_fixture_checkpoint", "Checkpoint subdirectory name, relative to -data.")
	flagSteps         = flag.Int("steps", 200, "Number of training steps to run.")
)

func ConvModel(scope *model.Scope, spec any, inputs []*Node) []*Node {
	images := inputs[0]
	batchSize := images.Shape().Dimensions[0]

	x := layers.Convolution(scope.In("conv1"), images).Filters(16).KernelSize(3).PadSame().Done()
	x = activation.Relu(x)
	x = MaxPool(x).Window(2).Done()

	x = layers.Convolution(scope.In("conv2"), x).Filters(32).KernelSize(3).PadSame().Done()
	x = activation.Relu(x)
	x = MaxPool(x).Window(2).Done()

	x = Reshape(x, batchSize, -1)

	x = layers.Dense(scope.In("dense1"), x, true, 128)
	x = activation.Relu(x)

	logits := layers.Dense(scope.In("logits"), x, true, mnist.NumClasses)
	return []*Node{logits}
}

func prepareDatasets(backend compute.Backend, dataDir string) (trainDS, testDS train.Dataset) {
	dataDir = fsutil.MustReplaceTildeInDir(dataDir)

	if err := mnist.Download(dataDir); err != nil {
		log.Fatalf("Failed to download MNIST dataset: %+v", err)
	}

	rawTrain, err := mnist.NewDataset(backend, "MNIST Train", dataDir, "train", dtypes.Float32)
	if err != nil {
		log.Fatalf("Failed to load training dataset: %+v", err)
	}
	rawTest, err := mnist.NewDataset(backend, "MNIST Test", dataDir, "test", dtypes.Float32)
	if err != nil {
		log.Fatalf("Failed to load test dataset: %+v", err)
	}

	// Infinite(true): we drive this with loop.RunSteps, not RunEpochs, so the
	// training stream must not run out mid-way through an arbitrary step count.
	trainDS = rawTrain.Shuffle().BatchSize(batchSize, true).Infinite(true)
	testDS = rawTest.BatchSize(batchSize, false)
	return
}

func main() {
	flag.Parse()

	backend := compute.MustNew()
	defer backend.Finalize()
	fmt.Printf("Backend: %s (%s)\n", backend.Name(), backend.Description())

	store := model.NewStore()
	trainDS, testDS := prepareDatasets(backend, *flagDataDir)

	accuracyMetric := metric.NewSparseCategoricalAccuracy("Accuracy", "acc")
	trainer := train.NewTrainer(
		backend,
		store,
		ConvModel,
		loss.SparseCategoricalCrossEntropyLogits,
		optimizer.Adam().LearningRate(1e-3).Done(),
		[]metric.Interface{accuracyMetric},
		[]metric.Interface{accuracyMetric},
	)

	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop)

	checkpointHandler, err := checkpoint.Build(store).
		DirFromBase(*flagCheckpointDir, *flagDataDir).
		Keep(3).
		Done()
	if err != nil {
		log.Fatalf("Failed to create checkpoint handler: %+v", err)
	}
	fmt.Printf("Checkpoint / plot-points directory: %s\n", checkpointHandler.Dir())

	// This is the wiring Milestone 1 asks for: record plot.Point values (train
	// and eval metrics) to training_plot_points.json as training progresses.
	_ = plotly.New().
		WithCheckpoint(checkpointHandler).
		WithDatasets(testDS).
		ScheduleEveryNSteps(loop, 20)

	fmt.Printf("Running %d training steps...\n", *flagSteps)
	if _, err := loop.RunSteps(trainDS, *flagSteps); err != nil {
		log.Fatalf("Training loop failed: %+v", err)
	}
	fmt.Println("Done.")
}
