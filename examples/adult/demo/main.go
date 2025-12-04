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
	"time"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/examples/adult"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/pkg/core/distributed"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/checkpoints"
	"github.com/gomlx/gomlx/pkg/ml/datasets"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/layers/fnn"
	"github.com/gomlx/gomlx/pkg/ml/layers/kan"
	"github.com/gomlx/gomlx/pkg/ml/layers/regularizers"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gomlx/pkg/ml/train/metrics"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers/cosineschedule"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gomlx/ui/gonb/margaid"
	"github.com/gomlx/gomlx/ui/gonb/plotly"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
	"github.com/schollz/progressbar/v3"
	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/default"
)

var (
	// ModelDType used for the model.
	ModelDType = dtypes.Float32
)

func createDefaultContext() *context.Context {
	ctx := context.New()
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
		//	$ gomlx_checkpoints -metrics -metrics_labels -metrics_types=accuracy --metrics_names='E(bat)/#loss,E(tes)/#loss' -loop=3s fnn
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
	flagDataDir = flag.String("data", "~/work/uci-adult",
		"Directory to save and load downloaded and generated dataset files.")
	flagCheckpoint = flag.String("checkpoint", "", "Checkpoint subdirectory under the --data directory. "+
		"If empty does not use checkpoints. If absolute path, use that instead.")
	flagForceDownload = flag.Bool("force_download", false, "Force re-download of Adult dataset files.")

	flagNumQuantiles = flag.Int("quantiles", 100,
		"Max number of quantiles to use for numeric features, used during piece-wise linear calibration. "+
			"It will only use unique values, so if there are fewer variability, fewer quantiles are used.")
	flagEmbeddingDim = flag.Int("embedding_dim", 8,
		"Default embedding dimension for categorical values.")
	flagVerbosity = flag.Int("verbosity", 1, "Level of verbosity, the higher the more verbose.")

	flagUseCategorical       = flag.Bool("use_categorical", true, "Use categorical features.")
	flagUseContinuous        = flag.Bool("use_continuous", true, "Use continuous features.")
	flagTrainableCalibration = flag.Bool("trainable_calibration", true,
		"Allow piece-wise linear calibration to adjust outputs.")
	flagDistributed = flag.Bool("distributed", false, "Use distributed training: it will use as many devices as "+
		"available in the backend.")
	flagNumDevices = flag.Int("num_devices", 0,
		"Number of devices to use for distributed training. The default is to use all devices available in the "+
			"backend. Setting this to > 1 automatically enables -distributed. If 0, it will use all "+
			"devices available in the backend.")
	flagPrefetchOnDevice = flag.Int("prefetch_on_device", 0,
		"Number of batches to prefetch and upload to the device in parallel to training.")
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
	backend := backends.MustNew()
	numDevices := backend.NumDevices()
	if *flagNumDevices > 0 {
		if *flagNumDevices > numDevices {
			return errors.Errorf("-num_devices: %d is greater than the number of devices in the backend (%d)", *flagNumDevices, numDevices)
		}
		numDevices = *flagNumDevices
		if numDevices > 1 {
			*flagDistributed = true
		}
	}
	dataDir = fsutil.MustReplaceTildeInDir(dataDir)
	if *flagVerbosity >= 1 {
		fmt.Printf("Backend: %s\n\t%s\n", backend.Name(), backend.Description())
		fmt.Println(commandline.SprintContextSettings(ctx))
	}

	// Checkpoints loading (and saving)
	var checkpoint *checkpoints.Handler
	const keepCheckpoints = 3
	if checkpointPath != "" {
		numCheckpointsToKeep := context.GetParamOr(ctx, "num_checkpoints", keepCheckpoints)
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

	// Batch size: it is affected by the distributed execution..
	batchSize := context.GetParamOr(ctx, "batch_size", 128)
	shardedBatchSize := batchSize
	if *flagDistributed {
		if numDevices < 2 {
			klog.Fatalf("-distributed: distributed training requires at least 2 devices")
		}
		fmt.Printf("- Using distributed execution across %d devices.\n", numDevices)
		shardedBatchSize = batchSize / numDevices
		newBatchSize := shardedBatchSize * numDevices
		if newBatchSize != batchSize {
			klog.Warningf(
				"-distributed: batch_size reduced from %d to %d to make it divisible by the number of devices=%d",
				batchSize, newBatchSize, numDevices)
			batchSize = newBatchSize
		}
	}

	// Create datasets for training and evaluation.
	inMemoryDS := adult.NewDataset(backend, adult.Data.Train, "batched train")
	var (
		trainEvalDS train.Dataset = inMemoryDS.Copy().BatchSize(batchSize, false)
		testEvalDS  train.Dataset = adult.NewDataset(backend, adult.Data.Test, "test").BatchSize(batchSize, false)
		// For training, we shuffle and loop indefinitely.
		trainDS train.Dataset = inMemoryDS.BatchSize(batchSize, true).Shuffle().Infinite(true)
	)

	// Convert to a distributed dataset, if -distributed is set.
	if *flagDistributed {
		// Specify how to distributed: AutoSharding, and shard the data along the batch axis.
		strategy := distributed.AutoSharding
		mesh, err := distributed.NewDeviceMesh([]int{numDevices}, []string{"shards"})
		if err != nil {
			return err
		}
		shardingSpec, err := distributed.NewShardingSpec(mesh, distributed.AxisSpec{"shards"})
		if err != nil {
			return err
		}
		inputShardingSpecs := []*distributed.ShardingSpec{shardingSpec}
		labelsShardingSpecs := []*distributed.ShardingSpec{shardingSpec}
		var deviceAssignment []backends.DeviceNum // nil, the default assignment will be used.
		trainDS, err = datasets.NewDistributedAccumulator(
			backend, trainDS, strategy, inputShardingSpecs, labelsShardingSpecs, deviceAssignment)
		if err != nil {
			return err
		}
		trainEvalDS, err = datasets.NewDistributedAccumulator(
			backend, trainEvalDS, strategy, inputShardingSpecs, labelsShardingSpecs, deviceAssignment)
		if err != nil {
			return err
		}
		testEvalDS, err = datasets.NewDistributedAccumulator(
			backend, testEvalDS, strategy, inputShardingSpecs, labelsShardingSpecs, deviceAssignment)
		if err != nil {
			return err
		}
	}

	// Prefetch batches and upload to device in parallel to training.
	if *flagPrefetchOnDevice > 0 {
		// Distributed datasets already prefetch on device, so we don't need to do it here.
		if !*flagDistributed {
			var err error
			trainDS, err = datasets.NewOnDevice(backend, trainDS, false, *flagPrefetchOnDevice, backends.DeviceNum(0))
			if err != nil {
				return err
			}
		}
	}

	// Metrics we are interested.
	meanAccuracyMetric := metrics.NewMeanBinaryLogitsAccuracy("Mean Accuracy", "#acc")
	movingAccuracyMetric := metrics.NewMovingAverageBinaryLogitsAccuracy("Moving Average Accuracy",
		"~acc", 0.01)

	// Create a train.Trainer: this object will orchestrate running the model, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	trainer := train.NewTrainer(backend, ctx, Model, losses.BinaryCrossentropyLogits,
		optimizers.FromContext(ctx),
		[]metrics.Interface{movingAccuracyMetric}, // trainMetrics
		[]metrics.Interface{meanAccuracyMetric})   // evalMetrics

	// Use a standard training loop, with a progress bar.
	loop := train.NewLoop(trainer)
	commandline.ProgressbarStyle = progressbar.ThemeUnicode
	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.

	// Attach a checkpoint saver at every minute of training.
	if checkpoint != nil {
		period := time.Minute * 1
		train.PeriodicCallback(loop, period, true, "saving checkpoint", 100,
			func(loop *train.Loop, metrics []*tensors.Tensor) error {
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
	trainSteps := context.GetParamOr(ctx, "train_steps", 0)
	globalStep := int(optimizers.GetGlobalStep(ctx))
	if globalStep != 0 {
		fmt.Printf("- Restarting training from global step %d\n", globalStep)
	}
	metrics, err := loop.RunToGlobalStep(trainDS, trainSteps)
	if err != nil {
		return err
	}
	if metrics == nil {
		fmt.Printf("- Target train_steps=%d already reached. To train further, set a number larger than "+
			"current global step %d (e.g.: -set=train_steps=1_000_000).\n", trainSteps, globalStep)
	}

	// Finally, print an evaluation on train and test datasets.
	return commandline.ReportEval(trainer, trainEvalDS, testEvalDS)
}
