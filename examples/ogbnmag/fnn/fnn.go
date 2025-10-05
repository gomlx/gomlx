// Package fnn implements a feed-forward neural network for the OGBN-MAG problem.
package fnn

import (
	"fmt"
	"github.com/gomlx/gomlx/backends"
	mag "github.com/gomlx/gomlx/examples/ogbnmag"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/checkpoints"
	mldata "github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activations"
	"github.com/gomlx/gomlx/ml/layers/kan"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/metrics"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gomlx/ui/gonb/plotly"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"time"
)

// FnnModelGraph builds a FnnModel for the OGBN-MAP dataset.
func FnnModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	seeds := inputs[0]
	g := seeds.Graph()
	getMagVar := func(name string) *Node {
		magVar := ctx.GetVariableByScopeAndName(mag.OgbnMagVariablesScope, name)
		if magVar == nil {
			exceptions.Panicf("Missing OGBN-MAG dataset variables (%q), pls call UploadOgbnMagVariables() on context first.", name)
		}
		return magVar.ValueGraph(g)
	}
	log1pMagVar := func(name string) *Node {
		return Log1p(ConvertDType(getMagVar(name), dtypes.Float32))
	}

	// Gather and concatenate all features from the seeds (indices of papers).
	logits := Concatenate([]*Node{
		Gather(getMagVar("PapersEmbeddings"), seeds),
		//Gather(log1pMagVar("CountPapersCites"), seeds),
		//Gather(log1pMagVar("CountPapersIsCited"), seeds),
		//Gather(log1pMagVar("CountPapersFieldsOfStudy"), seeds),
		Gather(log1pMagVar("CountPapersAuthors"), seeds),
	}, 1)

	// Build FNN.
	numLayers := context.GetParamOr(ctx, "hidden_layers", 2)
	numNodes := context.GetParamOr(ctx, "num_nodes", 128)
	useKan := context.GetParamOr(ctx, "kan", false)
	if useKan {
		logits = kan.New(ctx, logits, mag.NumLabels).NumHiddenLayers(numLayers, numNodes).Done()
	} else {
		// Normal FNN
		for layerNum := range numLayers {
			layerName := fmt.Sprintf("layer-%d", layerNum)
			logits = layers.DenseWithBias(ctx.In(layerName), logits, numNodes)
			logits = activations.LeakyRelu(logits)
			dropoutRate := context.GetParamOr(ctx, "dropout_rate", 0.0)
			if dropoutRate > 0 {
				dropoutRateNode := Scalar(g, dtypes.Float32, dropoutRate)
				logits = layers.Dropout(ctx, logits, dropoutRateNode)
			}
		}
		logits = layers.DenseWithBias(ctx.In("readout"), logits, mag.NumLabels)
	}

	return []*Node{logits} // Return only the logits.
}

var ModelFn = FnnModelGraph

// Train FNN model based on configuration in `ctx`.
func Train(backend backends.Backend, ctx *context.Context) error {
	trainDS, validDS, testDS, err := mag.PapersSeedDatasets(backend)
	mag.UploadOgbnMagVariables(backend, ctx)
	//ctx.EnumerateVariables(func(v *context.Variable) {
	//	fmt.Printf("%s :: %s:\t%s\n", v.Scope(), v.Name(), v.Value().Shape())
	//})

	if err != nil {
		return err
	}

	batchSize := context.GetParamOr(ctx, "batch_size", 128)
	trainEvalDS := trainDS.Copy()
	trainDS = trainDS.Shuffle().BatchSize(batchSize, true).Infinite(true)

	// Evaluation datasets.
	evalBatchSize := context.GetParamOr(ctx, "eval_batch_size", 1024)
	trainEvalDS = trainEvalDS.BatchSize(evalBatchSize, false).Infinite(false)
	validDS = validDS.BatchSize(evalBatchSize, false).Infinite(false)
	testDS = testDS.BatchSize(evalBatchSize, false).Infinite(false)

	// Get trainSteps before a checkpoint is loaded -- in which case it will be overwritten.
	trainSteps := context.GetParamOr(ctx, "train_steps", 100)

	// Checkpoint: it loads if already exists, and it will save as we train.
	checkpointPath := context.GetParamOr(ctx, "checkpoint", "")
	numCheckpointsToKeep := context.GetParamOr(ctx, "num_checkpoints", 10)
	var checkpoint *checkpoints.Handler
	var globalStep int64
	if checkpointPath != "" {
		checkpointPath = mldata.ReplaceTildeInDir(checkpointPath) // If the path starts with "~", it is replaced.
		var err error
		if numCheckpointsToKeep <= 1 {
			// Only limit the amount of checkpoints kept if >= 2.
			numCheckpointsToKeep = -1
		}
		if numCheckpointsToKeep > 0 {
			checkpoint, err = checkpoints.Build(ctx).Dir(checkpointPath).Keep(numCheckpointsToKeep).TakeMean(3, backend).Done()
		} else {
			checkpoint, err = checkpoints.Build(ctx).Dir(checkpointPath).Done()
		}
		if err != nil {
			return errors.WithMessagef(err, "while setting up checkpoint to %q (keep=%d)",
				checkpointPath, numCheckpointsToKeep)
		}
		globalStep = optimizers.GetGlobalStep(ctx)
		if globalStep != 0 {
			fmt.Printf("> restarting training from global_step=%d\n", globalStep)
			ctx = ctx.Reuse()
		}
		ctx.SetParam("train_steps", trainSteps)
	}

	// Loss: multi-class classification problem.
	lossFn := losses.SparseCategoricalCrossEntropyLogits

	// Metrics we are interested.
	meanAccuracyMetric := metrics.NewSparseCategoricalAccuracy("Mean Accuracy", "#acc")
	movingAccuracyMetric := metrics.NewMovingAverageSparseCategoricalAccuracy("Moving Average Accuracy", "~acc", 0.01)

	// Create a train.Trainer: this object will orchestrate running the model, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	optimizer := optimizers.ByName(ctx, context.GetParamOr(ctx, "optimizer", "adamw"))
	trainer := train.NewTrainer(backend, ctx, ModelFn,
		lossFn,
		optimizer,
		[]metrics.Interface{movingAccuracyMetric}, // trainMetrics
		[]metrics.Interface{meanAccuracyMetric})   // evalMetrics

	// Use standard training loop.
	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.

	// Attach a checkpoint: checkpoint every 1 minute of training.
	if checkpoint != nil && numCheckpointsToKeep > 1 {
		period := time.Minute * 1
		train.PeriodicCallback(loop, period, true, "saving checkpoint", 100,
			func(loop *train.Loop, metrics []*tensors.Tensor) error {
				return checkpoint.Save()
			})
	}

	// Attach a margaid plots: plot points at exponential steps.
	// The points generated are saved along the checkpoint directory (if one is given).
	usePlots := context.GetParamOr(ctx, "plots", false)
	if usePlots {
		_ = plotly.New().
			WithCheckpoint(checkpoint).
			Dynamic().
			WithDatasets(validDS, testDS, trainEvalDS).
			ScheduleExponential(loop, 200, 1.2).
			WithBatchNormalizationAveragesUpdate(trainEvalDS)
	}

	// Loop for given number of steps
	if int(globalStep) < trainSteps {
		_, err = loop.RunSteps(trainDS, trainSteps-int(globalStep))
		if err != nil {
			return errors.WithMessage(err, "while running steps")
		}
		fmt.Printf("\t[Step %d] median train step: %d microseconds\n",
			loop.LoopStep, loop.MedianTrainStepDuration().Microseconds())
		if checkpoint != nil && numCheckpointsToKeep <= 1 {
			// Save checkpoint at end of training.
			err = checkpoint.Save()
			if err != nil {
				klog.Errorf("Failed to save final checkpoint in %q: %+v", checkpointPath, err)
			}
		}
	}

	// Finally, print an evaluation on train and test datasets.
	fmt.Println()
	err = commandline.ReportEval(trainer, trainEvalDS, validDS, testDS)
	if err != nil {
		return errors.WithMessage(err, "while reporting eval")
	}
	fmt.Println()
	return nil
}
