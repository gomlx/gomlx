// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dogsvscats

import (
	"fmt"
	"os"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/core/graph/nanlogger"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/layers/fnn"
	"github.com/gomlx/gomlx/ml/layers/kan"
	"github.com/gomlx/gomlx/ml/layers/norm"
	"github.com/gomlx/gomlx/ml/layers/regularizer"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/metric"
	"github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/gomlx/gomlx/ml/train/optimizer/cosineschedule"
	"github.com/gomlx/gomlx/support/fsutil"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gomlx/ui/gonb/plotly"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

var (

	// ModelsPrep maps models to a preparation function called in TrainModel before training start.
	//
	// This can be extended for new models.
	ModelsPrep = map[string]func(scope *model.Scope, dataDir string, checkpointHandler *checkpoint.Handler) error{
		"inception": InceptionV3ModelPrep,
	}
)

// nanLogger is used for debugging, enabled with --nanlogger in the command line.
// See `nanlogger` package for details.
var nanLogger *nanlogger.NanLogger

// CreateModelStore sets the store with default hyperparameters to use with TrainWithStore.
func CreateModelStore() *model.Store {
	store := model.NewStore()
	store.ResetRNGState()
	store.SetParams(map[string]any{
		// Model type to use
		"model":           "cnn",
		"num_checkpoints": 3,
		"train_steps":     2000,

		// batch_size for training.
		"batch_size": DefaultConfig.BatchSize,

		// eval_batch_size can be larger than training, it's more efficient.
		"eval_batch_size": DefaultConfig.EvalBatchSize,

		// Debug parameters.
		"nan_logger": false, // Trigger nan error as soon as it happens -- expensive, but helps debugging.

		// Image augmentation parameters
		"augmentation_angle_stddev":   20.0,  // Standard deviation of noise used to rotate the image. Disabled if --augment=false.
		"augmentation_random_flips":   true,  // Randomly flip images horizontally.
		"augmentation_force_original": false, // Force reading from original images instead of pre-generated.

		// "plots" trigger generating intermediary eval data for plotting, and if running in GoNB, to actually
		// draw the plot with Plotly.
		//
		// From the command-line, an easy way to monitor the metrics being generated during the training of a model
		// is using the gomlx_checkpointss tool:
		//
		//	$ gomlx_checkpointss --metrics --metrics_labels --metrics_types=accuracy  --metrics_names='E(Tra)/#loss,E(Val)/#loss' --loop=3s "<checkpoint_path>"
		plotly.ParamPlots: true,

		// "normalization" is overridden by "fnn_normalization" and "cnn_normalization", if they are set.
		layers.ParamNormalization: "batch",

		optimizer.ParamOptimizer:        "adamw",
		optimizer.ParamLearningRate:     1e-4,
		optimizer.ParamAdamEpsilon:      1e-7,
		optimizer.ParamAdamDType:        "",
		cosineschedule.ParamPeriodSteps: 0,
		activation.ParamActivation:      "",
		layers.ParamDropoutRate:         0.1,
		regularizer.ParamL2:             0.0,
		regularizer.ParamL1:             0.0,

		// FNN network parameters:
		fnn.ParamNumHiddenLayers: 3,
		fnn.ParamNumHiddenNodes:  128,
		fnn.ParamResidual:        true,
		fnn.ParamNormalization:   "",   // Set to "none" for no normalization, otherwise it falls back to layers.ParamNormalization.
		fnn.ParamDropoutRate:     -1.0, // Set to 0.0 for no dropout, otherwise it falls back to layers.ParamDropoutRate.

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
		"cnn_num_layers":      5.0,
		"cnn_dropout_rate":    -1.0,
		"cnn_embeddings_size": 128,

		// BYOL (Build Your Own Latent) model configuration ("model": "byol")
		"byol_pretrain":            false, // Pre-train BYOL model, unsupervised.
		"byol_finetune":            false, // Finetune BYOL model. If set to false, only the linear model on top is trained.
		"byol_hidden_nodes":        4096,  // Number of nodes in the hidden layer.
		"byol_projection_nodes":    256,   // Number of nodes (dimension) in the projection to the target regularizing model.
		"byol_target_update_ratio": 0.99,  // Moving average update weight to the "target" sub-model for BYOL model.
		"byol_regularization_rate": 1.0,   // BYOL regularization loss rate, a simple multiplier.
		"byol_inception":           false, // Instead of using a CNN model with BYOL, uses InceptionV3.
		"byol_reg_len1":            0.01,  // BYOL regularize projections to length 1.

		// InceptionV3 model configuration ("model": "inception")
		"inception_pretrained": true, // Whether to use the pre-trained weights to transfer learn
		"inception_finetuning": true, // Whether to fine-tune the inception model
	})
	return store
}

// Train based on configuration and flags.
func Train(store *model.Store, dataDir, checkpointPath string, runEval bool, paramsSet []string) error {
	scope := store.RootScope()
	var err error
	dataDir, err = fsutil.ReplaceTildeInDir(dataDir)
	if err != nil {
		return err
	}
	exists, err := fsutil.FileExists(dataDir)
	if err != nil {
		return err
	}
	if !exists {
		if err = os.MkdirAll(dataDir, 0777); err != nil {
			return err
		}
	}

	// Checkpoint: it loads if already exists, and it will save as we train.
	var checkpointHandler *checkpoint.Handler
	if checkpointPath != "" {
		numCheckpoints := model.GetParamOr(scope, "num_checkpoints", 3)
		checkpointHandler, err = checkpoint.Build(store).
			DirFromBase(checkpointPath, dataDir).
			ExcludeParams(append(
				paramsSet,
				"data_dir",
				"train_steps",
				"plots",
				"nan_logger",
				"num_checkpoints",
				"byol_pretrain",
				"byol_finetune",
			)...).
			Keep(numCheckpoints).Done()
		if err != nil {
			return err
		}
	}

	// Generation of augmented images and create datasets.
	if err = Download(dataDir); err != nil {
		return err
	}
	if err = FilterValidImages(dataDir); err != nil {
		return err
	}
	config := NewPreprocessingConfiguration(store, dataDir)
	trainDS, trainEvalDS, validationEvalDS := CreateDatasets(config)

	// Metrics we are interested.
	meanAccuracyMetric := metric.NewMeanBinaryLogitsAccuracy("Mean Accuracy", "#acc")
	movingAccuracyMetric := metric.NewMovingAverageBinaryLogitsAccuracy("Moving Average Accuracy", "~acc", 0.01)

	// Select the model type we are using:
	modelType := model.GetParamOr(scope, "model", "")
	fmt.Printf("Model: %q\n", modelType)
	if modelPrep, found := ModelsPrep[modelType]; found {
		if err = modelPrep(scope, dataDir, checkpointHandler); err != nil {
			return err
		}
	}

	// BYOL may require pretraining.
	preTraining := modelType == "byol" && model.GetParamOr(scope, "byol_pretrain", false)
	if preTraining && checkpointHandler == nil {
		klog.Warning(
			"*** pre-training model but not saving (--checkpoint) the pretrained weights -- is only useful for debugging ***",
		)
	}

	// Create a train.Trainer: this object will orchestrate running the model, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	backend := compute.MustNew()
	var trainer *train.Trainer
	theOptimizer := optimizer.FromStore(store)
	switch modelType {
	case "cnn":
		trainer = train.NewTrainer(backend, store, CnnModelGraph,
			loss.BinaryCrossentropyLogits,
			theOptimizer,
			[]metric.Interface{movingAccuracyMetric}, // trainMetrics
			[]metric.Interface{meanAccuracyMetric})   // evalMetrics
	case "inception":
		trainer = train.NewTrainer(backend, store, InceptionV3ModelGraph,
			loss.BinaryCrossentropyLogits,
			theOptimizer,
			[]metric.Interface{movingAccuracyMetric}, // trainMetrics
			[]metric.Interface{meanAccuracyMetric})   // evalMetrics
	case "byol":
		if !preTraining {
			trainer = train.NewTrainer(backend, store, ByolCnnModelGraph,
				loss.BinaryCrossentropyLogits,
				theOptimizer,
				[]metric.Interface{movingAccuracyMetric}, // trainMetrics
				[]metric.Interface{meanAccuracyMetric})   // evalMetrics
		} else {
			// Pre-training: no loss, no metrics.
			trainer = train.NewTrainer(backend, store, ByolCnnModelGraph,
				nil,
				theOptimizer,
				nil, // trainMetrics
				nil) // evalMetrics
		}
	default:
		return errors.Errorf("Unknown model type %q: valid values are [\"cnn\", \"inception\", \"byol\"]", modelType)
	}

	// Debugging.
	if model.GetParamOr(scope, "nan_logger", false) {
		nanlogger.New().AttachToTrainer(trainer)
	}

	// Use standard training loop.
	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.

	// Attach a checkpoint: checkpoint every 1 minute of training.
	if checkpointHandler != nil {
		period := time.Minute * 3
		train.PeriodicCallback(loop, period, true, "saving checkpoint", 100,
			func(loop *train.Loop, metrics []*tensors.Tensor) error {
				return checkpointHandler.Save()
			})
	}

	// Attach Plotly plots: plot points at exponential steps.
	// The points generated are saved along the checkpoint directory (if one is given).
	if model.GetParamOr(scope, plotly.ParamPlots, false) {
		if preTraining {
			// Pre-training: no evaluation.
			_ = plotly.New().
				WithCheckpoint(checkpointHandler).
				ScheduleExponential(loop, 200, 1.2).
				Dynamic()
		} else {
			_ = plotly.New().
				WithCheckpoint(checkpointHandler).
				Dynamic().
				WithDatasets(trainEvalDS, validationEvalDS).
				ScheduleExponential(loop, 200, 1.2).
				WithBatchNormalizationAveragesUpdate(trainEvalDS)
		}
	}

	// Loop for given number of steps.
	numTrainSteps := model.GetParamOr(scope, "train_steps", 0)
	globalStep := int(optimizer.GetGlobalStep(store))
	if globalStep < numTrainSteps {
		if _, err = loop.RunSteps(trainDS, int(numTrainSteps-globalStep)); err != nil {
			return err
		}
		fmt.Printf(
			"\t[Step %d] median train step: %d microseconds\n",
			loop.LoopStep,
			loop.MedianTrainStepDuration().Microseconds(),
		)
		updated, err := norm.UpdateBatchNormAverages(trainer, trainEvalDS)
		if err != nil {
			return err
		}
		if updated {
			fmt.Println("\tUpdated batch normalization mean/variances averages.")
			if err = checkpointHandler.Save(); err != nil {
				return err
			}
		}
	} else {
		fmt.Printf("\t - target train_steps=%d already reached. To train further, set a number additional "+
			"to current global step.\n", numTrainSteps)
	}
	fmt.Printf("Training done (global_step=%d).\n", optimizer.GetGlobalStep(scope))

	if preTraining {
		// If pre-training (unsupervised), skip evaluation, and clear optimizer variables and global step.
		fmt.Println("Pre-training only, no evaluation.")
		if err = theOptimizer.Clear(scope); err != nil {
			return err
		}
		if err = optimizer.DeleteGlobalStep(scope); err != nil {
			return err
		}
		if err = checkpointHandler.Save(); err != nil {
			return err
		}
		return nil
	}

	// Finally, print an evaluation on train and test datasets.
	if runEval {
		fmt.Println()
		if err = commandline.ReportEval(trainer, trainEvalDS, validationEvalDS); err != nil {
			return err
		}
		fmt.Println()
	}
	return nil
}
