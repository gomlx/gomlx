package dogsvscats

import (
	"fmt"
	"os"
	"time"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/pkg/core/graph/nanlogger"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/checkpoints"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/layers/batchnorm"
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
	"github.com/gomlx/gomlx/ui/gonb/plotly"
	"golang.org/x/exp/maps"
	"k8s.io/klog/v2"
)

var (
	// ModelsFns maps a model name to its train model function.
	// It holds mappings to predefined models, but one can insert new ones.
	// It is used by the TrainModel function.
	//
	// Models are selected by the "model" hyperparameter.
	ModelsFns = map[string]train.ModelFn{
		"cnn":       CnnModelGraph,
		"inception": InceptionV3ModelGraph,
		"byol":      nil,
	}

	// ModelsPrep maps models to a preparation function called in TrainModel before training start.
	//
	// This can be extended for new models.
	ModelsPrep = map[string]func(ctx *context.Context, dataDir string, checkpoint *checkpoints.Handler){
		"inception": InceptionV3ModelPrep,
	}
)

// nanLogger is used for debugging, enabled with --nanlogger in the command line.
// See `nanlogger` package for details.
var nanLogger *nanlogger.NanLogger

// CreateDefaultContext sets the context with default hyperparameters to use with TrainModel.
func CreateDefaultContext() *context.Context {
	ctx := context.New()
	ctx.ResetRNGState()
	ctx.SetParams(map[string]any{
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
		// is using the gomlx_checkpoints tool:
		//
		//	$ gomlx_checkpoints --metrics --metrics_labels --metrics_types=accuracy  --metrics_names='E(Tra)/#loss,E(Val)/#loss' --loop=3s "<checkpoint_path>"
		plotly.ParamPlots: true,

		// "normalization" is overridden by "fnn_normalization" and "cnn_normalization", if they are set.
		layers.ParamNormalization: "batch",

		optimizers.ParamOptimizer:       "adamw",
		optimizers.ParamLearningRate:    1e-4,
		optimizers.ParamAdamEpsilon:     1e-7,
		optimizers.ParamAdamDType:       "",
		cosineschedule.ParamPeriodSteps: 0,
		activations.ParamActivation:     "",
		layers.ParamDropoutRate:         0.1,
		regularizers.ParamL2:            0.0,
		regularizers.ParamL1:            0.0,

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
	return ctx
}

// TrainModel based on configuration and flags.
func TrainModel(ctx *context.Context, dataDir, checkpointPath string, runEval bool, paramsSet []string) {
	dataDir = fsutil.MustReplaceTildeInDir(dataDir)
	if !fsutil.MustFileExists(dataDir) {
		must.M(os.MkdirAll(dataDir, 0777))
	}

	// Checkpoint: it loads if already exists, and it will save as we train.
	var checkpoint *checkpoints.Handler
	if checkpointPath != "" {
		numCheckpoints := context.GetParamOr(ctx, "num_checkpoints", 3)
		checkpoint = must.M1(checkpoints.Build(ctx).
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
			Keep(numCheckpoints).Done())
	}

	// Generation of augmented images and create datasets.
	must.M(Download(dataDir))
	must.M(FilterValidImages(dataDir))
	config := NewPreprocessingConfigurationFromContext(ctx, dataDir)
	trainDS, trainEvalDS, validationEvalDS := CreateDatasets(config)

	// Metrics we are interested.
	meanAccuracyMetric := metrics.NewMeanBinaryLogitsAccuracy("Mean Accuracy", "#acc")
	movingAccuracyMetric := metrics.NewMovingAverageBinaryLogitsAccuracy("Moving Average Accuracy", "~acc", 0.01)

	// Select the model type we are using:
	modelType := context.GetParamOr(ctx, "model", "")
	modelFn, found := ModelsFns[modelType]
	if !found {
		exceptions.Panicf("Unknown model type %q: valid values are %q", modelType, maps.Keys(ModelsFns))
	}
	fmt.Printf("Model: %q\n", modelType)
	if modelPrep, found := ModelsPrep[modelType]; found {
		modelPrep(ctx, dataDir, checkpoint)
	}
	// BYOL may require pretraining.
	preTraining := modelType == "byol" && context.GetParamOr(ctx, "byol_pretrain", false)
	if preTraining && checkpoint == nil {
		klog.Warning(
			"*** pre-training model but not saving (--checkpoint) the pretrained weights -- is only useful for debugging ***",
		)
	}

	// Create a train.Trainer: this object will orchestrate running the model, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	backend := backends.MustNew()
	var trainer *train.Trainer
	optimizer := optimizers.FromContext(ctx)
	if !preTraining {
		trainer = train.NewTrainer(backend, ctx, modelFn,
			losses.BinaryCrossentropyLogits,
			optimizer,
			[]metrics.Interface{movingAccuracyMetric}, // trainMetrics
			[]metrics.Interface{meanAccuracyMetric})   // evalMetrics
	} else {
		// Pre-training: no loss, no metrics.
		trainer = train.NewTrainer(backend, ctx, modelFn,
			nil,
			optimizer,
			nil, // trainMetrics
			nil) // evalMetrics
	}

	// Debugging.
	if context.GetParamOr(ctx, "nan_logger", false) {
		nanlogger.New().AttachToTrainer(trainer)
	}

	// Use standard training loop.
	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.

	// Attach a checkpoint: checkpoint every 1 minute of training.
	if checkpoint != nil {
		period := time.Minute * 3
		train.PeriodicCallback(loop, period, true, "saving checkpoint", 100,
			func(loop *train.Loop, metrics []*tensors.Tensor) error {
				return checkpoint.Save()
			})
	}

	// Attach Plotly plots: plot points at exponential steps.
	// The points generated are saved along the checkpoint directory (if one is given).
	if context.GetParamOr(ctx, plotly.ParamPlots, false) {
		if preTraining {
			// Pre-training: no evaluation.
			_ = plotly.New().
				WithCheckpoint(checkpoint).
				ScheduleExponential(loop, 200, 1.2).
				Dynamic()
		} else {
			_ = plotly.New().
				WithCheckpoint(checkpoint).
				Dynamic().
				WithDatasets(trainEvalDS, validationEvalDS).
				ScheduleExponential(loop, 200, 1.2).
				WithBatchNormalizationAveragesUpdate(trainEvalDS)
		}
	}

	// Loop for given number of steps.
	numTrainSteps := context.GetParamOr(ctx, "train_steps", 0)
	globalStep := int(optimizers.GetGlobalStep(ctx))
	if globalStep > 0 {
		trainer.SetContext(ctx.Reuse())
	}
	if globalStep < numTrainSteps {
		_ = must.M1(loop.RunSteps(trainDS, int(numTrainSteps-globalStep)))
		fmt.Printf(
			"\t[Step %d] median train step: %d microseconds\n",
			loop.LoopStep,
			loop.MedianTrainStepDuration().Microseconds(),
		)
		if must.M1(batchnorm.UpdateAverages(trainer, trainEvalDS)) {
			fmt.Println("\tUpdated batch normalization mean/variances averages.")
			must.M(checkpoint.Save())
		}
	} else {
		fmt.Printf("\t - target train_steps=%d already reached. To train further, set a number additional "+
			"to current global step.\n", numTrainSteps)
	}
	fmt.Printf("Training done (global_step=%d).\n", optimizers.GetGlobalStep(ctx))

	if preTraining {
		// If pre-training (unsupervised), skip evaluation, and clear optimizer variables and global step.
		fmt.Println("Pre-training only, no evaluation.")
		must.M(optimizer.Clear(ctx))
		must.M(optimizers.DeleteGlobalStep(ctx))
		must.M(checkpoint.Save())
		return
	}

	// Finally, print an evaluation on train and test datasets.
	if runEval {
		fmt.Println()
		must.M(commandline.ReportEval(trainer, trainEvalDS, validationEvalDS))
		fmt.Println()
	}
}
