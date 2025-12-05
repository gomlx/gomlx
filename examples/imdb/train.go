package imdb

import (
	"fmt"
	"os"
	"time"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/checkpoints"
	"github.com/gomlx/gomlx/pkg/ml/datasets"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/layers/batchnorm"
	"github.com/gomlx/gomlx/pkg/ml/layers/fnn"
	"github.com/gomlx/gomlx/pkg/ml/layers/regularizers"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gomlx/pkg/ml/train/metrics"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers/cosineschedule"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gomlx/ui/gonb/plotly"
	"github.com/gomlx/gopjrt/dtypes"

	"github.com/charmbracelet/lipgloss"
	"github.com/gomlx/gomlx/internal/must"
	"golang.org/x/exp/maps"
)

var (
	// ValidModels is the list of model types supported.
	ValidModels = map[string]train.ModelFn{
		"bow":         BagOfWordsModelGraph,
		"cnn":         Conv1DModelGraph,
		"transformer": TransformerModelGraph,
	}

	// ParamsExcludedFromLoading is the list of parameters (see CreateDefaultContext) that shouldn't be saved
	// along on the models checkpoints, and may be overwritten in further training sessions.
	ParamsExcludedFromLoading = []string{
		"data_dir", "train_steps", "num_checkpoints", "plots",
	}
)

// DType used in the mode.
var DType = dtypes.Float32

// CreateDefaultContext sets the context with default hyperparameters to use with TrainModel.
func CreateDefaultContext() *context.Context {
	ctx := context.New()
	ctx.ResetRNGState()
	ctx.SetParams(map[string]any{
		// Model type to use
		"model":           "bow", // One of the listed in ValidModels: the user can also inject (in ValidModels) new custom models.
		"train_steps":     5000,
		"num_checkpoints": 3,

		// batch_size for training.
		"batch_size": 32,

		// eval_batch_size can be larger than training, it's more efficient.
		"eval_batch_size": 200,

		// Imdb dataset parameters:
		"imdb_mask_word_task_weight": 0.0,    // Include "masked word" self-supervised task with this given weight.
		"imdb_use_unsupervised":      false,  // Use unsupervised dataset to pretrain with mask word task -- requires further fine-tuning later.
		"imdb_include_separators":    false,  // If true include the word separator symbols in the tokens.
		"imdb_content_max_len":       200,    // Maximum number of tokens to take from observation, per example.
		"imdb_max_vocab":             20_000, // Top most frequent words to consider, the rest is considered unknown.
		"imdb_token_embedding_size":  32,     // Size of token embedding table. There are ~140K unique tokens.
		"imdb_word_dropout_rate":     0.0,    // Special kind of dropout, used by all model types.

		// "plots" trigger generating intermediary eval data for plotting, and if running in GoNB, to actually
		// draw the plot with Plotly.
		//
		// From the command-line, an easy way to monitor the metrics being generated during the training of a model
		// is using the gomlx_checkpoints tool:
		//
		//	$ gomlx_checkpoints --metrics --metrics_labels --metrics_types=accuracy  --metrics_names='E(Tra)/#loss,E(Val)/#loss' --loop=3s "<checkpoint_path>"
		plotly.ParamPlots: true,

		// "normalization" is overridden by "fnn_normalization" and "cnn_normalization", if they are set.
		layers.ParamNormalization: "layer",

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
		fnn.ParamNumHiddenLayers: 2,
		fnn.ParamNumHiddenNodes:  32,
		fnn.ParamResidual:        true,
		fnn.ParamNormalization:   "",  // Set to "none" for no normalization. If "" it falls back to layers.ParamNormalization.
		fnn.ParamDropoutRate:     0.3, // Set to 0.0 for no dropout. If < 0 it falls back to layers.ParamDropoutRate.

		// CNN
		"cnn_num_layers":    5.0,
		"cnn_dropout_rate":  0.5, // Set to 0.0 for no dropout. If < 0 it falls back to layers.ParamDropoutRate.
		"cnn_normalization": "",  // Set to "none" for no normalization. If "" it falls back to layers.ParamNormalization.

		// Transformers
		"transformer_max_att_len":    200,  // Maximum attention length: input will be split in ranges of this size.
		"transformer_num_att_heads":  2,    // umber of attention heads,/ if --model=transformer.
		"transformer_num_att_layers": 1,    // Number of stacked attention layers, if --model=transformer.
		"transformer_att_key_size":   8,    // Dimension of the Key/Query attention embedding.
		"transformer_dropout_rate":   -1.0, // Set to 0.0 for no dropout. If < 0 it falls back to layers.ParamDropoutRate.
	})
	return ctx
}

// TrainModel with hyperparameters given in ctx.
func TrainModel(
	ctx *context.Context,
	dataDir, checkpointPath string,
	paramsSet []string,
	evaluateOnEnd bool,
	verbosity int,
) {
	// Data directory: datasets and top-level directory holding checkpoints for different models.
	dataDir = fsutil.MustReplaceTildeInDir(dataDir)
	if !fsutil.MustFileExists(dataDir) {
		must.M(os.MkdirAll(dataDir, 0777))
	}

	// Imdb data preparation.
	IncludeSeparators = context.GetParamOr(ctx, "imdb_include_separators", false)
	must.M(Download(dataDir))
	imdbUseUnsupervised := context.GetParamOr(ctx, "imdb_use_unsupervised", false)
	imdbMaskWordTaskWeight := context.GetParamOr(ctx, "imdb_mask_word_task_weight", 0.0)
	if imdbUseUnsupervised && imdbMaskWordTaskWeight <= 0 {
		exceptions.Panicf(
			`Parameter "imdb_use_unsupervised" is only useful together with parameter "imdb_mask_word_task" (=%g) > 0.0`,
			imdbMaskWordTaskWeight,
		)
	}

	// Backend handles creation of ML computation graphs, accelerator resources, etc.
	backend := backends.MustNew()
	if verbosity >= 1 {
		fmt.Printf("Backend %q:\t%s\n", backend.Name(), backend.Description())
	}

	// Create datasets used for training and evaluation.
	batchSize := context.GetParamOr(ctx, "batch_size", int(0))
	if batchSize <= 0 {
		exceptions.Panicf("Batch size must be > 0 (maybe it was not set?): %d", batchSize)
	}
	evalBatchSize := context.GetParamOr(ctx, "eval_batch_size", int(0))
	if evalBatchSize <= 0 {
		evalBatchSize = batchSize
	}
	var trainDS, trainEvalDS, testEvalDS train.Dataset
	maxLen := context.GetParamOr(ctx, "imdb_content_max_len", 200)
	if imdbUseUnsupervised {
		trainDS = NewUnsupervisedDataset("unsupervised-train", maxLen, batchSize, true).Shuffle()
	} else {
		trainDS = NewDataset("train", TypeTrain, maxLen, batchSize, true).Shuffle()
	}
	trainEvalDS = NewDataset("train-eval", TypeTrain, maxLen, batchSize, false)
	testEvalDS = NewDataset("test-eval", TypeTest, maxLen, batchSize, false)

	// Parallelize generation of batches.
	trainDS = datasets.Parallel(trainDS)
	trainEvalDS = datasets.Parallel(trainEvalDS)
	testEvalDS = datasets.Parallel(testEvalDS)

	// Checkpoints saving.
	var checkpoint *checkpoints.Handler
	if checkpointPath != "" {
		numCheckpointsToKeep := context.GetParamOr(ctx, "num_checkpoints", 3)
		checkpoint = must.M1(checkpoints.Build(ctx).
			DirFromBase(checkpointPath, dataDir).
			Keep(numCheckpointsToKeep).
			ExcludeParams(append(paramsSet, ParamsExcludedFromLoading...)...).
			Done())
		fmt.Printf("Checkpoint: %q\n", checkpoint.Dir())
	}
	if verbosity >= 2 {
		fmt.Println(commandline.SprintContextSettings(ctx))
	}

	// Select model graph building function.
	modelType := context.GetParamOr(ctx, "model", "bow")
	modelFn, found := ValidModels[modelType]
	if !found {
		exceptions.Panicf("Parameter \"model\" must take one value from %v, got %q", maps.Keys(ValidModels), modelType)
	}
	fmt.Printf("Model: %s\n", modelType)

	// Metrics we are interested.
	meanAccuracyMetric := metrics.NewMeanBinaryLogitsAccuracy("Mean Accuracy", "#acc")
	movingAccuracyMetric := metrics.NewMovingAverageBinaryLogitsAccuracy("Moving Average Accuracy", "~acc", 0.01)

	// Create a train.Trainer: this object will orchestrate running the model, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	ctx = ctx.In("model") // Convention scope used for model creation.
	var loss train.LossFn
	if !imdbUseUnsupervised {
		loss = losses.BinaryCrossentropyLogits
	}
	trainer := train.NewTrainer(backend, ctx, modelFn,
		loss,
		optimizers.FromContext(ctx),
		[]metrics.Interface{movingAccuracyMetric}, // trainMetrics
		[]metrics.Interface{meanAccuracyMetric})   // evalMetrics

	// Use standard training loop.
	loop := train.NewLoop(trainer)
	if verbosity >= 0 {
		commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.
	}

	// Checkpoint saving: every 3 minutes of training.
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
		_ = plotly.New().
			WithCheckpoint(checkpoint).
			Dynamic().
			WithDatasets(trainEvalDS, testEvalDS).
			ScheduleExponential(loop, 200, 1.2).
			WithBatchNormalizationAveragesUpdate(trainEvalDS)
	}

	// Loop for given number of steps.
	numTrainSteps := context.GetParamOr(ctx, "train_steps", 0)
	globalStep := int(optimizers.GetGlobalStep(ctx))
	if globalStep > 0 {
		trainer.SetContext(ctx.Reuse())
	}
	if globalStep < numTrainSteps {
		_ = must.M1(loop.RunSteps(trainDS, numTrainSteps-globalStep))
		if verbosity >= 1 {
			fmt.Printf("\t[Step %d] median train step: %d microseconds\n",
				loop.LoopStep, loop.MedianTrainStepDuration().Microseconds())
		}

		// Update batch normalization averages, if they are used.
		if must.M1(batchnorm.UpdateAverages(trainer, trainEvalDS)) {
			fmt.Println("\tUpdated batch normalization mean/variances averages.")
			if checkpoint != nil {
				must.M(checkpoint.Save())
			}
		}

	} else {
		fmt.Printf("\t - target train_steps=%d already reached. To train further, set a number additional "+
			"to current global step.\n", numTrainSteps)
	}

	// Finally, print an evaluation on train and test datasets.
	if evaluateOnEnd {
		if verbosity >= 1 {
			fmt.Println()
		}
		must.M(commandline.ReportEval(trainer, trainEvalDS, testEvalDS))
	}
}

var sampleStyle = lipgloss.NewStyle().
	Border(lipgloss.NormalBorder()).
	Padding(1, 4, 1, 4).
	Width(60)

// PrintSample of n examples.
func PrintSample(n int) {
	const maxLen = 200
	ds := NewDataset("TypeTest", TypeTest, maxLen, n, true).Shuffle()
	_, inputs, labels := must.M3(ds.Yield())
	tensors.MustConstFlatData[int8](labels[0], func(labelsData []int8) {
		for ii := range n {
			fmt.Println(sampleStyle.Render(
				fmt.Sprintf("[Sample %d - label %v]\n%s\n", ii, labelsData[ii], InputToString(inputs[0], ii))))
		}
	})
	fmt.Println()
}
