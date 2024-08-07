package diffusion

import (
	"github.com/gomlx/gomlx/examples/notebook/gonb/plotly"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activations"
	"github.com/gomlx/gomlx/ml/layers/fnn"
	"github.com/gomlx/gomlx/ml/layers/regularizers"
	"github.com/gomlx/gomlx/ml/train/optimizers"
)

var (
	// ParamsExcludedFromSaving is the list of parameters (see CreateDefaultContext) that shouldn't be saved
	// along on the models checkpoints, and may be overwritten in further training sessions.
	ParamsExcludedFromSaving = []string{
		"data_dir", "train_steps", "plots",
		"num_checkpoints", "checkpoint_frequency",
		"kid", "rng_reset",

		// Notice that "samples_during_training" is saved, since it can't change after a model started training.
		"samples_during_training_frequency", "samples_during_training_frequency_growth",
	}
)

// CreateDefaultContext sets the context with default hyperparameters to use with TrainModel.
func CreateDefaultContext() *context.Context {
	ctx := context.New()
	ctx.RngStateReset()
	ctx.SetParams(map[string]any{
		// Model type to use
		"model":                "bow", // One of the listed in ValidModels: the user can also inject (in ValidModels) new custom models.
		"train_steps":          5000,
		"num_checkpoints":      3,
		"checkpoint_frequency": "10m", // How often to save checkpoints. Default to 10 minutes. See time.ParseDuration.

		// batch_size for training.
		"batch_size": 64,

		// eval_batch_size can be larger than training, it's more efficient.
		"eval_batch_size": 128,

		// image_size of the images to use: since we'll cache them in the accelerator machine, it "+
		// is limited by its memory. Largest value (the original image sizes) is 500.
		"image_size": 64,

		// dtype to use for the model.
		"dtype": "float32",

		// samples_during_training is the number of images that are updated during training, to observe the evolution of the
		// model.
		// These start with noise, that gets de-noised to images at different stages of the training.
		"samples_during_training":                  64,
		"samples_during_training_frequency":        200, // Number of steps between regenerating samples. It's actually the period not the frequency.
		"samples_during_training_frequency_growth": 1.2, // Growth factor for samples_during_training_frequency.

		// kid enables calculating Kernel Inception Distance (KID) on evaluation -- it is quite expensive."
		"kid": false,

		// rng_reset enables resetting the random number generator state with a new random value -- useful when continuing training.
		"rng_reset": true,

		// Debugging: add a NanLogger to help debug where NaNs may appear in the model.
		"nan_logger": false,

		// Model parameters for the dataset:
		"flower_type_embed_size": 16,     // If > 0, use embedding of the flower type of the given dimension.
		"sinusoidal_embed_size":  32,     // Sinusoidal embedding size. It must be an even number.
		"sinusoidal_max_freq":    1000.0, // Sinusoidal embedding max frequency.
		"sinusoidal_min_freq":    1.0,    // Sinusoidal embedding min frequency.

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

		optimizers.ParamOptimizer:           "adamw",
		optimizers.ParamLearningRate:        1e-4,
		optimizers.ParamAdamEpsilon:         1e-7,
		optimizers.ParamAdamDType:           "",
		optimizers.ParamCosineScheduleSteps: 0,
		activations.ParamActivation:         "",
		layers.ParamDropoutRate:             0.1,
		regularizers.ParamL2:                0.0,
		regularizers.ParamL1:                0.0,

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
	})
	return ctx
}
