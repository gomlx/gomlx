package diffusion

import (
	"github.com/gomlx/gomlx/examples/notebook/gonb/plotly"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activations"
	"github.com/gomlx/gomlx/ml/layers/regularizers"
	"github.com/gomlx/gomlx/ml/train/optimizers"
)

var (
	// ParamsExcludedFromLoading is the list of parameters (see CreateDefaultContext) that shouldn't be loaded
	// from models checkpoints.
	//
	// These are appended to the list of settings given in the command line in the flag -set.
	ParamsExcludedFromLoading = []string{
		"data_dir", "train_steps", "plots",
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
		"num_checkpoints":      5,
		"checkpoint_frequency": "3m", // How often to save checkpoints. Default to 3 minutes. See time.ParseDuration.

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

		// Diffusion model:
		"diffusion_loss":                "mae",                  // "mse" (Mean-Squared-Error), "mae" (Mean-Absolute-Error) or "huber".
		"huber_delta":                   0.2,                    // If "huber" loss is selected, this is the delta, after which the loss becomes linear.
		"diffusion_context_features":    true,                   // Enable context features concatenated at every DownBlock convolution -- they are always included at the start of the model.
		"diffusion_num_residual_blocks": 2,                      // Number of residual blocks per image size in the U-Net model.
		"diffusion_channels_list":       []int{32, 64, 96, 128}, // Number of channels (features) for each image size (progressively smaller) in U-Net model.
		"diffusion_min_signal_ratio":    0.02,                   // Minimum of the signal-to-noise ratio when training. Must be > 0.
		"diffusion_max_signal_ratio":    0.95,                   // Maximum of the signal-to-noise ratio when training.
		"diffusion_balanced_dataset":    false,                  // Enable training on a balanced dataset: batch_size=102, one example per flower type.
		"diffusion_pool":                "mean",                 // Values are: "mean", "max", "sum", "concat"
		"diffusion_residual_version":    1,                      // Valid values are 1 or 2. See code in function ResidualBlock.
		"diffusion_ema":                 0.999,                  // Exponential Moving Average of the model weights to use during evaluation. Set to <= 0 to disable.

		// Model parameters for the dataset:
		"flower_type_embed_size": 16,     // If > 0, use embedding of the flower type of the given dimension.
		"sinusoidal_embed_size":  32,     // Sinusoidal embedding size. It must be an even number.
		"sinusoidal_max_freq":    1000.0, // Sinusoidal embedding max frequency.
		"sinusoidal_min_freq":    1.0,    // Sinusoidal embedding min frequency.

		// "normalization" is overridden by "fnn_normalization" and "cnn_normalization", if they are set.
		layers.ParamNormalization: "layer",

		optimizers.ParamOptimizer:           "adam",
		optimizers.ParamAdamEpsilon:         1e-7,
		optimizers.ParamAdamDType:           "",
		optimizers.ParamAdamWeightDecay:     1e-4,
		optimizers.ParamCosineScheduleSteps: 0,
		activations.ParamActivation:         "swish",
		layers.ParamDropoutRate:             0.15,
		regularizers.ParamL2:                0.0,
		regularizers.ParamL1:                0.0,

		optimizers.ParamLearningRate:                  1e-3,
		optimizers.ParamCosineScheduleSteps:           0, // Enabled if > 0, it sets the period of the cosine schedule. Typically, the same value as 'train_steps'.
		optimizers.ParamCosineScheduleMinLearningRate: 1e-5,

		// "plots" trigger generating intermediary eval data for plotting, and if running in GoNB, to actually
		// draw the plot with Plotly.
		//
		// From the command-line, an easy way to monitor the metrics being generated during the training of a model
		// is using the gomlx_checkpoints tool:
		//
		//	$ gomlx_checkpoints --metrics --metrics_labels --metrics_types=accuracy  --metrics_names='E(Tra)/#loss,E(Val)/#loss' --loop=3s "<checkpoint_path>"
		plotly.ParamPlots: true,
	})
	return ctx
}
