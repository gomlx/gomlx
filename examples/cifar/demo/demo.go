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

// CIFAR-10 demo trainer.
// It supports a CNN, FNN, KAN and DiscreteKAN models, with many different options.
package main

import (
	"flag"

	"github.com/gomlx/gomlx/examples/cifar"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/layers/fnn"
	"github.com/gomlx/gomlx/pkg/ml/layers/kan"
	"github.com/gomlx/gomlx/pkg/ml/layers/regularizers"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers/cosineschedule"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gomlx/ui/gonb/plotly"
	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/default"
)

var (
	flagDataDir    = flag.String("data", "~/work/cifar", "Directory to cache downloaded and generated dataset files.")
	flagEval       = flag.Bool("eval", true, "Whether to evaluate the model on the validation data in the end.")
	flagVerbosity  = flag.Int("verbosity", 1, "Level of verbosity, the higher the more verbose.")
	flagCheckpoint = flag.String("checkpoint", "", "Directory save and load checkpoints from. If left empty, no checkpoints are created.")
)

// createDefaultContext sets the context with default hyperparameters
func createDefaultContext() *context.Context {
	ctx := context.New()
	ctx.ResetRNGState()
	ctx.SetParams(map[string]any{
		// Model type to use
		"model":           cifar.C10ValidModels[0],
		"checkpoint":      "",
		"num_checkpoints": 3,
		"train_steps":     3000,

		// batch_size for training.
		"batch_size": 64,

		// eval_batch_size can be larger than training, it's more efficient.
		"eval_batch_size": 200,

		// "plots" trigger generating intermediary eval data for plotting, and if running in GoNB, to actually
		// draw the plot with Plotly.
		//
		// From the command-line, an easy way to monitor the metrics being generated during the training of a model
		// is using the gomlx_checkpoints tool:
		//
		//	$ gomlx_checkpoints --metrics --metrics_labels --metrics_types=accuracy  --metrics_names='E(Tra)/#loss,E(Val)/#loss' --loop=3s "<checkpoint_path>"
		plotly.ParamPlots: true,

		// "normalization" is overridden by "fnn_normalization" if set.
		layers.ParamNormalization: "none",

		optimizers.ParamOptimizer:       "adamw",
		optimizers.ParamLearningRate:    1e-4,
		optimizers.ParamAdamEpsilon:     1e-7,
		optimizers.ParamAdamDType:       "",
		cosineschedule.ParamPeriodSteps: 0,
		activations.ParamActivation:     "swish",
		layers.ParamDropoutRate:         0.0,
		regularizers.ParamL2:            0.0, // 1e-5,
		regularizers.ParamL1:            0.0, // 1e-5,

		// FNN network parameters:
		fnn.ParamNumHiddenLayers: 8,
		fnn.ParamNumHiddenNodes:  128,
		fnn.ParamResidual:        true,
		fnn.ParamNormalization:   "",   // Set to "none" for no normalization, otherwise it falls back to layers.ParamNormalization.
		fnn.ParamDropoutRate:     -1.0, // Set to 0.0 for no dropout, otherwise it falls back to layers.ParamDropoutRate.

		// KAN network parameters:
		kan.ParamRational:        false, // "kan_rational" (see GR-KAN) uses rational functions.
		kan.ParamPiecewiseLinear: false, // "kan_pwl" uses piecewise linear functions.
		kan.ParamDiscrete:        false, // "kan_discrete" uses discrete (piecewise-constant) functions: tricky to train.

		kan.ParamResidual:         true,
		kan.ParamMean:             true,
		kan.ParamNumControlPoints: 10, // Number of control points
		kan.ParamNumHiddenNodes:   64,
		kan.ParamNumHiddenLayers:  4,

		kan.ParamBSplineDegree:      2,
		kan.ParamBSplineMagnitudeL1: 1e-5,
		kan.ParamBSplineMagnitudeL2: 0.0,

		// Discrete-KAN
		kan.ParamDiscreteSoftness: 0.1,

		// KAN-PWL
		kan.ParamPWLSplitPointsTrainable: true, // Split points trainable.

		// CNN model
		cifar.ParamCNNNormalization: "batch",
	})
	return ctx
}

func main() {
	// Flags with context settings.
	ctx := createDefaultContext()
	settings := commandline.CreateContextSettingsFlag(ctx, "")
	klog.InitFlags(nil)
	flag.Parse()
	paramsSet := must.M1(commandline.ParseContextSettings(ctx, *settings))

	// Train.
	cifar.TrainCifar10Model(ctx, *flagDataDir, *flagCheckpoint, *flagEval, *flagVerbosity, paramsSet)
}
