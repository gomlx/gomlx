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

// Linear generates random synthetic data, based on some linear mode + noise.
// Then it learns the original weights used to generate the data.
package main

import (
	"flag"
	"fmt"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/must"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/datasets"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gopjrt/dtypes"
	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/default"
)

const (
	CoefficientMu    = 0.0
	CoefficientSigma = 5.0
	BiasMu           = 1.0
	BiasSigma        = 10.0
)

// initCoefficients chooses random coefficients and bias. These are the true values the model will
// attempt to learn.
func initCoefficients(backend backends.Backend, numVariables int) (coefficients, bias *tensors.Tensor) {
	e := MustNewExec(backend, func(g *Graph) (coefficients, bias *Node) {
		rngState := RNGStateForGraph(g)
		rngState, coefficients = RandomNormal(rngState, shapes.Make(dtypes.Float64, numVariables))
		coefficients = AddScalar(
			MulScalar(coefficients, CoefficientSigma),
			CoefficientMu)
		rngState, bias = RandomNormal(rngState, shapes.Make(dtypes.Float64))
		bias = AddScalar(MulScalar(bias, BiasSigma), BiasMu)
		return
	})
	results := e.MustExec()
	coefficients, bias = results[0], results[1]
	return
}

func buildExamples(
	backend backends.Backend,
	coef, bias *tensors.Tensor,
	numExamples int,
	noise float64,
) (inputs, labels *tensors.Tensor) {
	e := MustNewExec(backend, func(coef, bias *Node) (inputs, labels *Node) {
		g := coef.Graph()
		numFeatures := coef.Shape().Dimensions[0]

		// Random inputs (observations).
		rngState := RNGStateForGraph(g)
		rngState, inputs = RandomNormal(rngState, shapes.Make(dtypes.Float64, numExamples, numFeatures))
		coef = InsertAxes(coef, 0)

		// Calculate perfect labels.
		labels = ReduceAndKeep(Mul(inputs, coef), ReduceSum, -1)
		labels = Add(labels, bias)
		if noise > 0 {
			// Add some noise to the labels.
			var noiseVector *Node
			rngState, noiseVector = RandomNormal(rngState, labels.Shape())
			noiseVector = MulScalar(noiseVector, noise)
			labels = Add(labels, noiseVector)
		}
		return
	})
	examples := e.MustExec(coef, bias)
	inputs, labels = examples[0], examples[1]
	return
}

// modelGraph builds graph that returns predictions for inputs.
func modelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	_ = spec
	logits := layers.Dense(ctx, inputs[0], true, 1)
	return []*Node{logits}
}

var (
	flagNumExamples  = flag.Int("num_examples", 10000, "Number of examples to generate")
	flagNumFeatures  = flag.Int("num_features", 3, "Number of features")
	flagNoise        = flag.Float64("noise", 0.2, "Noise in synthetic data generation")
	flagNumSteps     = flag.Int("steps", 1000, "Number of gradient descent steps to perform")
	flagNumThreads   = flag.Int("num_threads", -1, "Number of threads. Leave as -1 to use as many as there are cores.")
	flagNumReplicas  = flag.Int("num_replicas", 1, "Number of replicas.")
	flagPlatform     = flag.String("platform", "", "PluginDescription to use, if empty uses the default one.")
	flagLearningRate = flag.Float64("learning_rate", 0.1, "Initial learning rate.")
)

func main() {
	flag.Parse()
	backend := backends.MustNew()

	fmt.Printf("Backend: %s, %s\n", backend.Name(), backend.Description())

	trueCoefficients, trueBias := initCoefficients(backend, *flagNumFeatures)
	fmt.Printf("Target coefficients: %0.5v\n", trueCoefficients.Value())
	fmt.Printf("Target bias: %0.5v\n\n", trueBias.Value())

	inputs, labels := buildExamples(backend, trueCoefficients, trueBias, *flagNumExamples, *flagNoise)
	fmt.Printf("Training data (inputs, labels): (%s, %s)\n\n", inputs.Shape(), labels.Shape())

	// Create an in-memory dataset from the tensors.
	dataset := must.M1(datasets.InMemoryFromData(backend, "linear dataset", []any{inputs}, []any{labels})).
		Infinite(true).Shuffle().BatchSize(*flagNumExamples, false)
	// dataset := &Dataset{"training", []*tensors.Tensor{inputs}, []*tensors.Tensor{labels}}

	// Creates Context with learned weights and bias.
	ctx := context.New()
	ctx.SetParam(optimizers.ParamLearningRate, *flagLearningRate)

	// train.Trainer executes a training step.
	trainer := train.NewTrainer(backend, ctx, modelGraph,
		losses.MeanSquaredError,
		optimizers.StochasticGradientDescent().Done(),
		nil, nil) // trainMetrics, evalMetrics

	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.

	// Loop for the given number of steps.
	_, err := loop.RunSteps(dataset, *flagNumSteps)
	if err != nil {
		klog.Fatalf("Failed with error: %+v", err)
	}

	// Print learned coefficients and bias -- from the weights in the dense layer.
	fmt.Println()
	coefVar, biasVar := ctx.GetVariableByScopeAndName(
		"/dense",
		"weights",
	), ctx.GetVariableByScopeAndName(
		"/dense",
		"biases",
	)
	learnedCoef, learnedBias := coefVar.MustValue(), biasVar.MustValue()
	fmt.Printf("Learned coefficients: %0.5v\n", learnedCoef.Value())
	fmt.Printf("Learned bias: %0.5v\n", learnedBias.Value())
}
