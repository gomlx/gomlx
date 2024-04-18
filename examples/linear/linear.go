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

// Linear generates random synthetic data, based on some linear mode + noise. Then it tries
// to learn the weights used to generate the data.
package main

import (
	"flag"
	"fmt"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/commandline"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"

	. "github.com/gomlx/gomlx/graph"
)

const (
	CoefficientMu    = 0.0
	CoefficientSigma = 5.0
	BiasMu           = 1.0
	BiasSigma        = 10.0
)

// initCoefficients chooses random coefficients and bias. These are the true values the model will
// attempt to learn.
func initCoefficients(manager *Manager, numVariables int) (coefficients, bias tensor.Tensor) {
	e := NewExec(manager, func(g *Graph) (coefficients, bias *Node) {
		rngState := Const(g, RngState())
		rngState, coefficients = RandomNormal(rngState, shapes.Make(shapes.F64, numVariables))
		coefficients = AddScalar(
			MulScalar(coefficients, CoefficientSigma),
			CoefficientMu)
		rngState, bias = RandomNormal(rngState, shapes.Make(shapes.F64))
		bias = AddScalar(MulScalar(bias, BiasSigma), BiasMu)
		return
	})
	results := e.Call()
	coefficients, bias = results[0], results[1]
	return
}

func buildExamples(manager *Manager, coef, bias tensor.Tensor, numExamples int, noise float64) (inputs, labels tensor.Tensor) {
	e := NewExec(manager, func(coef, bias *Node) (inputs, labels *Node) {
		g := coef.Graph()
		numFeatures := coef.Shape().Dimensions[0]

		// Random inputs (observations).
		rngState := Const(g, RngState())
		rngState, inputs = RandomNormal(rngState, shapes.Make(shapes.F64, numExamples, numFeatures))
		coef = ExpandDims(coef, 0)

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
	examples := e.Call(coef, bias)
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
	flagPlatform     = flag.String("platform", "", "Platform to use, if empty uses the default one.")
	flagLearningRate = flag.Float64("learning_rate", 0.1, "Initial learning rate.")
)

// Dataset is a trivial dataset that always returns the whole data.
type Dataset struct {
	name           string
	inputs, labels []tensor.Tensor
}

func (ds *Dataset) Name() string { return ds.name }

// Yield implements train.Dataset
func (ds *Dataset) Yield() (spec any, inputs, labels []tensor.Tensor, err error) {
	return nil, ds.inputs, ds.labels, nil
}

// Reset implements train.Dataset
func (ds *Dataset) Reset() {}

func main() {
	flag.Parse()
	manager := BuildManager().NumThreads(*flagNumThreads).NumReplicas(*flagNumReplicas).Platform(*flagPlatform).Done()

	trueCoefficients, trueBias := initCoefficients(manager, *flagNumFeatures)
	fmt.Printf("Target coefficients: %0.5v\n", trueCoefficients.Value())
	fmt.Printf("Target bias: %0.5v\n\n", trueBias.Value())

	inputs, labels := buildExamples(manager, trueCoefficients, trueBias, *flagNumExamples, *flagNoise)
	fmt.Printf("Training data (inputs, labels): (%s, %s)\n\n", inputs.Shape(), labels.Shape())
	dataset := &Dataset{"training", []tensor.Tensor{inputs}, []tensor.Tensor{labels}}

	// Creates Context with learned weights and bias.
	ctx := context.NewContext(manager)
	ctx.SetParam(optimizers.ParamLearningRate, *flagLearningRate)

	// train.Trainer executes a training step.
	trainer := train.NewTrainer(manager, ctx, modelGraph,
		losses.MeanSquaredError,
		optimizers.StochasticGradientDescent(),
		nil, nil) // trainMetrics, evalMetrics

	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.

	// Loop for given number of steps.
	_, err := loop.RunSteps(dataset, *flagNumSteps)
	if err != nil {
		panic(err)
	}

	// Print learned coefficients and bias -- from the weights in the dense layer.
	fmt.Println()
	coefVar, biasVar := ctx.InspectVariable("/dense", "weights"), ctx.InspectVariable("/dense", "biases")
	learnedCoef, learnedBias := coefVar.Value(), biasVar.Value()
	fmt.Printf("Learned coefficients: %0.5v\n", learnedCoef.Value())
	fmt.Printf("Learned bias: %0.5v\n", learnedBias.Value())
}
