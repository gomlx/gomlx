// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package kan

import (
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/rational"
)

// GR-KAN ("Group Rational Kolmogorov-Arnold Networks") is a variation of KAN that uses a rational functions as
// its univariate function family -- as opposed to the original spline functions. It is described in [1].
//
// See implementation details in package ml/layers/rational, [2] and [3].
//
// [1] "Kolmogorov-Arnold Transformer" by Xingyi Yang and Xinchao Wang, https://arxiv.org/abs/2409.10594
// [2] https://github.com/ml-research/rational_activations/
// [3] "Padé Activation Units: End-to-end Learning of Flexible Activation Functions in Deep Networks" by

var (
	// ParamRational indicates whether to use GR-KAN (KAN with rational functions) as the univariate function to learn.
	ParamRational = "kan_rational"

	// ParamRationalNumeratorDegree configures the default value for the numerator degree for GR-KAN.
	ParamRationalNumeratorDegree = "kan_rational_num"

	// ParamRationalDenominatorDegree configures the default value for the numerator degree for GR-KAN.
	ParamRationalDenominatorDegree = "kan_rational_den"

	// ParamRationalInitialApproximation configures how to initialize the values of the rational functions.
	// Valid values are "identity", "swish", "relu". See rational if the approximation is not given for the
	// given the degrees combination.
	ParamRationalInitialApproximation = "kan_rational_approximate"
)

// rationalConfig holds the configuration exclusive for GR-KANs
type rationalConfig struct {
	numDegree, denDegree          int
	initialApproximation, version string
	withMultiplier                bool
	multiplierInitialVariance     float64
}

// initRational initializes the GR-KAN configuration.
func (c *Config) initRational(ctx *context.Context) {
	c.rational.numDegree = context.GetParamOr(ctx, ParamRationalNumeratorDegree, 5)
	c.rational.denDegree = context.GetParamOr(ctx, ParamRationalDenominatorDegree, 4)
	c.rational.initialApproximation = context.GetParamOr(ctx, ParamRationalInitialApproximation, rational.IdentityApproximation)
	c.rational.version = "B"
	c.rational.withMultiplier = true
	c.rational.multiplierInitialVariance = 1.0
}

// Rational configures for GR-KAN, in which rational functions (as opposed to b-splines) are used to model \phi(x),
// the univariate function. See description in [1].
//
// See implementation details in package ml/layers/rational, [2] and [3].
//
// [1] "Kolmogorov-Arnold Transformer" by Xingyi Yang and Xinchao Wang, https://arxiv.org/abs/2409.10594
// [2] https://github.com/ml-research/rational_activations/
// [3] "Padé Activation Units: End-to-end Learning of Flexible Activation Functions in Deep Networks" by
func (c *Config) Rational() *Config {
	c.useRational = true
	return c
}

// Layer implements one GR-KAN layer. x is expected to be shaped [batchSize, numInputNodes].
func (c *Config) rationalLayer(ctx *context.Context, x *Node, numOutputNodes int) *Node {
	residual := x
	batchSize := x.Shape().Dimensions[0]
	numInputNodes := x.Shape().Dimensions[x.Rank()-1]
	numInputGroups := numInputNodes
	if c.inputGroupSize > 1 {
		numInputGroups = numInputNodes / c.inputGroupSize
	}
	output := rational.New(ctx, x).
		Version(c.rational.version).
		Approximate(c.rational.initialApproximation).
		WithMultiplier(c.rational.withMultiplier).
		WithMultiplierInitVariance(c.rational.multiplierInitialVariance).
		WithInputGroups(numInputGroups).
		WithMultipleOutputs(numOutputNodes).
		Done()
	if numOutputNodes > 1 {
		output.AssertDims(batchSize, numOutputNodes, numInputNodes)
	} else {
		output.AssertDims(batchSize, numInputNodes)
	}

	// Reduce the inputs to get the outputs:
	if c.useMean {
		output = ReduceMean(output, -1)
	} else {
		output = ReduceSum(output, -1)
	}
	if numOutputNodes > 1 {
		output.AssertDims(batchSize, numOutputNodes) // Shape=[batch, outputs]
	} else {
		output = Reshape(output, batchSize, 1)
	}

	if c.useResidual && numInputNodes == numOutputNodes {
		output = Add(output, residual)
	}
	return output
}
