// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package norm

import (
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/initializer"
)

// DynamicTanhBuilder holds the configuration for Dynamic Tanh (DyT).
// Once finished configuring, call DynamicTanhBuilder.Done().
type DynamicTanhBuilder struct {
	scope   *model.Scope
	operand *Node
	alpha   float64
}

var (
	// ParamDynamicTanhAlpha is the scope hyperparameter that defines the initial value for the
	// learnable alpha parameter. The default is 0.5.
	//
	// Alpha is a scalar shared across the input and scales values before the tanh nonlinearity.
	ParamDynamicTanhAlpha = "dynamic_tanh_alpha"
)

// DynamicTanh starts the configuration of a Dynamic Tanh (DyT) operation,
// as described in "Transformers without Normalization" (https://arxiv.org/abs/2503.10622).
//
// DyT is a drop-in replacement for layer normalization in transformer blocks. Unlike
// LayerNorm or RMSNorm, it does not compute mean or variance. Instead it applies:
//
//	DyT(x) = tanh(alpha * x) * gamma + beta
//
// Where:
//   - alpha is a learnable scalar (initialized to 0.5 by default)
//   - gamma is a learnable per-feature scale (initialized to 1)
//   - beta is a learnable per-feature offset (initialized to 0)
//
// Parameters are stored under the "dynamic_tanh" sub-scope as "alpha", "gamma", and "beta".
// Gamma and beta are shaped to the last dimension of the input (the feature dimension) and
// broadcast to the full input rank.
//
// For a transformer activation of shape [batch, seq, features], gamma and beta have shape
// [features] and are broadcast across batch and sequence positions.
func DynamicTanh(scope *model.Scope, operand *Node) *DynamicTanhBuilder {
	return &DynamicTanhBuilder{
		scope:   scope,
		operand: operand,
		alpha:   model.GetParamOr(scope, ParamDynamicTanhAlpha, 0.5),
	}
}

// WithAlpha sets the initial value for the learnable alpha parameter and returns the updated builder.
// The default is read from ParamDynamicTanhAlpha (0.5 if not set).
func (dyt *DynamicTanhBuilder) WithAlpha(alpha float64) *DynamicTanhBuilder {
	dyt.alpha = alpha
	return dyt
}

// Done uses the current configuration to apply Dynamic Tanh.
// It returns a node with the same shape as the input operand.
func (dyt *DynamicTanhBuilder) Done() *Node {
	scope := dyt.scope.In("dynamic_tanh")
	x := dyt.operand

	g := x.Graph()
	dtype := x.DType()

	features := x.Shape().Dim(-1)

	alphaVar := scope.VariableWithValue("alpha", shapes.CastAsDType(dyt.alpha, dtype))
	alpha := alphaVar.NodeValue(g)

	paramShape := shapes.Make(dtype, features)
	broadcastShape := x.Shape().Clone()

	for i := range len(broadcastShape.Dimensions) - 1 {
		broadcastShape.Dimensions[i] = 1
	}

	gammaVar := scope.WithInitializer(initializer.One).VariableWithShape("gamma", paramShape).SetTrainable(true)
	gamma := Reshape(gammaVar.NodeValue(g), broadcastShape.Dimensions...)

	betaVar := scope.WithInitializer(initializer.Zero).VariableWithShape("beta", paramShape).SetTrainable(true)
	beta := Reshape(betaVar.NodeValue(g), broadcastShape.Dimensions...)

	return Add(Mul(Tanh(Mul(x, alpha)), gamma), beta)
}
