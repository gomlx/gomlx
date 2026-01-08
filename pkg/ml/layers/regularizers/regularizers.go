// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package regularizers adds tools to facilitate add regularization to the weights learned.
//
// It defines a standard Regularizer interface and several methods that implement it.
//
// Layers like layers.Dense, layers.DenseWithBias and kan.Config will take regularizers as inputs.
package regularizers

import (
	. "github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"slices"
)

const (
	// ParamL2 context hyperparameter defines the amount of L2 regularization of kernels.
	// Each layer may decide independently to implement it or not.
	// layers.Dense, layers.DenseWithBias, layers.FNN, kan.New and layers.Convolution kernels look at this hyperparameter.
	// The value should be a float64.
	// The default is `0.0`.
	ParamL2 = "l2_regularization"

	// ParamL1 context hyperparameter defines the L1 regularizer of kernels.
	// Each layer may decide independently to implement it or not.
	// layers.Dense, layers.DenseWithBias, layers.FNN, kan.New and layers.Convolution kernels look at this hyperparameter.
	// The value should be a float64.
	// The default is `0.0`.
	ParamL1 = "l1_regularization"
)

// Regularizer is a function that will add a regularization term (train.AddLoss) to the loss relative to the given weights (Variables).
//
// Notice it takes variables (and not the nodes) as inputs, as some specialized regularizers may want to update the variables
// post gradient updates to impose constraints -- e.g.: L1 will reduce weights to 0 if they smaller than
// the amount of regularization; a monotonicity regularizer may force weights to be monotonic in some direction.
type Regularizer func(ctx *context.Context, g *Graph, weights ...*context.Variable)

// L2 creates a L2 regularizer (x^2 * amount) with the given static amount.
func L2(amount float64) Regularizer {
	if amount == 0 {
		return nil
	}
	return func(ctx *context.Context, g *Graph, weights ...*context.Variable) {
		if len(weights) == 0 {
			Panicf("no weights given to regularizers.L2")
		}
		var loss *Node
		for _, v := range weights {
			l2 := ReduceAllSum(Square(v.ValueGraph(g)))
			if loss == nil {
				loss = l2
			} else {
				loss = Add(loss, l2)
			}
		}
		loss = MulScalar(loss, amount)
		train.AddLoss(ctx, loss)
	}
}

// L1 creates a L1 regularizer (abs(x) * amount) with the given static amount.
//
// It also adds an update rule that sets values of x to 0 if they are smaller than the amount -- to avoid
// the flipping between positive/negative small values.
func L1(amount float64) Regularizer {
	if amount == 0 {
		return nil
	}
	return func(ctx *context.Context, g *Graph, weights ...*context.Variable) {
		if len(weights) == 0 {
			Panicf("no weights given to regularizers.L1")
		}
		var loss *Node
		for _, v := range weights {
			value := v.ValueGraph(g)
			l1 := ReduceAllSum(Abs(value))
			if loss == nil {
				loss = l1
			} else {
				loss = Add(loss, l1)
			}
		}
		loss = MulScalar(loss, amount)
		train.AddLoss(ctx, loss)

		// Update weights such that if they are smaller than the regularization amount, they are set to 0.
		// Part of this is finding a unique name for all weights, so we don't add updates to the same scope.
		for _, weight := range weights {
			scopedCtx := ctx.InAbsPath(weight.Scope()).Inf("regularizers.L1(%s)", weight.Name())
			train.AddPerStepUpdateGraphFn(scopedCtx, g,
				func(ctx *context.Context, g *Graph) {
					value := weight.ValueGraph(g)
					dtype := value.DType()
					smallWeights := LessThan(Abs(value), Scalar(g, dtype, amount))
					weight.SetValueGraph(Where(smallWeights, ZerosLike(value), value))
				})
		}
	}
}

// Combine the provided regularizers into one -- simply apply all of them.
// If regs is empty, this returns a nil regularizer.
// If regs has only one element, it is returned.
// If any of the regs is nil, it is skipped.
func Combine(regs ...Regularizer) Regularizer {
	// Filter out nil regularizers.
	regs = slices.DeleteFunc(regs, func(r Regularizer) bool { return r == nil })
	if len(regs) == 0 {
		return nil
	}
	if len(regs) == 1 {
		return regs[0]
	}
	return func(ctx *context.Context, g *Graph, weights ...*context.Variable) {
		for _, reg := range regs {
			if reg != nil {
				reg(ctx, g, weights...)
			}
		}
	}
}

// FromContext returns a regularizer from context hyperparameters.
// It may be nil if no regularization is configured.
//
// It looks at ParamL2 and ParamL1 regularizer for now.
func FromContext(ctx *context.Context) Regularizer {
	var regs []Regularizer

	amount := context.GetParamOr(ctx, ParamL2, 0.0)
	if amount > 0 {
		regs = append(regs, L2(amount))
	}
	amount = context.GetParamOr(ctx, ParamL1, 0.0)
	if amount > 0 {
		regs = append(regs, L1(amount))
	}

	if len(regs) == 0 {
		return nil
	}
	if len(regs) == 1 {
		return regs[0]
	}
	return Combine(regs...)
}

// ConstantL1 returns an L1 regularizer applied to the ConsecutiveDifference of the last axis of the weights.
// This has the effect of pushing each value towards its neighbours, that is, a constant function.
//
// This is useful for control points in piecewise-linear, piecewise-constant or b-spline functions, when one wants to make
// points that are not trained much to move towards the mean of its neighbours.
func ConstantL1(amount float64) Regularizer {
	return func(ctx *context.Context, g *Graph, weights ...*context.Variable) {
		var loss *Node
		for _, wVar := range weights {
			w := wVar.ValueGraph(g)
			diff := ConsecutiveDifference(w, -1, false)
			diff = MulScalar(diff, amount)
			l1 := ReduceAllSum(Abs(diff))
			if loss == nil {
				loss = l1
			} else {
				loss = Add(l1, loss)
			}
		}
		train.AddLoss(ctx, loss)
	}
}
