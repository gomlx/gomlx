// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn

import (
	"math"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// Dense performs a dense (linear) transformation: y = x @ weight + bias.
//
// weight has shape [in_features, out_features...]. bias is optional (nil means no
// bias). x's last axis contracts with weight's first axis.
//
// If the backend supports fused Dense (backends.OpTypeFusedDense), the optimized
// native implementation is used; otherwise the operation is decomposed into
// Einsum + Add.
func Dense(x, weight, bias *Node) *Node {
	if x.Graph().Backend().Capabilities().Operations[backends.OpTypeFusedDense] {
		return FusedDense(x, weight, bias)
	}

	// Decomposed: contract x's last axis with weight's first axis,
	// producing x @ weight.
	xShape := x.Shape()
	rank := xShape.Rank()
	var y *Node
	if rank <= 2 {
		y = Dot(x, weight)
	} else {
		// Reshape to 2D, Dot, reshape back.
		inFeatures := xShape.Dimensions[rank-1]
		batchSize := xShape.Size() / inFeatures
		x2d := Reshape(x, batchSize, inFeatures)
		y2d := Dot(x2d, weight)
		outDims := make([]int, rank-1+weight.Shape().Rank()-1)
		copy(outDims, xShape.Dimensions[:rank-1])
		copy(outDims[rank-1:], weight.Shape().Dimensions[1:])
		y = Reshape(y2d, outDims...)
	}
	if bias != nil {
		y = Add(y, ExpandLeftToRank(bias, y.Rank()))
	}
	return y
}

// DenseActivation performs a dense transformation followed by an activation
// function: activation(x @ weight + bias).
//
// weight has shape [in_features, out_features...]. bias is optional (nil means no
// bias). activation specifies the activation function to apply.
//
// If the backend supports fused DenseActivation
// (backends.OpTypeFusedDenseActivation), the optimized native implementation is
// used; otherwise the operation is decomposed into Dense + activation.
func DenseActivation(x, weight, bias *Node, activation backends.ActivationType) *Node {
	if x.Graph().Backend().Capabilities().Operations[backends.OpTypeFusedDenseActivation] {
		return FusedDenseActivation(x, weight, bias, activation)
	}

	// Decomposed: dense + activation.
	y := Dense(x, weight, bias)
	return applyActivation(y, activation)
}

// applyActivation applies the given activation function using decomposed graph ops.
func applyActivation(x *Node, activation backends.ActivationType) *Node {
	switch activation {
	case backends.ActivationNone:
		return x
	case backends.ActivationGelu:
		// Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
		cdfApprox := Add(x, MulScalar(PowScalar(x, 3), 0.044715))
		sqrt2ByPi := math.Sqrt(2.0 / math.Pi)
		cdfApprox = Tanh(MulScalar(cdfApprox, sqrt2ByPi))
		cdfApprox = MulScalar(OnePlus(cdfApprox), 0.5)
		return Mul(x, cdfApprox)
	case backends.ActivationRelu:
		zero := ScalarZero(x.Graph(), x.DType())
		return Max(x, zero)
	case backends.ActivationSilu:
		// SiLU = x * sigmoid(x)
		return Mul(x, Sigmoid(x))
	case backends.ActivationTanh:
		return Tanh(x)
	default:
		panic("nn.DenseActivation: unsupported activation type")
	}
}
