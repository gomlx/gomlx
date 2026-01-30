// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn

import (
	"math"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/graph"
)

// Linear performs a linear transformation: y = x @ weight^T + bias.
//
// weight has shape [out_features, in_features]. bias is optional (nil means no
// bias). This follows the same convention as graph.Linear.
//
// If the backend supports fused Linear (backends.OpTypeLinear), the optimized
// native implementation is used; otherwise the operation is decomposed into
// DotGeneral + Add.
func Linear(x, weight, bias *graph.Node) *graph.Node {
	if x.Graph().Backend().Capabilities().Operations[backends.OpTypeLinear] {
		return graph.Linear(x, weight, bias)
	}

	// Decomposed: contract x's last axis with weight's last axis (second dim
	// of [out, in]), producing x @ weight^T.
	y := graph.Einsum("...i,ji->...j", x, weight)
	if bias != nil {
		y = graph.Add(y, bias)
	}
	return y
}

// LinearActivation performs a linear transformation followed by an activation
// function: activation(x @ weight^T + bias).
//
// weight has shape [out_features, in_features]. bias is optional (nil means no
// bias). activation specifies the activation function to apply.
//
// If the backend supports fused LinearActivation
// (backends.OpTypeLinearActivation), the optimized native implementation is
// used; otherwise the operation is decomposed into Linear + activation.
func LinearActivation(x, weight, bias *graph.Node, activation backends.ActivationType) *graph.Node {
	if x.Graph().Backend().Capabilities().Operations[backends.OpTypeLinearActivation] {
		return graph.LinearActivation(x, weight, bias, activation)
	}

	// Decomposed: linear + activation.
	y := Linear(x, weight, bias)
	return applyActivation(y, activation)
}

// applyActivation applies the given activation function using decomposed graph ops.
func applyActivation(x *graph.Node, activation backends.ActivationType) *graph.Node {
	switch activation {
	case backends.ActivationNone:
		return x
	case backends.ActivationGelu:
		// Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
		cdfApprox := graph.Add(x, graph.MulScalar(graph.PowScalar(x, 3), 0.044715))
		sqrt2ByPi := math.Sqrt(2.0 / math.Pi)
		cdfApprox = graph.Tanh(graph.MulScalar(cdfApprox, sqrt2ByPi))
		cdfApprox = graph.MulScalar(graph.OnePlus(cdfApprox), 0.5)
		return graph.Mul(x, cdfApprox)
	case backends.ActivationRelu:
		zero := graph.ScalarZero(x.Graph(), x.DType())
		return graph.Max(x, zero)
	case backends.ActivationSilu:
		// SiLU = x * sigmoid(x)
		return graph.Mul(x, graph.Sigmoid(x))
	case backends.ActivationTanh:
		return graph.Tanh(x)
	default:
		panic("nn.LinearActivation: unsupported activation type")
	}
}
