// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backends

// ActivationType specifies the activation function for fused operations.
type ActivationType int

const (
	ActivationNone ActivationType = iota
	ActivationGelu
	ActivationRelu
	ActivationSilu
	ActivationTanh
)

// String returns the name of the activation type.
func (a ActivationType) String() string {
	switch a {
	case ActivationNone:
		return "none"
	case ActivationGelu:
		return "gelu"
	case ActivationRelu:
		return "relu"
	case ActivationSilu:
		return "silu"
	case ActivationTanh:
		return "tanh"
	default:
		return "unknown"
	}
}

// FusedOps defines optional fused operations. Backends may implement these for
// better performance; the graph layer falls back to decomposed operations when
// unavailable.
type FusedOps interface {

	// FusedSoftmax computes softmax along the specified axis.
	//
	// Note: unlike the generic softmax in GoMLX's graph package, the fused
	// softmax only accepts one axis. The axis must be non-negative (the caller
	// normalizes negative indices before calling).
	FusedSoftmax(x Value, axis int) (Value, error)

	// FusedGelu computes Gaussian Error Linear Unit activation.
	// If exact is true, the exact GELU (using erf) is computed;
	// otherwise the tanh approximation is used.
	FusedGelu(x Value, exact bool) (Value, error)

	// FusedLayerNorm applies layer normalization over specified axes.
	// gamma and beta can be nil if no learned scale/offset.
	// epsilon: numerical stability constant (typically 1e-5).
	FusedLayerNorm(x Value, axes []int, epsilon float64, gamma, beta Value) (Value, error)

	// FusedDense performs fused matmul + optional bias + optional activation:
	//   y = activation(x @ W + bias)
	// x: [batch..., in_features], weight: [in_features, out_features...],
	// bias: [out_features...] (nil-able).
	// Contracts x's last axis with weight's first axis.
	// activation specifies the activation function to apply after the matmul+bias.
	// Use ActivationNone for no activation.
	FusedDense(x, weight, bias Value, activation ActivationType) (Value, error)
}
