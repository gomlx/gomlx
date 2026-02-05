// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backends

import "github.com/pkg/errors"

// ErrNotImplemented indicates a fused op is not implemented for the given
// configuration (e.g. unsupported dtype or backend). Backends should wrap this
// error so InternalFusedOpCaller can distinguish "not supported" from genuine
// bugs and fall back to the decomposed implementation.
var ErrNotImplemented = errors.New("fused op not implemented")

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

	// FusedDense performs fused matmul + optional bias + optional activation.
	//
	// It does y = activation(x @ W + bias). Where @ is a standard matmul,
	// it contracts x's last axis with weight's first axis.
	//
	// - x: [batch..., in_features], weight: [in_features, out_features...],
	// - bias: [out_features...] (nil-able).
	// - activation: applied after the matmul+bias; set to ActivationNone for no activation.
	FusedDense(x, weight, bias Value, activation ActivationType) (Value, error)

	// FusedMultiHeadSDPA computes multi-head scaled dot-product attention.
	//
	// output = softmax(Q @ K^T * scale + mask) @ V, computed per-head with GQA support.
	//
	// Inputs:
	//   - q: [batch, numHeads, seqLen, headDim]
	//   - k: [batch, numKVHeads, kvLen, headDim]
	//   - v: [batch, numKVHeads, kvLen, headDim]
	//   - mask: [seqLen, kvLen] (optional, additive mask; nil for no mask)
	//
	// Parameters:
	//   - numHeads: number of query attention heads
	//   - numKVHeads: number of key/value attention heads (for GQA; numHeads must be divisible by numKVHeads)
	//   - scale: scaling factor applied to Q @ K^T (typically 1/sqrt(headDim))
	//   - causal: if true, apply causal (lower-triangular) mask
	//
	// Output: [batch, numHeads, seqLen, headDim]
	FusedMultiHeadSDPA(q, k, v, mask Value, numHeads, numKVHeads int, scale float64, causal bool) (Value, error)

	// FusedQKVDense performs fused QKV projection: a single large matmul followed by
	// scatter into separate Q, K, V outputs with optional per-projection bias.
	//
	// Inputs:
	//   - x: [batch, inFeatures]
	//   - wQKV: [inFeatures, qDim+2*kvDim] (Q/K/V weights concatenated along last axis)
	//   - biasQ: [qDim] (optional, nil for no bias)
	//   - biasK: [kvDim] (optional, nil for no bias)
	//   - biasV: [kvDim] (optional, nil for no bias)
	//
	// Parameters:
	//   - qDim: output dimension for Q projection
	//   - kvDim: output dimension for K and V projections
	//
	// Outputs: q [batch, qDim], k [batch, kvDim], v [batch, kvDim]
	FusedQKVDense(x, wQKV, biasQ, biasK, biasV Value, qDim, kvDim int) (q, k, v Value, err error)
}
