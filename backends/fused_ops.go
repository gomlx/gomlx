// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backends

import "github.com/pkg/errors"

// ErrNotImplemented indicates a fused op is not implemented for the given
// configuration (e.g. unsupported dtype or backend). Backends should wrap this
// error so InternalFusedOpCaller can distinguish "not supported" from genuine
// bugs and fall back to the decomposed implementation.
var ErrNotImplemented = errors.New("fused op not implemented")

// AxesLayout specifies the ordering of axes in 4D attention tensors.
type AxesLayout int

const (
	// AxesLayoutBHSD is the [batch, heads, seq, dim] layout used by PyTorch's F.scaled_dot_product_attention,
	// ONNX, and most inference runtimes.
	AxesLayoutBHSD AxesLayout = iota

	// AxesLayoutBSHD is the [batch, seq, heads, dim] layout used internally by MultiHeadAttention
	// (after Dense projections which produce [batch, seq, heads, dim]).
	AxesLayoutBSHD
)

// String returns the name of the layout.
func (l AxesLayout) String() string {
	switch l {
	case AxesLayoutBHSD:
		return "BHSD"
	case AxesLayoutBSHD:
		return "BSHD"
	default:
		return "unknown"
	}
}

// SeqAxis returns the axis index for the sequence dimension.
func (l AxesLayout) SeqAxis() int {
	switch l {
	case AxesLayoutBSHD:
		return 1
	default: // AxesLayoutBHSD
		return 2
	}
}

// HeadsAxis returns the axis index for the heads dimension.
func (l AxesLayout) HeadsAxis() int {
	switch l {
	case AxesLayoutBSHD:
		return 2
	default: // AxesLayoutBHSD
		return 1
	}
}

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

	// FusedScaledDotProductAttention computes multi-head scaled dot-product attention.
	//
	// output = softmax(query @ key^T * scale + mask) @ value, computed per-head with GQA support.
	//
	// Inputs:
	//   - query, key, value: 4D tensors whose axis ordering is determined by axesLayout.
	//     For AxesLayoutBHSD: query [batch, numHeads, seqLen, headDim],
	//                         key/value [batch, numKVHeads, kvLen, headDim].
	//     For AxesLayoutBSHD: query [batch, seqLen, numHeads, headDim],
	//                         key/value [batch, kvLen, numKVHeads, headDim].
	//   - mask: [seqLen, kvLen] (seqLen is the query sequence length): optional (can be nil) mask
	//     that can be either boolean or additive (any dtype other than Bool). See also causal below.
	//     Boolean mask: true = attend, false = ignore.
	//     Float/additive mask: added to scores before softmax.
	//     Must be broadcastable to the score tensor shape.
	//
	// Parameters:
	//   - numHeads: number of query attention heads
	//   - numKVHeads: number of key/value attention heads (for GQA; numHeads must be divisible by numKVHeads)
	//   - axesLayout: determines the axis ordering of query/key/value tensors
	//   - scale: scaling factor applied to query @ key^T (typically 1/sqrt(headDim))
	//   - causal: if true, apply causal (lower-triangular) mask. When both causal and mask are
	//     provided, they are combined: for boolean masks via logical-AND, for additive masks the
	//     causal additive mask is added to the explicit mask.
	//
	// Output: same shape as query.
	FusedScaledDotProductAttention(query, key, value, mask Value, numHeads, numKVHeads int, axesLayout AxesLayout, scale float64, causal bool) (Value, error)

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
