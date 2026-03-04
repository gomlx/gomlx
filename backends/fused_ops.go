// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backends

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

// QuantizationScheme specifies how quantized integer values map to floating-point values.
type QuantizationScheme int

const (
	// QuantLinear is standard affine quantization: float_value = int_value * scale + zeroPoint.
	// Used with Int4 weights (symmetric, zeroPoint=nil) or Int8 weights.
	QuantLinear QuantizationScheme = iota

	// QuantNF4 is 4-bit NormalFloat from QLoRA: nibble indices are looked up in a fixed
	// 16-entry table, then multiplied by scale.
	QuantNF4
)

// String returns the name of the quantization scheme.
func (q QuantizationScheme) String() string {
	switch q {
	case QuantLinear:
		return "Linear"
	case QuantNF4:
		return "NF4"
	default:
		return "unknown"
	}
}

// NF4LookupTable contains the 16 fixed QLoRA NormalFloat4 dequantization values.
// Used by both the fused executor and the decomposed graph-level fallback.
var NF4LookupTable = [16]float32{
	-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
	-0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0,
	0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
	0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
}

// ActivationType specifies the activation function for fused operations.
type ActivationType int

const (
	ActivationNone ActivationType = iota
	ActivationGelu
	ActivationRelu
	ActivationSilu
	ActivationHardSwish
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
	case ActivationHardSwish:
		return "hard_swish"
	case ActivationTanh:
		return "tanh"
	default:
		return "unknown"
	}
}

// FusedOps defines optional fused operations. Backends may implement these for
// better performance; the graph layer falls back to decomposed operations when
// unavailable.
//
// Like with standard ops, if the backend doesn't implement the fused op, return
// ErrNotImplemented (wrapped with a stack).
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
	//   - causal: if true, apply causal (lower-triangular) mask. Callers (e.g. attention.Core)
	//     treat causal and mask as mutually exclusive, folding causal into the mask before calling
	//     this method when both are needed. Backends may assume they won't both be set.
	//
	// Output: same shape as query.
	FusedScaledDotProductAttention(
		query, key, value, mask Value,
		numHeads, numKVHeads int,
		axesLayout AxesLayout,
		scale float64,
		causal bool) (Value, error)

	// FusedQuantizedDense performs fused dequantization + matmul + optional bias + optional activation.
	//
	// It computes y = activation(x @ dequant(weights, scales, zeroPoints) + bias), where the
	// dequantization and matmul are fused into a single pass to avoid materializing the
	// full float32 weight matrix.
	//
	// Inputs:
	//   - x: [batch..., K] float32 input activations.
	//   - weights: [K, N] with dtype reflecting storage precision (e.g. Int4, Int8).
	//     For sub-byte types the caller should Bitcast packed uint8 data to the correct dtype
	//     before calling.
	//   - scales: [K, numBlocks] float32, where numBlocks = ceil(N / blockSize).
	//     Each scale covers a contiguous block of blockSize output columns for one row.
	//   - zeroPoints: [K, numBlocks] float32 (nil-able). Additive offset applied after
	//     scaling: float_value = int_value * scale + zeroPoint. Nil for symmetric / NF4.
	//   - bias: [N] float32 (nil-able), added after matmul but before activation.
	//
	// Parameters:
	//   - scheme: QuantLinear or QuantNF4.
	//   - blockAxis: the dimension of the weights tensor along which blocking is applied
	//     (typically 1, the output-features axis).
	//   - blockSize: number of output columns sharing a single scale factor.
	//   - activation: applied after matmul+bias; set to ActivationNone for no activation.
	FusedQuantizedDense(x, weights, scales, zeroPoints, bias Value,
		scheme QuantizationScheme, blockAxis int, blockSize int,
		activation ActivationType) (Value, error)

	// FusedQuantizedScaledDotProductAttention computes multi-head SDPA using int8×int8
	// matmuls for Q@K^T and attn@V. Inputs are float32; quantization is internal.
	// Same interface as FusedScaledDotProductAttention.
	FusedQuantizedScaledDotProductAttention(
		query, key, value, mask Value,
		numHeads, numKVHeads int,
		axesLayout AxesLayout,
		scale float64,
		causal bool) (Value, error)

	// FusedAttentionQKVProjection performs fused Query-Key-Value projection: a single large matmul
	// merged with a scatter into separate query (Q), key (K), value (V) outputs with optional
	// per-projection bias.
	//
	// The caller is expected to flatten any leading dimensions (e.g. batch and sequence) into a
	// single "batch" axis before calling, and reshape the outputs afterwards. For example, with
	// BSHD layout the caller reshapes [batch, seqLen, inFeatures] → [batch*seqLen, inFeatures],
	// calls this method, then reshapes each output back to [batch, seqLen, ...].
	//
	// Inputs:
	//   - x: [batch, inFeatures] (batch may include a merged sequence dimension)
	//   - wQKV: [inFeatures, queryDim+2*keyValueDim] (Q/K/V weights concatenated along last axis)
	//   - biasQ: [queryDim] (optional, nil for no bias)
	//   - biasK: [keyValueDim] (optional, nil for no bias)
	//   - biasV: [keyValueDim] (optional, nil for no bias)
	//
	// Parameters:
	//   - queryDim: output dimension for query projection
	//   - keyValueDim: output dimension for key and value projections
	//
	// Outputs: query [batch, queryDim], key [batch, keyValueDim], value [batch, keyValueDim]
	FusedAttentionQKVProjection(
		x, wQKV, biasQ, biasK, biasV Value,
		queryDim, keyValueDim int) (
		query, key, value Value, err error)
}
