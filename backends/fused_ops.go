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

	// QuantGGML indicates that the weights are stored in native GGML block format
	// (e.g. Q4_0, Q8_0, K-quants). The scales and zero points are embedded in the
	// blocks themselves, so Scale/ZeroPoint/BlockAxis/BlockSize in Quantization are
	// unused; the GGMLType field specifies the concrete block format.
	// Weights must be [N, bytesPerRow] Uint8 with native GGML block layout.
	QuantGGML
)

// String returns the name of the quantization scheme.
func (q QuantizationScheme) String() string {
	switch q {
	case QuantLinear:
		return "Linear"
	case QuantNF4:
		return "NF4"
	case QuantGGML:
		return "GGML"
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

// GGMLQuantType identifies the specific GGML block quantization format.
// Enum values are aligned with go-highway's gguf.QuantType for future integration.
type GGMLQuantType int

const (
	GGMLQ4_0  GGMLQuantType = iota // 18 bytes/block, 32 values
	GGMLQ8_0                       // 34 bytes/block, 32 values
	GGMLIQ4NL                      // 18 bytes/block, 32 values (non-linear lookup)
	GGMLQ2_K                       // 84 bytes/block, 256 values
	GGMLQ3_K                       // 110 bytes/block, 256 values
	GGMLQ4_K                       // 144 bytes/block, 256 values
	GGMLQ5_K                       // 176 bytes/block, 256 values
	GGMLQ6_K                       // 210 bytes/block, 256 values
)

// ValuesPerBlock returns the number of float32 values represented by one block.
func (t GGMLQuantType) ValuesPerBlock() int {
	switch t {
	case GGMLQ4_0, GGMLQ8_0, GGMLIQ4NL:
		return 32
	default: // K-quant types
		return 256
	}
}

// BytesPerBlock returns the byte size of one quantized block.
func (t GGMLQuantType) BytesPerBlock() int {
	switch t {
	case GGMLQ4_0:
		return 18
	case GGMLQ8_0:
		return 34
	case GGMLIQ4NL:
		return 18
	case GGMLQ2_K:
		return 84
	case GGMLQ3_K:
		return 110
	case GGMLQ4_K:
		return 144
	case GGMLQ5_K:
		return 176
	case GGMLQ6_K:
		return 210
	default:
		return 0
	}
}

// String returns the name of the GGML quantization type.
func (t GGMLQuantType) String() string {
	switch t {
	case GGMLQ4_0:
		return "Q4_0"
	case GGMLQ8_0:
		return "Q8_0"
	case GGMLIQ4NL:
		return "IQ4_NL"
	case GGMLQ2_K:
		return "Q2_K"
	case GGMLQ3_K:
		return "Q3_K"
	case GGMLQ4_K:
		return "Q4_K"
	case GGMLQ5_K:
		return "Q5_K"
	case GGMLQ6_K:
		return "Q6_K"
	default:
		return "unknown"
	}
}

// Quantization describes how a value is quantized, and holds the information to dequantize it.
type Quantization struct {
	// Scheme: Linear, NF4, or GGML.
	Scheme QuantizationScheme

	// Scale is the multiplicative factor.
	// Shape: [K, NumBlocks] (block-wise), where K is the input-features
	// (contracting) dimension of the [K, N] weight matrix and
	// NumBlocks = ceil(N / BlockSize).
	// Unused for QuantGGML (scales are embedded in the blocks).
	Scale Value

	// ZeroPoint is the additive offset (only for Linear).
	// If nil, the quantization is assumed symmetric.
	// Unused for QuantGGML and QuantNF4.
	ZeroPoint Value

	// BlockAxis is the dimension of the quantized tensor that is blocked.
	// This is the output-features dimension (axis 1) of a [K, N] weight matrix.
	// Currently only BlockAxis=1 is supported.
	// Unused for QuantGGML.
	BlockAxis int

	// BlockSize is the number of elements in BlockAxis that share one scale.
	// If BlockSize == N, it's effectively per-axis quantization.
	// Unused for QuantGGML.
	BlockSize int

	// GGMLType specifies the concrete GGML block format (Q4_0, Q8_0, etc.).
	// Only used when Scheme == QuantGGML.
	GGMLType GGMLQuantType
}

// ScaledDotProductAttentionConfig holds optional optimization hints for FusedScaledDotProductAttention.
// A nil *ScaledDotProductAttentionConfig means "use defaults" (all optimizations disabled).
type ScaledDotProductAttentionConfig struct {
	// QuantizedMatmuls: if true, the backend may use dynamic per-head symmetric
	// affine quantization (scale-only, no zero point) to convert float32 Q/K/V slices
	// to uint8 for the Q@K^T and attn@V matmul stages. Accumulation is done in int32,
	// then dequantized back to float32. Softmax and masking remain in float32.
	// This matches ONNX DynamicQuantizeLinear semantics and trades some numerical
	// precision for throughput on hardware with fast integer dot-product instructions
	// (e.g. ARM SDOT/UDOT, x86 VNNI). Backends that do not support quantized matmuls
	// ignore this flag and use float arithmetic.
	QuantizedMatmuls bool
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
	//   - options: optional optimization hints (nil uses defaults). See ScaledDotProductAttentionConfig.
	//
	// Output: same shape as query.
	FusedScaledDotProductAttention(
		query, key, value, mask Value,
		numHeads, numKVHeads int,
		axesLayout AxesLayout,
		scale float64,
		causal bool,
		options *ScaledDotProductAttentionConfig) (Value, error)

	// FusedQuantizedGather performs a quantized embedding lookup: it gathers rows from a
	// quantized embedding table and dequantizes only the selected rows on-the-fly.
	// This is the quantized analogue of Gather for embedding lookups, similar to
	// llama.cpp's ggml_get_rows.
	//
	// Inputs:
	//   - table: [vocabSize, bytesPerRow] Uint8 with native GGML block layout.
	//   - indices: integer tensor with last dimension = 1 (same as Gather convention).
	//     For embeddings: [batch, seqLen, 1].
	//   - weightsQuantization: describes how to dequantize the table rows. Must not be nil.
	//     Only QuantGGML scheme is supported.
	//
	// Output: float32 tensor with shape [batch..., K] where K = (bytesPerRow / bytesPerBlock) * valuesPerBlock.
	//   For embeddings with indices [batch, seqLen, 1]: output is [batch, seqLen, K].
	FusedQuantizedGather(table, indices Value,
		weightsQuantization *Quantization) (Value, error)

	// FusedQuantizedDense performs fused dequantization + matmul + optional bias + optional activation.
	//
	// It computes y = activation(x @ dequant(weights, weightsQuantization) + bias), where the
	// dequantization and matmul are fused into a single pass to avoid materializing the
	// full float32 weight matrix.
	//
	// Inputs:
	//   - x: [batch..., K] float32 input activations.
	//   - weights: For Linear/NF4: [K, N] with dtype reflecting storage precision (e.g. Int4, Int8).
	//     For sub-byte types the caller should Bitcast packed uint8 data to the correct dtype
	//     before calling.
	//     For GGML: [N, bytesPerRow] Uint8 with native GGML block layout, where N is the
	//     output-features dimension and bytesPerRow = (K / valuesPerBlock) * bytesPerBlock.
	//   - bias: [N] float32 (nil-able), added after matmul but before activation.
	//   - weightsQuantization: describes how to dequantize the weights tensor. Must not be nil.
	//   - activation: applied after matmul+bias; set to ActivationNone for no activation.
	//
	// Future: inputQuantization, outputQuantization, and biasQuantization parameters may be
	// added to support fully quantized operations where the activations and/or output are
	// also quantized.
	FusedQuantizedDense(x, weights, bias Value,
		weightsQuantization *Quantization,
		activation ActivationType) (Value, error)

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
