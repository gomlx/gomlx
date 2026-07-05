// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/nn"
	. "github.com/gomlx/gomlx/support/exceptions"
	"k8s.io/klog/v2"
)

// AxesLayout is a type alias for compute.AxesLayout, re-exported for convenience.
type AxesLayout = compute.AxesLayout

const (
	// LayoutBHSD is the [batch, heads, seq, dim] layout.
	LayoutBHSD = compute.AxesLayoutBHSD

	// LayoutBSHD is the [batch, seq, heads, dim] layout.
	LayoutBSHD = compute.AxesLayoutBSHD
)

// Letters used in Einsum equations below:
//
//   - b: batch size
//   - h: number of heads for keys/values, and usually the same for queries (see g below)
//   - g: if using Grouped Query Attention (GQA), there are g*h query heads, while only h heads for keys/values
//   - q: queries sequence length
//   - k: keys/values sequence length
//   - d: features (aka embedding) dimension, contracted during the attention calculation

// scoreEquation returns the Einsum equation for computing attention scores (Q @ K^T).
// When isGQA is true, the query has been reshaped to split its heads axis into (numKVHeads, groupSize),
// so the equation uses a 5D query and treats the KV-heads axis as a batch dimension.
func scoreEquation(l AxesLayout, isGQA bool) string {
	if isGQA {
		switch l {
		case LayoutBSHD:
			// Q:[b,q,h,g,d] x K:[b,k,h,d] -> [b,q,h,g,k]
			return "bqhgd,bkhd->bqhgk"
		default: // LayoutBHSD
			// Q:[b,h,g,q,d] x K:[b,h,k,d] -> [b,h,g,q,k]
			return "bhgqd,bhkd->bhgqk"
		}
	}
	switch l {
	case LayoutBSHD:
		return "bqhd,bkhd->bqhk"
	default: // LayoutBHSD
		return "bhqd,bhkd->bhqk"
	}
}

// outputEquation returns the Einsum equation for computing the attention output (coefficients @ V).
// When isGQA is true, the coefficients have the split heads shape and V retains its original KV-heads shape.
func outputEquation(l AxesLayout, isGQA bool) string {
	if isGQA {
		switch l {
		case LayoutBSHD:
			// C:[b,q,h,g,k] x V:[b,k,h,d] -> [b,q,h,g,d]
			return "bqhgk,bkhd->bqhgd"
		default: // LayoutBHSD
			// C:[b,h,g,q,k] x V:[b,h,k,d] -> [b,h,g,q,d]
			return "bhgqk,bhkd->bhgqd"
		}
	}
	switch l {
	case LayoutBSHD:
		return "bqhk,bkhd->bqhd"
	default: // LayoutBHSD
		return "bhqk,bhkd->bhqd"
	}
}

// reshapeQueryForGQA splits the query's heads axis into (numKVHeads, groupSize) for
// Grouped Query Attention (GQA). This is a pure reshape (no data copy) that lets
// DotGeneral treat numKVHeads as a batch dimension, avoiding the need to
// broadcast-expand K/V tensors.
//
// For LayoutBHSD: [batch, numQueryHeads, seq, dim] → [batch, numKVHeads, groupSize, seq, dim]
// For LayoutBSHD: [batch, seq, numQueryHeads, dim] → [batch, seq, numKVHeads, groupSize, dim]
func reshapeQueryForGQA(query *Node, numQueryHeads, numKVHeads int, layout AxesLayout) *Node {
	groupSize := numQueryHeads / numKVHeads
	headsAxis := layout.HeadsAxis()
	newDims := slices.Clone(query.Shape().Dimensions)
	newDims = slices.Insert(newDims, headsAxis+1, groupSize)
	newDims[headsAxis] = numKVHeads
	return Reshape(query, newDims...)
}

// reshapeMaskForGQA adapts a 4D mask to the 5D score shape used during Grouped Query
// Attention (GQA). If the mask's heads dimension equals numQueryHeads, it is reshaped
// to split heads into (numKVHeads, groupSize). Otherwise an axis of size 1 is inserted
// for broadcasting.
func reshapeMaskForGQA(mask *Node, numQueryHeads, numKVHeads int, layout AxesLayout) *Node {
	headsAxis := layout.HeadsAxis()
	maskHeadDim := mask.Shape().Dimensions[headsAxis]
	if maskHeadDim == numQueryHeads {
		// Per-head mask: split heads axis into (numKVHeads, groupSize).
		groupSize := numQueryHeads / numKVHeads
		newDims := slices.Clone(mask.Shape().Dimensions)
		newDims = slices.Insert(newDims, headsAxis+1, groupSize)
		newDims[headsAxis] = numKVHeads
		return Reshape(mask, newDims...)
	}
	// Head dim is 1 or numKVHeads: insert a dim for groupSize that will broadcast.
	return InsertAxes(mask, headsAxis+1)
}

// mergeOutputGQAHeads merges the split (numKVHeads, groupSize) axes back into a single
// heads axis for the attention output tensor. The input is 5D and the output is 4D.
// This correctly handles d_v != d_q since it operates on the node's own dimensions.
func mergeOutputGQAHeads(node *Node, numQueryHeads int, layout AxesLayout) *Node {
	dims := node.Shape().Dimensions
	headsAxis := layout.HeadsAxis()
	mergedDims := slices.Clone(dims)
	mergedDims = slices.Delete(mergedDims, headsAxis+1, headsAxis+2)
	mergedDims[headsAxis] = numQueryHeads
	return Reshape(node, mergedDims...)
}

// mergeGQACoefficientHeads merges the split (numKVHeads, groupSize) axes back into a
// single heads axis for the attention coefficient tensor. The input is 5D and the output is 4D.
func mergeGQACoefficientHeads(node *Node, numQueryHeads int, layout AxesLayout) *Node {
	dims := node.Shape().Dimensions
	headsAxis := layout.HeadsAxis()
	mergedDims := slices.Clone(dims)
	mergedDims = slices.Delete(mergedDims, headsAxis+1, headsAxis+2)
	mergedDims[headsAxis] = numQueryHeads
	return Reshape(node, mergedDims...)
}

// CoreOptions packs all the optional parameters for Core.
// Default values (zero values) are sensible defaults.
type CoreOptions struct {
	// Scale is the scaling factor for attention scores. Typically 1/sqrt(head_dim).
	// If left as 0, it defaults to 1/sqrt(head_dim), where head_dim is the last dimension of query (head dimension).
	Scale float64

	// AttentionMask controls which positions can be attended to:
	//   - Boolean mask (DType.IsBoolean()): uses MaskedSoftmax — true means attend, false means ignore.
	//     This avoids the -1e9 additive trick which causes gradient and low-precision issues.
	//   - Float mask: added to scores before softmax (additive mask).
	//
	// The attentionMask must be broadcastable to the score tensor layout (using numHeads, not numKVHeads):
	//   - LayoutBHSD: broadcastable to [batch, numHeads, q_seq, kv_seq]
	//   - LayoutBSHD: broadcastable to [batch, q_seq, numHeads, kv_seq]
	//
	// The useCausalMask and mask parameters are mutually exclusive: providing both will panic.
	// If you need both causal masking and an explicit mask, combine them into a single mask
	// before calling Core (e.g. LogicalAnd a lower-triangular boolean mask with your mask).
	AttentionMask *Node

	// UseCausalMask controls whether to apply a causal (lower-triangular) mask.
	// It is mutually exclusive with AttentionMask.
	UseCausalMask bool

	// QuerySeqLen, KVSeqLen are an optional per-batch actual sequence length (int32 [B] node) for query
	// and Key/Value padding masking.
	// On the fused path, it is forwarded via the seqlen config.
	// On the decomposed path, it is materialized into a boolean padding mask.
	// Mutually exclusive with a non-nil AttentionMask.
	// If one is set the other must also be set.
	QuerySeqLen, KVSeqLen *Node

	// AttentionBias is an optional additive attention-score bias [batch, numHeads, q_seq, kv_seq]
	// (ALiBi / relative-position). It is DISTINCT from the Q/K/V projection bias (UseProjectionBias
	// on the builder) and from AttentionMask. On the fused path it selects the cuDNN fmhaScaleBias
	// variant; on the decomposed path it is added to the scores before softmax. It may combine with
	// UseCausalMask; a bias+seqlens combination the fused kernel can't take falls back to decomposed.
	//
	// For the fused path the bias dtype must match query/key/value (the backend does no automatic
	// conversion): a mismatched dtype falls back to the decomposed path.
	AttentionBias *Node

	// WantCoefficients, when true, forces the decomposed path to be used for the entire computation
	// (no fused op) and returns coefficients. When false, the fused op is attempted for
	// the output and coefficients is nil.
	WantCoefficients bool

	// ScoreSoftCap controls whether to cap the scores (of the attention softmax) using the [nn.SoftCap].
	// A value > 0 enables this feature.
	ScoreSoftCap float64

	// Scope provides the training/inference scope of the attention operation.
	// Currently, this is only needed for RNG state for dropout.
	// If not using dropout, just leave it at nil.
	Scope *model.Scope

	// DropoutRate (if not nil) applies dropout to the attention coefficients during training.
	// When dropout is active, the fused path is skipped (fused ops don't support dropout).
	// Notice: Dropout is rarely used in modern transformer models, and it prevents fusion (flash attention).
	// See discussion:
	// https://www.reddit.com/r/LocalLLaMA/comments/1pntkme/day_8_21_days_of_building_a_small_language_model/
	//
	// If using DropoutRate set also the Scope field.
	DropoutRate *Node

	// DisableFusion disables attempting to use the fused attention op (like flash-attention) when available.
	// This is mostly used for testing -- most of the times one wants fusion if available.
	DisableFusion bool
}

// String returns a representation of non-zero fields of CoreOptions.
func (o CoreOptions) String() string {
	var parts []string
	if o.Scale != 0 {
		parts = append(parts, fmt.Sprintf("Scale: %g", o.Scale))
	}
	if o.AttentionMask != nil {
		parts = append(parts, fmt.Sprintf("AttentionMask: %s", o.AttentionMask.Shape()))
	}
	if o.UseCausalMask {
		parts = append(parts, "UseCausalMask: true")
	}
	if o.QuerySeqLen != nil {
		parts = append(parts, fmt.Sprintf("QuerySeqLen: %s", o.QuerySeqLen.Shape()))
	}
	if o.KVSeqLen != nil {
		parts = append(parts, fmt.Sprintf("KVSeqLen: %s", o.KVSeqLen.Shape()))
	}
	if o.AttentionBias != nil {
		parts = append(parts, fmt.Sprintf("AttentionBias: %s", o.AttentionBias.Shape()))
	}
	if o.WantCoefficients {
		parts = append(parts, "WantCoefficients: true")
	}
	if o.ScoreSoftCap != 0 {
		parts = append(parts, fmt.Sprintf("ScoreSoftCap: %g", o.ScoreSoftCap))
	}
	if o.Scope != nil {
		parts = append(parts, fmt.Sprintf("Scope: %s", o.Scope.Scope()))
	}
	if o.DropoutRate != nil {
		parts = append(parts, fmt.Sprintf("DropoutRate: %s", o.DropoutRate.Shape()))
	}
	if o.DisableFusion {
		parts = append(parts, "DisableFusion: true")
	}
	return "{" + strings.Join(parts, ", ") + "}"
}

// Core computes the core scaled dot-product attention operation.
// This is the shared implementation used by MultiHeadAttention and ONNX op converters.
//
// The layout parameter determines the axis ordering of the input tensors:
//   - LayoutBHSD: [batch, heads, seq, dim] (PyTorch/ONNX convention)
//   - LayoutBSHD: [batch, seq, heads, dim] (MHA convention)
//
// Grouped Query Attention (GQA) is supported: key and value may have fewer heads than query.
// The number of query heads must be divisible by the number of key/value heads. Each group
// of query heads shares the same key/value head. This also covers Multi-Query Attention (MQA)
// where numKVHeads=1.
//
// Returns:
//   - output: same shape as query.
//   - coefficients: attention coefficients (nil when WantCoefficients is false) shaped
//     [batch, heads, q_seq, kv_seq] for LayoutBHSD or
//     [batch, q_seq, heads, kv_seq] for LayoutBSHD.
func Core(query, key, value *Node, layout AxesLayout, options CoreOptions) (output, coefficients *Node) {
	g := query.Graph()
	numQueryHeads := query.Shape().Dimensions[layout.HeadsAxis()]
	numKVHeads := key.Shape().Dimensions[layout.HeadsAxis()]

	if options.UseCausalMask && options.AttentionMask != nil {
		Panicf("attention.Core: useCausalMask and mask are mutually exclusive; combine them into a single mask before calling Core")
	}
	hasSeqLens := options.QuerySeqLen != nil || options.KVSeqLen != nil
	if hasSeqLens && options.AttentionMask != nil {
		Panicf("attention.Core: querySeqLen/keyValueSeqLen are mutually exclusive with a non-nil attentionMask; combine padding into a single mask upstream")
	}
	if hasSeqLens && layout != LayoutBSHD {
		Panicf("attention.Core: querySeqLen/keyValueSeqLen require LayoutBSHD (padding mask is BSHD-only), got %s", layout)
	}
	if numQueryHeads <= 0 || numKVHeads <= 0 || numQueryHeads%numKVHeads != 0 {
		Panicf("attention.Core: numQueryHeads (%d) must be positive and divisible by numKVHeads (%d)", numQueryHeads, numKVHeads)
	}

	dropoutActive := layers.IsDropoutActive(options.Scope, g) && options.DropoutRate != nil

	scale := options.Scale
	if scale == 0 {
		scale = 1.0 / math.Sqrt(float64(query.Shape().Dimensions[len(query.Shape().Dimensions)-1]))
	}

	if klog.V(2).Enabled() {
		scopePath := "<nil>"
		if options.Scope != nil {
			scopePath = options.Scope.Scope()
		}
		klog.Infof("attention.Core(scope=%s, query=%s, key=%s, value=%s, layout=%s, options=%s)",
			scopePath, query.Shape(), key.Shape(), value.Shape(), layout, options)
	}

	// Function to compute the attention "decomposed" (as in not-fused).
	// We use this as a closure (as opposed to calculating it directly), because it's required
	// by InternalFusedOpCaller() below.
	decomposedFn := func() (output *Node, coefficients *Node) {
		// Build causal mask for the decomposed path.
		decomposedMask := options.AttentionMask
		if options.UseCausalMask {
			seqLen := query.Shape().Dimensions[layout.SeqAxis()]
			causalBool := LowerTriangular(g, seqLen)
			// Reshape for correct broadcasting with score layout.
			switch layout {
			case LayoutBHSD:
				causalBool = Reshape(causalBool, 1, 1, seqLen, seqLen)
			default: // LayoutBSHD
				causalBool = Reshape(causalBool, 1, seqLen, 1, seqLen)
			}
			decomposedMask = causalBool
		}

		// Build padding mask from seqlen nodes and combine with existing decomposed mask.
		if options.QuerySeqLen != nil || options.KVSeqLen != nil {
			padMask := buildSeqLenPaddingMask(query, key, options.QuerySeqLen, options.KVSeqLen)
			if decomposedMask != nil {
				decomposedMask = LogicalAnd(decomposedMask, padMask)
			} else {
				decomposedMask = padMask
			}
		}

		// For Grouped Query Attention (GQA): reshape Q to split heads into
		// (numKVHeads, groupSize) so that DotGeneral treats numKVHeads as a batch
		// dimension. This avoids broadcasting K/V to match the query head count —
		// only a free reshape of Q is needed.
		isGQA := numQueryHeads != numKVHeads
		decomposedQuery := query
		// Bias convention is [B,H,Sq,Skv] (heads-major). BSHD scores are [B,Sq,H,Skv], so permute
		// [0,2,1,3] before Add. BHSD scores are already [B,H,Sq,Skv]: no permute.
		decomposedBias := options.AttentionBias
		if decomposedBias != nil && layout == LayoutBSHD {
			decomposedBias = TransposeAllAxes(decomposedBias, 0, 2, 1, 3)
		}
		if isGQA {
			decomposedQuery = reshapeQueryForGQA(query, numQueryHeads, numKVHeads, layout)
			if decomposedMask != nil {
				decomposedMask = reshapeMaskForGQA(decomposedMask, numQueryHeads, numKVHeads, layout)
			}
			if decomposedBias != nil {
				decomposedBias = reshapeMaskForGQA(decomposedBias, numQueryHeads, numKVHeads, layout)
			}
		}

		// Decomposed attention.
		scores := Einsum(scoreEquation(layout, isGQA), decomposedQuery, key)
		scores = MulScalar(scores, scale)

		if options.ScoreSoftCap > 0 {
			scores = nn.SoftCap(scores, options.ScoreSoftCap)
		}

		// Additive attention bias (ALiBi / relative-position) before softmax; reshaped above to the
		// score layout (and to 5D when GQA).
		if decomposedBias != nil {
			scores = Add(scores, decomposedBias)
		}

		if decomposedMask != nil && decomposedMask.DType() != dtypes.Bool {
			// Additive float mask.
			scores = Add(scores, decomposedMask)
			coefficients = Softmax(scores, -1)
		} else {
			// Boolean mask (or nil): MaskedSoftmax handles both.
			if decomposedMask != nil {
				decomposedMask = BroadcastToShape(decomposedMask, scores.Shape())
			}
			coefficients = MaskedSoftmax(scores, decomposedMask, -1)
		}
		if dropoutActive {
			coefficients = layers.Dropout(options.Scope, coefficients, options.DropoutRate)
		}

		decomposedOutput := Einsum(outputEquation(layout, isGQA), coefficients, value)

		if isGQA {
			// Merge (numKVHeads, groupSize) back into numQueryHeads for output and coefficients.
			decomposedOutput = mergeOutputGQAHeads(decomposedOutput, numQueryHeads, layout)
			coefficients = mergeGQACoefficientHeads(coefficients, numQueryHeads, layout)
		}

		return decomposedOutput, coefficients
	}

	// When coefficients are requested, use the decomposed path for everything
	// to avoid computing both paths (fused output + decomposed scores).
	capabilities := g.Backend().Capabilities()
	hasFused := capabilities.Operations[compute.OpTypeFusedScaledDotProductAttention]
	hasFusedVJP := capabilities.Operations[compute.OpTypeFusedScaledDotProductAttentionVJP]
	if options.WantCoefficients || dropoutActive || options.ScoreSoftCap > 0 || options.DisableFusion || !hasFused {
		klog.V(2).Info("attention.Core(): forced decomposed version.")
		output, coefficients = decomposedFn()
	} else {
		// Attempt fused version:
		var isFused bool
		useDecomposedVJP := !hasFusedVJP // If FusedVJP is not registered for backprop, fall back to use the decomposed.
		output, isFused = InternalFusedOpCaller(
			func() *Node {
				var fusedConfig *compute.ScaledDotProductAttentionConfig
				if options.QuerySeqLen != nil || options.KVSeqLen != nil || options.AttentionBias != nil {
					fusedConfig = &compute.ScaledDotProductAttentionConfig{}
					if options.QuerySeqLen != nil {
						fusedConfig.QuerySeqLen = InternalBackendOutputs(options.QuerySeqLen)[0]
					}
					if options.KVSeqLen != nil {
						fusedConfig.KeyValueSeqLen = InternalBackendOutputs(options.KVSeqLen)[0]
					}
					if options.AttentionBias != nil {
						fusedConfig.Bias = InternalBackendOutputs(options.AttentionBias)[0]
					}
				}
				klog.V(2).Info("attention.Core(): attempting to use FusedScaledDotProductAttention op.")
				out, _ := BackendFusedScaledDotProductAttention(
					query, key, value, options.AttentionMask,
					numQueryHeads, numKVHeads, layout, scale, options.UseCausalMask, fusedConfig)
				return out
			},
			func() *Node {
				output, _ := decomposedFn()
				return output
			},
			useDecomposedVJP,
		)
		if isFused {
			klog.V(2).Info("attention.Core(): using FusedScaledDotProductAttention op.")
		} else {
			klog.V(2).Info("attention.Core(): using decomposed op.")
		}
		coefficients = nil
	}
	return output, coefficients
}
