// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	. "github.com/gomlx/gomlx/pkg/support/exceptions"
)

// AxesLayout is a type alias for backends.AxesLayout, re-exported for convenience.
type AxesLayout = backends.AxesLayout

const (
	// LayoutBHSD is the [batch, heads, seq, dim] layout.
	LayoutBHSD = backends.AxesLayoutBHSD

	// LayoutBSHD is the [batch, seq, heads, dim] layout.
	LayoutBSHD = backends.AxesLayoutBSHD
)

// scoreEquation returns the Einsum equation for computing attention scores (Q @ K^T).
// When gqa is true, the query has been reshaped to split its heads axis into (numKVHeads, groupSize),
// so the equation uses a 5D query and treats the KV-heads axis as a batch dimension.
func scoreEquation(l AxesLayout, gqa bool) string {
	if gqa {
		switch l {
		case LayoutBSHD:
			// Q:[B,Sq,Hk,g,D] x K:[B,Sk,Hk,D] -> [B,Sq,Hk,g,Sk]
			return "bqHgd,bkHd->bqHgk"
		default: // LayoutBHSD
			// Q:[B,Hk,g,Sq,D] x K:[B,Hk,Sk,D] -> [B,Hk,g,Sq,Sk]
			return "bHgqd,bHkd->bHgqk"
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
// When gqa is true, the coefficients have the split heads shape and V retains its original KV-heads shape.
func outputEquation(l AxesLayout, gqa bool) string {
	if gqa {
		switch l {
		case LayoutBSHD:
			// C:[B,Sq,Hk,g,Sk] x V:[B,Sk,Hk,D] -> [B,Sq,Hk,g,D]
			return "bqHgk,bkHd->bqHgd"
		default: // LayoutBHSD
			// C:[B,Hk,g,Sq,Sk] x V:[B,Hk,Sk,D] -> [B,Hk,g,Sq,D]
			return "bHgqk,bHkd->bHgqd"
		}
	}
	switch l {
	case LayoutBSHD:
		return "bqhk,bkhd->bqhd"
	default: // LayoutBHSD
		return "bhqk,bhkd->bhqd"
	}
}

// reshapeQueryForGQA splits the query's heads axis into (numKVHeads, groupSize) for GQA.
// This is a pure reshape (no data copy) that lets DotGeneral treat numKVHeads as a batch
// dimension, avoiding the need to broadcast-expand K/V tensors.
//
// For LayoutBHSD: [batch, numHeads, seq, dim] → [batch, numKVHeads, groupSize, seq, dim]
// For LayoutBSHD: [batch, seq, numHeads, dim] → [batch, seq, numKVHeads, groupSize, dim]
func reshapeQueryForGQA(query *Node, numHeads, numKVHeads int, layout AxesLayout) *Node {
	dims := query.Shape().Dimensions
	groupSize := numHeads / numKVHeads
	headsAxis := layout.HeadsAxis()
	newDims := make([]int, 5)
	copy(newDims, dims[:headsAxis])
	newDims[headsAxis] = numKVHeads
	newDims[headsAxis+1] = groupSize
	copy(newDims[headsAxis+2:], dims[headsAxis+1:])
	return Reshape(query, newDims...)
}

// reshapeMaskForGQA adapts a 4D mask to the 5D score shape used during GQA.
// If the mask's heads dimension equals numHeads, it is reshaped to split heads into
// (numKVHeads, groupSize). Otherwise an axis of size 1 is inserted for broadcasting.
func reshapeMaskForGQA(mask *Node, numHeads, numKVHeads int, layout AxesLayout) *Node {
	headsAxis := layout.HeadsAxis()
	maskHeadDim := mask.Shape().Dimensions[headsAxis]
	if maskHeadDim == numHeads {
		// Per-head mask: split heads axis into (numKVHeads, groupSize).
		dims := mask.Shape().Dimensions
		groupSize := numHeads / numKVHeads
		newDims := make([]int, 5)
		copy(newDims, dims[:headsAxis])
		newDims[headsAxis] = numKVHeads
		newDims[headsAxis+1] = groupSize
		copy(newDims[headsAxis+2:], dims[headsAxis+1:])
		return Reshape(mask, newDims...)
	}
	// Head dim is 1 or numKVHeads: insert a dim for groupSize that will broadcast.
	return InsertAxes(mask, headsAxis+1)
}

// mergeGQAHeads merges the split (numKVHeads, groupSize) axes back into a single heads axis.
// The input is 5D and the output is the original 4D shape.
func mergeGQAHeads(node *Node, targetDims []int) *Node {
	return Reshape(node, targetDims...)
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
// The scale is typically 1/sqrt(head_dim).
//
// The mask parameter (if non-nil) controls which positions can be attended to:
//   - Boolean mask (DType.IsBoolean()): uses MaskedSoftmax — true means attend, false means ignore.
//     This avoids the -1e9 additive trick which causes gradient and low-precision issues.
//   - Float mask: added to scores before softmax (additive mask).
//
// The mask must be broadcastable to the score tensor layout (using numHeads, not numKVHeads):
//   - LayoutBHSD: broadcastable to [batch, numHeads, q_seq, kv_seq]
//   - LayoutBSHD: broadcastable to [batch, q_seq, numHeads, kv_seq]
//
// The causal and mask parameters are mutually exclusive: providing both will panic.
// If you need both causal masking and an explicit mask, combine them into a single mask
// before calling Core (e.g. LogicalAnd a lower-triangular boolean mask with your mask).
//
// The dropoutRate (if > 0) applies dropout to the attention coefficients during training.
// When dropout is active, the fused path is skipped (fused ops don't support dropout).
// The ctx parameter provides the training/inference context for dropout; it may be nil
// when dropoutRate is 0.
//
// When wantCoefficients is true, the decomposed path is used for the entire computation
// (no fused op) and coefficients are returned. When false, the fused op is attempted for
// the output and coefficients is nil.
//
// Returns:
//   - output: same shape as query.
//   - coefficients: attention coefficients (nil when wantCoefficients is false) shaped
//     [batch, heads, q_seq, kv_seq] for LayoutBHSD or
//     [batch, q_seq, heads, kv_seq] for LayoutBSHD.
func Core(ctx *context.Context, query, key, value *Node, scale float64, mask *Node, dropoutRate float64,
	layout AxesLayout, causal, wantCoefficients bool) (output, coefficients *Node) {
	g := query.Graph()
	numHeads := query.Shape().Dimensions[layout.HeadsAxis()]
	numKVHeads := key.Shape().Dimensions[layout.HeadsAxis()]

	if causal && mask != nil {
		Panicf("attention.Core: causal and mask are mutually exclusive; combine them into a single mask before calling Core")
	}
	if numHeads <= 0 || numKVHeads <= 0 || numHeads%numKVHeads != 0 {
		Panicf("attention.Core: numHeads (%d) must be positive and divisible by numKVHeads (%d)", numHeads, numKVHeads)
	}

	dropoutActive := layers.IsDropoutActive(ctx, g) && dropoutRate > 0

	// Function to compute the attention "decomposed" (as in not-fused).
	// We use this as a closure (as opposed to calculating it directly), because it's required
	// by InternalFusedOpCaller() below.
	decomposedFn := func() (output *Node, coefficients *Node) {
		// Build causal mask for the decomposed path.
		decomposedMask := mask
		if causal {
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

		// For GQA: reshape Q to split heads into (numKVHeads, groupSize) so that
		// DotGeneral treats numKVHeads as a batch dimension. This avoids broadcasting
		// K/V to match the query head count — only a free reshape of Q is needed.
		gqa := numHeads != numKVHeads
		decomposedQuery := query
		if gqa {
			decomposedQuery = reshapeQueryForGQA(query, numHeads, numKVHeads, layout)
			if decomposedMask != nil {
				decomposedMask = reshapeMaskForGQA(decomposedMask, numHeads, numKVHeads, layout)
			}
		}

		// Decomposed attention.
		scores := Einsum(scoreEquation(layout, gqa), decomposedQuery, key)
		scores = MulScalar(scores, scale)

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
			coefficients = layers.DropoutStatic(ctx, coefficients, dropoutRate)
		}

		decomposedOutput := Einsum(outputEquation(layout, gqa), coefficients, value)

		if gqa {
			// Merge (numKVHeads, groupSize) back into numHeads for output and coefficients.
			decomposedOutput = mergeGQAHeads(decomposedOutput, query.Shape().Dimensions)
			coeffDims := scores.Shape().Dimensions
			headsAxis := layout.HeadsAxis()
			mergedCoeffDims := make([]int, 4)
			copy(mergedCoeffDims, coeffDims[:headsAxis])
			mergedCoeffDims[headsAxis] = numHeads
			copy(mergedCoeffDims[headsAxis+1:], coeffDims[headsAxis+2:])
			coefficients = mergeGQAHeads(coefficients, mergedCoeffDims)
		}

		return decomposedOutput, coefficients
	}

	// When coefficients are requested, use the decomposed path for everything
	// to avoid computing both paths (fused output + decomposed scores).
	if wantCoefficients || dropoutActive {
		output, coefficients = decomposedFn()
	} else {
		output = InternalFusedOpCaller(
			func() *Node {
				return BackendFusedScaledDotProductAttention(
					query, key, value, mask,
					numHeads, numKVHeads, layout, scale, causal)
			},
			func() *Node {
				output, _ := decomposedFn()
				return output
			},
		)
		coefficients = nil
	}
	return output, coefficients
}
