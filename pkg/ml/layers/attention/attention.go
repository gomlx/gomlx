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
func scoreEquation(l AxesLayout) string {
	switch l {
	case LayoutBSHD:
		return "bqhd,bkhd->bqhk"
	default: // LayoutBHSD
		return "bhqd,bhkd->bhqk"
	}
}

// outputEquation returns the Einsum equation for computing the attention output (coefficients @ V).
func outputEquation(l AxesLayout) string {
	switch l {
	case LayoutBSHD:
		return "bqhk,bkhd->bqhd"
	default: // LayoutBHSD
		return "bhqk,bhkd->bhqd"
	}
}

// expandKVHeadsForGQA expands key or value tensors by repeating their heads to match the
// query head count for Grouped Query Attention (GQA). If numHeads == numKVHeads, this is a no-op.
//
// For LayoutBHSD: [batch, numKVHeads, seq, dim] → [batch, numHeads, seq, dim]
// For LayoutBSHD: [batch, seq, numKVHeads, dim] → [batch, seq, numHeads, dim]
func expandKVHeadsForGQA(kv *Node, numHeads, numKVHeads int, layout AxesLayout) *Node {
	if numHeads == numKVHeads {
		return kv
	}
	headsAxis := layout.HeadsAxis()
	repeats := numHeads / numKVHeads

	// Insert a new axis after the heads axis for the repeat dimension, broadcast, then merge.
	expanded := InsertAxes(kv, headsAxis+1)
	broadcastDims := expanded.Shape().Clone().Dimensions
	broadcastDims[headsAxis+1] = repeats
	expanded = BroadcastToDims(expanded, broadcastDims...)

	// Merge the KV-heads and repeats axes back into one heads axis.
	finalDims := kv.Shape().Clone().Dimensions
	finalDims[headsAxis] = numHeads
	return Reshape(expanded, finalDims...)
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

		// For GQA: expand K/V heads to match Q head count so that Einsum equations work.
		// The fused path handles GQA natively without expansion, but the decomposed
		// Einsum equations require the heads axis to match between Q and K/V.
		decomposedKey := expandKVHeadsForGQA(key, numHeads, numKVHeads, layout)
		decomposedValue := expandKVHeadsForGQA(value, numHeads, numKVHeads, layout)

		// Decomposed attention.
		scores := Einsum(scoreEquation(layout), query, decomposedKey)
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

		decomposedOutput := Einsum(outputEquation(layout), coefficients, decomposedValue)
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
