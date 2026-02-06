// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// AxesLayout specifies the ordering of axes in the 4D attention tensors.
// Different layouts use different Einsum equations but produce mathematically equivalent results.
type AxesLayout int

const (
	// LayoutBHSD is the [batch, heads, seq, dim] layout used by PyTorch's F.scaled_dot_product_attention,
	// ONNX, and most inference runtimes.
	LayoutBHSD AxesLayout = iota

	// LayoutBSHD is the [batch, seq, heads, dim] layout used internally by MultiHeadAttention
	// (after Dense projections which produce [batch, seq, heads, dim]).
	LayoutBSHD
)

// String returns the name of the layout.
func (l AxesLayout) String() string {
	switch l {
	case LayoutBHSD:
		return "BHSD"
	case LayoutBSHD:
		return "BSHD"
	default:
		return "unknown"
	}
}

// scoreEquation returns the Einsum equation for computing attention scores (Q @ K^T).
func (l AxesLayout) scoreEquation() string {
	switch l {
	case LayoutBSHD:
		return "bqhd,bkhd->bqhk"
	default: // LayoutBHSD
		return "bhqd,bhkd->bhqk"
	}
}

// outputEquation returns the Einsum equation for computing weighted values (weights @ V).
func (l AxesLayout) outputEquation() string {
	switch l {
	case LayoutBSHD:
		return "bqhk,bkhd->bqhd"
	default: // LayoutBHSD
		return "bhqk,bhkd->bhqd"
	}
}

// SeqAxis returns the axis index for the sequence dimension.
func (l AxesLayout) SeqAxis() int {
	switch l {
	case LayoutBSHD:
		return 1
	default: // LayoutBHSD
		return 2
	}
}

// AttentionCore computes the core scaled dot-product attention operation.
// This is the shared implementation used by MultiHeadAttention and ONNX op converters.
//
// The layout parameter determines the axis ordering of the input tensors:
//   - LayoutBHSD: [batch, heads, seq, dim] (PyTorch/ONNX convention)
//   - LayoutBSHD: [batch, seq, heads, dim] (MHA convention)
//
// The scale is typically 1/sqrt(head_dim).
//
// The additiveMask (if non-nil) is added to scores before softmax. It must match the
// score tensor layout:
//   - LayoutBHSD: broadcastable to [batch, heads, q_seq, kv_seq]
//   - LayoutBSHD: broadcastable to [batch, q_seq, heads, kv_seq]
//
// Returns output and optionally weights in the same layout as inputs.
func AttentionCore(query, key, value *Node, scale float64, additiveMask *Node, returnWeights bool, layout AxesLayout) (*Node, *Node) {
	// Compute attention scores using layout-specific equation
	scores := Einsum(layout.scoreEquation(), query, key)
	scores = MulScalar(scores, scale)

	// Apply additive mask
	if additiveMask != nil {
		scores = Add(scores, additiveMask)
	}

	// Softmax over kv_seq dimension (always the last axis in the score tensor)
	weights := Softmax(scores, -1)

	// Compute output using layout-specific equation
	output := Einsum(layout.outputEquation(), weights, value)

	if returnWeights {
		return output, weights
	}
	return output, nil
}

// BooleanToAdditiveMask converts a boolean attention mask to an additive mask.
// True means attend (adds 0), False means mask out (adds large negative value).
// The dtype parameter specifies the output dtype (should match the scores dtype).
func BooleanToAdditiveMask(booleanMask *Node, dtype dtypes.DType) *Node {
	g := booleanMask.Graph()

	largeNeg := ScalarOne(g, dtype)
	largeNeg = MulScalar(largeNeg, -1e9)
	zero := ScalarZero(g, dtype)

	// Where mask is true, add 0; where false, add large negative
	return Where(booleanMask, zero, largeNeg)
}
