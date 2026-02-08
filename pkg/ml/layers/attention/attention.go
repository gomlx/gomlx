// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
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

// outputEquation returns the Einsum equation for computing the attention output (coefficients @ V).
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

// BooleanToAdditiveMask converts a boolean mask to an additive float mask
// in the given dtype. True values become 0.0 (attend), false values become -1e9 (mask out).
// This is useful when interfacing with APIs that expect additive masks (like fused SDPA).
func BooleanToAdditiveMask(mask *Node, dtype dtypes.DType) *Node {
	g := mask.Graph()
	zero := ScalarZero(g, dtype)
	negInf := ConstAs(zero, float32(-1e9))
	return Where(mask, zero, negInf)
}

// Core computes the core scaled dot-product attention operation.
// This is the shared implementation used by MultiHeadAttention and ONNX op converters.
//
// The layout parameter determines the axis ordering of the input tensors:
//   - LayoutBHSD: [batch, heads, seq, dim] (PyTorch/ONNX convention)
//   - LayoutBSHD: [batch, seq, heads, dim] (MHA convention)
//
// The scale is typically 1/sqrt(head_dim).
//
// The mask parameter (if non-nil) controls which positions can be attended to:
//   - Boolean mask (DType.IsBoolean()): uses MaskedSoftmax — true means attend, false means ignore.
//     This avoids the -1e9 additive trick which causes gradient and low-precision issues.
//   - Float mask: added to scores before softmax (additive mask).
//
// The mask must be broadcastable to the score tensor layout:
//   - LayoutBHSD: broadcastable to [batch, heads, q_seq, kv_seq]
//   - LayoutBSHD: broadcastable to [batch, q_seq, heads, kv_seq]
//
// The dropoutRate (if > 0) applies dropout to the attention coefficients during training.
// The ctx parameter provides the training/inference context for dropout; it may be nil
// when dropoutRate is 0.
//
// Returns:
//   - output: same shape as value.
//   - coefficients: attention coefficients (from 0 to 1) shaped
//     [batch, heads, q_seq, kv_seq] for LayoutBHSD or
//     [batch, q_seq, heads, kv_seq] for LayoutBSHD.
func Core(ctx *context.Context, query, key, value *Node, scale float64, mask *Node, dropoutRate float64, layout AxesLayout) (output, coefficients *Node) {
	scores := Einsum(layout.scoreEquation(), query, key)
	scores = MulScalar(scores, scale)

	if mask != nil && mask.DType() != dtypes.Bool {
		// Additive float mask.
		scores = Add(scores, mask)
		coefficients = Softmax(scores, -1)
	} else {
		// Boolean mask (or nil): MaskedSoftmax handles both.
		if mask != nil {
			mask = BroadcastToShape(mask, scores.Shape())
		}
		coefficients = MaskedSoftmax(scores, mask, -1)
	}

	if dropoutRate > 0 && ctx != nil {
		coefficients = layers.Dropout(ctx, coefficients, ConstAs(coefficients, dropoutRate))
	}

	output = Einsum(layout.outputEquation(), coefficients, value)
	return output, coefficients
}
