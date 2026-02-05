// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"math"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	. "github.com/gomlx/gomlx/pkg/support/exceptions"
)

// AxesLayout specifies the ordering of axes in the 4D attention tensors.
// Different layouts use different Einsum equations but produce mathematically equivalent results.
type AxesLayout int

const (
	// LayoutBHSD is the [batch, heads, seq, dim] layout used by PyTorch's F.scaled_dot_product_attention,
	// ONNX, and most inference runtimes. This is the default for ScaledDotProductAttention.
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

// sdpaCore computes the core scaled dot-product attention operation.
// This is the shared implementation used by both ScaledDotProductAttention and MultiHeadAttention.
//
// The layout parameter determines the axis ordering of the input tensors:
//   - LayoutBHSD: [batch, heads, seq, dim] (PyTorch/ONNX convention)
//   - LayoutBSHD: [batch, seq, heads, dim] (MHA convention)
//
// The additiveMask must match the score tensor layout:
//   - LayoutBHSD: broadcastable to [batch, heads, q_seq, kv_seq]
//   - LayoutBSHD: broadcastable to [batch, q_seq, heads, kv_seq]
//
// Returns output and optionally weights in the same layout as inputs.
func sdpaCore(query, key, value *Node, scale float64, additiveMask *Node, returnWeights bool, layout AxesLayout) (*Node, *Node) {
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

// booleanToAdditiveMask converts a boolean attention mask to an additive mask.
// True means attend (adds 0), False means mask out (adds large negative value).
// The dtype parameter specifies the output dtype (should match the scores dtype).
func booleanToAdditiveMask(booleanMask *Node, dtype dtypes.DType) *Node {
	g := booleanMask.Graph()

	largeNeg := ScalarOne(g, dtype)
	largeNeg = MulScalar(largeNeg, -1e9)
	zero := ScalarZero(g, dtype)

	// Where mask is true, add 0; where false, add large negative
	return Where(booleanMask, zero, largeNeg)
}

// ScaledDotProductAttentionBuilder configures and executes scaled dot-product attention.
// Create it with ScaledDotProductAttention, configure with builder methods, and call Done.
type ScaledDotProductAttentionBuilder struct {
	query, key, value *Node
	scale             float64
	hasCustomScale    bool
	additiveMask      *Node
	booleanMask       *Node
	layout            AxesLayout // default LayoutBHSD
}

// ScaledDotProductAttention creates a builder for computing scaled dot-product attention.
//
// This follows the standard SDPA convention used by PyTorch's F.scaled_dot_product_attention,
// ONNX, and most inference runtimes.
//
// Input layout: [batch, heads, seq, dim]
//
// Parameters:
//   - query: [batch, heads, q_seq, head_dim]
//   - key: [batch, heads, kv_seq, head_dim]
//   - value: [batch, heads, kv_seq, v_head_dim]
//
// Returns a builder. Call Done() to get the output tensor.
func ScaledDotProductAttention(query, key, value *Node) *ScaledDotProductAttentionBuilder {
	return &ScaledDotProductAttentionBuilder{
		query: query,
		key:   key,
		value: value,
	}
}

// WithScale sets a custom scale factor. By default, scale is 1/sqrt(head_dim).
func (b *ScaledDotProductAttentionBuilder) WithScale(scale float64) *ScaledDotProductAttentionBuilder {
	b.scale = scale
	b.hasCustomScale = true
	return b
}

// WithAdditiveMask sets an additive attention mask that is added to the attention scores
// before softmax. The mask should be broadcastable to the score tensor shape, which depends on the layout:
//   - LayoutBHSD: [batch, heads, q_seq, kv_seq]
//   - LayoutBSHD: [batch, q_seq, heads, kv_seq]
//
// Typical values: 0 for positions to attend, large negative values (e.g., -1e9) for masked positions.
func (b *ScaledDotProductAttentionBuilder) WithAdditiveMask(mask *Node) *ScaledDotProductAttentionBuilder {
	b.additiveMask = mask
	return b
}

// WithLayout sets the axes layout for the input tensors.
// Default is LayoutBHSD [batch, heads, seq, dim].
// Use LayoutBSHD for [batch, seq, heads, dim] layout.
func (b *ScaledDotProductAttentionBuilder) WithLayout(layout AxesLayout) *ScaledDotProductAttentionBuilder {
	b.layout = layout
	return b
}

// WithBooleanMask sets a boolean attention mask. True means attend, false means mask out.
// The mask should be broadcastable to the score tensor shape, which depends on the layout:
//   - LayoutBHSD: [batch, heads, q_seq, kv_seq]
//   - LayoutBSHD: [batch, q_seq, heads, kv_seq]
func (b *ScaledDotProductAttentionBuilder) WithBooleanMask(mask *Node) *ScaledDotProductAttentionBuilder {
	b.booleanMask = mask
	return b
}

// Done executes the scaled dot-product attention and returns the output tensor.
//
// Returns:
//   - output: [batch, heads, q_seq, v_head_dim]
func (b *ScaledDotProductAttentionBuilder) Done() *Node {
	output, _ := b.execute(false)
	return output
}

// DoneWithCoefficients executes the scaled dot-product attention and returns both
// the output tensor and the attention weights.
//
// Returns:
//   - output: [batch, heads, q_seq, v_head_dim]
//   - weights: [batch, heads, q_seq, kv_seq] — attention coefficients after softmax
func (b *ScaledDotProductAttentionBuilder) DoneWithCoefficients() (*Node, *Node) {
	return b.execute(true)
}

func (b *ScaledDotProductAttentionBuilder) execute(returnWeights bool) (*Node, *Node) {
	query, key, value := b.query, b.key, b.value

	if query.Rank() != 4 || key.Rank() != 4 || value.Rank() != 4 {
		Panicf("ScaledDotProductAttention: query, key, value must be rank-4 [batch, heads, seq, dim], got ranks %d, %d, %d",
			query.Rank(), key.Rank(), value.Rank())
	}

	headDim := query.Shape().Dimensions[3]

	// Compute scale
	scale := b.scale
	if !b.hasCustomScale {
		scale = 1.0 / math.Sqrt(float64(headDim))
	}

	// Build combined additive mask from boolean and additive masks
	var additiveMask *Node
	if b.booleanMask != nil {
		additiveMask = booleanToAdditiveMask(b.booleanMask, query.DType())
	}
	if b.additiveMask != nil {
		if additiveMask != nil {
			additiveMask = Add(additiveMask, b.additiveMask)
		} else {
			additiveMask = b.additiveMask
		}
	}

	return sdpaCore(query, key, value, scale, additiveMask, returnWeights, b.layout)
}
