// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"math"

	. "github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// ScaledDotProductAttentionBuilder configures and executes scaled dot-product attention.
// Create it with ScaledDotProductAttention, configure with builder methods, and call Done.
type ScaledDotProductAttentionBuilder struct {
	query, key, value *Node
	scale             float64
	hasCustomScale    bool
	additiveMask      *Node
	booleanMask       *Node
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
// before softmax. The mask should be broadcastable to [batch, heads, q_seq, kv_seq].
// Typical values: 0 for positions to attend, large negative values (e.g., -1e9) for masked positions.
func (b *ScaledDotProductAttentionBuilder) WithAdditiveMask(mask *Node) *ScaledDotProductAttentionBuilder {
	b.additiveMask = mask
	return b
}

// WithBooleanMask sets a boolean attention mask. True means attend, false means mask out.
// The mask should be broadcastable to [batch, heads, q_seq, kv_seq].
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

	// Compute attention scores: Q @ K^T * scale
	// query: [batch, heads, q_seq, head_dim]
	// key:   [batch, heads, kv_seq, head_dim]
	// scores: [batch, heads, q_seq, kv_seq]
	scores := Einsum("bhqd,bhkd->bhqk", query, key)
	scores = MulScalar(scores, scale)

	// Apply boolean mask: convert to additive mask
	if b.booleanMask != nil {
		// Where mask is false, set to large negative value
		largeNeg := ScalarOne(scores.Graph(), scores.DType())
		largeNeg = MulScalar(largeNeg, -1e9)
		// Where mask is true, add 0; where false, add large negative
		maskAdditive := Where(b.booleanMask, ScalarZero(scores.Graph(), scores.DType()), largeNeg)
		scores = Add(scores, maskAdditive)
	}

	// Apply additive mask
	if b.additiveMask != nil {
		scores = Add(scores, b.additiveMask)
	}

	// Softmax over kv_seq dimension
	weights := Softmax(scores, -1)

	// Compute output: weights @ value
	// weights: [batch, heads, q_seq, kv_seq]
	// value:   [batch, heads, kv_seq, v_head_dim]
	// output:  [batch, heads, q_seq, v_head_dim]
	output := Einsum("bhqk,bhkd->bhqd", weights, value)

	if returnWeights {
		return output, weights
	}
	return output, nil
}
