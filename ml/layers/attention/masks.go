// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"reflect"
	"slices"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/support/xslices"
	. "github.com/gomlx/gomlx/core/graph"
	. "github.com/gomlx/gomlx/support/exceptions"
)

// WithMask is a simple shortcut to call WithKeyMask and WithQueryMask.
// Typically, it's used to handle padding in the input sequences, and can be combined with WithCausalMask.
//
// Do not use it with KVCache.
func (b *MultiHeadAttentionBuilder) WithMask(mask *Node) *MultiHeadAttentionBuilder {
	return b.WithKeyMask(mask).WithQueryMask(mask)
}

// WithKeyMask sets a mask for keys that are actually valid and can be attended.
//
// It defaults to no mask, meaning all keys are accessible. See also WithQueryMask.
//
// Shape should be `[batch_size, numHeads, <key_elements>]`,
// or `[batch_size, <key_elements>]` if the mask is the same
// for every head.
//
// Either use WithKeyMask and WithQueryMask separately or use WithKeyQueryMatrixMask, but not both.
// Optionally, one can also WithCausalMask, which is combined (logical-and) to any given mask.
func (b *MultiHeadAttentionBuilder) WithKeyMask(keyMask *Node) *MultiHeadAttentionBuilder {
	if b.queryKeyMatrixMask != nil {
		Panicf("a mask can be set either with SetKeyMask and SetQueryMask separately or with SetKeyQueryMatrixMask, but not both")
	}
	if b.querySeqLen != nil || b.keyValueSeqLen != nil {
		Panicf("MultiHeadAttention: an explicit mask is mutually exclusive with WithSeqLens; pass padding via seqlens only")
	}
	shape := keyMask.Shape()
	if shape.Rank() < 1+b.innerKeyAxes || shape.Rank() > 2+b.innerKeyAxes {
		Panicf("invalid keyMask shape (%s), expected rank to be %d or %d -- "+
			"`[batch_size, numHeads, <key_elements>]` or `[batch_size, <key_elements>]`",
			shape, 1+b.innerKeyAxes, 2+b.innerKeyAxes)
	}
	b.keyMask = keyMask
	return b
}

// WithQueryMask sets a mask for queries that are actually valid and should be used.
// Defaults to no mask, meaning all queries are used. See also WithKeyMask.
//
// Shape should be `[batch_size, numHeads, <query_elements>]`,
// or `[batch_size, <query_elements>]` if the mask is the same
// for every head.
//
// Either use WithKeyMask and WithQueryMask separately or use WithKeyQueryMatrixMask, but
// not both.
// Optionally, one can also WithCausalMask, which is combined (logical-and) to any given mask.
func (b *MultiHeadAttentionBuilder) WithQueryMask(queryMask *Node) *MultiHeadAttentionBuilder {
	if b.queryKeyMatrixMask != nil {
		Panicf("a mask can be set either with WithKeyMask and WithQueryMask separately or with WithKeyQueryMatrixMask, but not both")
	}
	if b.querySeqLen != nil || b.keyValueSeqLen != nil {
		Panicf("MultiHeadAttention: an explicit mask is mutually exclusive with WithSeqLens; pass padding via seqlens only")
	}
	shape := queryMask.Shape()
	if shape.Rank() < 1+b.innerQueryAxes || shape.Rank() > 2+b.innerQueryAxes {
		Panicf("invalid keyMask shape (%s), expected rank to be %d or %d -- "+
			"`[batch_size, numHeads, <query_elements>]` or `[batch_size, <query_elements>]`",
			shape, 1+b.innerQueryAxes, 2+b.innerQueryAxes)
	}
	b.queryMask = queryMask
	return b
}

// WithQueryKeyMatrixMask sets a mask matrix that defines which queries can attend to which
// keys. Defaults to no mask, meaning all queries are accessible.
//
// Shape should be `[batch_size, numHeads, <query_elements>, <key_elements>]`,
// or `[batch_size, <query_elements>, <key_elements>]` if the mask is the same
// for every head.
//
// Either use WithKeyMask and WithQueryMask separately or use WithKeyQueryMatrixMask, but
// not both.
//
// Optionally, one can also WithCausalMask, which is combined (logical-and) to any given mask.
func (b *MultiHeadAttentionBuilder) WithQueryKeyMatrixMask(queryKeyMatrixMask *Node) *MultiHeadAttentionBuilder {
	if b.useCausalMask {
		Panicf("MultiHeadAttention: SetQueryKeyMatrixMask is mutually exclusive with WithCausalMask; " +
			"combine them into a single mask if both causal and explicit masks are needed")
	}
	if b.keyMask != nil || b.queryMask != nil {
		Panicf("a mask can be set either with SetKeyMask and SetQueryMask separately or with SetKeyQueryMatrixMask, but not both")
	}
	if b.querySeqLen != nil || b.keyValueSeqLen != nil {
		Panicf("MultiHeadAttention: query/key matrix mask is mutually exclusive with WithSeqLens")
	}
	if slices.Equal(queryKeyMatrixMask.Shape().Dimensions, b.attentionShape.Dimensions) {
		// Simplest case: queryKeyMatrixMask provided with attentionShape.
		b.queryKeyMatrixMask = queryKeyMatrixMask
		return b
	}

	// shapeWithoutHeads = '[batch, <query_elements>, <key_elements>]` (without numHeads).
	shapeWithoutHeads := b.attentionShape.Clone()
	for ii := 1 + b.innerQueryAxes; ii < b.attentionShape.Rank()-1; ii++ {
		shapeWithoutHeads.Dimensions[ii] = shapeWithoutHeads.Dimensions[ii+1]
	}
	shapeWithoutHeads.Dimensions = shapeWithoutHeads.Dimensions[0 : b.attentionShape.Rank()-1]
	if !slices.Equal(queryKeyMatrixMask.Shape().Dimensions, shapeWithoutHeads.Dimensions) {
		Panicf("invalid shape for queryKeyMatrixMask %s: expected either %s (with per-head mask) or %s",
			queryKeyMatrixMask.Shape(), b.attentionShape, shapeWithoutHeads)
	}

	// Broadcast numHeads axes.
	queryKeyMatrixMask = InsertAxes(queryKeyMatrixMask, 1+b.innerQueryAxes)
	queryKeyMatrixMask = BroadcastToDims(queryKeyMatrixMask, b.attentionShape.Dimensions...)
	b.queryKeyMatrixMask = queryKeyMatrixMask
	return b
}

// WithCausalMask adds a mask where a query can only attend to keys with lower indices than itself.
// It assumes that query and key are either the same or have the same inner shape, and there is
// only one inner rank -- so key/query should have rank-3 shape `[batch, inner_dim, key/query_dim]`.
//
// WithCausalMask is mutually exclusive with WithKeyMask, WithQueryMask, and WithQueryKeyMatrixMask.
// If you need both causal masking and an explicit mask, combine them into a single mask
// before passing it (e.g. LogicalAnd a lower-triangular boolean mask with your mask).
func (b *MultiHeadAttentionBuilder) WithCausalMask(useCausalMask bool) *MultiHeadAttentionBuilder {
	b.useCausalMask = useCausalMask
	if !b.useCausalMask {
		// Nothing to check.
		return b
	}
	if b.queryKeyMatrixMask != nil {
		Panicf("MultiHeadAttention: WithCausalMask is mutually exclusive with WithQueryKeyMatrixMask; " +
			"combine them into a single mask if both causal and explicit masks are needed")
	}
	queryShape := b.query.Shape()
	keyShape := b.key.Shape()
	if queryShape.Rank() != 3 || keyShape.Rank() != 3 {
		// TODO: we could extrapolate and make this work for higher ranked tensors.
		Panicf("MultiHeadAttention's WithCausalMask requires key and query to be rank-3,"+
			" instead got query.shape=%s and key.shape=%s", queryShape, keyShape)
	}
	if !reflect.DeepEqual(keyShape.Dimensions[:keyShape.Rank()-1], queryShape.Dimensions[:queryShape.Rank()-1]) {
		Panicf("MultiHeadAttention's WithCausalMask requires inner shapes of query and key be the same,"+
			" instead got query.shape=%s and key.shape=%s", queryShape, keyShape)
	}
	return b
}

// buildAttentionMask returns a normalized or broadcastable mask for shape `[batch, <query_elements>, num_heads, <key_elements>]`.
//
// When not using KV cache or sliding window, the simple causal mask is handled by Core (via the causal flag),
// so buildAttentionMask only includes user-provided masks. When slidingWindow > 0 or causal masking needs to
// be combined with another mask, it is built here.
func (b *MultiHeadAttentionBuilder) buildAttentionMask() (mask *Node) {
	// User-provided masks.
	if b.queryMask != nil || b.keyMask != nil {
		mask = b.buildAttentionMaskFromSplitMasks()
	} else if b.queryKeyMatrixMask != nil {
		mask = b.queryKeyMatrixMask
	}

	// Causal mask is usually left to be calculated by Core, unless a sliding window is configured
	// or it needs to be combined with another mask.
	if b.slidingWindow > 0 || (b.useCausalMask && mask != nil) {
		localOrCausalMask := b.buildCausalAttentionMask()
		if mask != nil {
			mask = LogicalAnd(mask, localOrCausalMask)
		} else {
			mask = localOrCausalMask
		}
	}
	return
}

// broadcastToAttentionShape broadcasts x (either scalar or already expanded to attentionShape.Rank())
// to `[batch, <query_elements>, num_heads, <key_elements>]`, supporting both static and dynamic dimensions.
func (b *MultiHeadAttentionBuilder) broadcastToAttentionShape(x *Node) *Node {
	specs := b.attentionDimensionSpecs()
	if x.Rank() == 0 {
		return DynamicBroadcastInDim(x, nil, specs...)
	}
	broadcastAxes := make([]int, x.Rank())
	for i := range broadcastAxes {
		broadcastAxes[i] = i
	}
	return DynamicBroadcastInDim(x, broadcastAxes, specs...)
}

// buildAttentionMaskFromSplitMasks creates cross mask from split queryMask and keyMask.
// The shape should be `[batch, <query_elements>, num_heads, <key_elements>]`.
func (b *MultiHeadAttentionBuilder) buildAttentionMaskFromSplitMasks() (mask *Node) {
	var keyMask *Node
	if b.keyMask == nil {
		// keyMask nil, create a scalar true to be broadcast.
		keyMask = Const(b.g, true)
	} else {
		// Expand dims after the batch axis.
		// attentionShape=`[batch, <query_elements>, num_heads, <key_elements>]`
		// b.keyMask.shape=`[batch, <key_elements>]`
		keyMask = InsertAxes(b.keyMask, xslices.SliceWithValue(b.attentionShape.Rank()-b.keyMask.Rank(), 1)...)
		keyMask = b.broadcastToAttentionShape(keyMask)
	}
	var queryMask *Node
	if b.queryMask == nil {
		// queryMask nil, create a scalar true to be broadcast.
		queryMask = Const(b.g, true)
	} else {
		// Expand dims at the end.
		queryMask = InsertAxes(b.queryMask, xslices.SliceWithValue(b.attentionShape.Rank()-b.queryMask.Rank(), -1)...)
		queryMask = b.broadcastToAttentionShape(queryMask)
	}
	return LogicalAnd(queryMask, keyMask)
}

// buildSeqLenPaddingMask builds a boolean padding mask from per-batch sequence length tensors.
// querySeqLen and keyValueSeqLen are int32 [B] nodes; either may be nil (treated as full length).
// Returns a [B, Sq, 1, Skv] boolean mask (BSHD layout only) broadcastable to [B, Sq, H, Skv]:
// mask[b, q, 0, kv] = (q < querySeqLen[b]) AND (kv < keyValueSeqLen[b]).
// Nil seqlen means all positions in that axis are valid.
func buildSeqLenPaddingMask(query, key *Node, querySeqLen, keyValueSeqLen *Node) *Node {
	g := query.Graph()

	var kvMask, qMask *Node
	if keyValueSeqLen != nil {
		// kv positions: Iota [1, 1, 1, Skv], compare < kvLen[b] reshaped to [B, 1, 1, 1]
		kvIota := DynamicIota(g, dtypes.Int32, 0, DimensionSpecFor(key, 1))
		kvIota = ExpandAxes(kvIota, 0, 1, 2)
		kvLenBC := ExpandDims(ConvertDType(keyValueSeqLen, dtypes.Int32), 1, 2, 3)
		kvMask = LessThan(kvIota, kvLenBC) // [B, 1, 1, Skv]
	}
	if querySeqLen != nil {
		// q positions: Iota [1, Sq, 1, 1], compare < qLen[b] reshaped to [B, 1, 1, 1]
		qIota := DynamicIota(g, dtypes.Int32, 0, DimensionSpecFor(query, 1))
		qIota = ExpandAxes(qIota, 0, 2, 3)
		qLenBC := ExpandDims(ConvertDType(querySeqLen, dtypes.Int32), 1, 2, 3)
		qMask = LessThan(qIota, qLenBC) // [B, Sq, 1, 1]
	}

	switch {
	case kvMask != nil && qMask != nil:
		return LogicalAnd(qMask, kvMask) // [B, Sq, 1, Skv] via broadcast
	case kvMask != nil:
		return kvMask
	default:
		return qMask
	}
}

// buildCausalAttentionMask creates a causal and/or sliding-window mask shaped `[1, queryLen, 1, keyLen]`,
// broadcastable to `[batch, queryLen, numHeads, keyLen]` across both static and dynamic sequence dimensions.
func (b *MultiHeadAttentionBuilder) buildCausalAttentionMask() (mask *Node) {
	queryShape := b.query.Shape()
	keyShape := b.key.Shape()

	if queryShape.Rank() != 3 || keyShape.Rank() != 3 {
		Panicf("MultiHeadAttention's causal/sliding-window mask requires key and query to be rank-3,"+
			" instead got query.shape=%s and key.shape=%s", queryShape, keyShape)
	}
	if !reflect.DeepEqual(keyShape.Dimensions[:keyShape.Rank()-1], queryShape.Dimensions[:queryShape.Rank()-1]) {
		Panicf("MultiHeadAttention's causal/sliding-window mask requires inner shapes of query and key be the same,"+
			" instead got query.shape=%s and key.shape=%s", queryShape, keyShape)
	}

	// queryIndices: [0, 1, 2, ..., queryLen-1] -> shaped [queryLen, 1]
	queryIndices := DynamicIota(b.g, dtypes.Int32, 0, DimensionSpecFor(b.query, 1))
	queryIndices = ExpandDims(queryIndices, -1) // [queryLen, 1]

	// keyIndices: [0, 1, 2, ..., keyLen-1] -> shaped [1, keyLen]
	keyIndices := DynamicIota(b.g, dtypes.Int32, 0, DimensionSpecFor(b.key, 1))
	keyIndices = ExpandDims(keyIndices, 0) // [1, keyLen]

	if b.useCausalMask {
		mask = GreaterOrEqual(queryIndices, keyIndices) // [queryLen, keyLen]
		// If using a sliding window, add distance mask: (q - k) < slidingWindow
		if b.slidingWindow > 0 {
			qMinusK := Sub(queryIndices, keyIndices)
			slidingMask := LessThan(qMinusK, Const(b.g, int32(b.slidingWindow)))
			mask = LogicalAnd(mask, slidingMask)
		}
	} else {
		// Bidirectional sliding window: |q - k| < slidingWindow
		dist := Abs(Sub(queryIndices, keyIndices))
		mask = LessThan(dist, Const(b.g, int32(b.slidingWindow)))
	}

	// Expand to [1, queryLen, 1, keyLen], which broadcasts implicitly across batch and numHeads.
	return ExpandAxes(mask, 0, 2)
}
