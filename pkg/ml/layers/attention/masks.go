// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"reflect"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
	. "github.com/gomlx/gomlx/core/graph"
	. "github.com/gomlx/gomlx/pkg/support/exceptions"
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
	if queryKeyMatrixMask.Shape().Equal(b.attentionShape) {
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
	if !queryKeyMatrixMask.Shape().Equal(shapeWithoutHeads) {
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

// buildAttentionMask returns a normalized mask for shape `[batch, <query_elements>, num_heads, <key_elements>]`.
//
// When not using KV cache, the simple causal mask is handled by Core (via the causal flag),
// so buildAttentionMask only includes user-provided masks. When using KV cache, the position-aware
// causal mask is built here because Core doesn't know about cache positions.
func (b *MultiHeadAttentionBuilder) buildAttentionMask() (mask *Node) {
	// User-provided masks.
	if b.queryMask != nil || b.keyMask != nil {
		mask = b.buildAttentionMaskFromSplitMasks()
	} else if b.queryKeyMatrixMask != nil {
		mask = b.queryKeyMatrixMask
	}

	// Causal mask is usually left to be calculated by Core, but if KV cache is used, or if it needs to be combined
	// with another mask, it is built here.
	if b.useCausalMask && (b.kvCacheShape.Ok() || mask != nil) {
		causalMask := b.buildCausalAttentionMask()
		if mask == nil {
			mask = causalMask
		} else {
			mask = LogicalAnd(mask, causalMask)
		}
	}
	return
}

// buildAttentionMaskFromSplitMasks creates cross mask from split queryMask and keyMask.
// The shape should be `[batch, <query_elements>, num_heads, <key_elements>]`.
func (b *MultiHeadAttentionBuilder) buildAttentionMaskFromSplitMasks() (mask *Node) {
	var keyMask *Node
	if b.keyMask == nil {
		// keyMask nil, create a skeleton (to be broadcast) keyMask filled with `true`.
		keyMask = Ones(b.g, shapes.Make(dtypes.Bool, b.attentionShape.Dimensions...))
	} else {
		// Expand dims after the batch axis.
		// attentionShape=`[batch, <query_elements>, num_heads, <key_elements>]`
		// b.keyMask.shape=`[batch, <key_elements>]`
		keyMask = InsertAxes(b.keyMask, xslices.SliceWithValue(b.attentionShape.Rank()-b.keyMask.Rank(), 1)...)
		keyMask = BroadcastToDims(keyMask, b.attentionShape.Dimensions...)
	}
	var queryMask *Node
	if b.queryMask == nil {
		// queryMask nil, create a skeleton (to be broadcast) queryMask filled with `true`.
		queryMask = Ones(b.g, shapes.Make(dtypes.Bool, b.attentionShape.Dimensions...))
	} else {
		// Expand dims at the end.
		queryMask = InsertAxes(b.queryMask, xslices.SliceWithValue(b.attentionShape.Rank()-b.queryMask.Rank(), -1)...)
		queryMask = BroadcastToDims(queryMask, b.attentionShape.Dimensions...)
	}
	return LogicalAnd(queryMask, keyMask)
}

// buildCausalAttentionMask creates a mask where queries can only attend to keys with "smaller index" than itself.
// When using KV cache (incremental generation), the mask accounts for the current position.
func (b *MultiHeadAttentionBuilder) buildCausalAttentionMask() (mask *Node) {
	if b.kvCacheShape.Ok() {
		return b.buildCausalAttentionMaskForKVCache()
	}

	queryShape := b.query.Shape()
	queryLen := queryShape.Dimensions[1]
	keyShape := b.key.Shape()
	keyLen := keyShape.Dimensions[1]

	if queryShape.Rank() != 3 || keyShape.Rank() != 3 {
		Panicf("MultiHeadAttention's WithCausalMask requires key and query to be rank-3,"+
			" instead got query.shape=%s and key.shape=%s", queryShape, keyShape)
	}
	if !reflect.DeepEqual(keyShape.Dimensions[:keyShape.Rank()-1], queryShape.Dimensions[:queryShape.Rank()-1]) {
		Panicf("MultiHeadAttention's WithCausalMask requires inner shapes of query and key be the same,"+
			" instead got query.shape=%s and key.shape=%s", queryShape, keyShape)
	}
	if queryLen != keyLen {
		Panicf("causal mask in training mode requires query and key lengths to match, got query=%d, key=%d",
			queryLen, keyLen)
	}
	mask = LowerTriangular(b.g, queryLen)

	// If using a sliding window, we must also apply a band mask.
	// Since queryLen == keyLen, the difference q - k should be < slidingWindow.
	if b.slidingWindow > 0 {
		qIndices := Iota(b.g, shapes.Make(dtypes.Int32, queryLen), 0)
		kIndices := Iota(b.g, shapes.Make(dtypes.Int32, keyLen), 0)
		qIndices = ExpandDims(qIndices, -1) // [queryLen, 1]
		kIndices = ExpandDims(kIndices, 0)  // [1, keyLen]
		qMinusK := Sub(qIndices, kIndices)
		slidingMask := LessThan(qMinusK, Const(b.g, int32(b.slidingWindow)))
		mask = LogicalAnd(mask, slidingMask)
	}

	// Broadcast mask to target shape: [batch, <query_elements>, numHeads, <key_elements>]
	// mask is currently [queryLen, keyLen], need to add batch and numHeads dimensions
	// InsertAxes at beginning for batch dimension
	mask = ExpandDims(mask, 0) // [1, queryLen, keyLen]
	// Add dimension for numHeads at position 2 (after batch and query)
	mask = ExpandDims(mask, 2)                                   // [1, queryLen, 1, keyLen]
	mask = BroadcastToDims(mask, b.attentionShape.Dimensions...) // Broadcast to target dimensions
	return mask
}

// buildCausalAttentionMaskForKVCache creates a mask where queries can only attend to keys with "smaller index" than itself,
// accounting for the current position when using KV cache (incremental generation).
func (b *MultiHeadAttentionBuilder) buildCausalAttentionMaskForKVCache() (mask *Node) {
	queryShape := b.query.Shape()
	queryLen := queryShape.Dimensions[1]
	keyLen := b.kvCacheShape.Dimensions[2] // Static shape for graph building

	// Build position-aware causal mask for incremental generation
	// Query positions: [position, position+1, ..., position+queryLen-1]
	// Key positions: [0, 1, ..., actualCacheLen-1]
	// mask[q, k] = (q_pos >= k_pos) AND (k < actualCacheLen)

	queryPositions := Iota(b.g, shapes.Make(dtypes.Int32, queryLen), 0)
	// Add position offset using the position Node
	positionInt32 := ConvertDType(b.position, dtypes.Int32)
	if positionInt32.Rank() > 0 {
		positionInt32 = Squeeze(positionInt32) // Ensure scalar
	}
	positionBroadcast := ExpandDims(positionInt32, 0) // [1]
	positionBroadcast = BroadcastToShape(positionBroadcast, shapes.Make(dtypes.Int32, queryLen))
	queryPositions = Add(queryPositions, positionBroadcast)
	queryPositions = ExpandDims(queryPositions, -1) // [queryLen, 1]

	keyPositions := Iota(b.g, shapes.Make(dtypes.Int32, keyLen), 0)
	keyPositions = ExpandDims(keyPositions, 0) // [1, keyLen]

	causalMask := GreaterOrEqual(queryPositions, keyPositions) // [queryLen, keyLen]

	// If using a sliding window, add distance mask: (q - k) < slidingWindow
	if b.slidingWindow > 0 {
		qMinusK := Sub(queryPositions, keyPositions)
		slidingMask := LessThan(qMinusK, Const(b.g, int32(b.slidingWindow)))
		causalMask = LogicalAnd(causalMask, slidingMask)
	}

	// Additional mask for valid cached positions: k < actualCacheLen
	// actualCacheLen is scalar, broadcast to [keyLen]
	keyIndices := Iota(b.g, shapes.Make(dtypes.Int32, keyLen), 0)
	actualLenBroadcast := BroadcastToShape(b.actualCacheLen, keyIndices.Shape())
	validMask := LessThan(keyIndices, actualLenBroadcast) // [keyLen]

	// Broadcast to [queryLen, keyLen] and combine with causal mask
	validMask = ExpandDims(validMask, 0) // [1, keyLen]
	validMask = BroadcastToShape(validMask, causalMask.Shape())

	mask = LogicalAnd(causalMask, validMask) // [queryLen, keyLen]

	// Broadcast mask to target shape: [batch, <query_elements>, numHeads, <key_elements>]
	// mask is currently [queryLen, keyLen], need to add batch and numHeads dimensions
	// InsertAxes at beginning for batch dimension
	mask = ExpandDims(mask, 0) // [1, queryLen, keyLen]
	// Add dimension for numHeads at position 2 (after batch and query)
	mask = ExpandDims(mask, 2)                                   // [1, queryLen, 1, keyLen]
	mask = BroadcastToDims(mask, b.attentionShape.Dimensions...) // Broadcast to target dimensions

	// Combine with cache validity mask if using cache
	cacheValidMask := createKVCacheAttentionMask(b.g, b.kvCacheShape, b.position, queryLen, keyLen)
	// cacheValidMask shape: [batch, 1, queryLen, keyLen]
	// Remove the '1' dimension at position 1 and add numHeads dimension at position 2
	cacheValidMask = Squeeze(cacheValidMask, 1)    // [batch, queryLen, keyLen]
	cacheValidMask = ExpandDims(cacheValidMask, 2) // [batch, queryLen, 1, keyLen]
	cacheValidMask = BroadcastToDims(cacheValidMask, b.attentionShape.Dimensions...)
	mask = LogicalAnd(mask, cacheValidMask)
	return mask
}
