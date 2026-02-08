// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"math"
	"reflect"
	"slices"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention/pos"
	. "github.com/gomlx/gomlx/pkg/support/exceptions"
	"github.com/gomlx/gomlx/pkg/support/xslices"
)

// This file contains all parts of the layers.MultiHeadAttention implementation.

// MultiHeadAttentionBuilder is a helper to build a multi-head-attention computation.
// Create it with MultiHeadAttention, set the desired parameters, and when all is set, call Done.
type MultiHeadAttentionBuilder struct {
	ctx               *context.Context
	g                 *Graph
	query, key, value *Node
	numHeads          int
	keyQueryDim       int
	valueDim          int
	outputDim         int

	innerKeyAxes, innerQueryAxes int
	attentionShape               shapes.Shape

	useProjectionBias bool
	dropoutRate       float64

	// layout records the internal layout used for the attention core.
	// Defaults to LayoutBSHD (Dense projections produce [batch, seq, heads, dim]).
	layout AxesLayout

	// Mask related attributes.
	keyMask, queryMask *Node
	queryKeyMatrixMask *Node
	useCausalMask      bool

	// KV cache and incremental generation support.
	// kvCacheShape is the shape of the KV cache: [batch, heads, maxSeqLen, headDim].
	// If set, enables caching of key/value projections for incremental generation.
	kvCacheShape   shapes.Shape
	position       *Node // Position as a graph node (scalar int32) for graph caching
	actualCacheLen *Node // Actual filled cache length (scalar int32)

	// Positional Encoder to be used, e.g: RoPE.
	positionalEncoder pos.Encoder
}

// MultiHeadAttention defines a multi-head attention layers, as described in the paper
// "Attention Is All You Need", https://arxiv.org/abs/1706.03762,
// by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez,
// Lukasz Kaiser, Illia Polosukhin.
//
// It takes query, key, and value and project them numHead times, to a headDim sized embeddings.
// Then it uses the dot-product of query and key as weights, and returns a softmax sum
// of value, for each head.
//
// Typical shapes:
//
// - query: `[batch_size, <query_elements>, inputQueryDim]`.
// - key: `[batch_size, <num_key/value_elements>, inputKeyDim]`.
// - value: `[batch_size, <num_key/value_elements>, inputValueDim]`.
//
// And, when calling IsNil, after another output projection, it returns a node of shape
// `[batch_size, <num_queries>, inputValueDim]`, if no other settings are given.
// See settings in MultiHeadAttentionBuilder.to control various aspects.
//
// Notice it's common to use key=values, and even query=keys=values. For instance for
// encoding text, one may use the input sequence as all 3 (query, key and value).
//
// The function returns a MultiHeadAttentionBuilder that can be further configured,
// and the resulting Node is returned when MultiHeadAttentionBuilder.Done is called.
// Alternatively one can call MultiHeadAttentionBuilder.DoneWithCoefficients, in which
// case it returns both the updated state and the attention coefficients.
func MultiHeadAttention(ctx *context.Context, query, key, value *Node, numHeads int, headDim int) *MultiHeadAttentionBuilder {
	g := query.Graph()

	queryShape := query.Shape()
	keyShape := key.Shape()
	valueShape := value.Shape()
	innerKeyAxes := keyShape.Rank() - 2
	innerQueryAxes := queryShape.Rank() - 2

	b := &MultiHeadAttentionBuilder{
		ctx:               ctx.In("MultiHeadAttention"),
		g:                 g,
		query:             query,
		key:               key,
		value:             value,
		numHeads:          numHeads,
		valueDim:          headDim,
		keyQueryDim:       headDim,
		innerKeyAxes:      innerKeyAxes,
		innerQueryAxes:    innerQueryAxes,
		useProjectionBias: true,
		layout:            LayoutBSHD,
	}

	if queryShape.Rank() < 3 {
		Panicf("query rank is %d (shape=%s), but MultiHeadAttention requires at least rank 3",
			queryShape.Rank(), queryShape)
	}
	if keyShape.Rank() < 3 {
		Panicf("key rank is %d (shape=%s), but MultiHeadAttention requires at least rank 3",
			keyShape.Rank(), keyShape)
	}
	if valueShape.Rank() < 3 {
		Panicf("value rank is %d (shape=%s), but MultiHeadAttention requires at least rank 3",
			valueShape.Rank(), valueShape)
	}
	if keyShape.DType != queryShape.DType || keyShape.DType != valueShape.DType {
		Panicf("key, query and value should have the same dtype, instead got shapes key=%s, query=%s, value=%s",
			keyShape, queryShape, valueShape)
	}
	if !reflect.DeepEqual(keyShape.Dimensions[:keyShape.Rank()-1], valueShape.Dimensions[:valueShape.Rank()-1]) {
		Panicf("key and value shapes must be the same up to the one-before-last axis, instead got shapes key=%s, value=%s",
			keyShape, valueShape)
	}
	if valueShape.Ok() && valueShape.Rank() > 0 {
		b.outputDim = valueShape.Dimensions[valueShape.Rank()-1]
	}
	b.buildAttentionShape()
	return b
}

// SetKeyQueryDim allows finer configuration on the dimension of the projection used for the
// query/key pairs for each head. It defaults to the value given by `headDim`.
func (b *MultiHeadAttentionBuilder) SetKeyQueryDim(keyQueryDim int) *MultiHeadAttentionBuilder {
	b.keyQueryDim = keyQueryDim
	return b
}

// SetValueHeadDim allows finer configuration on the dimension of the projection used for the
// value for each head. It defaults to the value given by `headDim`.
func (b *MultiHeadAttentionBuilder) SetValueHeadDim(valueDim int) *MultiHeadAttentionBuilder {
	b.valueDim = valueDim
	return b
}

// SetOutputDim defines the output dimension of the final projection, from the flattened
// attention heads. It defaults to the value of the last dimension of `values` passed as input
// (`inputValueDim`).
func (b *MultiHeadAttentionBuilder) SetOutputDim(outputDim int) *MultiHeadAttentionBuilder {
	b.outputDim = outputDim
	return b
}

// SetKeyMask sets a mask for keys that are actually valid and can be attended.
// Defaults to no mask, meaning all keys are accessible. See also SetQueryMask.
//
// Shape should be `[batch_size, numHeads, <key_elements>]`,
// or `[batch_size, <key_elements>]` if the mask is the same
// for every head.
//
// Either use SetKeyMask and SetQueryMask separately or use SetKeyQueryMatrixMask, but
// not both. Optionally, one can also UseCausalMask, which is combined (logical-and) to
// any given mask.
func (b *MultiHeadAttentionBuilder) SetKeyMask(keyMask *Node) *MultiHeadAttentionBuilder {
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

// SetQueryMask sets a mask for queries that are actually valid and should be used.
// Defaults to no mask, meaning all queries are accessible. See also SetKeyMask.
//
// Shape should be `[batch_size, numHeads, <query_elements>]`,
// or `[batch_size, <query_elements>]` if the mask is the same
// for every head.
//
// Either use SetKeyMask and SetQueryMask separately or use SetKeyQueryMatrixMask, but
// not both.
// Optionally, one can also UseCausalMask, which is combined (logical-and) to any given mask.
func (b *MultiHeadAttentionBuilder) SetQueryMask(queryMask *Node) *MultiHeadAttentionBuilder {
	if b.queryKeyMatrixMask != nil {
		Panicf("a mask can be set either with SetKeyMask and SetQueryMask separately or with SetKeyQueryMatrixMask, but not both")
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

// SetQueryKeyMatrixMask sets a mask matrix that defines which queries can attend to which
// keys. Defaults to no mask, meaning all queries are accessible.
//
// Shape should be `[batch_size, numHeads, <query_elements>, <key_elements>]`,
// or `[batch_size, <query_elements>, <key_elements>]` if the mask is the same
// for every head.
//
// Either use SetKeyMask and SetQueryMask separately or use SetKeyQueryMatrixMask, but
// not both.
// Optionally, one can also UseCausalMask, which is combined (logical-and) to any given mask.
func (b *MultiHeadAttentionBuilder) SetQueryKeyMatrixMask(queryKeyMatrixMask *Node) *MultiHeadAttentionBuilder {
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
	return b
}

// UseCausalMask adds a mask where a query can only attend to keys with lower indices than itself.
// It assumes that query and key are either the same or have the same inner shape, and there is
// only one inner rank -- so key/query should have rank-3 shape `[batch, inner_dim, key/query_dim]`.
//
// This mask can be used in combination (logical-and) with other masks.
func (b *MultiHeadAttentionBuilder) UseCausalMask() *MultiHeadAttentionBuilder {
	queryShape := b.query.Shape()
	keyShape := b.key.Shape()
	if queryShape.Rank() != 3 || keyShape.Rank() != 3 {
		// TODO: we could extrapolate and make this work for higher ranked tensors.
		Panicf("MultiHeadAttention's UseCausalMask requires key and query to be rank-3,"+
			" instead got query.shape=%s and key.shape=%s", queryShape, keyShape)
	}
	if !reflect.DeepEqual(keyShape.Dimensions[:keyShape.Rank()-1], queryShape.Dimensions[:queryShape.Rank()-1]) {
		Panicf("MultiHeadAttention's UseCausalMask requires inner shapes of query and key be the same,"+
			" instead got query.shape=%s and key.shape=%s", queryShape, keyShape)
	}
	b.useCausalMask = true
	return b
}

// UseProjectionBias defines whether to use a bias term on the final output projection.
// Default is true.
func (b *MultiHeadAttentionBuilder) UseProjectionBias(useProjectionBias bool) *MultiHeadAttentionBuilder {
	b.useProjectionBias = useProjectionBias
	return b
}

// Dropout defines how much dropout to use in the attention coefficients calculation.
// If set to 0 or lower, it's simply disabled. Default is 0.
func (b *MultiHeadAttentionBuilder) Dropout(rate float64) *MultiHeadAttentionBuilder {
	b.dropoutRate = rate
	if b.dropoutRate >= 1 {
		Panicf("dropout rate %g >= 1 is undefined", rate)
	}
	return b
}

// WithKVCache enables incremental generation mode with KV caching.
// This is used for efficient token-by-token generation where past key/value
// projections are cached and reused, avoiding redundant computation.
//
// How it works:
//   - Input: Only the embeddings/features for the current token(s) being generated
//   - Output: The attention output for the current token(s)
//   - Side effect: New key/value projections are automatically stored in the cache
//   - Attention: Computed against all cached keys/values (including the new ones)
//
// Parameters:
//   - maxSeqLen: Maximum sequence length the cache can hold. The cache shape is
//     derived automatically from the attention configuration (batchSize, numHeads,
//     headDim, dtype are taken from the builder).
//   - position: Scalar Node (int32) representing the current position in the sequence.
//     This allows the same compiled graph to be reused for different positions.
//     For the initial prompt, position=0. For subsequent tokens, position increments.
//
// Returns:
//   - The builder for method chaining
//
// Usage pattern:
//
//	// Build generation graph:
//	builder := attention.MultiHeadAttention(ctx, embeddings, embeddings, embeddings, numHeads, headDim).
//	    WithKVCache(maxSeqLen, position).
//	    WithRoPE(10000.0).
//	    Done()
//
//	// Generate tokens:
//	// 1. First call with full prompt (e.g., 10 tokens), position=0
//	// 2. Each subsequent call with 1 new token, position=10, 11, 12, ...
//
//	// Before generating a new response, reset all caches in the model:
//	attention.KVCacheReset(ctx)
//
// The cache supports circular/rotating mode: when position exceeds maxSeqLen,
// new entries wrap around and overwrite the oldest entries. This allows efficient
// generation for sequences longer than maxSeqLen. Automatically enables causal masking.
func (b *MultiHeadAttentionBuilder) WithKVCache(maxSeqLen int, position *Node) *MultiHeadAttentionBuilder {
	// Derive cache shape from attention configuration
	batchSize := b.query.Shape().Dimensions[0]
	dtype := b.query.DType()
	b.kvCacheShape = shapes.Make(dtype, batchSize, b.numHeads, maxSeqLen, b.keyQueryDim)
	b.position = position
	// When using KV cache, automatically enable causal mask for proper attention
	b.useCausalMask = true
	// Rebuild attention shape with cache dimensions
	b.buildAttentionShape()
	return b
}

// KVCacheShape returns the shape of the KV cache configured for this attention layer.
// Shape is [batchSize, numHeads, maxSeqLen, headDim].
// Returns an invalid shape if WithKVCache was not called.
func (b *MultiHeadAttentionBuilder) KVCacheShape() shapes.Shape {
	return b.kvCacheShape
}

// WithRoPE enables Rotary Position Embeddings on query and key projections.
// baseFreq is typically 10000.0 (default from RoFormer paper).
// RoPE is applied after projections and works with both training and generation modes.
//
// Deprecated: Use WithPositionalEncoder, passing a pos.RoPE instead.
func (b *MultiHeadAttentionBuilder) WithRoPE(baseFreq float64) *MultiHeadAttentionBuilder {
	b.positionalEncoder = pos.NewRoPE(baseFreq)
	return b
}

// WithPositionalEncoder sets the positional encoder to use.
func (b *MultiHeadAttentionBuilder) WithPositionalEncoder(encoder pos.Encoder) *MultiHeadAttentionBuilder {
	b.positionalEncoder = encoder
	return b
}

// DoneWithCoefficients or Done should be called after all optional settings are configured.
// It returns both the attention output and the attention coefficients (matrix) used.
//
// `output` will be shaped `[batch_size, <query_elements>, output_dim]`, where `output_dim`
// can be configured by `SetOutputDim`.
//
// `coefficients` is shaped `[batch_size, <query_elements>, <num_heads>, <key_elements>]`
// with the attention weights (from 0 to 1).
func (b *MultiHeadAttentionBuilder) DoneWithCoefficients() (attentionOutput, attentionCoefficients *Node) {
	if b.layout != LayoutBSHD {
		Panicf("MultiHeadAttention only supports LayoutBSHD, got %s", b.layout)
	}
	seqAxis := b.layout.SeqAxis()

	// Flatten inner axes to a single sequence axis before projection.
	// This turns [batch, q1, q2, ..., inputDim] into [batch, q_flat, inputDim].
	batchSize := b.query.Shape().Dim(0)
	var queryInnerDims, keyInnerDims []int
	flatQuery, flatKey, flatValue := b.query, b.key, b.value
	needsUnflatten := b.innerKeyAxes > 1 || b.innerQueryAxes > 1
	if needsUnflatten {
		qDims := b.query.Shape().Dimensions
		kDims := b.key.Shape().Dimensions

		queryInnerDims = slices.Clone(qDims[1 : 1+b.innerQueryAxes])
		keyInnerDims = slices.Clone(kDims[1 : 1+b.innerKeyAxes])

		flatQuery = Reshape(b.query, batchSize, -1, b.query.Shape().Dim(-1))
		flatKey = Reshape(b.key, batchSize, -1, b.key.Shape().Dim(-1))
		flatValue = Reshape(b.value, batchSize, -1, b.value.Shape().Dim(-1))
	}

	projectedKey := layers.Dense(b.ctx.In("key"), flatKey, true, b.numHeads, b.keyQueryDim)
	projectedQuery := layers.Dense(b.ctx.In("query"), flatQuery, true, b.numHeads, b.keyQueryDim)
	projectedValue := layers.Dense(b.ctx.In("value"), flatValue, true, b.numHeads, b.valueDim)

	// Apply positional encoding (e.g. RoPE) if enabled.
	// Applied before KV cache so that rotated embeddings are cached.
	// projectedQuery/Key shape: [batch, seq, heads, dim] (BSHD layout).
	if b.positionalEncoder != nil {
		seqLen := projectedQuery.Shape().Dimensions[seqAxis]
		var posIndices *Node
		if b.position != nil {
			posIndices = pos.SequentialPositions(b.g, b.position, seqLen)
		} else {
			posIndices = pos.SequentialPositions(b.g, Const(b.g, int32(0)), seqLen)
		}
		projectedQuery = b.positionalEncoder.Apply(projectedQuery, posIndices, seqAxis)
		projectedKey = b.positionalEncoder.Apply(projectedKey, posIndices, seqAxis)
	}

	// Handle KV cache if in incremental generation mode
	if b.kvCacheShape.Ok() {
		// Cache expects shape: [batch, numHeads, seqLen, headDim]
		// projectedKey/Value shape: [batch, seqLen, numHeads, headDim]
		// Transpose: (0,1,2,3) -> (0,2,1,3)
		keyForCache := TransposeAllDims(projectedKey, 0, 2, 1, 3)
		valueForCache := TransposeAllDims(projectedValue, 0, 2, 1, 3)

		// Update cache and get full key/value (including past)
		// KV cache variables are stored in the context under "kv_cache" scope
		// Pass b.position so the cache knows where to write the new keys/values
		cacheCtx := b.ctx.In("kv_cache").Reuse().Checked(false)
		KVCacheUpdate(cacheCtx, b.g, b.kvCacheShape, b.position, keyForCache, valueForCache)
		fullKey, fullValue := getKVCache(cacheCtx, b.g, b.kvCacheShape)

		// b.position holds the current position in the sequence
		b.actualCacheLen = b.position

		// Transpose back to attention format: (0,2,1,3) -> (0,1,2,3)
		// Result: [batch, maxSeqLen, numHeads, headDim]
		projectedKey = TransposeAllDims(fullKey, 0, 2, 1, 3)
		projectedValue = TransposeAllDims(fullValue, 0, 2, 1, 3)
	}

	// Build the mask before any flattening, since buildMask uses the original attentionShape.
	// Mask shape: [batch, <query_elements>, num_heads, <key_elements>]
	mask := b.buildMask()

	// Flatten the mask to match the now-flat Q/K/V when inner axes > 1.
	if needsUnflatten && mask != nil {
		qFlat := projectedQuery.Shape().Dimensions[seqAxis]
		kFlat := projectedKey.Shape().Dimensions[seqAxis]
		mask = Reshape(mask, batchSize, qFlat, b.numHeads, kFlat)
	}

	scale := 1.0 / math.Sqrt(float64(b.keyQueryDim))
	// Pass causal to Core only when not using KV cache (Core builds a simple lower-triangular mask).
	// When using KV cache, the position-aware causal mask is already built in buildMask above.
	passCausal := b.useCausalMask && !b.kvCacheShape.Ok()
	attentionOutput, attentionCoefficients = Core(b.ctx, projectedQuery, projectedKey, projectedValue, scale, mask, b.dropoutRate, b.layout, passCausal)

	// Merge [numHeads, valueDim] into one axis and unflatten query inner dims if needed.
	// attentionOutput: [batch, q_flat, heads, value_dim] -> [batch, <query_elements>, numHeads*valueDim]
	// This is a no-op reshape when there are no extra inner axes to unflatten.
	dims := slices.Clone(b.query.Shape().Dimensions)
	dims[len(dims)-1] = -1
	attentionOutput = Reshape(attentionOutput, dims...)

	// Unflatten coefficients back to original inner axis structure when needed.
	if needsUnflatten {
		// attentionCoefficients: [batch, q_flat, heads, k_flat] -> [batch, <query_inner>, heads, <key_inner>]
		coefDims := attentionCoefficients.Shape().Dimensions
		unflatCoef := []int{coefDims[0]}
		unflatCoef = append(unflatCoef, queryInnerDims...)
		unflatCoef = append(unflatCoef, coefDims[2]) // heads
		unflatCoef = append(unflatCoef, keyInnerDims...)
		attentionCoefficients = Reshape(attentionCoefficients, unflatCoef...)
	}

	// Final shape: `[batch, <query_elements>, outputDim]`
	attentionOutput = layers.Dense(b.ctx.In("output"), attentionOutput, b.useProjectionBias, b.outputDim)

	return attentionOutput, attentionCoefficients
}

// Done should be called after all optional settings are configured.
// It returns the attention output shaped `[batch_size, <query_elements>, output_dim]`,
// where `output_dim` can be configured by `SetOutputDim`.
// Use DoneWithCoefficients if the attention coefficients are also needed.
func (b *MultiHeadAttentionBuilder) Done() (output *Node) {
	output, _ = b.DoneWithCoefficients()
	return output
}

// buildAttentionShape returns the shape of the attention coefficients and mask, and sets it to b.attentionShape.
// attentionShape is `[batch, <query_elements>, num_heads, <key_elements>]`.
func (b *MultiHeadAttentionBuilder) buildAttentionShape() {
	// Determine effective key length (may be larger than current key when caching)
	var keyLen int
	if b.kvCacheShape.Ok() {
		// In cache mode, key length is the full cache size (kvCacheShape is [batch, heads, maxSeqLen, headDim])
		keyLen = b.kvCacheShape.Dimensions[2]
	} else {
		keyLen = b.key.Shape().Dimensions[1]
	}

	finalDims := make([]int, 2+b.innerQueryAxes+b.innerKeyAxes)
	pos := 0
	finalDims[pos] = b.key.Shape().Dimensions[0] // batch
	pos += 1
	copy(finalDims[pos:], b.query.Shape().Dimensions[1:1+b.innerQueryAxes]) // <query_elements>
	pos += b.innerQueryAxes
	finalDims[pos] = b.numHeads
	pos += 1
	// Use effective key length
	finalDims[pos] = keyLen
	// Copy remaining inner key axes if any (though typically innerKeyAxes == 1)
	if b.innerKeyAxes > 1 {
		copy(finalDims[pos+1:], b.key.Shape().Dimensions[2:1+b.innerKeyAxes])
	}

	b.attentionShape = shapes.Make(b.key.DType(), finalDims...)
}

// buildMask returns a normalized mask for shape `[batch, <query_elements>, num_heads, <key_elements>]`.
//
// When not using KV cache, the simple causal mask is handled by Core (via the causal flag),
// so buildMask only includes user-provided masks. When using KV cache, the position-aware
// causal mask is built here because Core doesn't know about cache positions.
func (b *MultiHeadAttentionBuilder) buildMask() (mask *Node) {
	// User-provided masks.
	if b.queryMask != nil || b.keyMask != nil {
		mask = b.buildMaskFromSplitMasks()
	} else if b.queryKeyMatrixMask != nil {
		mask = b.queryKeyMatrixMask
	}

	// KV cache causal mask: position-aware, built here since Core doesn't know about cache.
	if b.useCausalMask && b.kvCacheShape.Ok() {
		causalMask := b.buildCausalMask()
		if mask == nil {
			mask = causalMask
		} else {
			mask = LogicalAnd(mask, causalMask)
		}
	}
	return
}

// buildMaskFromSplitMasks creates cross mask from split queryMask and keyMask.
// The shape should be `[batch, <query_elements>, num_heads, <key_elements>]`.
func (b *MultiHeadAttentionBuilder) buildMaskFromSplitMasks() (mask *Node) {
	trueNode := Const(b.g, true)
	var keyMask *Node
	if b.keyMask == nil {
		// keyMask nil, create a skeleton (to be broadcast) keyMask filled with `true`.
		keyMask = Reshape(trueNode, xslices.SliceWithValue(b.attentionShape.Rank(), 1)...)
		keyMask = BroadcastToDims(keyMask, b.attentionShape.Dimensions...)
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
		queryMask = Reshape(trueNode, xslices.SliceWithValue(b.attentionShape.Rank(), 1)...)
		queryMask = BroadcastToDims(queryMask, b.attentionShape.Dimensions...)
	} else {
		// Expand dims at the end.
		queryMask = InsertAxes(b.queryMask, xslices.SliceWithValue(b.attentionShape.Rank()-b.queryMask.Rank(), -1)...)
		queryMask = BroadcastToDims(queryMask, b.attentionShape.Dimensions...)
	}
	return LogicalAnd(queryMask, keyMask)
}

// buildCausalMask creates a mask where queries can only attend to keys with "smaller index" than itself.
// When using KV cache (incremental generation), the mask accounts for the current position.
func (b *MultiHeadAttentionBuilder) buildCausalMask() (mask *Node) {
	queryShape := b.query.Shape()
	queryLen := queryShape.Dimensions[1]

	var keyLen int
	if b.kvCacheShape.Ok() {
		// In incremental mode: queries are at position [position:position+queryLen]
		// keys span the cache [0:actualCacheLen] (not maxSeqLen!)
		// We'll use the actual length from the cache (kvCacheShape is [batch, heads, maxSeqLen, headDim])
		keyLen = b.kvCacheShape.Dimensions[2] // Static shape for graph building
	} else {
		// Training mode: key and query have same length
		keyShape := b.key.Shape()
		keyLen = keyShape.Dimensions[1]
	}

	if b.kvCacheShape.Ok() {
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

		// Additional mask for valid cached positions: k < actualCacheLen
		// actualCacheLen is scalar, broadcast to [keyLen]
		keyIndices := Iota(b.g, shapes.Make(dtypes.Int32, keyLen), 0)
		actualLenBroadcast := BroadcastToShape(b.actualCacheLen, keyIndices.Shape())
		validMask := LessThan(keyIndices, actualLenBroadcast) // [keyLen]

		// Broadcast to [queryLen, keyLen] and combine with causal mask
		validMask = ExpandDims(validMask, 0) // [1, keyLen]
		validMask = BroadcastToShape(validMask, causalMask.Shape())

		mask = LogicalAnd(causalMask, validMask) // [queryLen, keyLen]
	} else {
		// Original training mode logic
		keyShape := b.key.Shape()
		if queryShape.Rank() != 3 || keyShape.Rank() != 3 {
			Panicf("MultiHeadAttention's UseCausalMask requires key and query to be rank-3,"+
				" instead got query.shape=%s and key.shape=%s", queryShape, keyShape)
		}
		if !reflect.DeepEqual(keyShape.Dimensions[:keyShape.Rank()-1], queryShape.Dimensions[:queryShape.Rank()-1]) {
			Panicf("MultiHeadAttention's UseCausalMask requires inner shapes of query and key be the same,"+
				" instead got query.shape=%s and key.shape=%s", queryShape, keyShape)
		}
		if queryLen != keyLen {
			Panicf("causal mask in training mode requires query and key lengths to match, got query=%d, key=%d",
				queryLen, keyLen)
		}
		mask = LowerTriangular(b.g, queryLen)
	}

	// Broadcast mask to target shape: [batch, <query_elements>, numHeads, <key_elements>]
	// mask is currently [queryLen, keyLen], need to add batch and numHeads dimensions
	// InsertAxes at beginning for batch dimension
	mask = ExpandDims(mask, 0) // [1, queryLen, keyLen]
	// Add dimension for numHeads at position 2 (after batch and query)
	mask = ExpandDims(mask, 2)                                   // [1, queryLen, 1, keyLen]
	mask = BroadcastToDims(mask, b.attentionShape.Dimensions...) // Broadcast to target dimensions

	// Combine with cache validity mask if using cache
	if b.kvCacheShape.Ok() {
		cacheValidMask := createKVCacheAttentionMask(b.g, b.kvCacheShape, b.position, queryLen, keyLen)
		// cacheValidMask shape: [batch, 1, queryLen, keyLen]
		// Remove the '1' dimension at position 1 and add numHeads dimension at position 2
		cacheValidMask = Squeeze(cacheValidMask, 1)    // [batch, queryLen, keyLen]
		cacheValidMask = ExpandDims(cacheValidMask, 2) // [batch, queryLen, 1, keyLen]
		cacheValidMask = BroadcastToDims(cacheValidMask, b.attentionShape.Dimensions...)
		mask = LogicalAnd(mask, cacheValidMask)
	}

	return
}

// SelfAttention is a convenience wrapper for MultiHeadAttention where query, key, and value
// are all the same tensor (self-attention).
//
// This is equivalent to calling MultiHeadAttention(ctx, x, x, x, numHeads, headDim).
// Returns a builder that can be further configured with methods like WithKVCache, WithRoPE,
// UseCausalMask, etc.
//
// Example usage for training:
//
//	output := attention.SelfAttention(ctx, x, numHeads, headDim).
//	    UseCausalMask().
//	    Dropout(0.1).
//	    Done()
//
// Example usage for generation with KV cache:
//
//	output := attention.SelfAttention(ctx, x, numHeads, headDim).
//	    WithKVCache(maxCacheLen, position).
//	    WithRoPE(10000.0).
//	    Done()
func SelfAttention(ctx *context.Context, x *Node, numHeads int, headDim int) *MultiHeadAttentionBuilder {
	return MultiHeadAttention(ctx, x, x, x, numHeads, headDim)
}
