// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package attention implements multi-head attention and related utilities.
//
// The multi-head attention implements one attention layer, as described in the paper
// "Attention Is All You Need" (https://arxiv.org/abs/1706.03762), with some of the
// modern extensions.
//
// See MultiHeadAttention to create a builder for the the attention layer.
package attention

import (
	"math"
	"reflect"
	"slices"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention/pos"
	"github.com/gomlx/gomlx/pkg/ml/layers/regularizers"
	. "github.com/gomlx/gomlx/pkg/support/exceptions"
)

// This file contains all parts of the layers.MultiHeadAttention implementation.

// MultiHeadAttentionBuilder is a helper to build a multi-head-attention computation.
// Create it with MultiHeadAttention, set the desired parameters, and when all is set, call Done.
type MultiHeadAttentionBuilder struct {
	ctx               *context.Context
	g                 *Graph
	query, key, value *Node
	numHeads          int
	numKVHeads        int // 0 means same as numHeads (standard MHA).
	keyQueryDim       int
	valueDim          int
	outputDim         int

	innerKeyAxes, innerQueryAxes int
	attentionShape               shapes.Shape

	useProjectionBias bool
	dropoutRate       *Node

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

	// useQKVProjection replaces separate Dense Q/K/V projections with a single fused
	// QKVProjection (one large matmul + split). Only valid for self-attention.
	useQKVProjection bool

	withQKRMSNorm bool
	qkNormEpsilon float64

	useTransposedWeights bool
}

// MultiHeadAttention defines a multi-head attention layers, as described in [1], plus some modern extensions.
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
//
// Example:
//
//	positionEncoder := pos.NewRoPE(10000.0)
//	for layer := range numLayers {
//		ctx := ctx.In(fmt.Sprintf("layer_%d", layer))
//		// Use x for the source of query/key/values (self-attention).
//		x := attention.MultiHeadAttention(ctx, x, x, x, numHeads, headDim).
//		    WithPositionalEncoder(positionEncoder)
//		logits := attention.Done()
//		...  // normalization, residual connection, etc.
//	}
//
// [1] "Attention Is All You Need", https://arxiv.org/abs/1706.03762, by Ashish Vaswani, Noam Shazeer,
// Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.
func MultiHeadAttention(ctx *context.Context, query, key, value *Node, numHeads int, headDim int) *MultiHeadAttentionBuilder {
	g := query.Graph()

	queryShape := query.Shape()
	keyShape := key.Shape()
	valueShape := value.Shape()
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
	b := &MultiHeadAttentionBuilder{
		ctx:               ctx.In("MultiHeadAttention"),
		g:                 g,
		query:             query,
		key:               key,
		value:             value,
		outputDim:         valueShape.Dimensions[valueShape.Rank()-1],
		numHeads:          numHeads,
		valueDim:          headDim,
		keyQueryDim:       headDim,
		innerKeyAxes:      keyShape.Rank() - 2,
		innerQueryAxes:    queryShape.Rank() - 2,
		useProjectionBias: true,
		layout:            LayoutBSHD,
	}
	b.buildAttentionShape()
	return b
}

// effectiveNumKVHeads returns numKVHeads if set, or numHeads for standard MHA.
func (b *MultiHeadAttentionBuilder) effectiveNumKVHeads() int {
	if b.numKVHeads > 0 {
		return b.numKVHeads
	}
	return b.numHeads
}

// WithNumKVHeads sets the number of key/value heads for Grouped Query Attention (GQA).
//
// When numKVHeads < numHeads, each group of numHeads/numKVHeads query heads shares the
// same key/value head. This reduces the KV projection and KV cache size while retaining
// most of the quality of full multi-head attention.
//
// Special cases:
//   - numKVHeads == numHeads: standard multi-head attention (MHA), the default.
//   - numKVHeads == 1: multi-query attention (MQA), all query heads share one KV head.
//
// numHeads must be divisible by numKVHeads. This method is incompatible with
// UseQKVProjection when query == key == value and numKVHeads != numHeads,
// since the fused QKV path assumes equal head counts for Q and KV.
func (b *MultiHeadAttentionBuilder) WithNumKVHeads(numKVHeads int) *MultiHeadAttentionBuilder {
	if numKVHeads <= 0 {
		Panicf("MultiHeadAttention: numKVHeads (%d) must be positive", numKVHeads)
	}
	if b.numHeads%numKVHeads != 0 {
		Panicf("MultiHeadAttention: numHeads (%d) must be divisible by numKVHeads (%d)", b.numHeads, numKVHeads)
	}
	if b.useQKVProjection && numKVHeads != b.numHeads {
		Panicf("MultiHeadAttention: SetNumKVHeads(%d) is incompatible with UseQKVProjection (numHeads=%d); "+
			"use separate projections instead", numKVHeads, b.numHeads)
	}
	b.numKVHeads = numKVHeads
	return b
}

// WithKeyQueryDim allows finer configuration on the dimension of the projection used for the
// query/key pairs for each head. It defaults to the value given by `headDim`.
func (b *MultiHeadAttentionBuilder) WithKeyQueryDim(keyQueryDim int) *MultiHeadAttentionBuilder {
	b.keyQueryDim = keyQueryDim
	return b
}

// WithValueHeadDim allows finer configuration on the dimension of the projection used for the
// value for each head. It defaults to the value given by `headDim`.
func (b *MultiHeadAttentionBuilder) WithValueHeadDim(valueDim int) *MultiHeadAttentionBuilder {
	b.valueDim = valueDim
	return b
}

// WithOutputDim defines the output dimension of the final projection, from the flattened
// attention heads.
//
// It defaults to the value of the last dimension of `values` passed as input
// (`inputValueDim`).
func (b *MultiHeadAttentionBuilder) WithOutputDim(outputDim int) *MultiHeadAttentionBuilder {
	b.outputDim = outputDim
	return b
}

// UseProjectionBias defines whether to use a bias term on the final output projection.
// Default is true.
func (b *MultiHeadAttentionBuilder) UseProjectionBias(useProjectionBias bool) *MultiHeadAttentionBuilder {
	b.useProjectionBias = useProjectionBias
	return b
}

// WithDropout defines how much dropout to use in the attention coefficients calculation.
//
// If set to nil to disable it (the default). Values <= 0 are a no-op, and values > 1 are invalid.
//
// Notice the dropout rate is a *Node, since it can be set dynamically if needed.
func (b *MultiHeadAttentionBuilder) WithDropout(rate *Node) *MultiHeadAttentionBuilder {
	b.dropoutRate = rate
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
//		// Build generation graph:
//		attention := attention.MultiHeadAttention(ctx, embeddings, embeddings, embeddings, numHeads, headDim).
//		    WithKVCache(maxSeqLen, position).
//		    WithPositionalEncoder(pos.NewRoPE(10000.0))
//	 logits := attention.Done()
//
//		// Generate tokens:
//		// 1. First call with full prompt (e.g., 10 tokens), position=0
//		// 2. Each subsequent call with 1 new token, position=10, 11, 12, ...
//
//		// Before generating a new response, reset all caches in the model:
//		attention.KVCacheReset(ctx)
//
// The cache supports circular/rotating mode: when position exceeds maxSeqLen,
// new entries wrap around and overwrite the oldest entries. This allows efficient
// generation for sequences longer than maxSeqLen. Automatically enables causal masking.
func (b *MultiHeadAttentionBuilder) WithKVCache(maxSeqLen int, position *Node) *MultiHeadAttentionBuilder {
	// Derive cache shape from attention configuration.
	// KV cache stores with numKVHeads (not numHeads) to save memory with GQA.
	batchSize := b.query.Shape().Dimensions[0]
	dtype := b.query.DType()
	b.kvCacheShape = shapes.Make(dtype, batchSize, b.effectiveNumKVHeads(), maxSeqLen, b.keyQueryDim)
	b.position = position
	// When using KV cache, automatically enable causal mask for proper attention
	b.useCausalMask = true
	// Rebuild attention shape with cache dimensions
	b.buildAttentionShape()
	return b
}

// KVCacheShape returns the shape of the KV cache configured for this attention layer.
// Shape is [batchSize, numKVHeads, maxSeqLen, headDim] (numKVHeads equals numHeads unless SetNumKVHeads was called).
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

// WithPositionalEncoder sets a positional encoder to use.
//
// Note: Only pos.QKEncoder is actually called. A pos.PreEncoder is ignored, since they should be applied
// before the first MultiHeadAttention layer.
func (b *MultiHeadAttentionBuilder) WithPositionalEncoder(encoder pos.Encoder) *MultiHeadAttentionBuilder {
	b.positionalEncoder = encoder
	return b
}

// UseQKVProjection replaces the three separate Dense projections for query, key, and value
// with a single fused QKVProjection (one large matmul followed by split).
//
// This is only valid for self-attention where query, key, and value are the same node.
// Use SelfAttention() or pass the same node for all three to MultiHeadAttention().
//
// The fused weight is stored under the "qkv" context scope. This uses different variable
// names than the default separate projections (query/dense, key/dense, value/dense), so
// existing checkpoints using separate projections are not compatible.
func (b *MultiHeadAttentionBuilder) UseQKVProjection() *MultiHeadAttentionBuilder {
	if b.query != b.key || b.key != b.value {
		Panicf("MultiHeadAttention: UseQKVProjection requires self-attention (query, key, and value must be the same node)")
	}
	if b.keyQueryDim != b.valueDim {
		Panicf("MultiHeadAttention: UseQKVProjection requires keyQueryDim == valueDim (got %d != %d); "+
			"QKVProjection uses a single keyValueDim for both key and value projections",
			b.keyQueryDim, b.valueDim)
	}
	if b.numKVHeads > 0 && b.numKVHeads != b.numHeads {
		Panicf("MultiHeadAttention: UseQKVProjection is not supported with GQA (numKVHeads=%d != numHeads=%d); "+
			"use separate projections instead", b.numKVHeads, b.numHeads)
	}
	b.useQKVProjection = true
	return b
}

// UseTransposedWeights configures the builder to assume the query, key, value, and output projection weights
// are stored transposed (as [out_features, in_features]).
func (b *MultiHeadAttentionBuilder) UseTransposedWeights(transposed bool) *MultiHeadAttentionBuilder {
	b.useTransposedWeights = transposed
	return b
}

// WithQKRMSNorm applies RMSNorm to the query and key projections before attention (used by models like Gemma 2 and Gemma 3).
func (b *MultiHeadAttentionBuilder) WithQKRMSNorm(epsilon float64) *MultiHeadAttentionBuilder {
	b.withQKRMSNorm = true
	b.qkNormEpsilon = epsilon
	return b
}

// DoneWithCoefficients or Done should be called after all optional settings are configured.
// It returns both the attention output and the attention coefficients (matrix) used.
//
// Because coefficients are requested, the decomposed attention path is used (no fused
// SDPA op) so that the coefficient matrix is available. Use Done instead when coefficients
// are not needed to allow the fused path.
//
// `output` will be shaped `[batch_size, <query_elements>, output_dim]`, where `output_dim`
// can be configured by `SetOutputDim`.
//
// `coefficients` is shaped `[batch_size, <query_elements>, <num_heads>, <key_elements>]`
// with the attention weights (from 0 to 1).
func (b *MultiHeadAttentionBuilder) DoneWithCoefficients() (attentionOutput, attentionCoefficients *Node) {
	return b.doneInternal(true)
}

// Done should be called after all optional settings are configured.
// It returns the attention output shaped `[batch_size, <query_elements>, output_dim]`,
// where `output_dim` can be configured by `SetOutputDim`.
// Use DoneWithCoefficients if the attention coefficients are also needed.
func (b *MultiHeadAttentionBuilder) Done() (output *Node) {
	output, _ = b.doneInternal(false)
	return output
}

// doneInternal contains the shared implementation for Done and DoneWithCoefficients.
// When wantCoefficients is true the decomposed path is used so that coefficients are
// available; when false the fused SDPA op is attempted and coefficients is nil.
func (b *MultiHeadAttentionBuilder) doneInternal(wantCoefficients bool) (attentionOutput, attentionCoefficients *Node) {
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

	kvHeads := b.effectiveNumKVHeads()
	var projectedQuery, projectedKey, projectedValue *Node

	if b.useQKVProjection {
		projectedQuery, projectedKey, projectedValue = b.qkvProject(flatQuery)
	} else {
		projectedKey = b.dense(b.ctx.In("key"), flatKey, b.useProjectionBias, kvHeads, b.keyQueryDim)
		projectedQuery = b.dense(b.ctx.In("query"), flatQuery, b.useProjectionBias, b.numHeads, b.keyQueryDim)
		projectedValue = b.dense(b.ctx.In("value"), flatValue, b.useProjectionBias, kvHeads, b.valueDim)
	}

	if b.withQKRMSNorm {
		projectedQuery = layers.RMSNorm(b.ctx.In("query"), projectedQuery).WithEpsilon(b.qkNormEpsilon).WithNormalizationAxes(-1).WithScaleOffset(1.0).Done()
		projectedKey = layers.RMSNorm(b.ctx.In("key"), projectedKey).WithEpsilon(b.qkNormEpsilon).WithNormalizationAxes(-1).WithScaleOffset(1.0).Done()
	}

	// Apply positional encoding (e.g. RoPE) if enabled.
	// Applied before KV cache so that rotated embeddings are cached.
	// projectedQuery/Key shape: [batch, seq, heads, dim] (BSHD layout).
	if b.positionalEncoder != nil {
		qkEncoder, ok := b.positionalEncoder.(pos.QKEncoder)
		if ok {
			seqLen := projectedQuery.Shape().Dimensions[seqAxis]
			var posIndices *Node
			if b.position != nil {
				posIndices = pos.SequentialPositions(b.g, b.position, seqLen)
			} else {
				posIndices = pos.SequentialPositions(b.g, Const(b.g, int32(0)), seqLen)
			}
			projectedQuery, projectedKey = qkEncoder.EncodeQK(projectedQuery, projectedKey, posIndices, seqAxis)
		}
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
	mask := b.buildAttentionMask()

	// Flatten the mask to match the now-flat Q/K/V when inner axes > 1.
	if needsUnflatten && mask != nil {
		qFlat := projectedQuery.Shape().Dimensions[seqAxis]
		kFlat := projectedKey.Shape().Dimensions[seqAxis]
		mask = Reshape(mask, batchSize, qFlat, b.numHeads, kFlat)
	}

	scale := 1.0 / math.Sqrt(float64(b.keyQueryDim))
	// Pass causal to Core only when not using KV cache (Core builds a simple lower-triangular mask).
	// When using KV cache, the position-aware causal mask is already built in buildMask above.
	useCausalMask := b.useCausalMask && !b.kvCacheShape.Ok()
	attentionOutput, attentionCoefficients = Core(b.ctx, projectedQuery, projectedKey, projectedValue, scale, mask, b.dropoutRate, b.layout, useCausalMask, wantCoefficients)

	// Merge [numHeads, valueDim] into one axis and unflatten query inner dims if needed.
	// attentionOutput: [batch, q_flat, heads, value_dim] -> [batch, <query_elements>, numHeads*valueDim]
	// This is a no-op reshape when there are no extra inner axes to unflatten.
	dims := slices.Clone(b.query.Shape().Dimensions)
	dims[len(dims)-1] = -1
	attentionOutput = Reshape(attentionOutput, dims...)

	// Unflatten coefficients back to original inner axis structure when needed.
	if wantCoefficients && needsUnflatten {
		// attentionCoefficients: [batch, q_flat, heads, k_flat] -> [batch, <query_inner>, heads, <key_inner>]
		coefDims := attentionCoefficients.Shape().Dimensions
		unflatCoef := []int{coefDims[0]}
		unflatCoef = append(unflatCoef, queryInnerDims...)
		unflatCoef = append(unflatCoef, coefDims[2]) // heads
		unflatCoef = append(unflatCoef, keyInnerDims...)
		attentionCoefficients = Reshape(attentionCoefficients, unflatCoef...)
	}

	// Final shape: `[batch, <query_elements>, outputDim]`
	attentionOutput = b.dense(b.ctx.In("output"), attentionOutput, b.useProjectionBias, b.outputDim)

	return attentionOutput, attentionCoefficients
}

// dense executes a Dense projection layer.
// If UseTransposedWeights() has been called, it uses Einsum to compute the projection
// using PyTorch's default transposed weight schema ([outDim, inDim]).
func (b *MultiHeadAttentionBuilder) dense(ctx *context.Context, x *Node, useBias bool, outputDims ...int) *Node {
	if !b.useTransposedWeights {
		return layers.Dense(ctx, x, useBias, outputDims...)
	}
	ctx = ctx.In("dense")
	g := x.Graph()
	inDim := x.Shape().Dim(-1)
	outDim := 1
	for _, d := range outputDims {
		outDim *= d
	}
	wVar := ctx.VariableWithShape("weights", shapes.Make(x.DType(), outDim, inDim))
	w := wVar.ValueGraph(g)

	y := DotGeneral(x, []int{-1}, nil, w, []int{1}, nil)

	if useBias {
		bVar := ctx.VariableWithShape("biases", shapes.Make(x.DType(), outDim))
		y = Add(y, bVar.ValueGraph(g))
	}

	if len(outputDims) > 1 {
		newDims := make([]int, x.Rank()-1+len(outputDims))
		copy(newDims, x.Shape().Dimensions[:x.Rank()-1])
		copy(newDims[x.Rank()-1:], outputDims)
		y = Reshape(y, newDims...)
	}
	return y
}

// qkvProject performs a fused QKV projection using a single weight matrix.
// x has shape [batch, seq, inFeatures]. Returns projectedQuery, projectedKey, projectedValue
// each shaped [batch, seq, numHeads, headDim] matching the BSHD layout.
func (b *MultiHeadAttentionBuilder) qkvProject(x *Node) (projectedQuery, projectedKey, projectedValue *Node) {
	g := x.Graph()
	qkvCtx := b.ctx.In("qkv")
	dtype := x.DType()
	inFeatures := x.Shape().Dim(-1)
	queryDim := b.numHeads * b.keyQueryDim
	keyValueDim := b.numHeads * b.valueDim

	// Single fused weight: [inFeatures, queryDim + 2*keyValueDim].
	wQKVVar := qkvCtx.VariableWithShape("weights_qkv", shapes.Make(dtype, inFeatures, queryDim+2*keyValueDim))
	if regularizer := regularizers.FromContext(qkvCtx); regularizer != nil {
		regularizer(qkvCtx, g, wQKVVar)
	}
	wQKV := wQKVVar.ValueGraph(g)

	// Separate biases for Q, K, V (always enabled, matching the separate Dense path
	// which hardcodes useBias=true for Q/K/V projections).
	biasQ := qkvCtx.VariableWithShape("biases_q", shapes.Make(dtype, queryDim)).ValueGraph(g)
	biasK := qkvCtx.VariableWithShape("biases_k", shapes.Make(dtype, keyValueDim)).ValueGraph(g)
	biasV := qkvCtx.VariableWithShape("biases_v", shapes.Make(dtype, keyValueDim)).ValueGraph(g)

	// QKVProjection expects [..., inFeatures] and returns [..., dim] flat outputs.
	// With x=[batch, seq, inFeatures], outputs are [batch, seq, queryDim] etc.
	q, k, v := QKVProjection(x, wQKV, biasQ, biasK, biasV, queryDim, keyValueDim)

	// Reshape from [batch, seq, numHeads*headDim] to [batch, seq, numHeads, headDim].
	xDims := x.Shape().Dimensions
	batch := xDims[0]
	seqLen := xDims[1]
	projectedQuery = Reshape(q, batch, seqLen, b.numHeads, b.keyQueryDim)
	projectedKey = Reshape(k, batch, seqLen, b.numHeads, b.keyQueryDim)
	projectedValue = Reshape(v, batch, seqLen, b.numHeads, b.valueDim)
	return
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
