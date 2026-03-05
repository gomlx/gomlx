/*
 *	Copyright 2024 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package attention

import (
	"fmt"
	"strings"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/initializers"
	. "github.com/gomlx/gomlx/pkg/support/exceptions"
)

// KVCacheScopeName is the scope name used for KV cache variables in the context.
const KVCacheScopeName = "kv_cache"

// KV cache variable names
const (
	kvCacheKeyName   = "key"
	kvCacheValueName = "value"
)

// KVCacheReset clears all KV caches under the given context scope.
// Call this outside of graph execution before starting a new generation.
// This resets the key cache, value cache, and position for all attention layers
// under the provided context.
//
// Parameters:
//   - ctx: Context scope under which to find and reset KV cache variables.
//     Pass the root context to reset all caches in the model, or a
//     layer-specific context to reset only that layer's cache.
//
// Example:
//
//	// Reset all KV caches in the model before generating a new response:
//	attention.KVCacheReset(ctx)
//
//	// Or reset only a specific layer's cache:
//	attention.KVCacheReset(layerCtx)
func KVCacheReset(ctx *context.Context) {
	keySuffix := fmt.Sprintf("%s%s%s%s", context.ScopeSeparator, KVCacheScopeName, context.ScopeSeparator, kvCacheKeyName)
	valueSuffix := fmt.Sprintf("%s%s%s%s", context.ScopeSeparator, KVCacheScopeName, context.ScopeSeparator, kvCacheValueName)

	for v := range ctx.IterVariablesInScope() {
		scopeAndName := v.ScopeAndName()
		if strings.HasSuffix(scopeAndName, keySuffix) ||
			strings.HasSuffix(scopeAndName, valueSuffix) {
			// Reset to zero - tensors.FromShape creates a zero-initialized tensor
			v.SetValue(tensors.FromShape(v.Shape()))
		}
	}
}

// KVCacheGetVars returns the key and value variables for the KV cache.
// Variables are created on first access with zero-initialization.
// This is a low-level function; most users should use WithKVCache on the attention builder.
//
// Parameters:
//   - ctx: Context for storing/retrieving cache variables
//   - cacheShape: Shape [batchSize, numHeads, maxSeqLen, headDim]
//
// Returns:
//   - keyVar: Variable storing cached key projections
//   - valueVar: Variable storing cached value projections
func KVCacheGetVars(ctx *context.Context, cacheShape shapes.Shape) (keyVar, valueVar *context.Variable) {
	if cacheShape.Rank() != 4 {
		Panicf("KV cache shape must have rank 4, got %s", cacheShape)
	}

	ctx = ctx.In(KVCacheScopeName).WithInitializer(initializers.Zero)
	keyVar = ctx.VariableWithShape(kvCacheKeyName, cacheShape)
	valueVar = ctx.VariableWithShape(kvCacheValueName, cacheShape)
	return
}

// kvCacheWriteElement writes one batch element's keys/values to the cache using
// DynamicUpdateSlice. For multi-token updates, each token is written individually
// with wrap-around handling.
//
// Parameters:
//   - keyCache, valueCache: [batchSize, numHeads, maxSeqLen, headDim]
//   - batchIdx: scalar int32 — which batch element to update
//   - headsIdx, dimIdx: scalar int32(0) — zero start indices for heads and dim axes
//   - wrappedPos: scalar int32 — start position in cache (already mod maxSeqLen)
//   - newKeys, newValues: [1, numHeads, updateSeqLen, headDim]
//   - maxSeqLen: cache sequence length (for modulo wrapping in multi-token case)
//   - updateSeqLen: number of tokens being written
func kvCacheWriteElement(
	keyCache, valueCache *Node,
	batchIdx, headsIdx, dimIdx, wrappedPos *Node,
	newKeys, newValues *Node,
	maxSeqLen, updateSeqLen int,
) (*Node, *Node) {
	if updateSeqLen == 1 {
		keyCache = DynamicUpdateSlice(keyCache, newKeys, []*Node{batchIdx, headsIdx, wrappedPos, dimIdx})
		valueCache = DynamicUpdateSlice(valueCache, newValues, []*Node{batchIdx, headsIdx, wrappedPos, dimIdx})
	} else {
		for i := range updateSeqLen {
			tokenPos := AddScalar(wrappedPos, i)
			tokenWrappedPos := ModScalar(tokenPos, maxSeqLen)
			tokenKeys := Slice(newKeys, AxisRange(), AxisRange(), AxisElem(i), AxisRange())
			tokenValues := Slice(newValues, AxisRange(), AxisRange(), AxisElem(i), AxisRange())
			keyCache = DynamicUpdateSlice(keyCache, tokenKeys, []*Node{batchIdx, headsIdx, tokenWrappedPos, dimIdx})
			valueCache = DynamicUpdateSlice(valueCache, tokenValues, []*Node{batchIdx, headsIdx, tokenWrappedPos, dimIdx})
		}
	}
	return keyCache, valueCache
}

// KVCacheUpdate writes new keys/values to the cache at the specified position.
// This is used during incremental generation to accumulate key/value projections.
// Supports rolling (circular) cache: when position exceeds maxSeqLen, it wraps around.
// This is a low-level function; most users should use WithKVCache on the attention builder.
//
// Parameters:
//   - ctx: Context for storing/retrieving cache variables
//   - g: Graph for building the computation
//   - cacheShape: Shape [batchSize, numHeads, maxSeqLen, headDim]
//   - startPosition: Scalar Node (int32) indicating the absolute position in the sequence
//   - newKeysSlice: New key projections [batchSize, numHeads, seqLen, headDim]
//   - newValuesSlice: New value projections [batchSize, numHeads, seqLen, headDim]
//
// Example:
//
//	// During generation, update each layer's cache:
//	for _, layerCtx := range layerContexts {
//	    KVCacheUpdate(layerCtx, g, cacheShape, position, newKeys, newValues)
//	}
//	// Then increment position for next iteration (managed by caller)
func KVCacheUpdate(ctx *context.Context, g *Graph, cacheShape shapes.Shape, startPosition *Node, newKeysSlice, newValuesSlice *Node) {
	keyVar, valueVar := KVCacheGetVars(ctx, cacheShape)

	// Get current cache state
	keyCache := keyVar.ValueGraph(g)
	valueCache := valueVar.ValueGraph(g)

	// Convert startPosition to int32 scalar
	position := Reshape(ConvertDType(startPosition, dtypes.Int32)) // Ensure scalar shape

	// Get dimensions
	maxSeqLen := cacheShape.Dimensions[2]
	updateSeqLen := newKeysSlice.Shape().Dimensions[2]

	// Apply modulo for circular cache: write position = startPosition % maxSeqLen
	cacheWritePos := ModScalar(position, maxSeqLen)

	batchIdx := Const(g, int32(0))
	headsIdx := Const(g, int32(0))
	dimIdx := Const(g, int32(0))

	keyCache, valueCache = kvCacheWriteElement(
		keyCache, valueCache,
		batchIdx, headsIdx, dimIdx, cacheWritePos,
		newKeysSlice, newValuesSlice,
		maxSeqLen, updateSeqLen,
	)

	// Update the cache variables
	keyVar.SetValueGraph(keyCache)
	valueVar.SetValueGraph(valueCache)
}

// getKVCache returns key/value caches from the given context.
// cacheShape must be [batchSize, numHeads, maxSeqLen, headDim].
func getKVCache(ctx *context.Context, g *Graph, cacheShape shapes.Shape) (keys, values *Node) {
	keyVar, valueVar := KVCacheGetVars(ctx, cacheShape)
	return keyVar.ValueGraph(g), valueVar.ValueGraph(g)
}

// createKVCacheAttentionMask creates a mask for unfilled cache positions.
// Supports circular/rotating cache: when position >= maxSeqLen, all cache slots are valid.
//
// Parameters:
//   - g: Graph for building the computation
//   - cacheShape: Shape [batchSize, numHeads, maxSeqLen, headDim]
//   - currentPosition: Scalar Node indicating the current position in the sequence
//   - querySeqLen: Length of the query sequence
//   - keySeqLen: Length of the key sequence (typically maxSeqLen for cached attention)
//
// Returns:
//   - Boolean mask shaped [batchSize, 1, querySeqLen, keySeqLen] where True = attend, False = mask out
func createKVCacheAttentionMask(g *Graph, cacheShape shapes.Shape, currentPosition *Node, querySeqLen, keySeqLen int) *Node {
	batchSize := cacheShape.Dimensions[0]
	maxSeqLen := cacheShape.Dimensions[2]

	// Use Int32 for position arithmetic regardless of cache DType.
	// BFloat16 caches would lose integer precision past ~256, corrupting masks.
	currentPosition = ConvertDType(currentPosition, dtypes.Int32)
	currentPosition = Reshape(currentPosition) // Ensure scalar

	// Create position indices for keys: [0, 1, 2, ..., keySeqLen-1]
	// Shape: [keySeqLen]
	keyPositions := Iota(g, shapes.Make(dtypes.Int32, keySeqLen), 0)

	// For rotating cache: if position >= maxSeqLen, all slots are filled
	// effectivePosition = min(currentPosition, maxSeqLen)
	effectivePosition := MinScalar(currentPosition, maxSeqLen)

	// Create mask: True where key position < effectivePosition (i.e., slot is filled)
	// mask shape: [keySeqLen]
	mask := LessThan(keyPositions, effectivePosition)

	// Broadcast to [batch_size, 1, query_seq_len, key_seq_len]
	return BroadcastPrefix(mask, batchSize, 1, querySeqLen)
}

// BatchedKVCacheUpdate writes new keys/values to the cache where each batch element
// has an independent position. This is the key primitive for continuous batching:
// multiple requests at different generation stages share one forward pass.
//
// Parameters:
//   - ctx: Context for storing/retrieving cache variables
//   - g: Graph for building the computation
//   - cacheShape: Shape [batchSize, numHeads, maxSeqLen, headDim]
//   - positions: [batchSize] int32 tensor — per-element absolute position
//   - newKeysSlice: New key projections [batchSize, numHeads, seqLen, headDim]
//   - newValuesSlice: New value projections [batchSize, numHeads, seqLen, headDim]
func BatchedKVCacheUpdate(ctx *context.Context, g *Graph, cacheShape shapes.Shape, positions *Node, newKeysSlice, newValuesSlice *Node) {
	keyVar, valueVar := KVCacheGetVars(ctx, cacheShape)

	keyCache := keyVar.ValueGraph(g)
	valueCache := valueVar.ValueGraph(g)

	batchSize := cacheShape.Dimensions[0]
	maxSeqLen := cacheShape.Dimensions[2]
	updateSeqLen := newKeysSlice.Shape().Dimensions[2]

	// Convert positions to int32: [batchSize]
	positionsI32 := ConvertDType(positions, dtypes.Int32)

	headsIdx := Const(g, int32(0))
	dimIdx := Const(g, int32(0))

	// Process each batch element independently since DynamicUpdateSlice
	// takes scalar start indices.
	for b := range batchSize {
		batchIdxNode := Const(g, int32(b))

		// Get this batch element's position (scalar).
		batchPos := Reshape(Slice(positionsI32, AxisElem(b)))
		wrappedPos := ModScalar(batchPos, maxSeqLen)

		// Extract this batch element's new keys/values: [1, numHeads, seqLen, headDim]
		batchKeys := Slice(newKeysSlice, AxisElem(b), AxisRange(), AxisRange(), AxisRange())
		batchValues := Slice(newValuesSlice, AxisElem(b), AxisRange(), AxisRange(), AxisRange())

		keyCache, valueCache = kvCacheWriteElement(
			keyCache, valueCache,
			batchIdxNode, headsIdx, dimIdx, wrappedPos,
			batchKeys, batchValues,
			maxSeqLen, updateSeqLen,
		)
	}

	keyVar.SetValueGraph(keyCache)
	valueVar.SetValueGraph(valueCache)
}

// createBatchedKVCacheAttentionMask creates a validity mask where each batch element
// has an independent effective position. Used with BatchedKVCacheUpdate for
// continuous batching.
//
// Parameters:
//   - g: Graph for building the computation
//   - cacheShape: Shape [batchSize, numHeads, maxSeqLen, headDim]
//   - positions: [batchSize] int32 tensor — per-element current position
//   - querySeqLen: Length of the query sequence
//   - keySeqLen: Length of the key sequence (typically maxSeqLen)
//
// Returns:
//   - Boolean mask shaped [batchSize, 1, querySeqLen, keySeqLen]
func createBatchedKVCacheAttentionMask(g *Graph, cacheShape shapes.Shape, positions *Node, querySeqLen, keySeqLen int) *Node {
	maxSeqLen := cacheShape.Dimensions[2]

	// Use Int32 for position arithmetic regardless of cache DType.
	// BFloat16 caches would lose integer precision past ~256, corrupting masks.
	posI32 := ConvertDType(positions, dtypes.Int32)

	// effectivePositions = min(positions, maxSeqLen): [batchSize]
	effectivePositions := MinScalar(posI32, maxSeqLen)

	// Key indices: [keySeqLen]
	keyPositions := Iota(g, shapes.Make(dtypes.Int32, keySeqLen), 0)

	// Broadcast for comparison:
	//   effectivePositions: [batchSize] → [batchSize, 1]
	//   keyPositions:       [keySeqLen] → [1, keySeqLen]
	// Result: [batchSize, keySeqLen]
	effectivePositions = ExpandDims(effectivePositions, -1) // [batchSize, 1]
	keyPositions = ExpandDims(keyPositions, 0)              // [1, keySeqLen]
	mask := LessThan(keyPositions, effectivePositions)       // [batchSize, keySeqLen]

	// Reshape to [batchSize, 1, 1, keySeqLen] then broadcast to [batchSize, 1, querySeqLen, keySeqLen]
	mask = ExpandDims(mask, 1)
	mask = ExpandDims(mask, 2)
	mask = BroadcastToShape(mask, shapes.Make(dtypes.Bool, mask.Shape().Dimensions[0], 1, querySeqLen, keySeqLen))
	return mask
}
