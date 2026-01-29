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

	. "github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/initializers"
)

// KVCacheScopeName is the scope name used for KV cache variables in the context.
const KVCacheScopeName = "kv_cache"

// KV cache variable names
const (
	kvCacheKeyName      = "key"
	kvCacheValueName    = "value"
	kvCachePositionName = "position"
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
	positionSuffix := fmt.Sprintf("%s%s%s%s", context.ScopeSeparator, KVCacheScopeName, context.ScopeSeparator, kvCachePositionName)

	for v := range ctx.IterVariablesInScope() {
		scopeAndName := v.ScopeAndName()
		if strings.HasSuffix(scopeAndName, keySuffix) ||
			strings.HasSuffix(scopeAndName, valueSuffix) ||
			strings.HasSuffix(scopeAndName, positionSuffix) {
			// Reset to zero - tensors.FromShape creates a zero-initialized tensor
			v.SetValue(tensors.FromShape(v.Shape()))
		}
	}
}

// KVCacheGetVars returns the key, value, and position variables for the KV cache.
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
//   - positionVar: Variable storing current cache write position (shape [batchSize])
func KVCacheGetVars(ctx *context.Context, cacheShape shapes.Shape) (keyVar, valueVar, positionVar *context.Variable) {
	if cacheShape.Rank() != 4 {
		Panicf("KV cache shape must have rank 4, got %s", cacheShape)
	}

	batchSize := cacheShape.Dimensions[0]
	posShape := shapes.Make(dtypes.Int32, batchSize)

	ctx = ctx.In(KVCacheScopeName).WithInitializer(initializers.Zero)
	keyVar = ctx.VariableWithShape(kvCacheKeyName, cacheShape)
	valueVar = ctx.VariableWithShape(kvCacheValueName, cacheShape)
	positionVar = ctx.VariableWithShape(kvCachePositionName, posShape)
	return
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
	keyVar, valueVar, _ := KVCacheGetVars(ctx, cacheShape)

	// Get current cache state
	keyCache := keyVar.ValueGraph(g)
	valueCache := valueVar.ValueGraph(g)

	// Convert startPosition to int32 scalar
	position := ConvertDType(startPosition, dtypes.Int32)
	position = Reshape(position) // Ensure scalar shape

	// Get max cache size
	maxSeqLen := cacheShape.Dimensions[2]

	// Apply modulo for circular cache: write position = startPosition % maxSeqLen
	cacheWritePos := ModScalar(position, maxSeqLen)

	// Update the entire subsequence in one slice operation.
	// newKeysSlice/newValuesSlice have shape [batchSize, numHeads, updateSeqLen, headDim]
	// and we insert them at position [0, 0, cacheWritePos, 0] in the cache.
	batchIdx := Const(g, int32(0))
	headsIdx := Const(g, int32(0))
	dimIdx := Const(g, int32(0))
	keyCache = DynamicUpdateSlice(keyCache, newKeysSlice, []*Node{batchIdx, headsIdx, cacheWritePos, dimIdx})
	valueCache = DynamicUpdateSlice(valueCache, newValuesSlice, []*Node{batchIdx, headsIdx, cacheWritePos, dimIdx})

	// Update the cache variables
	keyVar.SetValueGraph(keyCache)
	valueVar.SetValueGraph(valueCache)
}

// getKVCache returns key/value caches and current position from the given context.
// cacheShape must be [batchSize, numHeads, maxSeqLen, headDim].
func getKVCache(ctx *context.Context, g *Graph, cacheShape shapes.Shape) (keys, values, currentPosition *Node) {
	keyVar, valueVar, posVar := KVCacheGetVars(ctx, cacheShape)

	// Return the full cache; the attention mechanism masks out positions >= currentPosition
	return keyVar.ValueGraph(g), valueVar.ValueGraph(g), posVar.ValueGraph(g)
}

// createKVCacheAttentionMask creates a mask for unfilled cache positions.
// Supports circular/rotating cache: when position >= maxSeqLen, all cache slots are valid.
//
// Parameters:
//   - ctx: Context containing the KV cache variables
//   - g: Graph for building the computation
//   - cacheShape: Shape [batchSize, numHeads, maxSeqLen, headDim]
//   - querySeqLen: Length of the query sequence
//   - keySeqLen: Length of the key sequence (typically maxSeqLen for cached attention)
//
// Returns:
//   - Boolean mask shaped [batchSize, 1, querySeqLen, keySeqLen] where True = attend, False = mask out
func createKVCacheAttentionMask(ctx *context.Context, g *Graph, cacheShape shapes.Shape, querySeqLen, keySeqLen int) *Node {
	_, _, posVar := KVCacheGetVars(ctx, cacheShape)

	batchSize := cacheShape.Dimensions[0]
	maxSeqLen := cacheShape.Dimensions[2]
	dtype := cacheShape.DType

	currentPosition := posVar.ValueGraph(g)
	currentPosition = ConvertDType(currentPosition, dtype)

	// Create position indices for keys: [0, 1, 2, ..., keySeqLen-1]
	// Shape: [keySeqLen]
	keyPositions := Iota(g, shapes.Make(dtype, keySeqLen), 0)

	// For rotating cache: if position >= maxSeqLen, all slots are filled
	// effectivePosition = min(currentPosition, maxSeqLen)
	maxSeqLenNode := Scalar(g, dtype, maxSeqLen)
	maxSeqLenNode = BroadcastToShape(maxSeqLenNode, currentPosition.Shape())
	effectivePosition := Min(currentPosition, maxSeqLenNode)

	// Expand for broadcasting
	// effectivePosition: [batch_size] -> [batch_size, 1, 1, 1]
	// keyPositions: [keySeqLen] -> [1, 1, 1, keySeqLen]
	effectivePosition = ExpandDims(ExpandDims(ExpandDims(effectivePosition, -1), -1), -1)
	keyPositions = ExpandDims(ExpandDims(ExpandDims(keyPositions, 0), 0), 0)

	// Create mask: True where key position < effectivePosition (i.e., slot is filled)
	mask := LessThan(keyPositions, effectivePosition)

	// Broadcast to [batch_size, 1, query_seq_len, key_seq_len]
	maskShape := shapes.Make(dtypes.Bool, batchSize, 1, querySeqLen, keySeqLen)
	mask = BroadcastToShape(mask, maskShape)

	return mask
}
