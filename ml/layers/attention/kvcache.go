// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"fmt"
	"strings"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/initializer"
	. "github.com/gomlx/gomlx/support/exceptions"
)

// KVCacheScopeName is the scope name used for KV cache variables in the model.
const KVCacheScopeName = "kv_cache"

// KV cache variable names
const (
	kvCacheKeyName   = "key"
	kvCacheValueName = "value"
)

// KVCacheReset clears all KV caches under the given scope scope.
// Call this outside of graph execution before starting a new generation.
// This resets the key cache, value cache, and position for all attention layers
// under the provided model.
//
// Parameters:
//   - scope: Scope under which to find and reset KV cache variables.
//     Pass the root scope to reset all caches in the model, or a
//     layer-specific scope to reset only that layer's cache.
//
// Example:
//
//	// Reset all KV caches in the model before generating a new response:
//	attention.KVCacheReset(scope)
//
//	// Or reset only a specific layer's cache:
//	attention.KVCacheReset(layerScope)
func KVCacheReset(scope *model.Scope) {
	keySuffix := fmt.Sprintf("%s%s%s%s", model.ScopeSeparator, KVCacheScopeName, model.ScopeSeparator, kvCacheKeyName)
	valueSuffix := fmt.Sprintf("%s%s%s%s", model.ScopeSeparator, KVCacheScopeName, model.ScopeSeparator, kvCacheValueName)

	for v := range scope.IterVariables() {
		scopeAndName := v.Path()
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
//   - scope: Scope for storing/retrieving cache variables
//   - cacheShape: Shape [batchSize, numHeads, maxSeqLen, headDim]
//
// Returns:
//   - keyVar: Variable storing cached key projections
//   - valueVar: Variable storing cached value projections
func KVCacheGetVars(scope *model.Scope, cacheShape shapes.Shape) (keyVar, valueVar *model.Variable) {
	if cacheShape.Rank() != 4 {
		Panicf("KV cache shape must have rank 4, got %s", cacheShape)
	}

	scope = scope.At(KVCacheScopeName).WithInitializer(initializer.Zero)
	keyVar = scope.VariableWithShape(kvCacheKeyName, cacheShape)
	valueVar = scope.VariableWithShape(kvCacheValueName, cacheShape)
	return
}

// KVCacheUpdate writes new keys/values to the cache at the specified position.
// This is used during incremental generation to accumulate key/value projections.
// Supports rolling (circular) cache: when position exceeds maxSeqLen, it wraps around.
// This is a low-level function; most users should use WithKVCache on the attention builder.
//
// Parameters:
//   - scope: Scope for storing/retrieving cache variables
//   - g: Graph for building the computation
//   - cacheShape: Shape [batchSize, numHeads, maxSeqLen, headDim]
//   - startPosition: Scalar Node (int32) indicating the absolute position in the sequence
//   - newKeysSlice: New key projections [batchSize, numHeads, seqLen, headDim]
//   - newValuesSlice: New value projections [batchSize, numHeads, seqLen, headDim]
//
// Example:
//
//	// During generation, update each layer's cache:
//	for _, layerScope := range layerScopes {
//	    KVCacheUpdate(layerScope, g, cacheShape, position, newKeys, newValues)
//	}
//	// Then increment position for next iteration (managed by caller)
func KVCacheUpdate(scope *model.Scope, g *Graph, cacheShape shapes.Shape, startPosition *Node, newKeysSlice, newValuesSlice *Node) {
	keyVar, valueVar := KVCacheGetVars(scope, cacheShape)

	// Get current cache state
	keyCache := keyVar.NodeValue(g)
	valueCache := valueVar.NodeValue(g)

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

	if updateSeqLen == 1 {
		// Common case: single token update, no wrap-around possible
		keyCache = DynamicUpdateSlice(keyCache, newKeysSlice, []*Node{batchIdx, headsIdx, cacheWritePos, dimIdx})
		valueCache = DynamicUpdateSlice(valueCache, newValuesSlice, []*Node{batchIdx, headsIdx, cacheWritePos, dimIdx})
	} else {
		// Multi-token update: handle wrap-around by writing each token individually
		// with its wrapped position. For example, with maxSeqLen=4, cacheWritePos=3, updateSeqLen=2:
		//   - Token 0 writes to position 3
		//   - Token 1 writes to position 0 (wrapped)
		for i := range updateSeqLen {
			// Calculate wrapped position for this token
			tokenPos := AddScalar(cacheWritePos, i)
			wrappedPos := ModScalar(tokenPos, maxSeqLen)

			// Extract single token slice: [batch, heads, 1, dim]
			tokenKeys := Slice(newKeysSlice, AxisRange(), AxisRange(), AxisElem(i), AxisRange())
			tokenValues := Slice(newValuesSlice, AxisRange(), AxisRange(), AxisElem(i), AxisRange())

			keyCache = DynamicUpdateSlice(keyCache, tokenKeys, []*Node{batchIdx, headsIdx, wrappedPos, dimIdx})
			valueCache = DynamicUpdateSlice(valueCache, tokenValues, []*Node{batchIdx, headsIdx, wrappedPos, dimIdx})
		}
	}

	// Update the cache variables
	keyVar.SetNodeValue(keyCache)
	valueVar.SetNodeValue(valueCache)
}

// getKVCache returns key/value caches from the given model.
// cacheShape must be [batchSize, numHeads, maxSeqLen, headDim].
func getKVCache(scope *model.Scope, g *Graph, cacheShape shapes.Shape) (keys, values *Node) {
	keyVar, valueVar := KVCacheGetVars(scope, cacheShape)
	return keyVar.NodeValue(g), valueVar.NodeValue(g)
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
	dtype := cacheShape.DType

	currentPosition = ConvertDType(currentPosition, dtype)
	currentPosition = Reshape(currentPosition) // Ensure scalar

	// Create position indices for keys: [0, 1, 2, ..., keySeqLen-1]
	// Shape: [keySeqLen]
	keyPositions := Iota(g, shapes.Make(dtype, keySeqLen), 0)

	// For rotating cache: if position >= maxSeqLen, all slots are filled
	// effectivePosition = min(currentPosition, maxSeqLen)
	effectivePosition := MinScalar(currentPosition, maxSeqLen)

	// Create mask: True where key position < effectivePosition (i.e., slot is filled)
	// mask shape: [keySeqLen]
	mask := LessThan(keyPositions, effectivePosition)

	// Broadcast to [batch_size, 1, query_seq_len, key_seq_len]
	return BroadcastPrefix(mask, batchSize, 1, querySeqLen)
}
