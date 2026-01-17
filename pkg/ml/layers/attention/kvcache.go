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
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

// KVCache stores past key/value states for autoregressive attention.
// Buffers: [batch, heads, max_len, head_dim].
type KVCache struct {
	ctx         *context.Context
	reuseCtx    *context.Context // Context in reuse mode for variable access
	scope       string
	batchSize   int
	numHeads    int
	maxSeqLen   int
	headDim     int
	dtype       dtypes.DType
	initialized bool
}

// NewKVCache creates pre-allocated KV buffers.
func NewKVCache(ctx *context.Context, scope string, batchSize, numHeads, maxSeqLen, headDim int, dtype dtypes.DType) *KVCache {
	scopedCtx := ctx.In(scope)
	// CRITICAL FIX: If context is already in reuse mode, don't call Reuse() again!
	// Calling Reuse() on an already-reused context creates a DIFFERENT reuse scope,
	// which prevents variables from persisting across graph executions.
	var reuseCtx *context.Context
	if scopedCtx.IsReuse() {
		// Bug fix: Don't call Reuse() on a context that's already in reuse mode.
		reuseCtx = scopedCtx
	} else {
		reuseCtx = scopedCtx.Reuse()
	}
	return &KVCache{
		ctx:       scopedCtx,
		reuseCtx:  reuseCtx,
		scope:     scope,
		batchSize: batchSize,
		numHeads:  numHeads,
		maxSeqLen: maxSeqLen,
		headDim:   headDim,
		dtype:     dtype,
	}
}

// Initialize creates variables (idempotent).
func (c *KVCache) Initialize(g *Graph) {
	if c.initialized {
		return
	}

	cacheShape := shapes.Make(c.dtype, c.batchSize, c.numHeads, c.maxSeqLen, c.headDim)

	// BUG FIX: Use reuseCtx consistently! Variables must be created and accessed with the same context.
	// Previously used c.ctx here but c.reuseCtx in Update/Get, causing scope mismatch!
	initCtx := c.reuseCtx.Checked(false)

	// Create key and value cache variables, initialized to zeros
	initCtx.VariableWithShape("key_cache", cacheShape)
	initCtx.VariableWithShape("value_cache", cacheShape)

	// Create position variable to track how many positions are filled
	// Shape: [batch_size] - tracks position for each batch element
	posShape := shapes.Make(dtypes.Int32, c.batchSize)
	posVar := initCtx.VariableWithShape("cache_position", posShape)
	// Initialize to zeros
	posVar.SetValueGraph(ZerosLike(posVar.ValueGraph(g)))

	c.initialized = true
}

// Reset sets all positions to 0.
func (c *KVCache) Reset(g *Graph) {
	c.Initialize(g)

	// Reset positions to zero
	posVar := c.reuseCtx.VariableWithShape("cache_position", shapes.Make(dtypes.Int32, c.batchSize))
	posVar.SetValueGraph(ZerosLike(posVar.ValueGraph(g)))
}

// Update appends new keys/values; returns updated position.
func (c *KVCache) Update(g *Graph, newKeys, newValues *Node) *Node {
	c.Initialize(g)

	// Get current cache state
	keyCache := c.reuseCtx.VariableWithShape("key_cache", shapes.Make(c.dtype, c.batchSize, c.numHeads, c.maxSeqLen, c.headDim)).ValueGraph(g)
	valueCache := c.reuseCtx.VariableWithShape("value_cache", shapes.Make(c.dtype, c.batchSize, c.numHeads, c.maxSeqLen, c.headDim)).ValueGraph(g)
	position := c.reuseCtx.VariableWithShape("cache_position", shapes.Make(dtypes.Int32, c.batchSize)).ValueGraph(g)

	// Get sequence length of new keys/values
	newSeqLen := newKeys.Shape().Dimensions[2]

	// For each position in the new sequence, update the cache
	// We'll update positions [current_pos : current_pos + newSeqLen]
	for i := 0; i < newSeqLen; i++ {
		// Extract single position from new keys/values
		// Shape: [batch_size, num_heads, head_dim]
		singleKey := Slice(newKeys, AxisRange(), AxisRange(), AxisRange(i, i+1), AxisRange())
		singleKey = Squeeze(singleKey, 2) // Remove seq dimension
		singleValue := Slice(newValues, AxisRange(), AxisRange(), AxisRange(i, i+1), AxisRange())
		singleValue = Squeeze(singleValue, 2)

		// Expand to match cache shape for dynamic update
		// We need shape [batch_size, num_heads, 1, head_dim]
		singleKey = ExpandDims(singleKey, 2)
		singleValue = ExpandDims(singleValue, 2)

		// Create start indices for dynamic slice update
		// All indices must be scalars (0D tensors)
		batchIdx := Const(g, int32(0)) // Always start at batch 0 (updates all batches)
		headsIdx := Const(g, int32(0)) // Always start at head 0
		dimIdx := Const(g, int32(0))   // Always start at dim 0

		// Position index varies per batch element - for now, use first batch element's position
		posIdx := Squeeze(Slice(position, AxisRange(0, 1)), 0) // Get first element and squeeze to scalar
		posIdx = AddScalar(posIdx, float64(i))
		posIdx = ConvertDType(posIdx, dtypes.Int32)

		// Update cache using DynamicUpdateSlice
		keyCache = DynamicUpdateSlice(keyCache, singleKey, []*Node{batchIdx, headsIdx, posIdx, dimIdx})
		valueCache = DynamicUpdateSlice(valueCache, singleValue, []*Node{batchIdx, headsIdx, posIdx, dimIdx})
	}

	// Update the cache variables
	c.reuseCtx.VariableWithShape("key_cache", keyCache.Shape()).SetValueGraph(keyCache)
	c.reuseCtx.VariableWithShape("value_cache", valueCache.Shape()).SetValueGraph(valueCache)

	// Update position
	updatedPosition := AddScalar(position, float64(newSeqLen))
	updatedPosition = ConvertDType(updatedPosition, dtypes.Int32)
	c.reuseCtx.VariableWithShape("cache_position", updatedPosition.Shape()).SetValueGraph(updatedPosition)

	return updatedPosition
}

// Get returns key/value caches and current position.
func (c *KVCache) Get(g *Graph) (keys, values, currentPosition *Node) {
	c.Initialize(g)

	keyCache := c.reuseCtx.VariableWithShape("key_cache", shapes.Make(c.dtype, c.batchSize, c.numHeads, c.maxSeqLen, c.headDim)).ValueGraph(g)
	valueCache := c.reuseCtx.VariableWithShape("value_cache", shapes.Make(c.dtype, c.batchSize, c.numHeads, c.maxSeqLen, c.headDim)).ValueGraph(g)
	currentPosition = c.reuseCtx.VariableWithShape("cache_position", shapes.Make(dtypes.Int32, c.batchSize)).ValueGraph(g)

	// For simplicity, we return the full cache
	// The attention mechanism should mask out positions >= currentPosition
	// In practice, we could use DynamicSlice to return only filled positions
	return keyCache, valueCache, currentPosition
}

// GetWithSlice returns filled portion up to maxPosition.
func (c *KVCache) GetWithSlice(g *Graph, maxPosition int) (keys, values *Node) {
	c.Initialize(g)

	keyCache := c.reuseCtx.VariableWithShape("key_cache", shapes.Make(c.dtype, c.batchSize, c.numHeads, c.maxSeqLen, c.headDim)).ValueGraph(g)
	valueCache := c.reuseCtx.VariableWithShape("value_cache", shapes.Make(c.dtype, c.batchSize, c.numHeads, c.maxSeqLen, c.headDim)).ValueGraph(g)

	// Slice to get only filled positions
	keys = Slice(keyCache, AxisRange(), AxisRange(), AxisRange(0, maxPosition), AxisRange())
	values = Slice(valueCache, AxisRange(), AxisRange(), AxisRange(0, maxPosition), AxisRange())

	return keys, values
}

// CreateAttentionMask masks unfilled cache positions: [batch,1,query_len,key_len].
func (c *KVCache) CreateAttentionMask(g *Graph, querySeqLen, keySeqLen int) *Node {
	c.Initialize(g)

	currentPosition := c.reuseCtx.VariableWithShape("cache_position", shapes.Make(dtypes.Int32, c.batchSize)).ValueGraph(g)
	currentPosition = ConvertDType(currentPosition, c.dtype)

	// Create position indices for keys: [0, 1, 2, ..., keySeqLen-1]
	// Shape: [keySeqLen]
	keyPositions := Iota(g, shapes.Make(c.dtype, keySeqLen), 0)

	// Expand for broadcasting
	// currentPosition: [batch_size] -> [batch_size, 1, 1, 1]
	// keyPositions: [keySeqLen] -> [1, 1, 1, keySeqLen]
	currentPosition = ExpandDims(ExpandDims(ExpandDims(currentPosition, -1), -1), -1)
	keyPositions = ExpandDims(ExpandDims(ExpandDims(keyPositions, 0), 0), 0)

	// Create mask: True where key position < current position
	mask := LessThan(keyPositions, currentPosition)

	// Broadcast to [batch_size, 1, query_seq_len, key_seq_len]
	maskShape := shapes.Make(dtypes.Bool, c.batchSize, 1, querySeqLen, keySeqLen)
	mask = BroadcastToShape(mask, maskShape)

	return mask
}
