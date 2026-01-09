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
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gopjrt/dtypes"
)

// IncrementalAttentionConfig configures incremental multi-head attention with KV cache.
type IncrementalAttentionConfig struct {
	ctx               *context.Context
	query             *Node
	numHeads          int
	headDim           int
	kvCache           *KVCache
	useRoPE           bool
	ropeBaseFreq      float64
	currentPosition   int
	useProjectionBias bool
	dropoutRate       float64
	keyQueryDim       int
	valueDim          int
	outputDim         int
}

// NewIncrementalAttention sets up autoregressive attention with optional KV cache.
func NewIncrementalAttention(ctx *context.Context, query *Node, numHeads, headDim int, kvCache *KVCache) *IncrementalAttentionConfig {
	return &IncrementalAttentionConfig{
		ctx:               ctx.In("IncrementalAttention"),
		query:             query,
		numHeads:          numHeads,
		headDim:           headDim,
		kvCache:           kvCache,
		useRoPE:           false,
		ropeBaseFreq:      10000.0,
		currentPosition:   0,
		useProjectionBias: true,
		dropoutRate:       0.0,
		keyQueryDim:       headDim,
		valueDim:          headDim,
		outputDim:         query.Shape().Dimensions[query.Shape().Rank()-1],
	}
}

// WithRoPE enables RoPE with base frequency.
func (c *IncrementalAttentionConfig) WithRoPE(baseFreq float64) *IncrementalAttentionConfig {
	c.useRoPE = true
	c.ropeBaseFreq = baseFreq
	return c
}

// WithPosition sets absolute position for RoPE and cache.
func (c *IncrementalAttentionConfig) WithPosition(position int) *IncrementalAttentionConfig {
	c.currentPosition = position
	return c
}

// WithProjectionBias toggles bias in output projection.
func (c *IncrementalAttentionConfig) WithProjectionBias(use bool) *IncrementalAttentionConfig {
	c.useProjectionBias = use
	return c
}

// WithDropout sets attention dropout rate.
func (c *IncrementalAttentionConfig) WithDropout(rate float64) *IncrementalAttentionConfig {
	c.dropoutRate = rate
	return c
}

// SetKeyQueryDim sets per-head query/key dimension.
func (c *IncrementalAttentionConfig) SetKeyQueryDim(dim int) *IncrementalAttentionConfig {
	c.keyQueryDim = dim
	return c
}

// SetValueDim sets per-head value dimension.
func (c *IncrementalAttentionConfig) SetValueDim(dim int) *IncrementalAttentionConfig {
	c.valueDim = dim
	return c
}

// SetOutputDim sets final output dimension.
func (c *IncrementalAttentionConfig) SetOutputDim(dim int) *IncrementalAttentionConfig {
	c.outputDim = dim
	return c
}

// Done runs incremental attention and updates KV cache.
func (c *IncrementalAttentionConfig) Done() *Node {
	output, _ := c.done(false)
	return output
}

// DoneWithAttentionWeights returns output and attention weights.
func (c *IncrementalAttentionConfig) DoneWithAttentionWeights() (output, attnWeights *Node) {
	return c.done(true)
}

// done is the internal implementation that optionally returns attention weights.
func (c *IncrementalAttentionConfig) done(returnWeights bool) (output, attnWeights *Node) {
	g := c.query.Graph()
	queryShape := c.query.Shape()
	_ = queryShape.Dimensions[2] // Check shape validity

	batchSize := queryShape.Dimensions[0]
	newSeqLen := queryShape.Dimensions[1]

	input := c.query

	queryProj := layers.Dense(c.ctx.In("query_proj"), input, true, c.numHeads*c.keyQueryDim)
	queryProj = Reshape(queryProj, batchSize, newSeqLen, c.numHeads, c.keyQueryDim)
	keyProj := layers.Dense(c.ctx.In("key_proj"), input, true, c.numHeads*c.keyQueryDim)
	keyProj = Reshape(keyProj, batchSize, newSeqLen, c.numHeads, c.keyQueryDim)
	valueProj := layers.Dense(c.ctx.In("value_proj"), input, true, c.numHeads*c.valueDim)
	valueProj = Reshape(valueProj, batchSize, newSeqLen, c.numHeads, c.valueDim)

	// RoPE
	if c.useRoPE {
		queryProj = RoPE(queryProj, c.currentPosition, c.ropeBaseFreq)
		keyProj = RoPE(keyProj, c.currentPosition, c.ropeBaseFreq)
	}

	// Caching
	var keys, values *Node
	if c.kvCache != nil {
		keyProjTransposed := TransposeAllDims(keyProj, 0, 2, 1, 3)
		valueProjTransposed := TransposeAllDims(valueProj, 0, 2, 1, 3)

		c.kvCache.Update(g, keyProjTransposed, valueProjTransposed)

		keys, values, _ = c.kvCache.Get(g)
		totalSeqLen := c.currentPosition + newSeqLen
		if totalSeqLen < c.kvCache.maxSeqLen {
			keys, values = c.kvCache.GetWithSlice(g, totalSeqLen)
		}
	} else {
		keys = TransposeAllDims(keyProj, 0, 2, 1, 3)
		values = TransposeAllDims(valueProj, 0, 2, 1, 3)
	}

	queryProj = TransposeAllDims(queryProj, 0, 2, 1, 3)

	// Compute attention scores: Q @ K^T / sqrt(key_dim)
	scores := Einsum("bhqd,bhkd->bhqk", queryProj, keys)
	scale := Sqrt(ConstAs(scores, float64(c.keyQueryDim)))
	scores = Div(scores, scale)

	// Causal masking - always apply for autoregressive models
	keySeqLen := keys.Shape().Dimensions[2]

	// Create causal mask: query at absolute position i can only attend to keys at absolute positions <= i
	// For cached generation: query positions are [currentPosition, ..., currentPosition + newSeqLen - 1],
	// key positions are [0, ..., keySeqLen - 1]
	// For training: query positions are [0, ..., newSeqLen - 1],
	// key positions are [0, ..., newSeqLen - 1]

	queryAbsPositions := IotaFull(g, shapes.Make(dtypes.Int32, newSeqLen))       // [0, ..., newSeqLen-1]
	queryAbsPositions = AddScalar(queryAbsPositions, float64(c.currentPosition)) // Add current position offset
	keyAbsPositions := IotaFull(g, shapes.Make(dtypes.Int32, keySeqLen))         // [0, ..., keySeqLen-1]

	queryAbsPositions = ExpandDims(queryAbsPositions, -1)            // [newSeqLen, 1]
	keyAbsPositions = ExpandDims(keyAbsPositions, 0)                 // [1, keySeqLen]
	causalMask := GreaterOrEqual(queryAbsPositions, keyAbsPositions) // [newSeqLen, keySeqLen]

	// Expand to [1, 1, newSeqLen, keySeqLen] for broadcasting
	causalMask = ExpandDims(ExpandDims(causalMask, 0), 0)

	var combinedMask *Node
	if c.kvCache != nil {
		// Also apply cache validity mask (don't attend to unfilled cache slots)
		cacheMask := c.kvCache.CreateAttentionMask(g, newSeqLen, keySeqLen)
		// Combine both masks: must satisfy both causal AND cache validity
		combinedMask = And(causalMask, cacheMask)
	} else {
		combinedMask = causalMask
	}

	// Apply mask to scores
	combinedMask = BroadcastToShape(combinedMask, scores.Shape())
	maskFloat := Where(combinedMask, ZerosLike(scores), ConstAs(scores, -1e10))
	scores = Add(scores, maskFloat)

	attnWeights = Softmax(scores)
	if c.dropoutRate > 0.0 {
		attnWeights = layers.DropoutFromContext(c.ctx, attnWeights)
	}

	// Apply attention to values:
	attnOutput := Einsum("bhqk,bhkd->bhqd", attnWeights, values)

	// Output
	attnOutput = TransposeAllDims(attnOutput, 0, 2, 1, 3)
	attnOutput = Reshape(attnOutput, batchSize, newSeqLen, c.numHeads*c.valueDim)
	output = layers.Dense(c.ctx.In("output_proj"), attnOutput, c.useProjectionBias, c.outputDim)

	if !returnWeights {
		attnWeights = nil
	}
	return output, attnWeights
}
