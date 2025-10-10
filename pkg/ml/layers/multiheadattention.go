/*
 *	Copyright 2023 Jan Pfeifer
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

package layers

import (
	"fmt"
	"math"
	"reflect"

	. "github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
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

	// Mask related attributes.
	keyMask, queryMask *Node
	queryKeyMatrixMask *Node
	useCausalMask      bool
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

// nextNAxes enumerates the next n consecutive axis, starting from nextAxis. It returns
// the string with the axis concatenated.
func nextNAxes(n int, nextAxis rune) string {
	var eq string
	for ii := 0; ii < n; ii++ {
		eq += string(nextAxis)
		nextAxis++
	}
	return eq
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
	projectedKey := Dense(b.ctx.In("key"), b.key, true, b.numHeads, b.keyQueryDim)
	projectedQuery := Dense(b.ctx.In("query"), b.query, true, b.numHeads, b.keyQueryDim)
	projectedValue := Dense(b.ctx.In("value"), b.value, true, b.numHeads, b.valueDim)

	// LearnedScale attentionLogits by 1/sqrt(keyQueryDim).
	projectedQuery = Mul(projectedQuery, ConstAs(projectedQuery, 1.0/math.Sqrt(float64(b.keyQueryDim))))

	// Build equation for attention Einsum.
	batchAxis := 'b'
	headsAxis := 'h'
	projectionAxis := 'd'

	numKeyAxes := b.key.Rank() - 2
	nextFreeAxis := 'i'
	keyInnerAxes := nextNAxes(numKeyAxes, nextFreeAxis)
	nextFreeAxis += rune(numKeyAxes)
	projectedKeyAxes := fmt.Sprintf("%c%s%c%c", batchAxis, keyInnerAxes, headsAxis, projectionAxis)
	numQueryAxes := b.query.Rank() - 2
	queryInnerAxes := nextNAxes(numKeyAxes, nextFreeAxis)
	nextFreeAxis += rune(numQueryAxes)
	projectedQueryAxes := fmt.Sprintf("%c%s%c%c", batchAxis, queryInnerAxes, headsAxis, projectionAxis)

	// Example of attention equation:
	//  - projectedKey.shape(rank 4)   = [batch, key_elements, numHeads, keyQueryDims]
	//  - projectedQuery.shape(rank 4) = [batch, query_elements, numHeads, keyQueryDim]
	//  - attentionEquation   = "bihd,bjhd->bjhi"
	attentionEquation := fmt.Sprintf("%s,%s->%c%s%c%s", projectedQueryAxes, projectedKeyAxes,
		batchAxis, queryInnerAxes, headsAxis, keyInnerAxes)

	// Attention logits: outer product of key/query inner dimensions, with a dot-product of their projections.
	// Shape: [batch, <query_elements>, num_heads, <key_elements>]
	attentionLogits := Einsum(attentionEquation, projectedQuery, projectedKey)
	normalizingFactor := math.Sqrt(float64(b.keyQueryDim))
	attentionLogits = DivScalar(attentionLogits, normalizingFactor)
	//fmt.Printf("\tattentionLogits: %s\n", attentionLogits.Shape())

	mask := b.buildMask()
	// Attention coefficients: Softmax over all the inner key axes (the last dimensions of attentionLogits)
	// Shape: [batch, <query_elements>, num_heads, <key_elements>]
	softmaxAxes := xslices.Iota(attentionLogits.Rank()-numKeyAxes, numKeyAxes)
	if mask == nil {
		attentionCoefficients = Softmax(attentionLogits, softmaxAxes...)
	} else {
		//fmt.Printf("\tmask=%s\n", mask.Shape())
		attentionCoefficients = MaskedSoftmax(attentionLogits, mask, softmaxAxes...)
	}
	//fmt.Printf("\tattentionCoefficients: %s\n", attentionLogits.Shape())
	if b.dropoutRate > 0 {
		attentionCoefficients = Dropout(b.ctx, attentionCoefficients, ConstAs(attentionCoefficients, b.dropoutRate))
	}

	// Build equation for the attention output Einsum.
	// - attentionCoefficients     = [batch, <query_elements>, num_heads, <key_elements>]
	// - projectedValue            = [batch, <key_elements>, num_heads, value_dim]
	// - resulting attentionOutput = [batch, <query_elements>, num_heads, value_dim]
	outputEquation := fmt.Sprintf("%c%s%c%s,%c%s%c%c->%c%s%c%c",
		batchAxis, queryInnerAxes, headsAxis, keyInnerAxes,
		batchAxis, keyInnerAxes, headsAxis, projectionAxis,
		batchAxis, queryInnerAxes, headsAxis, projectionAxis)
	//fmt.Printf("\toutputEquation (coef x value): %s\n", outputEquation)
	attentionOutput = Einsum(outputEquation, attentionCoefficients, projectedValue)

	// Final projection: flatten the heads and then do a final projection to the final
	// outputDim (set with `SetOutputDim`).
	//fmt.Printf("\tattentionOutput (1): %s\n", attentionOutput.Shape())
	flatDims := make([]int, attentionOutput.Rank()-1)
	copy(flatDims, attentionOutput.Shape().Dimensions[:len(flatDims)])
	flatDims[len(flatDims)-1] *= attentionOutput.Shape().Dimensions[attentionOutput.Rank()-1]
	// New shape: `[batch, <query_elements>, num_head*value_dim]`
	attentionOutput = Reshape(attentionOutput, flatDims...)
	// Final shape: `[batch, <query_elements>, outputDim]`
	attentionOutput = Dense(b.ctx.In("output"), attentionOutput, b.useProjectionBias, b.outputDim)

	return attentionOutput, attentionCoefficients
}

// Done or DoneWithCoefficients should be called after all optional settings are configured.
// It returns both the attention output and the attention coefficients (matrix) used.
//
// `output` will be shaped `[batch_size, <query_elements>, output_dim]`, where `output_dim`
// can be configured by `SetOutputDim`.
func (b *MultiHeadAttentionBuilder) Done() (output *Node) {
	output, _ = b.DoneWithCoefficients()
	return output
}

// buildAttentionShape returns the shape of the attention coefficients and mask, and sets it to b.attentionShape.
// attentionShape is `[batch, <query_elements>, num_heads, <key_elements>]`.
func (b *MultiHeadAttentionBuilder) buildAttentionShape() {
	finalDims := make([]int, 2+b.innerQueryAxes+b.innerKeyAxes)
	pos := 0
	finalDims[pos] = b.key.Shape().Dimensions[0]
	pos += 1
	copy(finalDims[pos:], b.query.Shape().Dimensions[1:1+b.innerQueryAxes]) // <query_elements>
	pos += b.innerQueryAxes
	finalDims[pos] = b.numHeads
	pos += 1
	copy(finalDims[pos:], b.key.Shape().Dimensions[1:1+b.innerKeyAxes]) // <query_elements>

	b.attentionShape = shapes.Make(b.key.DType(), finalDims...)
}

// buildMask returns a normalized mask for shape `[batch, <query_elements>, num_heads, <key_elements>]`.
func (b *MultiHeadAttentionBuilder) buildMask() (mask *Node) {
	// Mask defined in one of two ways.
	if b.queryMask != nil || b.keyMask != nil {
		mask = b.buildMaskFromSplitMasks()
	} else if b.queryKeyMatrixMask != nil {
		mask = b.queryKeyMatrixMask
	}

	// Combine causal mask.
	if b.useCausalMask {
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
func (b *MultiHeadAttentionBuilder) buildCausalMask() (mask *Node) {
	keyShape := b.key.Shape()
	dim := keyShape.Dimensions[1] // Same as queryShape.Dimensions[1].

	// mask is [<query_elements>, <key_elements>]
	mask = LowerTriangular(b.g, dim)

	// Broadcast mask to target shape of `[batch, <query_elements>, numHeads, <key_elements>]`
	mask = InsertAxes(mask, 0, 1)                                // Add batch and numHeads axes.
	mask = BroadcastToDims(mask, b.attentionShape.Dimensions...) // Broadcast to target dimensions.
	return
}
