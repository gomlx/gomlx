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
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
)

// LayerNormBuilder is a helper to build a layer normalization computation. Create it with LayerNormalization,
// set the desired parameters and when all is set, call Done.
// See LayerNormalization for details.
type LayerNormBuilder struct {
	ctx                *context.Context
	x                  *Node
	normalizingAxes    []int
	epsilon            float64
	center, scale      bool
	scaleNormalization bool
}

// LayerNormalization performs a layer normalization on the input. It includes a scaling and offset factor,
// and normalization over the feature entries.
//
// This is an alternative to BatchNormalization, that doesn't suffer from the problem of variance
// on small batch sizes, nor does it need to keep a moving average of the normalization parameters.
// Commonly used with transformer layers (see MultiHeadAttention).
//
// normalizingAxes are the axes over which to normalize: mean and variance are calculated over these
// axes and the values are then normalized. E.g: if your input is `[batch_size, features]` you should
// use `normalizingAxes=[1]` (same as -1) to normalize over the `features` axis; if your input is an image
// of shape `[batch_size, height, width, channels]` one common approach is to normalize over the image, so
// `normalizingAxes=[1 2]`, but not over the channels (or batch).
//
// Notice the difference between BatchNormalization, that normalizes over the batch dimension, as opposed
// to the feature dimensions.
//
// The layer norm may have a learned scale and offset, controlled by LayerNormBuilder.LearnedScale
// and LayerNormBuilder.LearnedOffset settings, enabled by default.
//
// To ease setting its parameters it returns a LayerNormBuilder object for configuration. Once it is
// set up call `LayerNormBuilder.Done` and it will return the normalized x. Browse through LayerNormBuilder to
// check for its capabilities, and the defaults.
//
// Layer normalization behaves the same during training and inference -- as opposed to batch normalization.
//
// Based on paper "Layer Normalization" (Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton),
// https://arxiv.org/abs/1607.06450
//
// FutureWork: support padding by not normalizing parts that weren't touched ...
func LayerNormalization(ctx *context.Context, x *Node, normalizingAxes ...int) *LayerNormBuilder {
	return &LayerNormBuilder{
		ctx:                ctx.In("layer_normalization"),
		x:                  x,
		normalizingAxes:    normalizingAxes,
		epsilon:            1e-3,
		center:             true,
		scale:              true,
		scaleNormalization: true,
	}
}

// Epsilon is a small float added to variance to avoid dividing by zero. It defaults to 1e-3.
//
// It is not used if ScaleNormalization is set to false.
func (builder *LayerNormBuilder) Epsilon(value float64) *LayerNormBuilder {
	builder.epsilon = value
	return builder
}

// LearnedOffset defines whether the layer normalization tries to center the input by adding a learned
// offset. It defaults to true.
//
// The offset will be learned separately for each axis that is not the batch (assumed to be axis 0 only)
// and not any of the normalizingAxes.
func (builder *LayerNormBuilder) LearnedOffset(value bool) *LayerNormBuilder {
	builder.center = value
	return builder
}

// LearnedScale defines whether the layer normalization tries to scale the input by adding a learned scale.
// It defaults to true.
//
// The scale will be learned separately for each axis that is not the batch (assumed to be axis 0 only)
// and not any of the normalizingAxes.
func (builder *LayerNormBuilder) LearnedScale(value bool) *LayerNormBuilder {
	builder.scale = value
	return builder
}

// ScaleNormalization defines whether the input's scale is normalized by the square root of the
// variance. The default is true, and this is the original paper specification, but in some cases
// it works best without it.
func (builder *LayerNormBuilder) ScaleNormalization(value bool) *LayerNormBuilder {
	builder.scaleNormalization = value
	return builder
}

// Done finishes configuring the LayerNormalization and generates the graph computation to normalize the input.
func (builder *LayerNormBuilder) Done() *Node {
	ctx := builder.ctx
	x := builder.x
	g := x.Graph()

	// Convert negative axes to their actual value.
	for ii := range builder.normalizingAxes {
		builder.normalizingAxes[ii] = AdjustAxis(x, builder.normalizingAxes[ii])
	}

	// LearnedScale and offset to be applied to the normalized value.
	var scale, offset *Node
	varShape := x.Shape().Copy()
	varShape.Dimensions[0] = 1 // Same value for all elements of the batch.
	for _, axis := range builder.normalizingAxes {
		varShape.Dimensions[axis] = 1 // Same value for the feature axes we are normalizing over.
	}
	var scaleVar *context.Variable
	if builder.scale {
		scaleVar = ctx.WithInitializer(initializers.One).VariableWithShape("scale", varShape).SetTrainable(true)
		scale = scaleVar.ValueGraph(g)
	} else {
		scale = Ones(g, varShape)
	}
	if builder.center {
		offsetVar := ctx.WithInitializer(initializers.Zero).VariableWithShape("offset", varShape).SetTrainable(true)
		offset = offsetVar.ValueGraph(g)
	} else {
		offset = Zeros(g, varShape)
	}

	// Add regularization to scale.
	if scaleVar != nil {
		if l2any, found := ctx.GetParam(L2RegularizationKey); found {
			l2 := l2any.(float64)
			if l2 > 0 {
				l2Node := ConstAs(x, l2)
				AddL2Regularization(ctx, l2Node, scaleVar.ValueGraph(g))
			}
		}
	}

	// Calculate mean and variance over normalizingAxes and normalize.
	mean := ReduceAndKeep(builder.x, ReduceMean, builder.normalizingAxes...)
	normalized := Sub(builder.x, mean)
	if builder.scaleNormalization {
		variance := ReduceAndKeep(Square(normalized), ReduceMean, builder.normalizingAxes...)
		epsilon := ConstAs(x, builder.epsilon)
		normalized = Div(normalized, Sqrt(Add(variance, epsilon)))
	}
	// Adjust if using learned offset/scale factors.
	if builder.scale {
		normalized = Mul(normalized, scale)
	}
	if builder.center {
		normalized = Add(normalized, offset)
	}

	return normalized
}
