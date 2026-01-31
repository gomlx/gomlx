// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package layers

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/initializers"
	"github.com/gomlx/gomlx/pkg/ml/layers/regularizers"
	"github.com/gomlx/gomlx/pkg/ml/nn"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/pkg/errors"
)

// LayerNormBuilder is a helper to build a layer normalization computation. Create it with LayerNormalization,
// set the desired parameters and when all is set, call Done.
// See LayerNormalization for details.
type LayerNormBuilder struct {
	ctx                *context.Context
	x, mask            *Node
	normalizingAxes    []int
	epsilon            float64
	center, gain       bool
	scaleNormalization bool
	regularizer        regularizers.Regularizer
	normalizationDType dtypes.DType
}

var (
	// ParamLayerNormEpsilon is the context parameter that defines the default layer normalization epsilon value.
	// The default is 1e-3.
	ParamLayerNormEpsilon = "layer_norm_epsilon"

	// ParamLayerNormCenter is the context parameter that defines whether to center the norm by default.
	// The default is true.
	ParamLayerNormCenter = "layer_norm_center"

	// ParamLayerNormLearnedGain is the context parameter that defines whether to learn a gain for the
	// layer norm, that multiplies its output.
	// The default is true.
	ParamLayerNormLearnedGain = "layer_norm_learned_gain"

	// ParamLayerNormLearnedScale is an alias to ParamLayerNormLearnedGain.
	// Deprecated: renamed to follow original papers nomenclature.
	ParamLayerNormLearnedScale = ParamLayerNormLearnedGain

	// ParamLayerNormRescale is the context parameter that defines whether to rescale the layer
	// by dividing it by the square root of the variance.
	// The default is true.
	ParamLayerNormRescale = "layer_norm_rescale"

	// ParamLayerNormL2Regularization is the context parameter that defines the amount of L2 regularization
	// to apply to the learned gain, if one is defined.
	// The default is 0.0.
	ParamLayerNormL2Regularization = "layer_norm_l2_regularization"

	// ParamLayerNormNormalizationDType is the dtype to use when doing the normalization.
	//
	// Low precision dtypes (Float16, BFloat16) usually lead to NaN when doing ReduceSum/ReduceMean,
	// which layer normalization does.
	//
	// If this value is different than the dtype of the input, the normalization will convert the values
	// to this dtype first.
	//
	// The default value (and empty string) means to use Float32 if the input dtype is 16 bits only, otherwise
	// use the same dtype as the input.
	ParamLayerNormNormalizationDType = "layer_norm_dtype"
)

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
// of shape [batch_size, height, width, channels] one common approach is to normalize over the image, so
// `normalizingAxes=[1 2]`, but not over the channels.
//
// Notice the difference between BatchNormalization, that normalizes over the batch dimension, as opposed
// to the feature dimensions.
//
// The layer norm may have a learned gain and offset, controlled by LayerNormBuilder.LearnedGain
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
	builder := &LayerNormBuilder{
		ctx:                ctx.In("layer_normalization"),
		x:                  x,
		normalizingAxes:    normalizingAxes,
		epsilon:            context.GetParamOr(ctx, ParamLayerNormEpsilon, 1e-3),
		center:             context.GetParamOr(ctx, ParamLayerNormCenter, true),
		gain:               context.GetParamOr(ctx, ParamLayerNormLearnedGain, true),
		scaleNormalization: context.GetParamOr(ctx, ParamLayerNormRescale, true),
		normalizationDType: x.DType(),
	}

	// Create default regularizer.
	if l2 := context.GetParamOr(ctx, ParamLayerNormL2Regularization, 0.0); l2 > 0 {
		builder.regularizer = regularizers.Combine(builder.regularizer, regularizers.L2(l2))
	}

	// Convert negative axes to their actual value.
	for ii := range builder.normalizingAxes {
		builder.normalizingAxes[ii] = MustAdjustAxis(builder.normalizingAxes[ii], x)
	}

	// Add default dtype for normalization.
	normDTypeStr := context.GetParamOr(ctx, ParamLayerNormNormalizationDType, "")
	if normDTypeStr == "" {
		if x.DType() == dtypes.Float16 || x.DType() == dtypes.BFloat16 {
			builder.normalizationDType = dtypes.Float32
		}
	} else {
		var err error
		builder.normalizationDType, err = dtypes.DTypeString(normDTypeStr)
		if err != nil {
			panic(errors.WithMessagef(err, "Invalid dtype configured for hyperparameter %q", ParamLayerNormNormalizationDType))
		}
	}
	return builder
}

// Epsilon is a small float added to variance to avoid dividing by zero.
// It defaults to the value given by [ParamLayerNormEpsilon].
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

// LearnedGain defines whether the layer normalization tries to apply a multiplying gain the input, a tensor
// with the shape of the combined normalizing axes -- so it changes the direction of the inputs, it's not
// simply a gain.
//
// Default is true.
func (builder *LayerNormBuilder) LearnedGain(value bool) *LayerNormBuilder {
	builder.gain = value
	return builder
}

// ScaleNormalization defines whether the input's scale is normalized by the square root of the
// variance. The default is true, and this is the original paper specification, but in some cases
// it works best without it.
func (builder *LayerNormBuilder) ScaleNormalization(value bool) *LayerNormBuilder {
	builder.scaleNormalization = value
	return builder
}

// Mask sets the mask for the input values. False values in the mask should be ignored for
// the normalization.
func (builder *LayerNormBuilder) Mask(mask *Node) *LayerNormBuilder {
	builder.mask = mask
	return builder
}

// Done finishes configuring the LayerNormalization and generates the graph computation to normalize the input.
func (builder *LayerNormBuilder) Done() *Node {
	ctx := builder.ctx
	x := builder.x
	mask := builder.mask
	g := x.Graph()

	// Fast path: dispatch to nn.LayerNorm when no special features are needed
	// (no mask, scale normalization enabled, no dtype conversion).
	if mask == nil && builder.scaleNormalization && builder.normalizationDType == x.DType() {
		normShape := shapes.Make(x.DType(), xslices.Map(builder.normalizingAxes, func(axis int) int {
			return x.Shape().Dimensions[axis]
		})...)
		broadcastNormShape := x.Shape().Clone()
		for ii := range broadcastNormShape.Dimensions {
			broadcastNormShape.Dimensions[ii] = 1
		}
		for _, axis := range builder.normalizingAxes {
			broadcastNormShape.Dimensions[axis] = x.Shape().Dimensions[axis]
		}

		var gamma, beta *Node
		if builder.gain {
			gainVar := ctx.WithInitializer(initializers.One).VariableWithShape("gain", normShape).SetTrainable(true)
			if builder.regularizer != nil {
				builder.regularizer(ctx, g, gainVar)
			}
			gamma = Reshape(gainVar.ValueGraph(g), broadcastNormShape.Dimensions...)
		}
		if builder.center {
			offsetVar := ctx.WithInitializer(initializers.Zero).VariableWithShape("offset", normShape).SetTrainable(true)
			beta = Reshape(offsetVar.ValueGraph(g), broadcastNormShape.Dimensions...)
		}
		return nn.LayerNorm(x, builder.normalizingAxes, builder.epsilon, gamma, beta, nil)
	}

	// LearnedGain and offset to be applied to the normalized value.
	var gain, offset *Node
	normShape := shapes.Make(x.DType(), xslices.Map(builder.normalizingAxes, func(axis int) int {
		return x.Shape().Dimensions[axis]
	})...)
	broadcastNormShape := x.Shape().Clone() // the shape `normShape` will need to be reshaped to be combined with `x`.
	for ii := range broadcastNormShape.Dimensions {
		broadcastNormShape.Dimensions[ii] = 1
	}
	for _, axis := range builder.normalizingAxes {
		broadcastNormShape.Dimensions[axis] = x.Shape().Dimensions[axis] // Same value for the feature axes we are normalizing over.
	}
	var gainVar *context.Variable
	if builder.gain {
		gainVar = ctx.WithInitializer(initializers.One).VariableWithShape("gain", normShape).SetTrainable(true)
		if builder.regularizer != nil {
			builder.regularizer(ctx, g, gainVar) // Apply regularizer.
		}
		gain = Reshape(gainVar.ValueGraph(g), broadcastNormShape.Dimensions...)
	} else {
		gain = Ones(g, broadcastNormShape)
	}
	if builder.center {
		offsetVar := ctx.WithInitializer(initializers.Zero).VariableWithShape("offset", normShape).SetTrainable(true)
		offset = Reshape(offsetVar.ValueGraph(g), broadcastNormShape.Dimensions...)
	} else {
		offset = Zeros(g, broadcastNormShape)
	}

	// Calculate mean and variance over normalizingAxes and normalize using configured normalizationDType
	// Notice: ConvertDType is a no-op if the dtype is not changed.
	var mean *Node
	convertedX := ConvertDType(x, builder.normalizationDType)
	if mask == nil {
		mean = ReduceAndKeep(convertedX, ReduceMean, builder.normalizingAxes...)
	} else {
		mean = MaskedReduceAndKeep(convertedX, mask, MaskedReduceMean, builder.normalizingAxes...)
	}
	normalized := Sub(convertedX, mean)
	if mask != nil {
		normalized = Where(mask, normalized, ZerosLike(normalized))
	}
	if builder.scaleNormalization {
		var variance *Node
		if mask == nil {
			variance = ReduceAndKeep(Square(normalized), ReduceMean, builder.normalizingAxes...)
		} else {
			variance = MaskedReduceAndKeep(Square(normalized), mask, MaskedReduceMean, builder.normalizingAxes...)
		}
		epsilon := ConstAs(convertedX, builder.epsilon)
		normalized = Div(normalized, Sqrt(Add(variance, epsilon)))
	}
	normalized = ConvertDType(normalized, x.DType()) // Convert back to the input's DType.

	// Adjust if using learned offset/gain factors.
	if builder.gain {
		normalized = Mul(normalized, gain)
	}
	if builder.center {
		normalized = Add(normalized, offset)
	}

	return normalized
}
