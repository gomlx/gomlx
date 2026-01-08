// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package layers

import (
	. "github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors/images"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/regularizers"
	"github.com/gomlx/gomlx/pkg/support/xslices"
)

// This file contains all parts of the layers.Convolve implementation.

// ConvBuilder is a helper to build a convolution computation. Create it with Convolution, set the desired parameters,
// and when all is set, call Done.
type ConvBuilder struct {
	ctx                                *context.Context
	graph                              *Graph
	x                                  *Node
	numSpatialDims                     int
	channelsAxisConfig                 images.ChannelsAxisConfig
	outputChannels                     int
	kernelSize                         []int
	bias                               bool
	strides                            []int
	padSame                            bool
	dilations                          []int
	newScope                           bool
	regularizer                        regularizers.Regularizer
	channelGroupCount, batchGroupCount int
}

// Convolution prepares one convolution on x with the given kernel for an arbitrary
// number of spatial dimensions (1D, 2D, 3D, etc.).
//
// It returns a ConvBuilder object for configuration.
// Once it is set up, call `ConvBuilder.Done` and it will return the convolved x.
//
// It includes support for padding, strides, dilations, grouping, and an added bias.
// Browse through ConvBuilder to see the capabilities, and their defaults.
//
// Two parameters need setting: Channels (or channels) and KernelSize. It will fail
// if they are not set.
//
// The shape of x should be `[batch, <spatial_dimensions...>, input_channels]` if
// configured with `ConvBuilder.ChannelsAxis(images.ChannelsLast)`, the default.
//
// If one sets `ConvBuilder.ChannelsAxis(images.ChannelsFirst)`, the shape should be
// `[batch, input_channels, <spatial_dimensions...>]` instead.
//
// The output rank and order of the output axes are the same as the input's.
// Their dimensions depend on the configuration options.
func Convolution(ctx *context.Context, x *Node) *ConvBuilder {
	conv := &ConvBuilder{
		ctx:               ctx,
		graph:             x.Graph(),
		x:                 x,
		newScope:          true,
		regularizer:       regularizers.FromContext(ctx),
		channelGroupCount: 1,
		batchGroupCount:   1,
	}
	conv.numSpatialDims = x.Rank() - 2
	if conv.numSpatialDims < 0 {
		Panicf("Input x must have rank >= 3, shaped by default as [batch, <spatial_dimensions...>, channels], "+
			"but x rank is %d", x.Rank())
	}
	return conv.ChannelsAxis(images.ChannelsLast).NoPadding().UseBias(true).Strides(1)
}

// Channels sets the number of output channels.
// There is no default, and this number must be set before Done is called.
func (conv *ConvBuilder) Channels(filters int) *ConvBuilder {
	conv.outputChannels = filters
	if filters <= 0 {
		Panicf("number of outputChannels must be > 0, it was set to %d", filters)
	}
	return conv
}

// Filters is a deprecated alias for Channels.
//
// Deprecated: Use Channels instead.
func (conv *ConvBuilder) Filters(channels int) *ConvBuilder {
	return conv.Channels(channels)
}

// KernelSize sets the kernel size for every axis.
// There is no default, and this value must be set before Done is called.
//
// You can also use KernelSizePerAxis to set the kernel size per axis individually.
func (conv *ConvBuilder) KernelSize(size int) *ConvBuilder {
	perDim := xslices.SliceWithValue(conv.numSpatialDims, size)
	return conv.KernelSizePerAxis(perDim...)
}

// KernelSizePerAxis sets the kernel size for each axis (axis).
// There is no default, and this value must be set before Done is called.
//
// You can also use KernelSize to set the kernel size the same for all dimensions.
func (conv *ConvBuilder) KernelSizePerAxis(dimensions ...int) *ConvBuilder {
	if len(dimensions) != conv.numSpatialDims {
		Panicf("received %d kernel dimensions, but x has %d spatial dimensions",
			len(dimensions), conv.numSpatialDims)
		return conv
	}
	conv.kernelSize = dimensions
	return conv
}

// KernelSizePerDim is a deprecated alias for KernelSizePerAxis.
//
// Deprecated: Use KernelSizePerAxis instead.
func (conv *ConvBuilder) KernelSizePerDim(dimensions ...int) *ConvBuilder {
	return conv.KernelSizePerAxis(dimensions...)
}

// UseBias sets whether to add a trainable bias term to the convolution. Default is true.
func (conv *ConvBuilder) UseBias(useBias bool) *ConvBuilder {
	conv.bias = useBias
	return conv
}

// ChannelsAxis configures the axis for the channels (aka. "depth" or "features") dimension. The default is
// `images.ChannelsLast`, meaning the "channels" dimension comes last.
//
// Note: `images` refers to package `github.com/gomlx/gomlx/core/tensors/images`.
//
// It returns the modified Config object, so calls can be cascaded.
func (conv *ConvBuilder) ChannelsAxis(channelsAxisConfig images.ChannelsAxisConfig) *ConvBuilder {
	conv.channelsAxisConfig = channelsAxisConfig
	return conv
}

// PadSame adds paddings on the edges of x such that in the end the output
// of the convolution has the same shape as the input (assuming strides=1).
//
// The default is NoPadding.
func (conv *ConvBuilder) PadSame() *ConvBuilder {
	conv.padSame = true
	return conv
}

// NoPadding removes any paddings, so if the kernel spatial dimensions > 1,
// the output shape will be reduced on the edges.
//
// This is the default.
func (conv *ConvBuilder) NoPadding() *ConvBuilder {
	conv.padSame = false
	return conv
}

// Strides sets the strides of the convolution. It sets the same value for every dimension.
// The default is 1.
//
// The stride is how many steps to move after a convolution. A value of 2 will half the input
// size, since a convolution will be done at every other position, and so on. It can be defined
// separately per dimension.
//
// One cannot use strides and dilation at the same time.
func (conv *ConvBuilder) Strides(strides int) *ConvBuilder {
	perDim := xslices.SliceWithValue(conv.numSpatialDims, strides)
	return conv.StridePerAxis(perDim...)
}

// StridePerAxis sets the strides for each spatial dimension of the convolution.
// The default is 1 for every dimension.
//
// The stride is how many steps to move after a convolution. A value of 2 will half the input
// size, since a convolution will be done at every other position, and so on. It can be defined
// separately per dimension.
//
// One cannot use strides and dilation at the same time.
func (conv *ConvBuilder) StridePerAxis(strides ...int) *ConvBuilder {
	if len(strides) != conv.numSpatialDims {
		Panicf("received %d strides in StridePerAxis, but x has %d spatial dimensions",
			len(strides), conv.numSpatialDims)
		return conv
	}
	conv.strides = strides
	return conv
}

// StridePerDim is a deprecated alias for StridePerAxis.
//
// Deprecated: Use StridePerAxis instead.
func (conv *ConvBuilder) StridePerDim(strides ...int) *ConvBuilder {
	return conv.StridePerAxis(strides...)
}

// Dilations sets the dilations of the convolution. It sets the same value for every dimension.
//
// The default is 1. A value > 1 is also called "atrous convolution".
//
// It specifies the kernel up-sampling rate. In the literature, the same parameter
// is sometimes called input stride or kernel dilation. The effective kernel size used for the convolution
// will be `kernel_shape + (kernel_shape - 1) * (dilation - 1)`, obtained by inserting (dilation-1) zeros
// between consecutive elements of the original filter in the spatial dimension.
//
// One cannot use strides and dilation at the same time.
func (conv *ConvBuilder) Dilations(dilation int) *ConvBuilder {
	dilationsPerDim := xslices.SliceWithValue(conv.numSpatialDims, dilation)
	return conv.DilationPerAxis(dilationsPerDim...)
}

// DilationPerAxis sets the kernel dilations for each spatial axis of the convolution.
// The default is 1 for every axis.
//
// Specifies the kernel up-sampling rate. In the literature, the same parameter
// is sometimes called input stride or dilation. The effective kernel size used for the convolution
// will be `kernel_shape + (kernel_shape - 1) * (dilation - 1)`, obtained by inserting (dilation-1) zeros
// between consecutive elements of the original filter in the spatial dimension.
//
// One cannot use strides and dilation at the same time.
func (conv *ConvBuilder) DilationPerAxis(dilations ...int) *ConvBuilder {
	if len(dilations) != conv.numSpatialDims {
		Panicf("received %d dilations in DilationPerAxis, but x has %d spatial dimensions",
			len(dilations), conv.numSpatialDims)
		return conv
	}
	conv.dilations = dilations
	return conv
}

// DilationPerDim is a deprecated alias for DilationPerAxis.
//
// Deprecated: Use DilationPerAxis instead.
func (conv *ConvBuilder) DilationPerDim(dilations ...int) *ConvBuilder {
	return conv.DilationPerAxis(dilations...)
}

// CurrentScope configures the convolution not to create a sub-scope for the kernel weights it needs,
// and instead use the current one provided in Convolution.
//
// By default, Convolution will create a sub-scope named "conv".
func (conv *ConvBuilder) CurrentScope() *ConvBuilder {
	conv.newScope = false
	return conv
}

// ChannelGroupCount splits input/output channels into independent groups.
// Equivalent to TensorFlow's "groups" parameter in tf.nn.convNd operations.
//
// When groupCount != 1, the kernel's shape changes: the input channels
// dimension of the kernel must equal (input_channels / group_count).
// This effectively creates separate convolution groups where each group
// processes a subset of input channels and produces a subset of output channels.
//
// For depthwise convolution, set groups = input_channels (see tf.nn.depthwise_conv2d).
// The output shape will have the same spatial dimensions as a regular convolution
// but with channel dimensions affected by the grouping.
//
// Side effects:
//   - Kernel shape: The kernel's input channel dimension becomes (input_channels / group_count)
//   - Output shape: The output maintains the same spatial dimensions as regular convolution,
//     but each group independently maps its input channels to output channels
//   - Performance: Can reduce computation cost by limiting connections between channels
//   - Memory usage: Reduces the number of parameters in the kernel
//
// Note: Back-propagation is not yet implemented for this feature.
//
// Reference: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#group_by_window
func (conv *ConvBuilder) ChannelGroupCount(groupCount int) *ConvBuilder {
	if groupCount < 1 {
		Panicf("FeatureGroupCount must be >= 1, got %d", groupCount)
	}

	conv.channelGroupCount = groupCount
	return conv
}

// BatchGroupCount splits batches into independent processing groups.
// Used for cross-batch interactions like ShuffleNet's channel shuffle.
//
// When BatchGroupCount != 1, the kernel's shape changes: the batch dimension
// of the input is divided by the group count, creating separate convolution
// groups where each group processes a subset of the batch.
//
// The output shape will have the same spatial dimensions as a regular convolution
// but with the batch dimension affected by the grouping.
//
// Note: Back-propagation is not yet implemented for this feature.
//
// Reference: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch
func (conv *ConvBuilder) BatchGroupCount(groupCount int) *ConvBuilder {
	if groupCount < 1 {
		Panicf("BatchGroupCount must be >= 1, got %d", groupCount)
	}

	conv.batchGroupCount = groupCount
	return conv
}

// Regularizer to be applied to the learned weights (but not the biases).
// Default is none.
//
// To use more than one type of Regularizer, use regularizers.Combine, and set the returned combined regularizer here.
//
// The default is regularizers.FromContext, which is configured by regularizers.ParamL1 and regularizers.ParamL2.
func (conv *ConvBuilder) Regularizer(regularizer regularizers.Regularizer) *ConvBuilder {
	conv.regularizer = regularizer
	return conv
}

// Done indicates that the Convolution layer is finished being configured. It then
// creates the convolution and it's kernels (variables) and returns the resulting
// Node.
func (conv *ConvBuilder) Done() *Node {
	// Default is to create a sub-scope for the convolution variables.
	ctxInScope := conv.ctx
	if conv.newScope {
		ctxInScope = ctxInScope.In("conv")
	}

	if len(conv.kernelSize) == 0 || conv.outputChannels <= 0 {
		Panicf("layers.Convolution requires Filters and KernelSize to be set")
	}
	if conv.numSpatialDims <= 0 {
		Panicf("invalid x shape %s, can't figure spatial dimensions", conv.x.Shape())
	}

	// Check only one of strides / dilations are set.
	var dilationsSet, stridesSet bool
	if conv.strides != nil {
		for _, stride := range conv.strides {
			if stride != 1 {
				stridesSet = true
			}
		}
	}
	if conv.dilations != nil {
		for _, dilation := range conv.dilations {
			if dilation != 1 {
				dilationsSet = true
			}
		}
	}
	if dilationsSet && stridesSet {
		Panicf("both strides (%v) and dilations (%v) are set, but only one can be used at a time",
			conv.strides, conv.dilations)
	}

	// Create and apply kernel.
	xShape := conv.x.Shape()
	dtype := xShape.DType
	kernelShape := shapes.Make(dtype)
	kernelShape.Dimensions = make([]int, 0, conv.numSpatialDims+2)
	channelsAxis := images.GetChannelsAxis(xShape, conv.channelsAxisConfig)
	inputChannels := xShape.Dimensions[channelsAxis]
	if conv.channelsAxisConfig == images.ChannelsFirst {
		kernelShape.Dimensions = append(kernelShape.Dimensions, conv.outputChannels)
		kernelShape.Dimensions = append(kernelShape.Dimensions, inputChannels)
		kernelShape.Dimensions = append(kernelShape.Dimensions, conv.kernelSize...)
	} else {
		kernelShape.Dimensions = append(kernelShape.Dimensions, conv.kernelSize...)
		kernelShape.Dimensions = append(kernelShape.Dimensions, inputChannels)
		kernelShape.Dimensions = append(kernelShape.Dimensions, conv.outputChannels)
	}
	kernelVar := ctxInScope.VariableWithShape("weights", kernelShape)
	if conv.regularizer != nil {
		conv.regularizer(ctxInScope, conv.graph, kernelVar)
	}
	kernel := kernelVar.ValueGraph(conv.graph)
	convOpts := Convolve(conv.x, kernel).
		StridePerAxis(conv.strides...).
		ChannelsAxis(conv.channelsAxisConfig).
		ChannelGroupCount(conv.channelGroupCount).
		BatchGroupCount(conv.batchGroupCount)
	if len(conv.dilations) > 0 {
		convOpts.DilationPerAxis(conv.dilations...)
	}
	if conv.padSame {
		convOpts.PadSame()
	} else {
		convOpts.NoPadding()
	}
	output := convOpts.Done()

	// Create and apply bias.
	if conv.bias {
		biasVar := ctxInScope.VariableWithShape("biases", shapes.Make(dtype, conv.outputChannels))
		bias := biasVar.ValueGraph(conv.graph)
		expandedDims := xslices.SliceWithValue(output.Rank(), 1)
		outputChannelsAxis := images.GetChannelsAxis(output, conv.channelsAxisConfig)
		expandedDims[outputChannelsAxis] = conv.outputChannels
		bias = Reshape(bias, expandedDims...)
		output = Add(output, bias)
	}

	// Add regularization.
	if l2any, found := ctxInScope.GetParam(ParamL2Regularization); found {
		l2 := l2any.(float64)
		if l2 > 0 {
			regularizers.L2(l2)(ctxInScope, conv.graph, kernelVar)
		}
	}
	return output
}
