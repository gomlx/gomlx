// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/support/xslices"
	timage "github.com/gomlx/gomlx/core/tensors/images"
	"github.com/gomlx/gomlx/internal/graphimpl"
	. "github.com/gomlx/gomlx/support/exceptions"
)

// This file contains all parts of the Convolve implementation.

// ConvolutionBuilder is a helper to build a convolution computation.
// Create it with Convolve, set the desired parameters and
// when set, call `IsNil()`.
type ConvolutionBuilder struct {
	graph                              *Graph
	x, kernel                          *Node
	numSpatialDims                     int
	strides                            []int
	paddings                           [][2]int
	padSame                            bool
	inputDilations, kernelDilations    []int
	channelGroupCount, batchGroupCount int

	channelsAxisConfig timage.ChannelsAxisConfig
	axes               ConvolveAxesConfig
}

// Convolve prepares a convolution on x with the given kernel for an arbitrary
// number of spatial dimensions (1D, 2D, 3D, etc.).
//
// It returns a ConvolutionBuilder object that can be further configured. Once the
// configuration is finished, call ConvolutionBuilder.Done, and it will return
// the convolved x. Browse through ConvolutionBuilder to see its capabilities
// and defaults.
//
// * Shapes:
//
// The shape of x should be [batch, <spatial_dimensions...>, input_channels] if
// configured with ConvolutionBuilder.ChannelsAxis(timage.ChannelsLast), the default.
// If one sets ConvolutionBuilder.ChannelsAxis(timage.ChannelsFirst), then the shape should be
// [batch, input_channels, <spatial_dimensions...>] instead.
//
// Note: package timage refers to package github.com/gomlx/gomlx/core/tensors/images.
//
// Alternatively, it provides the method ConvolutionBuilder.AxesConfig which allows arbitrary shape (axes order)
// configuration of x and the kernel.
//
// The shape of the kernel should be [<spatial_dimensions...>, input_channels, output_channels] if
// configured with ConvolutionBuilder.ChannelsAxis(timage.ChannelsLast), the default.
// If one sets ConvolutionBuilder.ChannelsAxis(timage.ChannelsFirst), the shape should be
// [input_channels, output_channels, <spatial_dimensions...>] instead.
//
// Notice x and kernel must have the same rank.
//
// We follow the Keras convention of calling "channels" the axis that is sometimes referred to by "features" or "depth".
// The "kernel" is also referred to as "filters" by some.
//
// Additional features:
//   - Group operations: Use ConvolutionBuilder.ChannelGroupCount to split channels
//     or BatchGroupCount to split batches into independent processing groups.
//     When using either feature, the kernel's shape changes.
func Convolve(x, kernel *Node) *ConvolutionBuilder {
	conv := &ConvolutionBuilder{
		graph:             validateBuildingGraphFromInputs(x, kernel),
		x:                 x,
		kernel:            kernel,
		channelGroupCount: 1,
		batchGroupCount:   1,
	}

	conv.numSpatialDims = x.Rank() - 2
	if conv.numSpatialDims < 0 {
		Panicf("the input x must have rank >= 3, shaped by default as [batch, <spatial_dimensions...>, channels], "+
			"but x rank is %d", x.Rank())
	}

	if kernel.Rank() != x.Rank() {
		Panicf("the kernel (rank %d) must have the same rank as the input x (rank %d) -- x has a batch dimension, "+
			"and kernel has an output_channels dimension", kernel.Rank(), x.Rank())
	}

	return conv.ChannelsAxis(timage.ChannelsLast).NoPadding()
}

// gatherSlice returns a slice of int values by gathering values from the params slices indexed by indices.
// Eg: gatherSlice([]int{1,3}, []int{10, 20, 30, 40, 50}) -> []int{20, 40}
func gatherSlice(indices, params []int) (slice []int) {
	slice = make([]int, len(indices))
	for ii := range slice {
		slice[ii] = params[indices[ii]]
	}
	return
}

// ChannelsAxis configures the axis for the channels (aka. "depth" or "features") dimension. The default is
// `timage.ChannelsLast`, meaning the "channels" dimension comes last.
//
// Note: `timage` refers to the package github.com/gomlx/gomlx/core/tensors/images
//
// For more fine-control, see AxesConfig.
//
// It returns the modified Config object, so calls can be cascaded.
func (conv *ConvolutionBuilder) ChannelsAxis(channelsAxisConfig timage.ChannelsAxisConfig) *ConvolutionBuilder {
	conv.channelsAxisConfig = channelsAxisConfig
	switch channelsAxisConfig {
	case timage.ChannelsFirst:
		conv.axes.InputBatch = 0
		conv.axes.InputChannels = 1
		conv.axes.InputSpatial = xslices.Iota(2, conv.numSpatialDims)

		conv.axes.KernelOutputChannels = 0
		conv.axes.KernelInputChannels = 1
		conv.axes.KernelSpatial = xslices.Iota(2, conv.numSpatialDims)

		conv.axes.OutputBatch = 0
		conv.axes.OutputChannels = 1
		conv.axes.OutputSpatial = xslices.Iota(2, conv.numSpatialDims)

	case timage.ChannelsLast:
		conv.axes.InputBatch = 0
		conv.axes.InputSpatial = xslices.Iota(1, conv.numSpatialDims)
		conv.axes.InputChannels = conv.numSpatialDims + 1

		conv.axes.KernelInputChannels = conv.numSpatialDims
		conv.axes.KernelOutputChannels = conv.numSpatialDims + 1
		conv.axes.KernelSpatial = xslices.Iota(0, conv.numSpatialDims)

		conv.axes.OutputBatch = 0
		conv.axes.OutputSpatial = xslices.Iota(1, conv.numSpatialDims)
		conv.axes.OutputChannels = conv.numSpatialDims + 1
	}
	return conv
}

// AxesConfig specify the exact configuration of the axes on the input (x/input and kernel) and output of
// the Convolve operation. This is advanced (and may not be supported in every backend), but it's powerful.
// Consider using `ConvolutionBuilder.ChannelsAxis` instead.
//
// The default is `ChannelsAxis(timage.ChannelsLast)`.
func (conv *ConvolutionBuilder) AxesConfig(axes ConvolveAxesConfig) *ConvolutionBuilder {
	conv.axes = axes
	return conv
}

// Strides sets the strides of the convolution. It sets the same value for every dimension.
// The default is 1.
//
// The stride is how many steps to move after a convolution. A value of 2 will halve the input
// size, since a convolution will be done at every other position, and so on. It can be defined
// separately per dimension.
//
// One cannot use strides and dilation at the same time.
func (conv *ConvolutionBuilder) Strides(strides int) *ConvolutionBuilder {
	perDim := xslices.SliceWithValue(conv.numSpatialDims, strides)
	return conv.StridePerAxis(perDim...)
}

// StridePerDim sets the strides for each spatial dimension of the convolution.
//
// Deprecated: Use StridePerAxis instead.
func (conv *ConvolutionBuilder) StridePerDim(strides ...int) *ConvolutionBuilder {
	return conv.StridePerAxis(strides...)
}

// StridePerAxis sets the strides for each spatial dimension of the convolution.
// The default is 1 for every dimension.
//
// The stride is how many steps to move after a convolution.
// A value of 2 will halve the input size, since a convolution will be done at every other position, and so on.
// It can be defined separately per dimension.
//
// One cannot use strides and dilation at the same time.
func (conv *ConvolutionBuilder) StridePerAxis(strides ...int) *ConvolutionBuilder {
	if len(strides) != conv.numSpatialDims {
		Panicf("received %d strides in StridePerAxis, but x has %d spatial dimensions",
			len(strides), conv.numSpatialDims)
	}
	conv.strides = strides
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
// Reference: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#group_by_window
func (conv *ConvolutionBuilder) ChannelGroupCount(groupCount int) *ConvolutionBuilder {
	if groupCount < 1 {
		Panicf("FeatureGroupCount must be >= 1, got %d", groupCount)
	}

	conv.channelGroupCount = groupCount
	return conv
}

// FeatureGroupCount is an alias for ChannelGroupCount.
//
// Deprecated: Use ChannelGroupCount instead.
func (conv *ConvolutionBuilder) FeatureGroupCount(groupCount int) *ConvolutionBuilder {
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
// The kernel output channels dimensions must be divisible by the batch group count,
// each slice applies to one batch, and they are concatenated in the end.
//
// The output shape will have the same spatial dimensions as a regular convolution
// but with the batch dimension divided by groupCount.
//
// Reference: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch
func (conv *ConvolutionBuilder) BatchGroupCount(groupCount int) *ConvolutionBuilder {
	if groupCount < 1 {
		Panicf("BatchGroupCount must be >= 1, got %d", groupCount)
	}

	conv.batchGroupCount = groupCount
	return conv
}

// PadSame adds paddings on the edges of x such that in the end the output
// of the convolution has the same shapes as the input (assuming strides=1).
//
// The default is no padding. See also NoPadding and PaddingPerDim.
func (conv *ConvolutionBuilder) PadSame() *ConvolutionBuilder {
	conv.paddings = nil
	conv.padSame = true
	return conv
}

// NoPadding removes any paddings, so if the kernel spatial dimensions > 1,
// the output shapes will be reduced on the edges. This is the default.
//
// See also PadSame and PaddingPerDim.
func (conv *ConvolutionBuilder) NoPadding() *ConvolutionBuilder {
	conv.paddings = nil
	conv.padSame = false
	return conv
}

// PaddingPerDim specifies the paddings at the start and at the end to use per spatial dimension,
// that means one pair ([2]int) per spatial dimension.
//
// If a nil value for paddings is given, this has no effect.
//
// The default is no padding. See also NoPadding and PadSame.
func (conv *ConvolutionBuilder) PaddingPerDim(paddings [][2]int) *ConvolutionBuilder {
	if paddings == nil {
		return conv
	}
	if len(paddings) != conv.numSpatialDims {
		Panicf("received %d paddings in PaddingPerDim, but x has %d spatial dimensions",
			len(paddings), conv.numSpatialDims)
	}
	conv.paddings = paddings
	conv.padSame = false
	return conv
}

// Dilations sets the dilations of the convolution: the same value is used for every dimension.
//
// The default is 1. A value > 1 is also called "atrous convolution".
//
// It specifies the kernel's up-sampling rate. In the literature, the same parameter
// is sometimes called input stride or kernel dilation. The effective kernel size used for the convolution
// will be `kernel_shape + (kernel_shape - 1) * (dilation - 1)`, obtained by inserting (dilation-1) zeros
// between consecutive elements of the original kernel in the spatial dimension.
//
// One cannot use strides and dilation at the same time.
func (conv *ConvolutionBuilder) Dilations(dilation int) *ConvolutionBuilder {
	dilationsPerDim := xslices.SliceWithValue(conv.numSpatialDims, dilation)
	return conv.DilationPerAxis(dilationsPerDim...)
}

// DilationPerAxis sets the kernel's dilations for each spatial dimension of the convolution.
//
// The default is 1 for every axis. A value > 1 is also called "atrous convolution".
//
// It specifies the kernel's up-sampling rate. In the literature, the same parameter
// is sometimes called input stride or dilation. The effective kernel size used for the convolution
// will be `kernel_shape + (kernel_shape - 1) * (dilation - 1)`, obtained by inserting (dilation-1) zeros
// between consecutive elements of the original kernel in the spatial dimension.
//
// One cannot use strides and dilation at the same time.
func (conv *ConvolutionBuilder) DilationPerAxis(dilations ...int) *ConvolutionBuilder {
	if len(dilations) == 0 {
		conv.kernelDilations = nil
		return conv
	}
	if len(dilations) != conv.numSpatialDims {
		Panicf("received %d dilations in DilationPerAxis, but x has %d spatial dimensions",
			len(dilations), conv.numSpatialDims)
	}
	conv.kernelDilations = dilations
	return conv
}

// DilationPerDim sets the kernel's dilations for each spatial dimension of the convolution.
//
// Deprecated: Use DilationPerAxis instead.
func (conv *ConvolutionBuilder) DilationPerDim(dilations ...int) *ConvolutionBuilder {
	return conv.DilationPerAxis(dilations...)
}

// InputDilationPerAxis virtually inserts zeros between input values, making the input effectively larger.
func (conv *ConvolutionBuilder) InputDilationPerAxis(dilations ...int) *ConvolutionBuilder {
	if len(dilations) == 0 {
		conv.inputDilations = nil
		return conv
	}
	if len(dilations) != conv.numSpatialDims {
		Panicf("received %d dilations in inputDilationPerDim, but x has %d spatial dimensions",
			len(dilations), conv.numSpatialDims)
	}
	conv.inputDilations = dilations
	return conv
}

// Done indicates that the convolve operation is finished being configured, and
// it updates the computation graph with convolution and returns the resulting
// Node.
func (conv *ConvolutionBuilder) Done() *Node {
	// Select the kernel spatial dimensions.
	kernelSpatialDims := gatherSlice(conv.axes.KernelSpatial, conv.kernel.Shape().Dimensions)

	// paddings can only be calculated after we are sure about the channels positioning.
	paddings := conv.paddings
	if paddings == nil && conv.padSame {
		// Pad such that the output is shaped the same as the input.
		paddings = make([][2]int, conv.numSpatialDims)
		dilation := 1
		for dim := range paddings {
			kernelSize := kernelSpatialDims[dim] // for this dimension.
			if conv.kernelDilations != nil {
				dilation = conv.kernelDilations[dim]
			}
			kernelSize = (kernelSize-1)*dilation + 1
			paddings[dim][0] = (kernelSize - 1) / 2 // For an even-sized kernel, the padding is asymmetric.
			paddings[dim][1] = kernelSize / 2
		}
	}

	// Check only one of the "strides" or "dilations" are set.
	var dilationsSet, stridesSet bool
	if len(conv.strides) > 0 {
		for _, stride := range conv.strides {
			if stride != 1 {
				stridesSet = true
				break
			}
		}
	}
	if conv.kernelDilations != nil {
		for _, dilation := range conv.kernelDilations {
			if dilation != 1 {
				dilationsSet = true
				break
			}
		}
	}
	if dilationsSet && stridesSet {
		Panicf("both strides (%v) and dilations (%v) are set, but only one can be used at a time",
			conv.strides, conv.kernelDilations)
	}

	// Validate feature group count
	if conv.channelGroupCount > 1 {
		inputChannels := conv.x.Shape().Dimensions[conv.axes.InputChannels]
		if inputChannels%conv.channelGroupCount != 0 {
			Panicf("input channels (%d) not divisible by FeatureGroupCount (%d)",
				inputChannels, conv.channelGroupCount)
		}

		// Validate that the kernel's input channel axis matches the feature group count.
		kernelInputChannels := conv.kernel.Shape().Dimensions[conv.axes.KernelInputChannels]
		if kernelInputChannels != inputChannels/conv.channelGroupCount {
			Panicf("kernel input channels (%d) must equal input channels (%d) divided by FeatureGroupCount (%d)",
				kernelInputChannels, inputChannels, conv.channelGroupCount)
		}
	}

	// Validate batch group count.
	if conv.batchGroupCount > 1 {
		batchSize := conv.x.Shape().Dimensions[conv.axes.InputBatch]
		if batchSize%conv.batchGroupCount != 0 {
			Panicf("batch size (%d) not divisible by BatchGroupCount (%d)",
				batchSize, conv.batchGroupCount)
		}
	}

	return ConvGeneral(conv.x, conv.kernel,
		conv.axes, conv.strides,
		paddings, conv.inputDilations, conv.kernelDilations,
		conv.channelGroupCount, conv.batchGroupCount)
}

// ConvolveAxesConfig defines the interpretation of the input/kernel/output tensor axes.
// There must be the same number of spatial dimensions (axes) for each of the 3 tensors.
// Input and output have batch and channels axes. Filters have "inputChannels" and "outputChannels" axes.
type ConvolveAxesConfig = compute.ConvolveAxesConfig

// ConvGeneral provides direct access to the backend implementation of convolutions.
// Consider using Convolve instead since this is mostly used for testing.
//
// It implements a generic convolution operation with support for:
//
// - Arbitrary number of spatial axes.
// - Arbitrary transposition of axes.
// - Strides and padding.
// - Dilations of the input.
// - Dilations of the kernel, aka. atrous convolution.
// - Filter grouping (on the input channels).
// - Batch grouping.
//
// Some details in https://www.tensorflow.org/xla/operation_semantics#convwithgeneralpadding_convolution.
// There operand and filter are called lhs and rhs.
// (XLA documentation is unfortunately poor, much is guess-work).
// Also useful, https://arxiv.org/pdf/1603.07285v1.pdf.
//
// Note: input is aka. operand; kernel is aka. "filters". The input and output "channels" are also known as "features dimensions".
func ConvGeneral(input, kernel *Node, axes ConvolveAxesConfig,
	strides []int, paddings [][2]int, inputDilations, kernelDilations []int,
	channelGroupCount, batchGroupCount int) *Node {
	_ = validateBuildingGraphFromInputs(input, kernel)
	numSpatialDims := input.Rank() - 2
	if len(axes.InputSpatial) != numSpatialDims || len(axes.OutputSpatial) != numSpatialDims || len(axes.KernelSpatial) != numSpatialDims {
		Panicf("ConvGeneral: input has %d spatial dimensions, but axes configuration has %d, %d, %d spatial axes configured "+
			"for input/kernel/output", numSpatialDims, len(axes.InputSpatial), len(axes.KernelSpatial), len(axes.OutputSpatial))
	}
	return wrapNode(graphimpl.BackendConvGeneral(unwrapNode(input), unwrapNode(kernel), axes, strides, paddings, inputDilations, kernelDilations, channelGroupCount, batchGroupCount))
}

// ConvGeneralDilated is a deprecated an alias to ConvGeneral.
//
// Deprecated: use ConvGeneral instead.
var ConvGeneralDilated = ConvGeneral
