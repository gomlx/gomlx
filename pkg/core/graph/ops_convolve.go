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

package graph

import (
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	timage "github.com/gomlx/gomlx/pkg/core/tensors/images"
	"github.com/gomlx/gomlx/pkg/support/xslices"
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
//     When using either feature, the kernel's shape changes and back-propagation
//     is not yet supported.
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

// InputDilationPerAxis is used when generating the gradient of a convolution with strides.
// It effectively inserts zeros in the input, making it effectively larger than it actually is.
//
// The gradient of Convolve with input dilation is not implemented yet, be careful.
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
	if conv.strides != nil {
		for _, stride := range conv.strides {
			if stride != 1 {
				stridesSet = true
			}
		}
	}
	if conv.kernelDilations != nil {
		for _, dilation := range conv.kernelDilations {
			if dilation != 1 {
				dilationsSet = true
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
type ConvolveAxesConfig = backends.ConvolveAxesConfig

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
	return backendConvGeneral(input, kernel, axes, strides, paddings, inputDilations, kernelDilations, channelGroupCount, batchGroupCount)
}

// ConvGeneralDilated is a deprecated an alias to ConvGeneral.
//
// Deprecated: use ConvGeneral instead.
var ConvGeneralDilated = ConvGeneral

func convGeneralVJP(node, v *Node, _ shapes.Shape) []*Node {
	// TODO: backward propagation is not working in this function

	// Recover parameters from serialized node.
	x := node.inputNodes[0]
	kernel := node.inputNodes[1]
	params := node.inputs.(*nodeInputsConvGeneral)
	numSpatialDims := x.Rank() - 2
	if len(params.inputDilations) > 0 {
		Panicf("gradient of Convolve with input dilation not defined, " +
			"usually it's only used to calculate the gradient of a convolution, so " +
			"this may occur when trying to do the gradient of a gradient.")
	}

	// Notice one can't have batch and channel grouping at the same time.
	var vjpX, vjpKernel *Node
	if params.channelGroupCount > 1 {
		vjpX, vjpKernel = convGeneralWithChannelGroupingVJP(node, x, kernel, v, numSpatialDims, params.axes,
			params.strides, params.paddings, params.kernelDilations, params.channelGroupCount)
	} else if params.batchGroupCount > 1 {
		vjpX, vjpKernel = convGeneralWithBatchGroupingVJP(node, x, kernel, v, numSpatialDims, params.axes,
			params.strides, params.paddings, params.kernelDilations, params.batchGroupCount)
	} else {
		vjpX = convVJPWrtX(node, x, kernel, v, numSpatialDims, params.axes,
			params.strides, params.paddings, params.kernelDilations)
		vjpKernel = convVJPWrtKernel(node, x, kernel, v, numSpatialDims, params.axes,
			params.strides, params.paddings, params.kernelDilations)
	}
	return []*Node{vjpX, vjpKernel}
}

func convGeneralWithChannelGroupingVJP(node, x, kernel, v *Node, numSpatialDims int, axes ConvolveAxesConfig,
	strides []int, paddings [][2]int, kernelDilations []int, channelGroupCount int) (vjpX, vjpKernel *Node) {
	// Split v, x and kernel into groups.
	outputChannelsAxis := axes.OutputChannels
	vSlices := Split(v, outputChannelsAxis, channelGroupCount)
	inputChannelsAxis := axes.InputChannels
	xSlices := Split(x, inputChannelsAxis, channelGroupCount)
	kernelOutputChannelsAxis := axes.KernelOutputChannels
	kernelSlices := Split(kernel, kernelOutputChannelsAxis, channelGroupCount)

	// Compute the gradients for each slice of the input (x).
	var xGradSlices []*Node
	for i := 0; i < channelGroupCount; i++ {
		gradSlice := convVJPWrtX(node, xSlices[i], kernelSlices[i], vSlices[i], numSpatialDims, axes, strides, paddings, kernelDilations)
		xGradSlices = append(xGradSlices, gradSlice)
	}
	vjpX = Concatenate(xGradSlices, inputChannelsAxis)

	// Compute the gradients for each slice of the kernel.
	var kernelGradSlices []*Node
	for i := 0; i < channelGroupCount; i++ {
		gradSlice := convVJPWrtKernel(vSlices[i], xSlices[i], kernelSlices[i], vSlices[i], numSpatialDims, axes, strides, paddings, kernelDilations)
		kernelGradSlices = append(kernelGradSlices, gradSlice)
	}
	vjpKernel = Concatenate(kernelGradSlices, kernelOutputChannelsAxis)
	return
}

// convGeneralWithBatchGroupingVJP handles the gradient calculation when using batch grouping.
func convGeneralWithBatchGroupingVJP(node, x, kernel, v *Node, numSpatialDims int, axes ConvolveAxesConfig,
	strides []int, paddings [][2]int, kernelDilations []int, batchGroupCount int) (vjpX, vjpKernel *Node) {
	// Split v, x and kernel into batch groups.
	vSlices := Split(v, axes.OutputChannels, batchGroupCount)
	xSlices := Split(x, axes.InputBatch, batchGroupCount)
	kernelSlices := Split(kernel, axes.KernelOutputChannels, batchGroupCount)

	// Compute the gradients for each slice of the input (x).
	var xGradSlices []*Node
	for i := 0; i < batchGroupCount; i++ {
		gradSlice := convVJPWrtX(node, xSlices[i], kernelSlices[i], vSlices[i], numSpatialDims, axes, strides, paddings, kernelDilations)
		xGradSlices = append(xGradSlices, gradSlice)
	}
	vjpX = Concatenate(xGradSlices, axes.InputBatch)

	// Compute the gradients for each slice of the kernel.
	var kernelGradSlices []*Node
	for i := 0; i < batchGroupCount; i++ {
		gradSlice := convVJPWrtKernel(vSlices[i], xSlices[i], kernelSlices[i], vSlices[i], numSpatialDims, axes, strides, paddings, kernelDilations)
		kernelGradSlices = append(kernelGradSlices, gradSlice)
	}
	vjpKernel = Concatenate(kernelGradSlices, axes.KernelOutputChannels)
	return
}

// convVJPWrtX creates the Vector-Jacobian (for backpropagation) of the
// output with respect to (==wrt) the input (x). See also convVJPWrtKernel.
func convVJPWrtX(node, x, kernel, v *Node, numSpatialDims int, axes ConvolveAxesConfig, strides []int, paddings [][2]int, kernelDilations []int) *Node {
	// Get output and input spatial dimensions.
	inputSpatialDims := gatherSlice(axes.InputSpatial, x.Shape().Dimensions)
	outputSpatialDims := gatherSlice(axes.OutputSpatial, node.Shape().Dimensions)
	kernelSpatialDims := gatherSlice(axes.KernelSpatial, kernel.Shape().Dimensions)

	// Gradient of the output with respect to x:
	// (1) we need to reverse the convolution, which involves the reverse kernel: spatial dimensions are reversed,
	//     and input/output channel weights are transposed.
	reverseKernel := Reverse(kernel, axes.KernelSpatial...)

	// Instead of transposing output/input channels, just swap their indices in the reverseAxes. Effectively this does:
	//     reverseKernel = Transpose(reverseKernel, axes.KernelOutputChannels, axes.KernelInputChannels)

	reverseAxes := axes
	reverseAxes.KernelInputChannels, reverseAxes.KernelOutputChannels = axes.KernelOutputChannels, axes.KernelInputChannels

	// (2) we need to pad the reverse convolution to match get the original input.
	reversePaddings := make([][2]int, numSpatialDims)
	dilation := 1
	for axis := 0; axis < numSpatialDims; axis++ {
		//fmt.Printf("\taxis %d\n", axis)
		// Effective kernel size.
		kernelSize := kernelSpatialDims[axis]
		if len(kernelDilations) > 0 {
			dilation = kernelDilations[axis]
		}
		kernelSize = (kernelSize-1)*dilation + 1
		inputDimSize := inputSpatialDims[axis]
		outputDimSize := outputSpatialDims[axis]
		dimStride := 1
		if len(strides) > 0 {
			dimStride = strides[axis]
		}

		// Start/End positions on the original input.
		var dimPadding [2]int
		if len(paddings) > 0 {
			dimPadding = paddings[axis]
		}
		inputDimStart := (kernelSize-1)/2 - dimPadding[0]
		inputDimEnd := inputDimSize - kernelSize/2 + dimPadding[1]

		if (inputDimEnd-inputDimStart+(dimStride-1))/dimStride != outputDimSize {
			Panicf("failed to set up reverse Convolve() for gradient in spatial dimension %d: "+
				"outputDimSize=%d, but input convolution window is %d with stride for this dimension %d",
				axis, outputDimSize, inputDimEnd-inputDimStart, dimStride)
		}

		// Start/End positions on the output for the reverse convolution.
		// Values below 0 or above outputDimSize means padding. It has to be such that it will regenerate
		// the original input spatial shapes.
		// Stride in the input becomes inputDilations in the reverse convolution.
		outputDimStart := -inputDimStart - ((kernelSize - 1) / 2)
		if outputDimStart > 0 {
			Panicf("failed to set up reverse Convolve() for gradient: spatial dimension %d "+
				"outputDimStart=%d > 0, which is out-of-bounds", axis, outputDimStart)
		}
		outputDimEnd := inputDimSize - inputDimStart + kernelSize/2
		// So far outputDimEnd and outputDimStart haven't considered the strides converted to input dilation
		// on the reverse convolution -- effectively injecting zeros.
		numInjectedZeros := (outputDimSize - 1) * (dimStride - 1)
		outputDimEnd -= numInjectedZeros
		if outputDimEnd < outputDimSize {
			Panicf("failed to set up reverse Convolve() for gradient: spatial dimension %d "+
				"outputDimEnd=%d < outputDimSize=%d, which is out-of-bounds", axis, outputDimEnd, outputDimSize)
		}

		// Set padding to the output to match its start/end positions.
		reversePaddings[axis][0] = -outputDimStart
		reversePaddings[axis][1] = outputDimEnd - outputDimSize
	}
	// (3) Run2 the reverse convolution of the VJP.
	revConv := Convolve(v, reverseKernel).PaddingPerDim(reversePaddings).DilationPerAxis(kernelDilations...).AxesConfig(reverseAxes)
	if len(strides) > 0 {
		revConv.InputDilationPerAxis(strides...)
	}
	return revConv.Done()
}

func expectedOutputSize(inputSize, kernelSize, dilation, stride int, padding [2]int) int {
	effectiveKernelSize := (kernelSize-1)*dilation + 1
	// Start/End positions on the original input.
	inputStart := (effectiveKernelSize-1)/2 - padding[0]
	inputEnd := inputSize - effectiveKernelSize/2 + padding[1]
	return (inputEnd - inputStart + (stride - 1)) / stride
}

// convVJPWrtKernel creates the Vector-Jacobian (for backpropagation) of the
// output with respect to (==wrt) the kernel (aka kernel). See also convVJPWrtX.
// It works for one group of convolution.
func convVJPWrtKernel(node, x, kernel, v *Node, numSpatialDims int, axes ConvolveAxesConfig,
	strides []int, paddings [][2]int, kernelDilations []int) *Node {
	// (1) For the Gradient of the output with respect to kernel, we need a reverse convolution of
	// the original input using v (the term from VJP, shaped as the original output) as the
	// "reverseKernel". Since we need to multiply it by most of the inputNodes to get the VJP wrt
	// to the kernel. The output of this reverse convolution will be shaped like the original
	// convolution kernel if we correctly adjust the axes. See below.
	reverseKernel := v

	// The batch dimension becomes like a channel: since they must be all added. But the channels that need
	// to be generated separately become a batch dimension.
	var reverseAxes ConvolveAxesConfig
	reverseAxes.InputBatch, reverseAxes.InputChannels = axes.InputChannels, axes.InputBatch
	reverseAxes.InputSpatial = axes.InputSpatial

	// The output of the reverse convolve is the original kernel shapes. The kernel's input channels axis
	// is the reverse output batch axis. The output channel of the reverse convolve goes into
	// the original kernel output channel.
	reverseAxes.OutputBatch, reverseAxes.OutputChannels = axes.KernelInputChannels, axes.KernelOutputChannels
	reverseAxes.OutputSpatial = axes.KernelSpatial

	// The kernel of the reverse node is shaped like the output of the original convolution.
	reverseAxes.KernelInputChannels, reverseAxes.KernelOutputChannels = axes.OutputBatch, axes.OutputChannels
	reverseAxes.KernelSpatial = axes.OutputSpatial

	// Strides in the original convolution become dilations for the backward convolution to the kernel.
	reverseDilations := strides
	reverseStrides := kernelDilations

	// (2) we need to pad the reverse convolution to match get the original input.
	reversePaddings := make([][2]int, numSpatialDims)
	copy(reversePaddings, paddings) // Safe even if paddings is nil (in which case nothing is copied)

	// Get output and input spatial dimensions.
	inputSpatialDims := gatherSlice(axes.InputSpatial, x.Shape().Dimensions)
	outputSpatialDims := gatherSlice(axes.OutputSpatial, node.Shape().Dimensions)
	kernelSpatialDims := gatherSlice(axes.KernelSpatial, kernel.Shape().Dimensions)
	for axisIdx := 0; axisIdx < numSpatialDims; axisIdx++ {
		// Get all the metrics for this spatial dimension.
		kernelDimSize := kernelSpatialDims[axisIdx]
		dimFilterDilation := 1
		if len(kernelDilations) > 0 {
			dimFilterDilation = kernelDilations[axisIdx]
		}
		inputDimSize := inputSpatialDims[axisIdx]
		outputDimSize := outputSpatialDims[axisIdx]
		dimStride := 1
		if len(strides) > 0 {
			dimStride = strides[axisIdx]
		}
		var dimPadding [2]int
		if len(paddings) > 0 {
			dimPadding = paddings[axisIdx]
		}
		expectedOutputDimSize := expectedOutputSize(inputDimSize, kernelDimSize, dimFilterDilation, dimStride, dimPadding)

		if expectedOutputDimSize != outputDimSize {
			Panicf("failed to set up reverse Convolve() for gradient in spatial dimension %d: "+
				"outputDimSize=%d, but input size is %d, filter size is %d, dilation is %d, and stride is %d",
				axisIdx, outputDimSize, inputDimSize, kernelDimSize, dimFilterDilation, dimStride)
		}

		// Reverse kernel size: the output of the original, modified by the kernelDilations (set to the
		// original convolution strides)
		revKernelDimSize := outputDimSize
		revDimDilation := 1
		if len(reverseDilations) > 0 {
			revDimDilation = reverseDilations[axisIdx]
		}
		revDimStride := 1
		if len(reverseStrides) > 0 {
			revDimStride = reverseStrides[axisIdx]
		}
		revDimPadding := reversePaddings[axisIdx]
		revExpectedOutputDimSize := expectedOutputSize(inputDimSize, revKernelDimSize, revDimDilation, revDimStride, revDimPadding)

		// Adjust revPadding to make revOutputSize to match the original
		revDimExtraPadding := (kernelDimSize - revExpectedOutputDimSize) * revDimStride
		reversePaddings[axisIdx][1] += revDimExtraPadding // Adjustment made to the end.
	}

	// (3) Run the reverse convolution of the VJP.
	revConv := Convolve(x, reverseKernel).
		AxesConfig(reverseAxes)
	if len(reversePaddings) > 0 {
		revConv.PaddingPerDim(reversePaddings)
	} else {
		revConv.NoPadding()
	}
	if len(reverseStrides) > 0 {
		revConv.StridePerAxis(reverseStrides...)
	}
	if len(reverseDilations) > 0 {
		revConv.DilationPerAxis(reverseDilations...)
	}
	output := revConv.Done()
	return output
}
