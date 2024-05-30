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
	. "github.com/gomlx/gomlx/types/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	timage "github.com/gomlx/gomlx/types/tensor/image"
	"github.com/gomlx/gomlx/xla"
)

// This file contains all parts of the Convolve implementation.

// ConvolutionBuilder is a helper to build a convolution computation.
// Create it with Convolve, set the desired parameters and
// when set, call `IsNil()`.
type ConvolutionBuilder struct {
	graph                             *Graph
	x, kernel                         *Node
	numSpatialDims                    int
	strides                           []int
	paddings                          [][2]int
	padSame                           bool
	inputDilation, filterDilation     []int
	filterGroupCount, batchGroupCount int

	channelsAxisConfig timage.ChannelsAxisConfig
	axes               ConvolveAxesConfig
}

// Convolve prepares a convolution on x with the given kernel for arbitrary
// number of spatial dimensions (1D, 2D, 3D, etc.).
//
// It returns a ConvolutionBuilder object that can be further configured. Once the
// configuration is finished, call `ConvolutionBuilder.Done` and it will return
// the convolved x. Browse through ConvolutionBuilder to see its capabilities
// and defaults.
//
// The shape of x should be `[batch, <spatial_dimensions...>, input_channels]` if
// configured with `ConvolutionBuilder.ChannelsAxis(timage.ChannelsLast)`, the default.
// If one sets `ConvolutionBuilder.ChannelsAxis(timage.ChannelsFirst)`, the shape should be
// `[batch, input_channels, <spatial_dimensions...>]` instead.
//
// Note: `timage` refers to package `github.com/gomlx/gomlx/types/tensor/image`.
//
// The shape of kernel should be `[<spatial_dimensions...>, input_channels, output_channels]` if
// configured with `ConvolutionBuilder.ChannelsAxis(timage.ChannelsLast)`, the default. If one
// sets `ConvolutionBuilder.ChannelsAxis(timage.ChannelsFirst)`, the shape should be
// `[input_channels, <spatial_dimensions...>, output_channels]` instead.
//
// Notice x and kernel must have the same rank.
//
// We follow the Keras convention of calling the "depth" or "feature" or "channels" dimension
// "channels". Likewise, we use "kernel" instead of "filters" -- but they mean the same.
func Convolve(x, kernel *Node) *ConvolutionBuilder {
	conv := &ConvolutionBuilder{
		graph:            validateGraphFromInputs(x, kernel),
		x:                x,
		kernel:           kernel,
		filterGroupCount: 1,
		batchGroupCount:  1,
	}
	conv.numSpatialDims = x.Rank() - 2
	if conv.numSpatialDims < 0 {
		Panicf("Input x must have rank >= 3, shaped by default as [batch, <spatial_dimensions...>, channels], "+
			"but x rank is %d", x.Rank())
	}
	if kernel.Rank() != x.Rank() {
		Panicf("Input x (rank %d) must have same rank as the kernel (rank %d) -- x has a batch dimension, "+
			"and kernel has an output_channels dimension", x.Rank(), kernel.Rank())
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
// Note: `timage` refers to package `github.com/gomlx/gomlx/types/tensor/image`.
//
// For more fine-control, see AxesConfig.
//
// It returns the modified Config object, so calls can be cascaded.
func (conv *ConvolutionBuilder) ChannelsAxis(channelsAxisConfig timage.ChannelsAxisConfig) *ConvolutionBuilder {
	conv.channelsAxisConfig = channelsAxisConfig
	conv.axes.InputBatch = 0
	conv.axes.InputChannel = timage.GetChannelsAxis(conv.x, channelsAxisConfig)
	conv.axes.InputSpatial = timage.GetSpatialAxes(conv.x, channelsAxisConfig)

	switch channelsAxisConfig {
	case timage.ChannelsFirst:
		conv.axes.KernelInputChannel = 0
		conv.axes.KernelSpatial = slices.Iota(1, conv.numSpatialDims)
		conv.axes.KernelOutputChannel = conv.numSpatialDims + 1

		conv.axes.OutputBatch = 0
		conv.axes.OutputChannel = 1
		conv.axes.OutputSpatial = slices.Iota(2, conv.numSpatialDims)

	case timage.ChannelsLast:
		conv.axes.KernelInputChannel = conv.numSpatialDims
		conv.axes.KernelOutputChannel = conv.numSpatialDims + 1
		conv.axes.KernelSpatial = slices.Iota(0, conv.numSpatialDims)

		conv.axes.OutputBatch = 0
		conv.axes.OutputSpatial = slices.Iota(1, conv.numSpatialDims)
		conv.axes.OutputChannel = conv.numSpatialDims + 1
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
	perDim := slices.SliceWithValue(conv.numSpatialDims, strides)
	return conv.StridePerDim(perDim...)
}

// StridePerDim sets the strides for each spatial dimension of the convolution.
// The default is 1 for every dimension.
//
// The stride is how many steps to move after a convolution.
// A value of 2 will halve the input size, since a convolution will be done at every other position, and so on.
// It can be defined separately per dimension.
//
// One cannot use strides and dilation at the same time.
func (conv *ConvolutionBuilder) StridePerDim(strides ...int) *ConvolutionBuilder {
	if len(strides) != conv.numSpatialDims {
		Panicf("received %d strides in StridePerDim, but x has %d spatial dimensions",
			len(strides), conv.numSpatialDims)
	}
	conv.strides = strides
	return conv
}

// PadSame adds paddings on the edges of x such that in the end the output
// of the convolution has the same shape as the input (assuming strides=1).
//
// The default is no padding. See also NoPadding and PaddingPerDim.
func (conv *ConvolutionBuilder) PadSame() *ConvolutionBuilder {
	conv.paddings = nil
	conv.padSame = true
	return conv
}

// NoPadding removes any paddings, so if the kernel spatial dimensions > 1,
// the output shape will be reduced on the edges. This is the default.
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
// The default is no padding. See also NoPadding and PadSame.
func (conv *ConvolutionBuilder) PaddingPerDim(paddings [][2]int) *ConvolutionBuilder {
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
// The default is 1.
//
// It specifies the kernel up-sampling rate. In the literature, the same parameter
// is sometimes called input stride or dilation. The effective kernel size used for the convolution
// will be `kernel_shape + (kernel_shape - 1) * (dilation - 1)`, obtained by inserting (dilation-1) zeros
// between consecutive elements of the original filter in the spatial dimension.
//
// One cannot use strides and dilation at the same time.
func (conv *ConvolutionBuilder) Dilations(dilation int) *ConvolutionBuilder {
	dilationsPerDim := slices.SliceWithValue(conv.numSpatialDims, dilation)
	return conv.DilationPerDim(dilationsPerDim...)
}

// DilationPerDim sets the kernel dilations for each spatial dimension of the convolution.
// The default is 1 for every dimension.
//
// It specifies the kernel up-sampling rate. In the literature, the same parameter
// is sometimes called input stride or dilation. The effective kernel size used for the convolution
// will be `kernel_shape + (kernel_shape - 1) * (dilation - 1)`, obtained by inserting (dilation-1) zeros
// between consecutive elements of the original filter in the spatial dimension.
//
// One cannot use strides and dilation at the same time.
func (conv *ConvolutionBuilder) DilationPerDim(dilations ...int) *ConvolutionBuilder {
	if len(dilations) == 0 {
		conv.filterDilation = nil
		return conv
	}
	if len(dilations) != conv.numSpatialDims {
		Panicf("received %d dilations in DilationPerDim, but x has %d spatial dimensions",
			len(dilations), conv.numSpatialDims)
	}
	conv.filterDilation = dilations
	return conv
}

// InputDilationPerDim is used when generating the gradient of a convolution with strides.
// It effectively inserts zeros in the input, making it effectively larger than it actually is.
// The gradient of Convolve with input dilation is not implemented yet, careful.
func (conv *ConvolutionBuilder) InputDilationPerDim(dilations ...int) *ConvolutionBuilder {
	if len(dilations) == 0 {
		conv.inputDilation = nil
		return conv
	}
	if len(dilations) != conv.numSpatialDims {
		Panicf("received %d dilations in inputDilationPerDim, but x has %d spatial dimensions",
			len(dilations), conv.numSpatialDims)
	}
	conv.inputDilation = dilations
	return conv
}

// Done indicates that the convolve operation is finished being configured and
// it updates the computation graph with convolution, and returns the resulting
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
			if conv.filterDilation != nil {
				dilation = conv.filterDilation[dim]
			}
			kernelSize = (kernelSize-1)*dilation + 1
			paddings[dim][0] = (kernelSize - 1) / 2 // For even sized kernels, the padding is asymmetric.
			paddings[dim][1] = kernelSize / 2
		}
	}

	// Check only one of "strides" or "dilations" are set.
	var dilationsSet, stridesSet bool
	if conv.strides != nil {
		for _, stride := range conv.strides {
			if stride != 1 {
				stridesSet = true
			}
		}
	}
	if conv.filterDilation != nil {
		for _, dilation := range conv.filterDilation {
			if dilation != 1 {
				dilationsSet = true
			}
		}
	}
	if dilationsSet && stridesSet {
		Panicf("both strides (%v) and dilations (%v) are set, but only one can be used at a time",
			conv.strides, conv.filterDilation)
	}

	return convGeneralDilatedXLA(conv.x, conv.kernel,
		conv.axes, conv.strides,
		paddings, conv.inputDilation, conv.filterDilation,
		conv.filterGroupCount, conv.batchGroupCount)
}

// ConvolveAxesConfig defines the interpretation of the input/kernel/output tensor axes.
// There must be the same number of spatial dimensions (axes) for each of the 3 tensors.
// Input and output has batch and channel axes. Kernel has inputChannel and outputChannel axes.
type ConvolveAxesConfig struct {
	InputBatch, InputChannel int
	InputSpatial             []int

	KernelInputChannel, KernelOutputChannel int
	KernelSpatial                           []int

	OutputBatch, OutputChannel int
	OutputSpatial              []int
}

// convGeneralDilatedXLA is a generic Convolution operation offered by XLA. See Convolve
// for the simpler version.
// featureAxisAfter defines whether the features (aka. channels or depth) axis comes after the
// spatial dimension. Example: a 2D input can be one of the two:
//
//   - featureAxisAfter=false: input=[batch_size, features, height, width], filter=[output_features, input_features, height, width]
//   - featureAxisAfter=true:  input=[batch_size, height, width, features], filter=[output_features, height, width, input_features]
//
// Some details in https://www.tensorflow.org/xla/operation_semantics#convwithgeneralpadding_convolution.
// (XLA documentation is really poor here, much is guess-work).
// Also useful is https://arxiv.org/pdf/1603.07285v1.pdf.
// Not exported for now, hopefully Convolve will suffice.
func convGeneralDilatedXLA(input, filter *Node, axes ConvolveAxesConfig,
	strides []int, paddings [][2]int, inputDilation, filterDilation []int,
	filterGroupCount, batchGroupCount int) *Node {
	g := validateGraphFromInputs(input, filter)
	numSpatialDims := input.Rank() - 2
	if len(axes.InputSpatial) != numSpatialDims || len(axes.OutputSpatial) != numSpatialDims || len(axes.KernelSpatial) != numSpatialDims {
		Panicf("convGeneralDilatedXLA: input has %d spatial dimensions, but axes configuration has %d, %d, %d spatial axes configured "+
			"for input/kernel/output", numSpatialDims, len(axes.InputSpatial), len(axes.KernelSpatial), len(axes.OutputSpatial))
	}

	// Encoding of the values as follows. IMPORTANT: this code needs to be in sync with corresponding
	// decoding code in c/gomlx/computation.cpp, in function ComputationAddOp, under GatherNode case.
	//  * 8 first elements store the various parameters and lengths:
	axesConfigLen := 3 * (numSpatialDims + 2)
	ints := make([]int, 0, 7+axesConfigLen+len(strides)+2*len(paddings)+len(inputDilation)+len(filterDilation))
	encode := func(values ...int) {
		ints = append(ints, values...)
	}

	// Encode dimensions.
	encode(numSpatialDims, filterGroupCount, batchGroupCount)
	encode(len(strides), len(paddings), len(inputDilation), len(filterDilation))

	// Append axes configuration.
	encode(axes.InputBatch, axes.InputChannel)
	encode(axes.InputSpatial...)
	encode(axes.KernelInputChannel, axes.KernelOutputChannel)
	encode(axes.KernelSpatial...)
	encode(axes.OutputBatch, axes.OutputChannel)
	encode(axes.OutputSpatial...)

	// Append arrays of ints.
	encode(strides...)
	for _, pair := range paddings {
		encode(pair[0], pair[1])
	}
	encode(inputDilation...)
	encode(filterDilation...)

	return newNode(g, &xla.SerializedNode{
		Type: xla.ConvGeneralDilatedNode,
		Ints: ints,
	}, []*Node{input, filter})
}

func convGeneralDilatedVJP(node, v *Node, _ shapes.Shape) []*Node {
	// Recover parameters from serialized node.
	x := node.inputs[0]
	kernel := node.inputs[1]
	packedPos := 0
	decode := func() int {
		i := node.serializedNode.Ints[packedPos]
		packedPos++
		return i
	}
	decodeN := func(n int) []int {
		slice := node.serializedNode.Ints[packedPos : packedPos+n]
		packedPos += n
		return slice
	}

	numSpatialDims := decode()
	filterGroupCount, batchGroupCount := decode(), decode()
	lenStrides := decode()
	lenPadding := decode()
	lenInputDilation := decode()
	lenFilterDilation := decode()

	var axes ConvolveAxesConfig
	axes.InputBatch, axes.InputChannel = decode(), decode()
	axes.InputSpatial = decodeN(numSpatialDims)
	axes.KernelInputChannel, axes.KernelOutputChannel = decode(), decode()
	axes.KernelSpatial = decodeN(numSpatialDims)
	axes.OutputBatch, axes.OutputChannel = decode(), decode()
	axes.OutputSpatial = decodeN(numSpatialDims)

	strides := decodeN(lenStrides)
	paddings := make([][2]int, lenPadding)
	for ii := range paddings {
		paddings[ii][0] = decode()
		paddings[ii][1] = decode()
	}
	inputDilation := decodeN(lenInputDilation)
	filterDilation := decodeN(lenFilterDilation)

	// Not used.
	_, _ = filterGroupCount, batchGroupCount
	_ = inputDilation

	if lenInputDilation > 0 {
		Panicf("gradient of Convolve with input dilation not defined, " +
			"usually it's only used to calculate the gradient of a convolution, so " +
			"this may occur when trying to do the gradient of a gradient.")
	}

	//fmt.Printf("\tx.shape=%s\n", x.Shape())
	//fmt.Printf("\tkernel.shape=%s\n", kernel.Shape())
	//fmt.Printf("\tnode.shape=%s\n", node.Shape())

	vjpX := convVJPWrtX(node, x, kernel, v, numSpatialDims, axes,
		strides, paddings, filterDilation)
	vjpKernel := convVJPWrtKernel(node, x, kernel, v, numSpatialDims, axes,
		strides, paddings, filterDilation)
	return []*Node{vjpX, vjpKernel}
}

// convVJPWrtX creates the Vector-Jacobian (for backpropagation) of the
// output with respect to (==wrt) the input (x). See also convVJPWrtKernel.
func convVJPWrtX(node, x, kernel, v *Node, numSpatialDims int, axes ConvolveAxesConfig,
	strides []int, paddings [][2]int, filterDilation []int) *Node {
	// Get output and input spatial dimensions.
	inputSpatialDims := gatherSlice(axes.InputSpatial, x.Shape().Dimensions)
	outputSpatialDims := gatherSlice(axes.OutputSpatial, node.Shape().Dimensions)
	kernelSpatialDims := gatherSlice(axes.KernelSpatial, kernel.Shape().Dimensions)
	//fmt.Printf("\tinputSpatialDims=%v\n", inputSpatialDims)
	//fmt.Printf("\toutputSpatialDims=%v\n", outputSpatialDims)

	// Gradient of the output with respect to x:
	// (1) we need to reverse the convolution, which involves the reverse kernel: spatial dimensions are reversed,
	//     and input/output channel weights are transposed.
	reverseKernel := Reverse(kernel, axes.KernelSpatial...)

	// Instead of transposing output/input channels, just swap their indices in the reverseAxes. Effectively this does:
	//     reverseKernel = Transpose(reverseKernel, axes.KernelOutputChannel, axes.KernelInputChannel)
	reverseAxes := axes
	reverseAxes.KernelInputChannel, reverseAxes.KernelOutputChannel = axes.KernelOutputChannel, axes.KernelInputChannel

	// (2) we need to pad the reverse convolution to match get the original input.
	reversePaddings := make([][2]int, numSpatialDims)
	dilation := 1
	for axis := 0; axis < numSpatialDims; axis++ {
		//fmt.Printf("\taxis %d\n", axis)
		// Effective kernel size.
		kernelSize := kernelSpatialDims[axis]
		if len(filterDilation) > 0 {
			dilation = filterDilation[axis]
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
		//fmt.Printf("\t\tinput start/end: %d, %d\n", inputDimStart, inputDimEnd)

		if (inputDimEnd-inputDimStart+(dimStride-1))/dimStride != outputDimSize {
			Panicf("failed to set up reverse Convolve() for gradient in spatial dimension %d: "+
				"outputDimSize=%d, but input convolution window is %d with stride for this dimension %d",
				axis, outputDimSize, inputDimEnd-inputDimStart, dimStride)
		}

		// Start/End positions on the output for the reverse convolution.
		// Values below 0 or above outputDimSize means padding. It has to be such that it will regenerate
		// the original input spatial shape.
		// Stride in the input will be become inputDilation in the reverse convolution.
		outputDimStart := -inputDimStart - ((kernelSize - 1) / 2)
		if outputDimStart > 0 {
			Panicf("failed to set up reverse Convolve() for gradient: spatial dimension %d "+
				"outputDimStart=%d > 0, which is out-of-bounds", axis, outputDimStart)
		}
		outputDimEnd := inputDimSize - inputDimStart + kernelSize/2
		// So far outputDimEnd and outputDimStart hasn't considered the strides converted to input dilation
		// on the reverse convolution -- effectively injecting zeros.
		//fmt.Printf("\t\tno strides output start/end: %d, %d\n", outputDimStart, outputDimEnd)
		numInjectedZeros := (outputDimSize - 1) * (dimStride - 1)
		outputDimEnd -= numInjectedZeros
		if outputDimEnd < outputDimSize {
			Panicf("failed to set up reverse Convolve() for gradient: spatial dimension %d "+
				"outputDimEnd=%d < outputDimSize=%d, which is out-of-bounds", axis, outputDimEnd, outputDimSize)
		}
		//fmt.Printf("\t\toutput start/end: %d, %d\n", outputDimStart, outputDimEnd)

		// Set padding to the output to match it's start/end positions.
		reversePaddings[axis][0] = -outputDimStart
		reversePaddings[axis][1] = outputDimEnd - outputDimSize
	}
	// (3) Run2 the reverse convolution of the VJP.
	//fmt.Printf("\treversePaddings=%v\n", reversePaddings)
	//fmt.Printf("\tfilterDilation=%v\n", filterDilation)
	revConv := Convolve(v, reverseKernel).PaddingPerDim(reversePaddings).DilationPerDim(filterDilation...).AxesConfig(reverseAxes)
	if len(strides) > 0 {
		revConv.InputDilationPerDim(strides...)
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

// convVJPWrtX creates the Vector-Jacobian (for backpropagation) of the
// output with respect to (==wrt) the kernel (aka filters). See also convVJPWrtX.
func convVJPWrtKernel(node, x, kernel, v *Node, numSpatialDims int, axes ConvolveAxesConfig,
	strides []int, paddings [][2]int, filterDilation []int) *Node {
	//fmt.Printf("\nconvVJPWrtKernel input:\n")
	//fmt.Printf("\tnode.shape=%s, x.shape=%s, kernel.shape=%s, v.shape=%s\n", node.Shape(), x.Shape(), kernel.Shape(), v.Shape())
	//fmt.Printf("\tnumSpatialDims=%d, axes=%+v\n", numSpatialDims, axes)
	//fmt.Printf("\tstrides=%v, paddings=%v, filterDilation=%v\n", strides, paddings, filterDilation)

	// (1) For the Gradient of the output with respect to kernel we need a reverse convolution of
	// the original input using v (the term from VJP, shaped as the original output) as the
	// "reverseKernel". Since we need to multiply it by most of the inputs to get the VJP wrt
	// to the kernel. The output of this reverse convolution will be shaped like original
	// convolution kernel, if we adjust correctly the axes, see below.
	reverseKernel := v

	// The batch dimension becomes like a channel: since they must be all added. But the channels, that need
	// to be generated separately become a batch dimension.
	var reverseAxes ConvolveAxesConfig
	reverseAxes.InputBatch, reverseAxes.InputChannel = axes.InputChannel, axes.InputBatch
	reverseAxes.InputSpatial = axes.InputSpatial

	// The output of the reverse convolve is the original kernel shape. The kernel input channels axis
	// is the reverse output batch axis. The output channel of the reverse convolve will goes into
	// the original kernel output channel.
	reverseAxes.OutputBatch, reverseAxes.OutputChannel = axes.KernelInputChannel, axes.KernelOutputChannel
	reverseAxes.OutputSpatial = axes.KernelSpatial

	// The kernel of the reverse node is shaped like the output of the original convolution.
	reverseAxes.KernelInputChannel, reverseAxes.KernelOutputChannel = axes.OutputBatch, axes.OutputChannel
	reverseAxes.KernelSpatial = axes.OutputSpatial

	// Strides in the original convolution become dilations for the backward convolution to the kernel.
	reverseDialations := strides
	reverseStrides := filterDilation

	// (2) we need to pad the reverse convolution to match get the original input.
	reversePaddings := make([][2]int, numSpatialDims)
	if paddings != nil {
		for ii, padding := range paddings {
			reversePaddings[ii] = padding
		}
	}

	// Get output and input spatial dimensions.
	inputSpatialDims := gatherSlice(axes.InputSpatial, x.Shape().Dimensions)
	outputSpatialDims := gatherSlice(axes.OutputSpatial, node.Shape().Dimensions)
	kernelSpatialDims := gatherSlice(axes.KernelSpatial, kernel.Shape().Dimensions)
	for axisIdx := 0; axisIdx < numSpatialDims; axisIdx++ {
		//fmt.Printf("\taxis %d\n", axisIdx)
		// Get all meetrics for this spatial dimension..
		kernelDimSize := kernelSpatialDims[axisIdx]
		dimFilterDilation := 1
		if len(filterDilation) > 0 {
			dimFilterDilation = filterDilation[axisIdx]
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
				"outputDimSize=%d, but input size is %d, kernel size is %d, dilation is %d, and stride is %d",
				axisIdx, outputDimSize, inputDimSize, kernelDimSize, dimFilterDilation, dimStride)
		}

		// Reverse kernel size: the output of the original, modified by the filterDilation (set to the
		// original convolution strides)
		revKernelDimSize := outputDimSize
		revDimDilation := 1
		if len(reverseDialations) > 0 {
			revDimDilation = reverseDialations[axisIdx]
		}
		revDimStride := 1
		if len(reverseStrides) > 0 {
			revDimStride = reverseStrides[axisIdx]
		}
		revDimPadding := reversePaddings[axisIdx]
		revExpectedOutputDimSize := expectedOutputSize(inputDimSize, revKernelDimSize, revDimDilation, revDimStride, revDimPadding)
		//fmt.Printf("\t\trevOut=%d, kernelSize=%d\n", revExpectedOutputDimSize, kernelDimSize)

		// Adjust revPadding to make revOutputSize to match the original
		revDimExtraPadding := (kernelDimSize - revExpectedOutputDimSize) * revDimStride
		//fmt.Printf("\t\textraPad=%d\n", revDimExtraPadding)
		reversePaddings[axisIdx][1] += revDimExtraPadding // Adjustment made to the end.
	}

	// (3) Run2 the reverse convolution of the VJP.
	//fmt.Printf("\treversePaddings=%v\n", reversePaddings)
	//fmt.Printf("\tfilterDilation=%v\n", filterDilation)
	revConv := Convolve(x, reverseKernel).
		AxesConfig(reverseAxes)
	if len(reversePaddings) > 0 {
		revConv.PaddingPerDim(reversePaddings)
	} else {
		revConv.NoPadding()
	}
	if len(reverseStrides) > 0 {
		revConv.StridePerDim(reverseStrides...)
	}
	if len(reverseDialations) > 0 {
		revConv.DilationPerDim(reverseDialations...)
	}
	output := revConv.Done()
	//fmt.Printf("convVJPWrtKernel output:\n")
	//fmt.Printf("\trev: x.shape=%s, kernel.shape=%s, output.shape=%s\n", x.Shape(), reverseKernel.Shape(), output.Shape())
	//fmt.Printf("\tnumSpatialDims=%d, axes=%+v\n", revConv.numSpatialDims, revConv.axes)
	//fmt.Printf("\tstrides=%v, paddings=%v, filterDilation=%v\n", revConv.strides, revConv.paddings, revConv.filterDilation)
	//fmt.Printf("\n")
	return output
}
