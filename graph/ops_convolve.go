package graph

import (
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	timage "github.com/gomlx/gomlx/types/tensors/images"
	"github.com/gomlx/gomlx/types/xslices"
)

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

func (conv *ConvolutionBuilder) FilterGroupCount(groupCount int) *ConvolutionBuilder {
	conv.filterGroupCount = groupCount
	return conv
}

func Convolve(x, kernel *Node) *ConvolutionBuilder {
	conv := &ConvolutionBuilder{
		graph:            validateBuildingGraphFromInputs(x, kernel),
		x:                x,
		kernel:           kernel,
		filterGroupCount: 1,
		batchGroupCount:  1,
	}

	conv.numSpatialDims = x.Rank() - 2
	if conv.numSpatialDims < 0 {
		Panicf("Input x must have rank >= 3, shaped by default as [batch, , channels], "+
			"but x rank is %d", x.Rank())
	}

	if kernel.Rank() != x.Rank() {
		Panicf("Input x (rank %d) must have same rank as the kernel (rank %d) -- x has a batch dimension, "+
			"and kernel has an output_channels dimension", x.Rank(), kernel.Rank())
	}

	return conv.ChannelsAxis(timage.ChannelsLast).NoPadding()
}

func gatherSlice(indices, params []int) (slice []int) {
	slice = make([]int, len(indices))
	for ii := range slice {
		slice[ii] = params[indices[ii]]
	}
	return
}

func (conv *ConvolutionBuilder) ChannelsAxis(channelsAxisConfig timage.ChannelsAxisConfig) *ConvolutionBuilder {
	conv.channelsAxisConfig = channelsAxisConfig
	conv.axes.InputBatch = 0
	conv.axes.InputChannel = timage.GetChannelsAxis(conv.x, channelsAxisConfig)
	conv.axes.InputSpatial = timage.GetSpatialAxes(conv.x, channelsAxisConfig)

	switch channelsAxisConfig {
	case timage.ChannelsFirst:
		conv.axes.KernelInputChannel = 0
		conv.axes.KernelSpatial = xslices.Iota(1, conv.numSpatialDims)
		conv.axes.KernelOutputChannel = conv.numSpatialDims + 1

		conv.axes.OutputBatch = 0
		conv.axes.OutputChannel = 1
		conv.axes.OutputSpatial = xslices.Iota(2, conv.numSpatialDims)

	case timage.ChannelsLast:
		conv.axes.KernelInputChannel = conv.numSpatialDims
		conv.axes.KernelOutputChannel = conv.numSpatialDims + 1
		conv.axes.KernelSpatial = xslices.Iota(0, conv.numSpatialDims)

		conv.axes.OutputBatch = 0
		conv.axes.OutputSpatial = xslices.Iota(1, conv.numSpatialDims)
		conv.axes.OutputChannel = conv.numSpatialDims + 1
	}
	return conv
}

func (conv *ConvolutionBuilder) AxesConfig(axes ConvolveAxesConfig) *ConvolutionBuilder {
	conv.axes = axes
	return conv
}

func (conv *ConvolutionBuilder) Strides(strides int) *ConvolutionBuilder {
	perDim := xslices.SliceWithValue(conv.numSpatialDims, strides)
	return conv.StridePerDim(perDim...)
}

func (conv *ConvolutionBuilder) StridePerDim(strides ...int) *ConvolutionBuilder {
	if len(strides) != conv.numSpatialDims {
		Panicf("received %d strides in StridePerAxis, but x has %d spatial dimensions",
			len(strides), conv.numSpatialDims)
	}
	conv.strides = strides
	return conv
}

func (conv *ConvolutionBuilder) PadSame() *ConvolutionBuilder {
	conv.paddings = nil
	conv.padSame = true
	return conv
}

func (conv *ConvolutionBuilder) NoPadding() *ConvolutionBuilder {
	conv.paddings = nil
	conv.padSame = false
	return conv
}

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

func (conv *ConvolutionBuilder) Dilations(dilation int) *ConvolutionBuilder {
	dilationsPerDim := xslices.SliceWithValue(conv.numSpatialDims, dilation)
	return conv.DilationPerDim(dilationsPerDim...)
}

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

	return ConvGeneralDilated(conv.x, conv.kernel,
		conv.axes, conv.strides,
		paddings, conv.inputDilation, conv.filterDilation,
		conv.filterGroupCount, conv.batchGroupCount)
}

type ConvolveAxesConfig = backends.ConvolveAxesConfig

func ConvGeneralDilated(input, kernel *Node, axes ConvolveAxesConfig,
	strides []int, paddings [][2]int, inputDilation, filterDilation []int,
	filterGroupCount, batchGroupCount int) *Node {
	_ = validateBuildingGraphFromInputs(input, kernel)
	numSpatialDims := input.Rank() - 2
	if len(axes.InputSpatial) != numSpatialDims || len(axes.OutputSpatial) != numSpatialDims || len(axes.KernelSpatial) != numSpatialDims {
		Panicf("ConvGeneralDilated: input has %d spatial dimensions, but axes configuration has %d, %d, %d spatial axes configured "+
			"for input/kernel/output", numSpatialDims, len(axes.InputSpatial), len(axes.KernelSpatial), len(axes.OutputSpatial))
	}
	return backendConvGeneralDilated(input, kernel, axes, strides, paddings, inputDilation, filterDilation, filterGroupCount, batchGroupCount)
}

func convGeneralDilatedVJP(node, v *Node, _ shapes.Shape) []*Node {
	// Recover parameters from serialized node.
	x := node.inputNodes[0]
	kernel := node.inputNodes[1]
	params := node.inputs.(*nodeInputsConvGeneralDilated)
	numSpatialDims := x.Rank() - 2
	if len(params.inputDilation) > 0 {
		Panicf("gradient of Convolve with input dilation not defined, " +
			"usually it's only used to calculate the gradient of a convolution, so " +
			"this may occur when trying to do the gradient of a gradient.")
	}

	//fmt.Printf("\tx.shapes=%s\n", x.Shape())
	//fmt.Printf("\tkernel.shapes=%s\n", kernel.Shape())
	//fmt.Printf("\tnode.shapes=%s\n", node.Shape())

	vjpX := convVJPWrtX(node, x, kernel, v, numSpatialDims, params.axes,
		params.strides, params.paddings, params.filterDilation)
	vjpKernel := convVJPWrtKernel(node, x, kernel, v, numSpatialDims, params.axes,
		params.strides, params.paddings, params.filterDilation)
	return []*Node{vjpX, vjpKernel}
}

func convVJPWrtX(node, x, kernel, v *Node, numSpatialDims int, axes ConvolveAxesConfig,
	strides []int, paddings [][2]int, filterDilation []int) *Node {
	inputSpatialDims := gatherSlice(axes.InputSpatial, x.Shape().Dimensions)
	outputSpatialDims := gatherSlice(axes.OutputSpatial, node.Shape().Dimensions)
	kernelSpatialDims := gatherSlice(axes.KernelSpatial, kernel.Shape().Dimensions)

	reverseKernel := Reverse(kernel, axes.KernelSpatial...)

	reverseAxes := axes
	reverseAxes.KernelInputChannel, reverseAxes.KernelOutputChannel = axes.KernelOutputChannel, axes.KernelInputChannel

	reversePaddings := make([][2]int, numSpatialDims)
	dilation := 1
	for axis := 0; axis < numSpatialDims; axis++ {
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

		outputDimStart := -inputDimStart - ((kernelSize - 1) / 2)
		if outputDimStart > 0 {
			Panicf("failed to set up reverse Convolve() for gradient: spatial dimension %d "+
				"outputDimStart=%d > 0, which is out-of-bounds", axis, outputDimStart)
		}
		outputDimEnd := inputDimSize - inputDimStart + kernelSize/2
		numInjectedZeros := (outputDimSize - 1) * (dimStride - 1)
		outputDimEnd -= numInjectedZeros
		if outputDimEnd < outputDimSize {
			Panicf("failed to set up reverse Convolve() for gradient: spatial dimension %d "+
				"outputDimEnd=%d < outputDimSize=%d, which is out-of-bounds", axis, outputDimEnd, outputDimSize)
		}
		reversePaddings[axis][0] = -outputDimStart
		reversePaddings[axis][1] = outputDimEnd - outputDimSize
	}
	revConv := Convolve(v, reverseKernel).PaddingPerDim(reversePaddings).DilationPerDim(filterDilation...).AxesConfig(reverseAxes)
	if len(strides) > 0 {
		revConv.InputDilationPerDim(strides...)
	}
	return revConv.Done()
}

func expectedOutputSize(inputSize, kernelSize, dilation, stride int, padding [2]int) int {
	effectiveKernelSize := (kernelSize-1)*dilation + 1
	inputStart := (effectiveKernelSize-1)/2 - padding[0]
	inputEnd := inputSize - effectiveKernelSize/2 + padding[1]
	return (inputEnd - inputStart + (stride - 1)) / stride
}

func convVJPWrtKernel(node, x, kernel, v *Node, numSpatialDims int, axes ConvolveAxesConfig,
	strides []int, paddings [][2]int, filterDilation []int) *Node {

	reverseKernel := v

	var reverseAxes ConvolveAxesConfig
	reverseAxes.InputBatch, reverseAxes.InputChannel = axes.InputChannel, axes.InputBatch
	reverseAxes.InputSpatial = axes.InputSpatial

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
	return output
}
