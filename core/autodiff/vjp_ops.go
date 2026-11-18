// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package autodiff

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/internal/graphbridge"
	"github.com/gomlx/gomlx/internal/graphimpl"
	. "github.com/gomlx/gomlx/support/exceptions"
	"github.com/gomlx/gomlx/support/sets"
)

func validateBuildingGraphFromInputs(inputs ...*graph.Node) *graph.Graph {
	return graphbridge.Graph(graphimpl.ValidateBuildingGraphFromInputs(graphbridge.NodeImpls(inputs)...))
}

func splitNode(node *graph.Node) []*graph.Node {
	return graphbridge.Nodes(graphimpl.SplitNode(graphbridge.NodeImpl(node)))
}

// concatenateVJP implements a VJP function for the Concatenate operation.
func concatenateVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	inputs := node.Inputs()
	vjps := make([]*graph.Node, 0, len(inputs))
	params := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsConcatenate)
	concatAxis := params.Axis
	shape := node.Shape()

	ranges := make([]graph.SliceAxisSpec, shape.Rank())
	for dim := 0; dim < shape.Rank(); dim++ {
		if dim == concatAxis {
			continue
		}
		ranges[dim] = graph.AxisRange()
	}

	concatDimStart := 0
	for _, input := range inputs {
		concatDimEnd := concatDimStart + input.Shape().Dimensions[concatAxis]
		ranges[concatAxis] = graph.AxisRange(concatDimStart, concatDimEnd)
		concatDimStart = concatDimEnd
		vjps = append(vjps, graph.Slice(v, ranges...))
	}
	return vjps
}

func dotCrossAxes(input *graph.Node, contractingAxes, batchAxes []int) (crossAxes []int) {
	rank := input.Rank()
	used := make([]bool, rank)
	for _, axis := range contractingAxes {
		used[axis] = true
	}
	for _, axis := range batchAxes {
		used[axis] = true
	}

	crossAxes = make([]int, 0, rank-len(contractingAxes)-len(batchAxes))
	for ii := range used {
		if !used[ii] {
			crossAxes = append(crossAxes, ii)
		}
	}
	return
}

// dotGeneralVJP generates the gradient with respect to the lhs (left-hand-side) and rhs (right-hand-side) operands.
func dotGeneralVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	params := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsDotGeneral)
	lhs, rhs := graphbridge.Node(params.Lhs), graphbridge.Node(params.Rhs)
	lhsCrossAxes := dotCrossAxes(lhs, params.LhsContractingAxes, params.LhsBatchAxes)
	rhsCrossAxes := dotCrossAxes(rhs, params.RhsContractingAxes, params.RhsBatchAxes)

	computationDType := v.DType()
	if params.Config.AccumulatorDType != dtypes.InvalidDType {
		computationDType = params.Config.AccumulatorDType
	} else if params.Config.OutputDType != dtypes.InvalidDType {
		computationDType = params.Config.OutputDType
	}

	gradFn := func(thisInput *graph.Node, thisBatchAxes, thisContractingAxes, thisCrossAxes []int, thisCrossesFirst bool,
		otherInput *graph.Node, otherBatchAxes, otherContractingAxes, otherCrossAxes []int) *graph.Node {
		_ = thisInput
		numBatchAxes := len(thisBatchAxes)
		numContractionAxes := len(thisContractingAxes)

		var thisVJP *graph.Node
		if len(otherCrossAxes) > 0 {
			vv := v
			if vv.DType() != computationDType {
				vv = graph.ConvertDType(vv, computationDType)
			}
			oo := otherInput
			if oo.DType() != computationDType {
				oo = graph.ConvertDType(oo, computationDType)
			}
			vOtherCrossStart := numBatchAxes
			if thisCrossesFirst {
				vOtherCrossStart += len(thisCrossAxes)
			}
			thisVJP = graph.DotGeneral(
				vv, xslices.Iota(vOtherCrossStart, len(otherCrossAxes)), xslices.Iota(0, numBatchAxes),
				oo, otherCrossAxes, otherBatchAxes)
		} else {
			thisVJP = v
			if thisVJP.DType() != computationDType {
				thisVJP = graph.ConvertDType(thisVJP, computationDType)
			}
			if numContractionAxes > 0 {
				thisVJP = graph.InsertAxes(thisVJP, xslices.SliceWithValue(numContractionAxes, -1)...)
			}
			otherProjected := otherInput
			if otherProjected.DType() != computationDType {
				otherProjected = graph.ConvertDType(otherProjected, computationDType)
			}
			otherRank := otherProjected.Rank()
			permutations := make([]int, 0, otherRank)
			permutations = append(permutations, otherBatchAxes...)
			permutations = append(permutations, otherCrossAxes...)
			permutations = append(permutations, otherContractingAxes...)
			changed := false
			for ii, axis := range permutations {
				if ii != axis {
					changed = true
				}
			}
			if changed {
				otherProjected = graph.TransposeAllAxes(otherProjected, permutations...)
			}
			if len(thisCrossAxes) > 0 {
				pos := numBatchAxes
				if !thisCrossesFirst {
					pos += len(otherCrossAxes)
				}
				otherProjected = graph.InsertAxes(otherProjected, xslices.SliceWithValue(len(thisCrossAxes), pos)...)
			}
			thisVJP = graph.Mul(thisVJP, otherProjected)
		}

		thisRank := thisVJP.Rank()
		{
			permutation := make([]int, thisRank)
			for ii, axis := range thisBatchAxes {
				permutation[axis] = ii
			}
			for ii, axis := range thisCrossAxes {
				permutation[axis] = ii + numBatchAxes
			}
			for ii, axis := range thisContractingAxes {
				permutation[axis] = ii + numBatchAxes + len(thisCrossAxes)
			}
			thisVJP = graph.TransposeAllAxes(thisVJP, permutation...)
		}
		if thisVJP.DType() != thisInput.DType() {
			thisVJP = graph.ConvertDType(thisVJP, thisInput.DType())
		}
		return thisVJP
	}

	return []*graph.Node{
		gradFn(lhs, params.LhsBatchAxes, params.LhsContractingAxes, lhsCrossAxes, true, rhs, params.RhsBatchAxes, params.RhsContractingAxes, rhsCrossAxes),
		gradFn(rhs, params.RhsBatchAxes, params.RhsContractingAxes, rhsCrossAxes, false, lhs, params.LhsBatchAxes, params.LhsContractingAxes, lhsCrossAxes),
	}
}

func gatherSlice(indices, params []int) (slice []int) {
	slice = make([]int, len(indices))
	for ii, idx := range indices {
		slice[ii] = params[idx]
	}
	return
}

func convGeneralVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	inputs := node.Inputs()
	x := inputs[0]
	kernel := inputs[1]
	params := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsConvGeneral)
	numSpatialDims := x.Rank() - 2
	var vjpX, vjpKernel *graph.Node
	if params.ChannelGroupCount > 1 {
		vjpX, vjpKernel = convGeneralWithChannelGroupingVJP(node, x, kernel, v, numSpatialDims, params.Axes,
			params.Strides, params.Paddings, params.InputDilations, params.KernelDilations, params.ChannelGroupCount)
	} else if params.BatchGroupCount > 1 {
		vjpX, vjpKernel = convGeneralWithBatchGroupingVJP(node, x, kernel, v, numSpatialDims, params.Axes,
			params.Strides, params.Paddings, params.InputDilations, params.KernelDilations, params.BatchGroupCount)
	} else {
		vjpX = convVJPWrtX(node, x, kernel, v, numSpatialDims, params.Axes,
			params.Strides, params.Paddings, params.InputDilations, params.KernelDilations)
		vjpKernel = convVJPWrtKernel(node, x, kernel, v, numSpatialDims, params.Axes,
			params.Strides, params.Paddings, params.InputDilations, params.KernelDilations)
	}
	return []*graph.Node{vjpX, vjpKernel}
}

func convGeneralWithChannelGroupingVJP(node, x, kernel, v *graph.Node, numSpatialDims int, axes graph.ConvolveAxesConfig,
	strides []int, paddings [][2]int, inputDilations, kernelDilations []int, channelGroupCount int) (vjpX, vjpKernel *graph.Node) {
	outputChannelsAxis := axes.OutputChannels
	vSlices := graph.Split(v, outputChannelsAxis, channelGroupCount)
	inputChannelsAxis := axes.InputChannels
	xSlices := graph.Split(x, inputChannelsAxis, channelGroupCount)
	kernelOutputChannelsAxis := axes.KernelOutputChannels
	kernelSlices := graph.Split(kernel, kernelOutputChannelsAxis, channelGroupCount)

	var xGradSlices []*graph.Node
	for i := range channelGroupCount {
		gradSlice := convVJPWrtX(node, xSlices[i], kernelSlices[i], vSlices[i], numSpatialDims, axes, strides, paddings, inputDilations, kernelDilations)
		xGradSlices = append(xGradSlices, gradSlice)
	}
	vjpX = graph.Concatenate(xGradSlices, inputChannelsAxis)

	var kernelGradSlices []*graph.Node
	for i := range channelGroupCount {
		gradSlice := convVJPWrtKernel(vSlices[i], xSlices[i], kernelSlices[i], vSlices[i], numSpatialDims, axes, strides, paddings, inputDilations, kernelDilations)
		kernelGradSlices = append(kernelGradSlices, gradSlice)
	}
	vjpKernel = graph.Concatenate(kernelGradSlices, kernelOutputChannelsAxis)
	return
}

func convGeneralWithBatchGroupingVJP(node, x, kernel, v *graph.Node, numSpatialDims int, axes graph.ConvolveAxesConfig,
	strides []int, paddings [][2]int, inputDilations, kernelDilations []int, batchGroupCount int) (vjpX, vjpKernel *graph.Node) {
	vSlices := graph.Split(v, axes.OutputChannels, batchGroupCount)
	xSlices := graph.Split(x, axes.InputBatch, batchGroupCount)
	kernelSlices := graph.Split(kernel, axes.KernelOutputChannels, batchGroupCount)

	var xGradSlices []*graph.Node
	for i := range batchGroupCount {
		gradSlice := convVJPWrtX(node, xSlices[i], kernelSlices[i], vSlices[i], numSpatialDims, axes, strides, paddings, inputDilations, kernelDilations)
		xGradSlices = append(xGradSlices, gradSlice)
	}
	vjpX = graph.Concatenate(xGradSlices, axes.InputBatch)

	var kernelGradSlices []*graph.Node
	for i := range batchGroupCount {
		gradSlice := convVJPWrtKernel(vSlices[i], xSlices[i], kernelSlices[i], vSlices[i], numSpatialDims, axes, strides, paddings, inputDilations, kernelDilations)
		kernelGradSlices = append(kernelGradSlices, gradSlice)
	}
	vjpKernel = graph.Concatenate(kernelGradSlices, axes.KernelOutputChannels)
	return
}

func convVJPWrtX(node, x, kernel, v *graph.Node, numSpatialDims int, axes graph.ConvolveAxesConfig, strides []int, paddings [][2]int, inputDilations, kernelDilations []int) *graph.Node {
	inputSpatialDims := gatherSlice(axes.InputSpatial, x.Shape().Dimensions)
	outputSpatialDims := gatherSlice(axes.OutputSpatial, node.Shape().Dimensions)
	kernelSpatialDims := gatherSlice(axes.KernelSpatial, kernel.Shape().Dimensions)

	reverseKernel := graph.Reverse(kernel, axes.KernelSpatial...)
	reverseAxes := axes
	reverseAxes.KernelInputChannels, reverseAxes.KernelOutputChannels = axes.KernelOutputChannels, axes.KernelInputChannels

	reversePaddings := make([][2]int, numSpatialDims)
	dilation := 1
	for axis := range numSpatialDims {
		kernelSize := kernelSpatialDims[axis]
		if len(kernelDilations) > 0 {
			dilation = kernelDilations[axis]
		}
		kernelSize = (kernelSize-1)*dilation + 1
		inputDimSize := inputSpatialDims[axis]
		if len(inputDilations) > 0 {
			inputDimSize = (inputDimSize-1)*inputDilations[axis] + 1
		}
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
	revConv := graph.Convolve(v, reverseKernel).PaddingPerDim(reversePaddings).DilationPerAxis(kernelDilations...).AxesConfig(reverseAxes)
	if len(strides) > 0 {
		revConv.InputDilationPerAxis(strides...)
	}
	output := revConv.Done()
	for spatialIdx, inputAxis := range axes.InputSpatial {
		if len(inputDilations) > 0 && inputDilations[spatialIdx] != 1 {
			output = graph.SliceAxis(output, inputAxis, graph.AxisRange().Stride(inputDilations[spatialIdx]))
		}
	}
	return output
}

func expectedOutputSize(inputSize, kernelSize, dilation, stride int, padding [2]int) int {
	effectiveKernelSize := (kernelSize-1)*dilation + 1
	inputStart := (effectiveKernelSize-1)/2 - padding[0]
	inputEnd := inputSize - effectiveKernelSize/2 + padding[1]
	return (inputEnd - inputStart + (stride - 1)) / stride
}

func convVJPWrtKernel(node, x, kernel, v *graph.Node, numSpatialDims int, axes graph.ConvolveAxesConfig,
	strides []int, paddings [][2]int, inputDilations, kernelDilations []int) *graph.Node {
	reverseKernel := v

	var reverseAxes graph.ConvolveAxesConfig
	reverseAxes.InputBatch, reverseAxes.InputChannels = axes.InputChannels, axes.InputBatch
	reverseAxes.InputSpatial = axes.InputSpatial
	reverseAxes.OutputBatch, reverseAxes.OutputChannels = axes.KernelInputChannels, axes.KernelOutputChannels
	reverseAxes.OutputSpatial = axes.KernelSpatial
	reverseAxes.KernelInputChannels, reverseAxes.KernelOutputChannels = axes.OutputBatch, axes.OutputChannels
	reverseAxes.KernelSpatial = axes.OutputSpatial

	reverseDilations := strides
	reverseStrides := kernelDilations

	reversePaddings := make([][2]int, numSpatialDims)
	copy(reversePaddings, paddings)

	inputSpatialDims := gatherSlice(axes.InputSpatial, x.Shape().Dimensions)
	outputSpatialDims := gatherSlice(axes.OutputSpatial, node.Shape().Dimensions)
	kernelSpatialDims := gatherSlice(axes.KernelSpatial, kernel.Shape().Dimensions)
	for axisIdx := range numSpatialDims {
		kernelDimSize := kernelSpatialDims[axisIdx]
		dimFilterDilation := 1
		if len(kernelDilations) > 0 {
			dimFilterDilation = kernelDilations[axisIdx]
		}
		inputDimSize := inputSpatialDims[axisIdx]
		if len(inputDilations) > 0 {
			inputDimSize = (inputDimSize-1)*inputDilations[axisIdx] + 1
		}
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

		revDimExtraPadding := (kernelDimSize - revExpectedOutputDimSize) * revDimStride
		reversePaddings[axisIdx][1] += revDimExtraPadding
	}

	return graph.ConvGeneral(x, reverseKernel, reverseAxes, reverseStrides, reversePaddings,
		inputDilations, reverseDilations, 1, 1)
}

func checkedSelectAndScatter(x, source *graph.Node, reduceOp graph.ReduceOpType, windowDimensions, strides []int, paddings [][2]int) *graph.Node {
	_ = validateBuildingGraphFromInputs(x, source)
	rank := x.Rank()
	if len(windowDimensions) != rank {
		Panicf("windowSizes (length %d) must have same length as rank of input x (rank %d)", len(windowDimensions), rank)
	}
	if len(strides) != rank {
		Panicf("strides (length %d) must have same length as rank of input x (rank %d)", len(strides), rank)
	}
	if len(paddings) > 0 && len(paddings) != rank {
		Panicf("paddings (length %d) if given must have same length as rank of input x (rank %d)", len(paddings), rank)
	}

	switch reduceOp {
	case compute.ReduceOpMax:
		return graphbridge.Node(graphimpl.BackendSelectAndScatterMax(graphbridge.NodeImpl(x), graphbridge.NodeImpl(source), windowDimensions, strides, paddings))
	case compute.ReduceOpMin:
		return graphbridge.Node(graphimpl.BackendSelectAndScatterMin(graphbridge.NodeImpl(x), graphbridge.NodeImpl(source), windowDimensions, strides, paddings))
	default:
		Panicf("SelectAndScatter not defined for original reduce operation %s", reduceOp)
		panic(nil)
	}
}

var vjpReductionTypes = sets.MakeWith(compute.ReduceOpMax, compute.ReduceOpMin, compute.ReduceOpSum)

func reduceWindowVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	params := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsReduceWindow)
	if !vjpReductionTypes.Has(params.ReductionType) {
		Panicf("ReduceWindow gradient only defined for the operations %q, instead got %q", vjpReductionTypes, params.ReductionType)
	}

	if len(params.InputDilations) > 0 || len(params.WindowDilations) > 0 {
		Panicf("gradient of ReduceWindow with base or window dilations is not defined")
	}
	input := graphbridge.Node(params.Input)
	var vjpX *graph.Node
	switch params.ReductionType {
	case compute.ReduceOpMax, compute.ReduceOpMin:
		vjpX = checkedSelectAndScatter(input, v, params.ReductionType, params.WindowDimensions, params.Strides, params.Paddings)
	case compute.ReduceOpSum:
		vjpX = dilateConvolveToMatchSumPooling(input, v, params.WindowDimensions, params.Strides, params.Paddings)
	default:
		Panicf("ReduceWindow gradient not defined for reduction %q, pls create an issue in github.com/gomlx/gomlx if you need this", params.ReductionType)
	}
	return []*graph.Node{vjpX}
}

func dilateConvolveToMatchSumPooling(x, backProp *graph.Node, windowDimensions, strides []int, paddings [][2]int) *graph.Node {
	g := validateBuildingGraphFromInputs(x, backProp)
	dtype := x.DType()
	rank := x.Rank()

	if len(windowDimensions) != rank {
		Panicf("windowSizes (length %d) must have the same length as the rank of x (rank %d)", len(windowDimensions), rank)
	}
	if len(strides) != rank {
		Panicf("strides (length %d) must have the same length as the rank of x (rank %d)", len(strides), rank)
	}
	if len(paddings) > 0 && len(paddings) != rank {
		Panicf("paddings (length %d) if given must have the same length as the rank of x (rank %d)", len(paddings), rank)
	}

	padConfig := make([]graph.PadAxis, rank)
	for axis := range padConfig {
		conf := &padConfig[axis]
		conf.Interior = strides[axis] - 1
		conf.Start = windowDimensions[axis] / 2
		conf.End = (windowDimensions[axis] - 1) / 2
	}
	zero := graph.ScalarZero(g, dtype)
	expanded := graph.Pad(backProp, zero, padConfig...)

	padSame := make([][2]int, rank)
	for axis := range padSame {
		windowSize := windowDimensions[axis]
		padSame[axis][0] = (windowSize - 1) / 2
		padSame[axis][1] = windowSize / 2
	}
	stridesOne := xslices.SliceWithValue(rank, 1)
	expanded = graphbridge.Node(graphimpl.BackendReduceWindow(graphbridge.NodeImpl(expanded), compute.ReduceOpSum, windowDimensions, stridesOne, nil, nil, padSame))

	var requiresAdjustment bool
	for axis := range padConfig {
		conf := &padConfig[axis]
		*conf = graph.PadAxis{}

		amountToAdjust := x.Shape().Dimensions[axis] - expanded.Shape().Dimensions[axis]
		paddingStart := 0
		if len(paddings) > 0 {
			paddingStart = paddings[axis][0]
		}
		conf.Start = -paddingStart
		conf.End = amountToAdjust - conf.Start
		if conf.Start != 0 || conf.End != 0 {
			requiresAdjustment = true
		}
	}
	var grad *graph.Node
	if requiresAdjustment {
		grad = graph.Pad(expanded, zero, padConfig...)
	} else {
		grad = expanded
	}
	return grad
}

func scatterSumVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	params := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsScatterSum)
	operandVJP := v
	updatesVJP := graph.Gather(v, graphbridge.Node(params.ScatterIndices), params.IndicesAreSorted)
	return []*graph.Node{operandVJP, nil, updatesVJP}
}

func scatterMaxOrMinVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	inputs := node.Inputs()
	operand, scatterIndices, updates := inputs[0], inputs[1], inputs[2]
	var indicesAreSorted bool
	nodeImpl := graphbridge.NodeImpl(node)
	if nodeImpl.Type() == graphimpl.NodeTypeScatterMax {
		indicesAreSorted = nodeImpl.NodeInputs().(*graphimpl.NodeInputsScatterMax).IndicesAreSorted
	} else if nodeImpl.Type() == graphimpl.NodeTypeScatterMin {
		indicesAreSorted = nodeImpl.NodeInputs().(*graphimpl.NodeInputsScatterMin).IndicesAreSorted
	}

	operandMask := graph.Equal(node, operand)
	operandVJP := graph.Where(operandMask, v, graph.ZerosLike(v))

	maxForUpdates := graph.Gather(node, scatterIndices, indicesAreSorted)
	updatesMask := graph.Equal(maxForUpdates, updates)
	updatesVJP := graph.Gather(v, scatterIndices)
	updatesVJP = graph.Where(updatesMask, updatesVJP, graph.ZerosLike(updatesVJP))
	return []*graph.Node{operandVJP, nil, updatesVJP}
}

func fftVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	params := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsFFT)
	switch params.FftType {
	case compute.FFTForward:
		size := float64(xslices.At(v.Shape().Dimensions, -1))
		return []*graph.Node{graph.MulScalar(graph.InverseFFT(v), size)}
	case compute.FFTInverse:
		invSize := 1.0 / float64(xslices.At(v.Shape().Dimensions, -1))
		return []*graph.Node{graph.MulScalar(graph.FFT(v), invSize)}
	case compute.FFTForwardReal:
		return realFftVJP(node, v)
	case compute.FFTInverseReal:
		return inverseRealFftVJP(node, v)
	}

	Panicf("Sorry, gradient for FFT of type %s not implemented, please add an issue in github.com/gomlx/gomlx if you need it!",
		params.FftType)
	return nil
}

func realFftVJP(node, v *graph.Node) []*graph.Node {
	params := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsFFT)
	fftLength := params.FftLength
	isEven := 1.0 - float64(xslices.Last(fftLength)%2)
	operand := node.Inputs()[0]
	rank := operand.Rank()
	complexDType := v.DType()
	operandLastDim := operand.Shape().Dimensions[rank-1]

	g := v.Graph()
	yMaskShape := operand.Shape().Clone()
	yMaskShape.DType = dtypes.Int32
	for ii := 0; ii < rank-1; ii++ {
		yMaskShape.Dimensions[ii] = 1
	}
	yMask := graph.Iota(g, yMaskShape, -1)
	yMask = graph.ModScalar(yMask, 2.0)
	yMask = graph.Neg(graph.AddScalar(graph.MulScalar(yMask, 2), -1))
	yMask = graph.ConvertDType(yMask, complexDType)

	y0 := graph.Slice(v, graph.AxisRange().Spacer(), graph.AxisElem(0))
	yLast := graph.Slice(v, graph.AxisRange().Spacer(), graph.AxisElem(-1))
	extraTerms := graph.Add(y0, graph.MulScalar(graph.Mul(yLast, yMask), isEven))

	irFft := graphbridge.Node(graphimpl.BackendFFT(graphbridge.NodeImpl(v), compute.FFTInverseReal, fftLength))
	newVJP := graph.MulScalar(graph.Add(graph.MulScalar(irFft, float64(operandLastDim)), graph.Real(extraTerms)), 0.5)
	return []*graph.Node{newVJP}
}

func inverseRealFftVJP(node, v *graph.Node) []*graph.Node {
	g := node.Graph()
	params := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsFFT)
	realDType := v.DType()
	fftValue := graphbridge.Node(params.Operand)
	complexDType := fftValue.DType()
	fftValueLastDim := xslices.Last(fftValue.Shape().Dimensions)
	fftLength := params.FftLength
	isFftLengthOdd := xslices.Last(fftLength) % 2

	innerMask := graph.AddScalar(graph.Ones(g, shapes.Make(realDType, fftValueLastDim-2+isFftLengthOdd)), 1.0)
	edgeMask := graph.Ones(g, shapes.Make(realDType, 1))
	var mask *graph.Node
	if isFftLengthOdd == 1 {
		mask = graph.Concatenate([]*graph.Node{edgeMask, innerMask}, 0)
	} else {
		mask = graph.Concatenate([]*graph.Node{edgeMask, innerMask, edgeMask}, 0)
	}
	mask.AssertDims(fftValueLastDim)
	mask = graph.ExpandLeftToRank(mask, fftValue.Rank())
	sizeNormalization := 1.0 / float64(xslices.Last(v.Shape().Dimensions))
	normalizedMask := graph.ConvertDType(graph.MulScalar(mask, sizeNormalization), complexDType)
	vjp := graph.Mul(graph.RealFFT(v), normalizedMask)
	return []*graph.Node{vjp}
}

func scatterAlongAxis(operand, indices, updates *graph.Node, axis int) *graph.Node {
	rank := operand.Rank()
	axis = graph.MustAdjustAxis(axis, operand)

	coordParts := make([]*graph.Node, rank)
	for i := range rank {
		if i == axis {
			coordParts[i] = indices
		} else {
			coordParts[i] = graph.ConvertDType(graph.IotaLike(indices, i), indices.DType())
		}
	}
	coords := graph.Stack(coordParts, -1)
	return graph.ScatterSum(graph.ZerosLike(operand), coords, updates, false, true)
}

func fusedTopKVJP(node *graph.Node, vjps []*graph.Node, _ shapes.Shape) []*graph.Node {
	inputs := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsFusedTopK)
	splitOutputs := splitNode(node)
	indices := splitOutputs[1]
	operand := graphbridge.Node(inputs.Operand)
	axis := graph.MustAdjustAxis(inputs.Cfg.Axis, operand)
	return []*graph.Node{scatterAlongAxis(operand, indices, vjps[0], axis)}
}

func fusedLayerNormVJP(node *graph.Node, vjp *graph.Node, _ shapes.Shape) []*graph.Node {
	inputs := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsFusedLayerNorm)
	dx, dGamma, dBeta := graphimpl.BackendFusedLayerNormVJP(inputs.X, graphbridge.NodeImpl(vjp), inputs.Axes, inputs.Epsilon, inputs.Gamma, inputs.Beta)
	res := []*graph.Node{graph.ConvertDType(graphbridge.Node(dx), inputs.X.DType())}
	if inputs.Gamma != nil {
		res = append(res, graph.ConvertDType(graphbridge.Node(dGamma), inputs.Gamma.DType()))
	}
	if inputs.Beta != nil {
		res = append(res, graph.ConvertDType(graphbridge.Node(dBeta), inputs.Beta.DType()))
	}
	return res
}

func fusedDenseVJP(node *graph.Node, vjp *graph.Node, _ shapes.Shape) []*graph.Node {
	nodeImpl := graphbridge.NodeImpl(node)
	inputs := nodeImpl.NodeInputs().(*graphimpl.NodeInputsFusedDense)
	dx, dWeight, dBias := graphimpl.BackendFusedDenseVJP(inputs.X, inputs.Weight, inputs.Bias, nodeImpl, graphbridge.NodeImpl(vjp), inputs.Options)
	if inputs.Bias != nil {
		return []*graph.Node{
			graph.ConvertDType(graphbridge.Node(dx), inputs.X.DType()),
			graph.ConvertDType(graphbridge.Node(dWeight), inputs.Weight.DType()),
			graph.ConvertDType(graphbridge.Node(dBias), inputs.Bias.DType()),
		}
	}
	return []*graph.Node{
		graph.ConvertDType(graphbridge.Node(dx), inputs.X.DType()),
		graph.ConvertDType(graphbridge.Node(dWeight), inputs.Weight.DType()),
	}
}

func fusedActivationVJP(node *graph.Node, vjp *graph.Node, _ shapes.Shape) []*graph.Node {
	nodeImpl := graphbridge.NodeImpl(node)
	inputs := nodeImpl.NodeInputs().(*graphimpl.NodeInputsFusedActivation)
	dx := graphimpl.BackendFusedActivationVJP(nodeImpl, inputs.X, graphbridge.NodeImpl(vjp), inputs.Cfg)
	return []*graph.Node{graph.ConvertDType(graphbridge.Node(dx), inputs.X.DType())}
}

func fusedScaledDotProductAttentionVJP(node *graph.Node, vjps []*graph.Node, _ shapes.Shape) []*graph.Node {
	inputs := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsFusedScaledDotProductAttention)
	splitOutputs := splitNode(node)
	output, statesForVJP := splitOutputs[0], splitOutputs[1:]
	dOutput := vjps[0]
	dQuery, dKey, dValue := graphimpl.BackendFusedScaledDotProductAttentionVJP(
		inputs.Query, inputs.Key, inputs.Value,
		inputs.AxesLayout, inputs.Options,
		graphbridge.NodeImpl(output), graphbridge.NodeImpls(statesForVJP), graphbridge.NodeImpl(dOutput))
	return []*graph.Node{
		graph.ConvertDType(graphbridge.Node(dQuery), inputs.Query.DType()),
		graph.ConvertDType(graphbridge.Node(dKey), inputs.Key.DType()),
		graph.ConvertDType(graphbridge.Node(dValue), inputs.Value.DType()),
	}
}

func fusedQuantizedDenseVJP(node *graph.Node, vjp *graph.Node, _ shapes.Shape) []*graph.Node {
	nodeImpl := graphbridge.NodeImpl(node)
	inputs := nodeImpl.NodeInputs().(*graphimpl.NodeInputsFusedQuantizedDense)
	var xForVJP, yForVJP *graphimpl.NodeImpl
	if inputs.X.Shape().IsDynamic() || inputs.X.DType() != vjp.DType() || len(inputs.X.Shape().AxisNames) > 0 {
		xForVJP = inputs.X
	}
	if inputs.Cfg.Activation.Type != compute.ActivationNone {
		yForVJP = nodeImpl
	}
	dx, dBias := graphimpl.BackendFusedQuantizedDenseVJP(xForVJP, inputs.Weight, inputs.Bias, yForVJP, graphbridge.NodeImpl(vjp), inputs.Cfg)
	vjps := make([]*graph.Node, len(node.Inputs()))
	vjps[0] = graph.ConvertDType(graphbridge.Node(dx), inputs.X.DType())
	if inputs.Bias != nil {
		vjps[len(vjps)-1] = graph.ConvertDType(graphbridge.Node(dBias), inputs.Bias.DType())
	}
	return vjps
}
