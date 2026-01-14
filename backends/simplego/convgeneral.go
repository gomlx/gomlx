// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"fmt"
	"slices"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/shapeinference"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/pkg/errors"
)

func init() {
	setNodeExecutor(backends.OpTypeConvGeneral, priorityGeneric, execConvGeneral)
}

// Auto-generate alternate specialized versions of execConvGeneral, with small changes.
// (that can't easily be refactored into smaller functions due to latency penalities)
//go:generate go run ../../internal/cmd/alternates_generator -base=convgeneral_exec.go -tags=bf16,full,full_bf16

// ConvGeneral is a generic Convolution operation with support for:
//
// - Arbitrary number of spatial axes.
// - Arbitrary transposition of axes.
// - Strides and padding.
// - Dilations of the input.
// - Dilations of the kernel, aka. atrous convolution.
// - Feature grouping (on the input channels).
// - Batch grouping.
//
// Some details in https://www.tensorflow.org/xla/operation_semantics#convwithgeneralpadding_convolution.
// There operand and filter are called lhs and rhs.
// (XLA documentation is unfortunately poor, much is guess-work).
// Also useful, https://arxiv.org/pdf/1603.07285v1.pdf.
//
// Note: input is aka. operand; kernel is aka. "filters". The input and output "channels" are also known as "features dimensions".
func (f *Function) ConvGeneral(inputOp, kernelOp backends.Value, axes backends.ConvolveAxesConfig,
	strides []int, paddings [][2]int,
	inputDilations, kernelDilations []int,
	channelGroupCount, batchGroupCount int) (backends.Value, error) {
	// Sanitize group count.
	channelGroupCount = max(channelGroupCount, 1)
	batchGroupCount = max(batchGroupCount, 1)

	opType := backends.OpTypeConvGeneral
	inputs, err := f.builder.checkOps(opType.String(), inputOp, kernelOp)
	if err != nil {
		return nil, err
	}
	input, kernel := inputs[0], inputs[1]

	outputShape, err := shapeinference.ConvGeneralOp(input.shape, kernel.shape, axes, strides, paddings, inputDilations, kernelDilations, channelGroupCount, batchGroupCount)
	if err != nil {
		fmt.Printf("ConvGeneral: input=%s, kernel=%s, output=%s, axes=%+v, strides=%v, paddings=%v, inputDilations=%v, kernelDilations=%v, channelGroupCount=%d, batchGroupCount=%d\n",
			input.shape, kernel.shape, outputShape, axes, strides, paddings, inputDilations, kernelDilations, channelGroupCount, batchGroupCount)
		return nil, err
	}

	// Sanitize parameters.
	spatialRank := outputShape.Rank() - 2
	if strides == nil {
		strides = xslices.SliceWithValue(spatialRank, 1)
	} else {
		strides = slices.Clone(strides)
	}
	if paddings == nil {
		paddings = make([][2]int, spatialRank)
	} else {
		paddings = slices.Clone(paddings)
	}
	if len(inputDilations) > 0 {
		inputDilations = slices.Clone(inputDilations)
		for i, dilation := range inputDilations {
			if dilation <= 0 {
				inputDilations[i] = 1
			}
		}
	} else {
		inputDilations = xslices.SliceWithValue(spatialRank, 1)
	}
	if len(kernelDilations) > 0 {
		kernelDilations = slices.Clone(kernelDilations)
		for i, dilation := range kernelDilations {
			if dilation <= 0 {
				kernelDilations[i] = 1
			}
		}
	}
	params := &convNode{
		axes:              axes.Clone(),
		strides:           strides,
		paddings:          paddings,
		inputDilations:    inputDilations,
		kernelDilations:   kernelDilations,
		channelGroupCount: max(channelGroupCount, 1),
		batchGroupCount:   max(batchGroupCount, 1),

		hasInputDilations:       len(inputDilations) > 0 && slices.Max(inputDilations) > 1,
		hasKernelDilations:      len(kernelDilations) > 0 && slices.Max(kernelDilations) > 1,
		inputStrides:            input.shape.Strides(),
		kernelStrides:           kernel.shape.Strides(),
		dilatedInputSpatialDims: outputShape.Dimensions,
	}

	// Generate static derived data that will be used during execution.
	params.dilatedInputSpatialDims = make([]int, spatialRank)
	params.inputSpatialStrides = make([]int, spatialRank)
	for spatialIdx, inputAxis := range axes.InputSpatial {
		params.inputSpatialStrides[spatialIdx] = params.inputStrides[inputAxis]
		dim := input.shape.Dimensions[inputAxis]
		if dim > 0 {
			params.dilatedInputSpatialDims[spatialIdx] = (dim-1)*inputDilations[spatialIdx] + 1
		}
	}
	node, _ := f.builder.getOrCreateNode(f, opType, outputShape, []*Node{input, kernel}, params)
	return node, nil
}

type convNode struct {
	axes              backends.ConvolveAxesConfig
	strides           []int
	paddings          [][2]int
	inputDilations    []int
	kernelDilations   []int
	channelGroupCount int
	batchGroupCount   int

	hasInputDilations, hasKernelDilations            bool
	inputStrides, inputSpatialStrides, kernelStrides []int

	// dilatedInputSpatialDims holds the dimensions of the input spatial axes after applying the dilations.
	// For non-dilated dimensions it's the same as the original dimension.
	dilatedInputSpatialDims []int
}

// EqualNodeData implements nodeDataComparable for convNode.
func (c *convNode) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*convNode)
	if c.channelGroupCount != o.channelGroupCount ||
		c.batchGroupCount != o.batchGroupCount ||
		c.hasInputDilations != o.hasInputDilations ||
		c.hasKernelDilations != o.hasKernelDilations {
		return false
	}
	// Compare ConvolveAxesConfig
	if c.axes.InputBatch != o.axes.InputBatch ||
		c.axes.InputChannels != o.axes.InputChannels ||
		c.axes.KernelInputChannels != o.axes.KernelInputChannels ||
		c.axes.KernelOutputChannels != o.axes.KernelOutputChannels ||
		c.axes.OutputBatch != o.axes.OutputBatch ||
		c.axes.OutputChannels != o.axes.OutputChannels {
		return false
	}
	return slices.Equal(c.axes.InputSpatial, o.axes.InputSpatial) &&
		slices.Equal(c.axes.KernelSpatial, o.axes.KernelSpatial) &&
		slices.Equal(c.axes.OutputSpatial, o.axes.OutputSpatial) &&
		slices.Equal(c.strides, o.strides) &&
		slices.Equal(c.paddings, o.paddings) &&
		slices.Equal(c.inputDilations, o.inputDilations) &&
		slices.Equal(c.kernelDilations, o.kernelDilations) &&
		slices.Equal(c.inputStrides, o.inputStrides) &&
		slices.Equal(c.inputSpatialStrides, o.inputSpatialStrides) &&
		slices.Equal(c.kernelStrides, o.kernelStrides) &&
		slices.Equal(c.dilatedInputSpatialDims, o.dilatedInputSpatialDims)
}

// ConvGeneralDilated is a deprecated an alias to ConvGeneral.
//
// Deprecated: use ConvGeneral instead.
func (f *Function) ConvGeneralDilated(inputOp, kernelOp backends.Value, axes backends.ConvolveAxesConfig,
	strides []int, paddings [][2]int,
	inputDilations, kernelDilations []int,
	channelGroupCount, batchGroupCount int) (backends.Value, error) {
	return f.ConvGeneral(inputOp, kernelOp, axes, strides, paddings, inputDilations, kernelDilations, channelGroupCount, batchGroupCount)
}

// execConvGeneral executes the DotGeneral by first normalizing and repackaging the tensors into blocks.
func execConvGeneral(backend *Backend, node *Node, inputs []*Buffer, _ []bool) (*Buffer, error) {
	input, kernel := inputs[0], inputs[1]
	params := node.data.(*convNode)
	outputShape := node.shape
	dtype := input.shape.DType
	output := backend.getBufferForShape(outputShape)
	if output == nil {
		return nil, errors.Errorf("failed allocating (out-of-memory?) output buffer shaped %s", outputShape)
	}
	output.Zeros()

	// TODO(optimizations):
	// - Optimize order of axes iterations.
	// - Split input into cache-fitting buckets ?

	// Find execution plan:
	// - We iterate the axes in order they are laid out in memory for the **output**: so we prioritize visiting the output
	//   sequentially, and each output position is visited only once -- minimizing the number of cache flushes -- cache
	//   misses will only happen in the input (or kernel, if it is large).
	plan := convGeneralExecPlan{
		backend:     backend,
		dtype:       dtype,
		inputFlat:   input.flat,
		inputShape:  input.shape,
		kernelFlat:  kernel.flat,
		kernelShape: kernel.shape,
		outputFlat:  output.flat,
		outputShape: outputShape,
		params:      params,
	}
	var convFn func(convGeneralExecPlan) error
	if params.hasInputDilations || params.hasKernelDilations || params.channelGroupCount > 1 || params.batchGroupCount > 1 {
		// Full version.
		convFn = convDTypeMap.Get(dtype).(func(convGeneralExecPlan) error)
	} else {
		// Faster, but no dilation or grouping version.
		convFn = convNoDilationDTypeMap.Get(dtype).(func(plan convGeneralExecPlan) error)
	}
	err := convFn(plan)
	if err != nil {
		backend.putBuffer(output)
		return nil, err
	}
	return output, nil
}

type convGeneralExecPlan struct {
	backend                              backends.Backend
	inputFlat, kernelFlat, outputFlat    any
	inputShape, kernelShape, outputShape shapes.Shape
	params                               *convNode
	dtype                                dtypes.DType
}

var (
	convNoDilationDTypeMap = NewDTypeMap("ConvNoDilation")
	convDTypeMap           = NewDTypeMap("ConvGeneral")
)

func init() {
	convNoDilationDTypeMap.Register(dtypes.BFloat16, priorityTyped, execConvNoDilationBFloat16)
	convDTypeMap.Register(dtypes.BFloat16, priorityTyped, execConvBFloat16)
}
