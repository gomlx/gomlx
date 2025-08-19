package simplego

import (
	"slices"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/shapeinference"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/pkg/errors"
)

func init() {
	nodeExecutors[backends.OpTypeConvGeneralDilated] = execConvGeneral
}

// ConvGeneral is a generic Convolution operation with support for:
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
func (b *Builder) ConvGeneral(inputOp, kernelOp backends.Op, axes backends.ConvolveAxesConfig,
	strides []int, paddings [][2]int,
	inputDilations, kernelDilations []int,
	filterGroupCount, batchGroupCount int) (backends.Op, error) {
	opType := backends.OpTypeConvGeneralDilated
	inputs, err := b.checkOps(opType.String(), inputOp, kernelOp)
	if err != nil {
		return nil, err
	}
	input, kernel := inputs[0], inputs[1]

	outputShape, err := shapeinference.ConvGeneralOp(input.shape, kernel.shape, axes, strides, paddings, inputDilations, kernelDilations, filterGroupCount, batchGroupCount)
	if err != nil {
		return nil, err
	}

	node := b.newNode(opType, outputShape, input, kernel)
	node.data = &convNode{
		axes:             axes.Clone(),
		strides:          slices.Clone(strides),
		paddings:         slices.Clone(paddings),
		inputDilations:   slices.Clone(inputDilations),
		kernelDilations:  slices.Clone(kernelDilations),
		filterGroupCount: filterGroupCount,
		batchGroupCount:  batchGroupCount,
	}
	return node, nil
}

type convNode struct {
	axes             backends.ConvolveAxesConfig
	strides          []int
	paddings         [][2]int
	inputDilations   []int
	kernelDilations  []int
	filterGroupCount int
	batchGroupCount  int
}

// ConvGeneralDilated is a deprecated an alias to ConvGeneral.
//
// Deprecated: use ConvGeneral instead.
func (b *Builder) ConvGeneralDilated(inputOp, kernelOp backends.Op, axes backends.ConvolveAxesConfig,
	strides []int, paddings [][2]int,
	inputDilations, kernelDilations []int,
	filterGroupCount, batchGroupCount int) (backends.Op, error) {
	return b.ConvGeneral(inputOp, kernelOp, axes, strides, paddings, inputDilations, kernelDilations, filterGroupCount, batchGroupCount)
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

	if slices.Max(params.inputDilations) > 1 || slices.Max(params.kernelDilations) > 1 || params.filterGroupCount > 1 || params.batchGroupCount > 1 {
		return nil, errors.Errorf("SimpleGo backend doesn't yet support convolution (ConvGeneral) with dilations or groupings (filter or batch) -- please open a bug report if needed")
	}
	convFn := convNoDilationDTypeMap.Get(dtype).(func(plan convGeneralExecPlan) error)
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

// execConvNoDilationGeneric executes a ConvGeneral without any dilation or grouping, for generic Go native numeric types,
// so this excludes BFloat16.
func execConvNoDilationGeneric[T PODNumericConstraints](plan convGeneralExecPlan) error {
	// Shortcuts (and maybe move these values to the stack for faster access)
	inputFlat := plan.inputFlat.([]T)
	inputShape := plan.inputShape
	kernelFlat := plan.kernelFlat.([]T)
	kernelShape := plan.kernelShape
	outputFlat := plan.outputFlat.([]T)
	outputShape := plan.outputShape
	rank := outputShape.Rank() // same rank for input and kernel.
	//spatialRank := rank - 2
	params := plan.params
	axes := params.axes
	paddings := params.paddings
	convStrides := params.strides

	inputBatchAxis := axes.InputBatch
	inputChannelsAxis := axes.InputChannels
	inputSpatialAxes := axes.InputSpatial
	outputBatchAxis := axes.OutputBatch
	outputChannelsAxis := axes.OutputChannels
	outputSpatialAxes := axes.OutputSpatial
	kernelInputChannelsAxis := axes.KernelInputChannels
	kernelOutputChannelsAxis := axes.KernelOutputChannels
	kernelSpatialAxes := axes.KernelSpatial
	numInputChannels := kernelShape.Dimensions[kernelInputChannelsAxis]

	// Indices we'll be iterating over.
	var outputFlatIdx int

	// Indices and strides: note we don't use an inputIndices because we only keep an inputFlatIndex.
	outputIndices := make([]int, rank)
	kernelIndices := make([]int, rank)

	inputStrides := inputShape.Strides()
	kernelStrides := kernelShape.Strides()

	// Loop sequentially over all output positions:
	for outputFlatIdx, outputIndices = range outputShape.IterOn(outputIndices) {
		batchIdx := outputIndices[outputBatchAxis]
		outputChannel := outputIndices[outputChannelsAxis]
		baseInputFlatIdx := batchIdx * inputStrides[inputBatchAxis]

		// Loop over the kernel spatial axes, with the outputChannel given by the output loop.
		kernelIndices[kernelOutputChannelsAxis] = outputChannel
		var outputValue T
		var kernelFlatIdx int
	kernelLoop:
		for kernelFlatIdx, kernelIndices = range kernelShape.IterOnAxes(kernelSpatialAxes, kernelStrides, kernelIndices) {
			// Calculate the corresponding position in the input.
			inputFlatIdx := baseInputFlatIdx
			for spatialIdx, outputSpatialAxis := range outputSpatialAxes {
				inputSpatialAxis := inputSpatialAxes[spatialIdx]
				kernelSpatialAxes := axes.KernelSpatial[spatialIdx]
				outputIdx := outputIndices[outputSpatialAxis]
				kernelIdx := kernelIndices[kernelSpatialAxes]
				inputIdx := outputIdx*convStrides[spatialIdx] + kernelIdx - paddings[spatialIdx][0]
				if inputIdx < 0 || inputIdx >= inputShape.Dimensions[inputSpatialAxis] {
					// Index is in the padded area, we can move to the next kernel position.
					continue kernelLoop
				}
				inputFlatIdx += inputIdx * inputStrides[inputSpatialAxis]
			}

			// Accumulate over all the kernel/input channels.
			inputChannelStride := inputStrides[inputChannelsAxis]
			kernelChannelStride := kernelStrides[kernelInputChannelsAxis]
			for range numInputChannels {
				inputValue := inputFlat[inputFlatIdx]
				kernelValue := kernelFlat[kernelFlatIdx]
				outputValue += inputValue * kernelValue
				inputFlatIdx += inputChannelStride
				kernelFlatIdx += kernelChannelStride
			}
		}

		// Update output with accumulated value from the convolution of the kernel at this position.
		outputFlat[outputFlatIdx] = outputValue
	}
	return nil
}

func init() {
	convNoDilationDTypeMap.Register(dtypes.BFloat16, execConvNoDilationBFloat16)
}

// execConvNoDilation for BFloat16.
func execConvNoDilationBFloat16(plan convGeneralExecPlan) error {
	// Shortcuts (and maybe move these values to the stack for faster access)
	inputFlat := plan.inputFlat.([]bfloat16.BFloat16)
	inputShape := plan.inputShape
	kernelFlat := plan.kernelFlat.([]bfloat16.BFloat16)
	kernelShape := plan.kernelShape
	outputFlat := plan.outputFlat.([]bfloat16.BFloat16)
	outputShape := plan.outputShape
	rank := outputShape.Rank() // same rank for input and kernel.
	//spatialRank := rank - 2
	params := plan.params
	axes := params.axes
	paddings := params.paddings
	convStrides := params.strides

	inputBatchAxis := axes.InputBatch
	inputChannelsAxis := axes.InputChannels
	inputSpatialAxes := axes.InputSpatial
	outputBatchAxis := axes.OutputBatch
	outputChannelsAxis := axes.OutputChannels
	outputSpatialAxes := axes.OutputSpatial
	kernelInputChannelsAxis := axes.KernelInputChannels
	kernelOutputChannelsAxis := axes.KernelOutputChannels
	kernelSpatialAxes := axes.KernelSpatial
	numInputChannels := kernelShape.Dimensions[kernelInputChannelsAxis]

	// Indices we'll be iterating over.
	var outputFlatIdx int

	// Indices and strides: note we don't use an inputIndices because we only keep an inputFlatIndex.
	outputIndices := make([]int, rank)
	kernelIndices := make([]int, rank)

	inputStrides := inputShape.Strides()
	kernelStrides := kernelShape.Strides()

	// Loop sequentially over all output positions:
	for outputFlatIdx, outputIndices = range outputShape.IterOn(outputIndices) {
		batchIdx := outputIndices[outputBatchAxis]
		outputChannel := outputIndices[outputChannelsAxis]
		baseInputFlatIdx := batchIdx * inputStrides[inputBatchAxis]

		// Loop over the kernel spatial axes, with the outputChannel given by the output loop.
		kernelIndices[kernelOutputChannelsAxis] = outputChannel
		var outputValue float32
		var kernelFlatIdx int
	kernelLoop:
		for kernelFlatIdx, kernelIndices = range kernelShape.IterOnAxes(kernelSpatialAxes, kernelStrides, kernelIndices) {
			// Calculate the corresponding position in the input.
			inputFlatIdx := baseInputFlatIdx
			for spatialIdx, outputSpatialAxis := range outputSpatialAxes {
				inputSpatialAxis := inputSpatialAxes[spatialIdx]
				kernelSpatialAxes := axes.KernelSpatial[spatialIdx]
				outputIdx := outputIndices[outputSpatialAxis]
				kernelIdx := kernelIndices[kernelSpatialAxes]
				inputIdx := outputIdx*convStrides[spatialIdx] + kernelIdx - paddings[spatialIdx][0]
				if inputIdx < 0 || inputIdx >= inputShape.Dimensions[inputSpatialAxis] {
					// Index is in the padded area, we can move to the next kernel position.
					continue kernelLoop
				}
				inputFlatIdx += inputIdx * inputStrides[inputSpatialAxis]
			}

			// Accumulate over all the kernel/input channels.
			inputChannelStride := inputStrides[inputChannelsAxis]
			kernelChannelStride := kernelStrides[kernelInputChannelsAxis]
			for range numInputChannels {
				inputValue := inputFlat[inputFlatIdx]
				kernelValue := kernelFlat[kernelFlatIdx]
				outputValue += inputValue.Float32() * kernelValue.Float32()
				inputFlatIdx += inputChannelStride
				kernelFlatIdx += kernelChannelStride
			}
		}

		// Update output with accumulated value from the convolution of the kernel at this position.
		outputFlat[outputFlatIdx] = bfloat16.FromFloat32(outputValue)
	}
	return nil
}

// execConv executes a ConvGeneral with support for dilation and grouping (and slower for that).
// This is the generic version for Go native numeric types -- so this excludes BFloat16.
func execConvGeneric[T PODNumericConstraints](plan convGeneralExecPlan) error {
	// Shortcuts (and maybe move these values to the stack for faster access)
	inputFlat := plan.inputFlat.([]T)
	inputShape := plan.inputShape
	kernelFlat := plan.kernelFlat.([]T)
	kernelShape := plan.kernelShape
	outputFlat := plan.outputFlat.([]T)
	outputShape := plan.outputShape
	rank := outputShape.Rank() // same rank for input and kernel.
	//spatialRank := rank - 2
	params := plan.params
	axes := params.axes
	paddings := params.paddings
	convStrides := params.strides

	inputBatchAxis := axes.InputBatch
	inputChannelsAxis := axes.InputChannels
	inputSpatialAxes := axes.InputSpatial
	outputBatchAxis := axes.OutputBatch
	outputChannelsAxis := axes.OutputChannels
	outputSpatialAxes := axes.OutputSpatial
	kernelInputChannelsAxis := axes.KernelInputChannels
	kernelOutputChannelsAxis := axes.KernelOutputChannels
	kernelSpatialAxes := axes.KernelSpatial
	numInputChannels := kernelShape.Dimensions[kernelInputChannelsAxis]

	// Indices we'll be iterating over.
	var outputFlatIdx int

	// Indices and strides: note we don't use an inputIndices because we only keep an inputFlatIndex.
	outputIndices := make([]int, rank)
	kernelIndices := make([]int, rank)

	inputStrides := inputShape.Strides()
	kernelStrides := kernelShape.Strides()

	// Loop sequentially over all output positions:
	for outputFlatIdx, outputIndices = range outputShape.IterOn(outputIndices) {
		batchIdx := outputIndices[outputBatchAxis]
		outputChannel := outputIndices[outputChannelsAxis]
		baseInputFlatIdx := batchIdx * inputStrides[inputBatchAxis]

		// Loop over the kernel spatial axes, with the outputChannel given by the output loop.
		kernelIndices[kernelOutputChannelsAxis] = outputChannel
		var outputValue T
		var kernelFlatIdx int
	kernelLoop:
		for kernelFlatIdx, kernelIndices = range kernelShape.IterOnAxes(kernelSpatialAxes, kernelStrides, kernelIndices) {
			// Calculate the corresponding position in the input.
			inputFlatIdx := baseInputFlatIdx
			for spatialIdx, outputSpatialAxis := range outputSpatialAxes {
				inputSpatialAxis := inputSpatialAxes[spatialIdx]
				kernelSpatialAxes := axes.KernelSpatial[spatialIdx]
				outputIdx := outputIndices[outputSpatialAxis]
				kernelIdx := kernelIndices[kernelSpatialAxes]
				inputIdx := outputIdx*convStrides[spatialIdx] + kernelIdx - paddings[spatialIdx][0]
				if inputIdx < 0 || inputIdx >= inputShape.Dimensions[inputSpatialAxis] {
					// Index is in the padded area, we can move to the next kernel position.
					continue kernelLoop
				}
				inputFlatIdx += inputIdx * inputStrides[inputSpatialAxis]
			}

			// Accumulate over all the kernel/input channels.
			inputChannelStride := inputStrides[inputChannelsAxis]
			kernelChannelStride := kernelStrides[kernelInputChannelsAxis]
			for range numInputChannels {
				inputValue := inputFlat[inputFlatIdx]
				kernelValue := kernelFlat[kernelFlatIdx]
				outputValue += inputValue * kernelValue
				inputFlatIdx += inputChannelStride
				kernelFlatIdx += kernelChannelStride
			}
		}

		// Update output with accumulated value from the convolution of the kernel at this position.
		outputFlat[outputFlatIdx] = outputValue
	}
	return nil
}
