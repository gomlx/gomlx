package simplego

import (
	"slices"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/shapeinference"
	"github.com/gomlx/gopjrt/dtypes"
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
	output.Zeros()

	// TODO(optimizations):
	// - Optimize order of axes iterations.
	// - Split input into cache-fitting buckets ?

	// Find execution plan:
	// - We iterate the axes in order they are laid out in memory for the **output**: so we prioritize visiting the output
	//   sequentially, and each output position is visited only once -- minimizing the number of cache flushes -- cache
	//   misses will only happen in the input (or kernel, if it is large).
	rank := input.shape.Rank()
	plan := convGeneralExecPlan{
		backend:         backend,
		dtype:           dtype,
		inputFlat:       input.flat,
		kernelFlat:      kernel.flat,
		outputFlat:      output.flat,
		params:          params,
		inputAxesStarts: make([]int, rank),
	}

	if slices.Max(params.inputDilations) > 1 || slices.Max(params.kernelDilations) > 1 || params.filterGroupCount > 1 || params.batchGroupCount > 1 {
		return nil, errors.Errorf("SimpleGo backend doesn't yet support convolution (ConvGeneral) with dilations or groupings (filter or batch) -- please open a bug report if needed")
	}
	convFn := convNoDilationDTypeMap.Get(dtype).(func(plan *convGeneralExecPlan) error)
	err := convFn(&plan)
	if err != nil {
		backend.putBuffer(output)
		return nil, err
	}
	return output, nil
}

type convGeneralExecPlan struct {
	backend                           backends.Backend
	inputFlat, kernelFlat, outputFlat any
	params                            *convNode
	dtype                             dtypes.DType
	inputAxesStarts                   []int // For each axis,
}

var (
	convNoDilationDTypeMap = NewDTypeMap("ConvNoDilation")
)

func execConvNoDilationGeneric[T PODIntegerConstraints](plan *convGeneralExecPlan) error {
	inputFlat := plan.inputFlat.([]T)
	kernelFlat := plan.kernelFlat.([]T)
	outputFlat := plan.outputFlat.([]T)
	_, _, _ = inputFlat, kernelFlat, outputFlat
	return nil
}
