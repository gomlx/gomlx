package xla

import (
	"github.com/gomlx/go-xla/pkg/stablehlo"
	"github.com/gomlx/gomlx/backends"
)

func scatterOpToReduceOp(opType backends.OpType) backends.OpType {
	switch opType {
	case backends.OpTypeScatterMax:
		return backends.OpTypeReduceMax
	case backends.OpTypeScatterMin:
		return backends.OpTypeReduceMin
	case backends.OpTypeScatterSum:
		return backends.OpTypeReduceSum
	default:
		return -1
	}
}

func (f *Function) scatter(opType backends.OpType,
	operandOp, scatterIndicesOp, updatesOp backends.Op, indexVectorAxis int, updateWindowAxes,
	insertedWindowAxes, scatterAxesToOperandAxes []int, indicesAreSorted, uniqueIndices bool) (backends.Op, error) {
	nodes, err := f.verifyAndCastValues(opType.String(), operandOp, scatterIndicesOp, updatesOp)
	if err != nil {
		return nil, err
	}
	operand, scatterIndices, updates := nodes[0], nodes[1], nodes[2]
	dtype := operand.shape.DType

	// Get update closure: it's the same as a reduction of the corresponding type.
	reductionFn, err := f.getReductionFn(dtype, scatterOpToReduceOp(opType))
	if err != nil {
		return nil, err
	}

	// Batching axes are left empty, since in GoMLX we don't support them.
	var inputBatchingAxes, scatterIndicesBatchingAxes []int

	value, err := stablehlo.Scatter(
		operand.value, scatterIndices.value, updates.value,
		updateWindowAxes, insertedWindowAxes,
		inputBatchingAxes, scatterIndicesBatchingAxes,
		scatterAxesToOperandAxes, indexVectorAxis,
		indicesAreSorted, uniqueIndices,
		reductionFn)
	if err != nil {
		return nil, err
	}
	return f.newNode(value), nil
}

// ScatterMax scatter values from updates pointed by scatterIndices to operand, by taking the Max.
func (f *Function) ScatterMax(operand, scatterIndices, updates backends.Op, indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int, indicesAreSorted, uniqueIndices bool) (backends.Op, error) {
	return f.scatter(backends.OpTypeScatterMax,
		operand, scatterIndices, updates, indexVectorAxis,
		updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes, indicesAreSorted, uniqueIndices)
}

// ScatterMin scatter values from updates pointed by scatterIndices to operand, by taking the Min.
func (f *Function) ScatterMin(operand, scatterIndices, updates backends.Op, indexVectorAxis int, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int, indicesAreSorted, uniqueIndices bool) (backends.Op, error) {
	return f.scatter(backends.OpTypeScatterMin,
		operand, scatterIndices, updates, indexVectorAxis,
		updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes, indicesAreSorted, uniqueIndices)
}

// ScatterSum values from updates pointed by scatterIndices to operand.
func (f *Function) ScatterSum(operand, scatterIndices, updates backends.Op, indexVectorAxis int, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int, indicesAreSorted, uniqueIndices bool) (backends.Op, error) {
	return f.scatter(backends.OpTypeScatterSum,
		operand, scatterIndices, updates, indexVectorAxis,
		updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes, indicesAreSorted, uniqueIndices)
}
