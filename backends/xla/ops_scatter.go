// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xla

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/go-xla/pkg/stablehlo"
)

func scatterOpToReduceOp(opType compute.OpType) compute.OpType {
	switch opType {
	case compute.OpTypeScatterMax:
		return compute.OpTypeReduceMax
	case compute.OpTypeScatterMin:
		return compute.OpTypeReduceMin
	case compute.OpTypeScatterSum:
		return compute.OpTypeReduceSum
	default:
		return -1
	}
}

func (f *Function) scatter(opType compute.OpType,
	operandOp, scatterIndicesOp, updatesOp compute.Value, indexVectorAxis int, updateWindowAxes,
	insertedWindowAxes, scatterAxesToOperandAxes []int, indicesAreSorted, uniqueIndices bool) (compute.Value, error) {
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
func (f *Function) ScatterMax(operand, scatterIndices, updates compute.Value, indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int, indicesAreSorted, uniqueIndices bool) (compute.Value, error) {
	return f.scatter(compute.OpTypeScatterMax,
		operand, scatterIndices, updates, indexVectorAxis,
		updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes, indicesAreSorted, uniqueIndices)
}

// ScatterMin scatter values from updates pointed by scatterIndices to operand, by taking the Min.
func (f *Function) ScatterMin(operand, scatterIndices, updates compute.Value, indexVectorAxis int, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int, indicesAreSorted, uniqueIndices bool) (compute.Value, error) {
	return f.scatter(compute.OpTypeScatterMin,
		operand, scatterIndices, updates, indexVectorAxis,
		updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes, indicesAreSorted, uniqueIndices)
}

// ScatterSum values from updates pointed by scatterIndices to operand.
func (f *Function) ScatterSum(operand, scatterIndices, updates compute.Value, indexVectorAxis int, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int, indicesAreSorted, uniqueIndices bool) (compute.Value, error) {
	return f.scatter(compute.OpTypeScatterSum,
		operand, scatterIndices, updates, indexVectorAxis,
		updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes, indicesAreSorted, uniqueIndices)
}
