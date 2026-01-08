// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

// Float16 binary operations support.
// These wrap the generic binary executors to handle Float16 dtype.

import (
	"math"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/x448/float16"
)

// Float16 binary operations

func execBinaryFloat16[OpFn func(a, b float32) float32](opFn OpFn, lhs, rhs []float16.Float16, output []float16.Float16,
	lhsShape, rhsShape, outputShape shapes.Shape) {
	if len(rhs) == 1 {
		// Case 1: One side (rhs) is a scalar: only iterate over the lhs.
		c := rhs[0].Float32()
		for ii, input := range lhs {
			a := input.Float32()
			output[ii] = float16.Fromfloat32(opFn(a, c))
		}
		return
	} else if len(lhs) == 1 {
		// Case 1b: One side (lhs) is a scalar: only iterate over the rhs.
		// This is needed for non-commutative operations like Sub and Div.
		c := lhs[0].Float32()
		for ii, input := range rhs {
			b := input.Float32()
			output[ii] = float16.Fromfloat32(opFn(c, b))
		}
		return
	} else if lhsShape.Equal(rhsShape) {
		// Case 2: Exact same shapes, no broadcasting.
		for outputIdx := range output {
			a := lhs[outputIdx].Float32()
			b := rhs[outputIdx].Float32()
			output[outputIdx] = float16.Fromfloat32(opFn(a, b))
		}
		return
	} else {
		// Case 3: with broadcasting non-scalar tensors:
		lhsIter := newBroadcastIterator(lhsShape, outputShape)
		rhsIter := newBroadcastIterator(rhsShape, outputShape)
		for outputIdx := range output {
			lhsIdx := lhsIter.Next()
			rhsIdx := rhsIter.Next()
			a := lhs[lhsIdx].Float32()
			b := rhs[rhsIdx].Float32()
			output[outputIdx] = float16.Fromfloat32(opFn(a, b))
		}
	}
}

func execCompareFloat16[OpFn func(a, b float32) bool](opFn OpFn, lhs, rhs []float16.Float16, output []bool,
	lhsShape, rhsShape, outputShape shapes.Shape) {
	if len(rhs) == 1 {
		// Case 1: One side (rhs) is a scalar.
		c := rhs[0].Float32()
		for ii, input := range lhs {
			a := input.Float32()
			output[ii] = opFn(a, c)
		}
		return
	} else if len(lhs) == 1 {
		// Case 1b: One side (lhs) is a scalar.
		c := lhs[0].Float32()
		for ii, input := range rhs {
			b := input.Float32()
			output[ii] = opFn(c, b)
		}
		return
	} else if lhsShape.Equal(rhsShape) {
		// Case 2: Exact same shapes.
		for outputIdx := range output {
			a := lhs[outputIdx].Float32()
			b := rhs[outputIdx].Float32()
			output[outputIdx] = opFn(a, b)
		}
		return
	} else {
		// Case 3: Broadcasting.
		lhsIter := newBroadcastIterator(lhsShape, outputShape)
		rhsIter := newBroadcastIterator(rhsShape, outputShape)
		for outputIdx := range output {
			lhsIdx := lhsIter.Next()
			rhsIdx := rhsIter.Next()
			a := lhs[lhsIdx].Float32()
			b := rhs[rhsIdx].Float32()
			output[outputIdx] = opFn(a, b)
		}
	}
}


func makeFloat16BinaryWrapper(
	origExec func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error),
	opFn func(a, b float32) float32,
) func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error) {
	return func(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
		if inputs[0].shape.DType != dtypes.Float16 {
			return origExec(backend, node, inputs, inputsOwned)
		}
		lhs, rhs, output, _, _ := binaryOperandsAndOutput(backend, inputs, inputsOwned, node.shape)
		execBinaryFloat16(opFn, lhs.flat.([]float16.Float16), rhs.flat.([]float16.Float16),
			output.flat.([]float16.Float16), lhs.shape, rhs.shape, output.shape)
		return output, nil
	}
}

func makeFloat16CompareWrapper(
	origExec func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error),
	opFn func(a, b float32) bool,
) func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error) {
	return func(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
		if inputs[0].shape.DType != dtypes.Float16 {
			return origExec(backend, node, inputs, inputsOwned)
		}
		lhs, rhs := inputs[0], inputs[1]
		output := backend.getBuffer(node.shape.DType, node.shape.Size())
		output.shape = node.shape
		execCompareFloat16(opFn, lhs.flat.([]float16.Float16), rhs.flat.([]float16.Float16),
			output.flat.([]bool), lhs.shape, rhs.shape, output.shape)
		return output, nil
	}
}

func init() {
	// Register Float16 wrappers with priorityTyped.
	// These wrap the generic executors (from gen_exec_binary.go) to handle Float16 dtype.
	// NEON implementations in float16_binary_neon_arm64.go use priorityArch to override these.
	setNodeExecutor(backends.OpTypeAdd, priorityTyped, makeFloat16BinaryWrapper(execAdd, func(a, b float32) float32 { return a + b }))
	setNodeExecutor(backends.OpTypeSub, priorityTyped, makeFloat16BinaryWrapper(execSub, func(a, b float32) float32 { return a - b }))
	setNodeExecutor(backends.OpTypeMul, priorityTyped, makeFloat16BinaryWrapper(execMul, func(a, b float32) float32 { return a * b }))
	setNodeExecutor(backends.OpTypeDiv, priorityTyped, makeFloat16BinaryWrapper(execDiv, func(a, b float32) float32 { return a / b }))
	setNodeExecutor(backends.OpTypeMax, priorityTyped, makeFloat16BinaryWrapper(execMax, func(a, b float32) float32 {
		if a > b {
			return a
		}
		return b
	}))
	setNodeExecutor(backends.OpTypeMin, priorityTyped, makeFloat16BinaryWrapper(execMin, func(a, b float32) float32 {
		if a < b {
			return a
		}
		return b
	}))
	setNodeExecutor(backends.OpTypePow, priorityTyped, makeFloat16BinaryWrapper(execPow, func(a, b float32) float32 {
		return float32(math.Pow(float64(a), float64(b)))
	}))
	setNodeExecutor(backends.OpTypeEqual, priorityTyped, makeFloat16CompareWrapper(execEqual, func(a, b float32) bool { return a == b }))
	setNodeExecutor(backends.OpTypeNotEqual, priorityTyped, makeFloat16CompareWrapper(execNotEqual, func(a, b float32) bool { return a != b }))
	setNodeExecutor(backends.OpTypeGreaterOrEqual, priorityTyped, makeFloat16CompareWrapper(execGreaterOrEqual, func(a, b float32) bool { return a >= b }))
	setNodeExecutor(backends.OpTypeGreaterThan, priorityTyped, makeFloat16CompareWrapper(execGreaterThan, func(a, b float32) bool { return a > b }))
	setNodeExecutor(backends.OpTypeLessOrEqual, priorityTyped, makeFloat16CompareWrapper(execLessOrEqual, func(a, b float32) bool { return a <= b }))
	setNodeExecutor(backends.OpTypeLessThan, priorityTyped, makeFloat16CompareWrapper(execLessThan, func(a, b float32) bool { return a < b }))
}
