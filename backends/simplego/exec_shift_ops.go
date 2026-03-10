// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

func init() {
	setNodeExecutor(backends.OpTypeShiftLeft, priorityGeneric, execShiftLeft)
	setNodeExecutor(backends.OpTypeShiftRightArithmetic, priorityGeneric, execShiftRightArithmetic)
	setNodeExecutor(backends.OpTypeShiftRightLogical, priorityGeneric, execShiftRightLogical)
}

// execShiftLeft executes lhs << rhs for integer types.
func execShiftLeft(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	lhs, rhs, output, _, _ := BinaryOperandsAndOutput(backend, inputs, inputsOwned, node.shape)
	switch lhs.shape.DType {
	case dtypes.Uint8:
		shiftOp(lhs.flat.([]uint8), rhs.flat.([]uint8), output.flat.([]uint8), lhs.shape, rhs.shape, output.shape,
			func(a, b uint8) uint8 { return a << b })
	case dtypes.Uint16:
		shiftOp(lhs.flat.([]uint16), rhs.flat.([]uint16), output.flat.([]uint16), lhs.shape, rhs.shape, output.shape,
			func(a, b uint16) uint16 { return a << b })
	case dtypes.Uint32:
		shiftOp(lhs.flat.([]uint32), rhs.flat.([]uint32), output.flat.([]uint32), lhs.shape, rhs.shape, output.shape,
			func(a, b uint32) uint32 { return a << b })
	case dtypes.Uint64:
		shiftOp(lhs.flat.([]uint64), rhs.flat.([]uint64), output.flat.([]uint64), lhs.shape, rhs.shape, output.shape,
			func(a, b uint64) uint64 { return a << b })
	case dtypes.Int8:
		shiftOp(lhs.flat.([]int8), rhs.flat.([]int8), output.flat.([]int8), lhs.shape, rhs.shape, output.shape,
			func(a, b int8) int8 { return a << uint(b) })
	case dtypes.Int16:
		shiftOp(lhs.flat.([]int16), rhs.flat.([]int16), output.flat.([]int16), lhs.shape, rhs.shape, output.shape,
			func(a, b int16) int16 { return a << uint(b) })
	case dtypes.Int32:
		shiftOp(lhs.flat.([]int32), rhs.flat.([]int32), output.flat.([]int32), lhs.shape, rhs.shape, output.shape,
			func(a, b int32) int32 { return a << uint(b) })
	case dtypes.Int64:
		shiftOp(lhs.flat.([]int64), rhs.flat.([]int64), output.flat.([]int64), lhs.shape, rhs.shape, output.shape,
			func(a, b int64) int64 { return a << uint(b) })
	default:
		return nil, errors.Errorf("unsupported data type %s for %s", output.shape.DType, node.opType)
	}
	return output, nil
}

// execShiftRightArithmetic executes arithmetic right shift (preserves sign bit for signed types).
func execShiftRightArithmetic(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	lhs, rhs, output, _, _ := BinaryOperandsAndOutput(backend, inputs, inputsOwned, node.shape)
	switch lhs.shape.DType {
	case dtypes.Uint8:
		shiftOp(lhs.flat.([]uint8), rhs.flat.([]uint8), output.flat.([]uint8), lhs.shape, rhs.shape, output.shape,
			func(a, b uint8) uint8 { return a >> b })
	case dtypes.Uint16:
		shiftOp(lhs.flat.([]uint16), rhs.flat.([]uint16), output.flat.([]uint16), lhs.shape, rhs.shape, output.shape,
			func(a, b uint16) uint16 { return a >> b })
	case dtypes.Uint32:
		shiftOp(lhs.flat.([]uint32), rhs.flat.([]uint32), output.flat.([]uint32), lhs.shape, rhs.shape, output.shape,
			func(a, b uint32) uint32 { return a >> b })
	case dtypes.Uint64:
		shiftOp(lhs.flat.([]uint64), rhs.flat.([]uint64), output.flat.([]uint64), lhs.shape, rhs.shape, output.shape,
			func(a, b uint64) uint64 { return a >> b })
	case dtypes.Int8:
		shiftOp(lhs.flat.([]int8), rhs.flat.([]int8), output.flat.([]int8), lhs.shape, rhs.shape, output.shape,
			func(a, b int8) int8 { return a >> uint(b) })
	case dtypes.Int16:
		shiftOp(lhs.flat.([]int16), rhs.flat.([]int16), output.flat.([]int16), lhs.shape, rhs.shape, output.shape,
			func(a, b int16) int16 { return a >> uint(b) })
	case dtypes.Int32:
		shiftOp(lhs.flat.([]int32), rhs.flat.([]int32), output.flat.([]int32), lhs.shape, rhs.shape, output.shape,
			func(a, b int32) int32 { return a >> uint(b) })
	case dtypes.Int64:
		shiftOp(lhs.flat.([]int64), rhs.flat.([]int64), output.flat.([]int64), lhs.shape, rhs.shape, output.shape,
			func(a, b int64) int64 { return a >> uint(b) })
	default:
		return nil, errors.Errorf("unsupported data type %s for %s", output.shape.DType, node.opType)
	}
	return output, nil
}

// execShiftRightLogical executes logical right shift (zero-fills from the left, ignoring sign).
// For signed types, we reinterpret as unsigned, shift, then reinterpret back.
func execShiftRightLogical(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	lhs, rhs, output, _, _ := BinaryOperandsAndOutput(backend, inputs, inputsOwned, node.shape)
	switch lhs.shape.DType {
	case dtypes.Uint8:
		shiftOp(lhs.flat.([]uint8), rhs.flat.([]uint8), output.flat.([]uint8), lhs.shape, rhs.shape, output.shape,
			func(a, b uint8) uint8 { return a >> b })
	case dtypes.Uint16:
		shiftOp(lhs.flat.([]uint16), rhs.flat.([]uint16), output.flat.([]uint16), lhs.shape, rhs.shape, output.shape,
			func(a, b uint16) uint16 { return a >> b })
	case dtypes.Uint32:
		shiftOp(lhs.flat.([]uint32), rhs.flat.([]uint32), output.flat.([]uint32), lhs.shape, rhs.shape, output.shape,
			func(a, b uint32) uint32 { return a >> b })
	case dtypes.Uint64:
		shiftOp(lhs.flat.([]uint64), rhs.flat.([]uint64), output.flat.([]uint64), lhs.shape, rhs.shape, output.shape,
			func(a, b uint64) uint64 { return a >> b })
	case dtypes.Int8:
		shiftOp(lhs.flat.([]int8), rhs.flat.([]int8), output.flat.([]int8), lhs.shape, rhs.shape, output.shape,
			func(a, b int8) int8 { return int8(uint8(a) >> uint(b)) })
	case dtypes.Int16:
		shiftOp(lhs.flat.([]int16), rhs.flat.([]int16), output.flat.([]int16), lhs.shape, rhs.shape, output.shape,
			func(a, b int16) int16 { return int16(uint16(a) >> uint(b)) })
	case dtypes.Int32:
		shiftOp(lhs.flat.([]int32), rhs.flat.([]int32), output.flat.([]int32), lhs.shape, rhs.shape, output.shape,
			func(a, b int32) int32 { return int32(uint32(a) >> uint(b)) })
	case dtypes.Int64:
		shiftOp(lhs.flat.([]int64), rhs.flat.([]int64), output.flat.([]int64), lhs.shape, rhs.shape, output.shape,
			func(a, b int64) int64 { return int64(uint64(a) >> uint(b)) })
	default:
		return nil, errors.Errorf("unsupported data type %s for %s", output.shape.DType, node.opType)
	}
	return output, nil
}

// shiftOp applies a binary shift operation with broadcasting support.
func shiftOp[T PODIntegerConstraints](lhs, rhs, output []T,
	lhsShape, rhsShape, outputShape shapes.Shape, op func(T, T) T) {
	switch {
	case len(rhs) == 1:
		c := rhs[0]
		for i, v := range lhs {
			output[i] = op(v, c)
		}
	case len(lhs) == 1:
		c := lhs[0]
		for i, v := range rhs {
			output[i] = op(c, v)
		}
	case lhsShape.Equal(rhsShape):
		for i, v := range lhs {
			output[i] = op(v, rhs[i])
		}
	default:
		lhsIter := NewBroadcastIterator(lhsShape, outputShape)
		rhsIter := NewBroadcastIterator(rhsShape, outputShape)
		for i := range output {
			lhsIdx := lhsIter.Next()
			rhsIdx := rhsIter.Next()
			output[i] = op(lhs[lhsIdx], rhs[rhsIdx])
		}
	}
}
