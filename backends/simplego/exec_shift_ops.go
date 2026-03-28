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
	lhs, rhs, output, _, _ := binaryOperandsAndOutput(backend, inputs, inputsOwned, node.shape)
	switch lhs.shape.DType { //nolint:exhaustive
	case dtypes.Uint8:
		shiftLeftOp(lhs.flat.([]uint8), rhs.flat.([]uint8), output.flat.([]uint8), lhs.shape, rhs.shape, output.shape)
	case dtypes.Uint16:
		shiftLeftOp(lhs.flat.([]uint16), rhs.flat.([]uint16), output.flat.([]uint16), lhs.shape, rhs.shape, output.shape)
	case dtypes.Uint32:
		shiftLeftOp(lhs.flat.([]uint32), rhs.flat.([]uint32), output.flat.([]uint32), lhs.shape, rhs.shape, output.shape)
	case dtypes.Uint64:
		shiftLeftOp(lhs.flat.([]uint64), rhs.flat.([]uint64), output.flat.([]uint64), lhs.shape, rhs.shape, output.shape)
	case dtypes.Int8:
		shiftLeftOp(lhs.flat.([]int8), rhs.flat.([]int8), output.flat.([]int8), lhs.shape, rhs.shape, output.shape)
	case dtypes.Int16:
		shiftLeftOp(lhs.flat.([]int16), rhs.flat.([]int16), output.flat.([]int16), lhs.shape, rhs.shape, output.shape)
	case dtypes.Int32:
		shiftLeftOp(lhs.flat.([]int32), rhs.flat.([]int32), output.flat.([]int32), lhs.shape, rhs.shape, output.shape)
	case dtypes.Int64:
		shiftLeftOp(lhs.flat.([]int64), rhs.flat.([]int64), output.flat.([]int64), lhs.shape, rhs.shape, output.shape)
	default:
		return nil, errors.Errorf("unsupported data type %s for %s", output.shape.DType, node.opType)
	}
	return output, nil
}

// execShiftRightArithmetic executes arithmetic right shift (preserves sign bit for signed types).
func execShiftRightArithmetic(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	lhs, rhs, output, _, _ := binaryOperandsAndOutput(backend, inputs, inputsOwned, node.shape)
	switch lhs.shape.DType { //nolint:exhaustive
	case dtypes.Uint8:
		shiftRightArithmeticOp(lhs.flat.([]uint8), rhs.flat.([]uint8), output.flat.([]uint8), lhs.shape, rhs.shape, output.shape)
	case dtypes.Uint16:
		shiftRightArithmeticOp(lhs.flat.([]uint16), rhs.flat.([]uint16), output.flat.([]uint16), lhs.shape, rhs.shape, output.shape)
	case dtypes.Uint32:
		shiftRightArithmeticOp(lhs.flat.([]uint32), rhs.flat.([]uint32), output.flat.([]uint32), lhs.shape, rhs.shape, output.shape)
	case dtypes.Uint64:
		shiftRightArithmeticOp(lhs.flat.([]uint64), rhs.flat.([]uint64), output.flat.([]uint64), lhs.shape, rhs.shape, output.shape)
	case dtypes.Int8:
		shiftRightArithmeticOp(lhs.flat.([]int8), rhs.flat.([]int8), output.flat.([]int8), lhs.shape, rhs.shape, output.shape)
	case dtypes.Int16:
		shiftRightArithmeticOp(lhs.flat.([]int16), rhs.flat.([]int16), output.flat.([]int16), lhs.shape, rhs.shape, output.shape)
	case dtypes.Int32:
		shiftRightArithmeticOp(lhs.flat.([]int32), rhs.flat.([]int32), output.flat.([]int32), lhs.shape, rhs.shape, output.shape)
	case dtypes.Int64:
		shiftRightArithmeticOp(lhs.flat.([]int64), rhs.flat.([]int64), output.flat.([]int64), lhs.shape, rhs.shape, output.shape)
	default:
		return nil, errors.Errorf("unsupported data type %s for %s", output.shape.DType, node.opType)
	}
	return output, nil
}

// execShiftRightLogical executes logical right shift (zero-fills from the left, ignoring sign).
// For signed types, we reinterpret as unsigned, shift, then reinterpret back.
func execShiftRightLogical(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	lhs, rhs, output, _, _ := binaryOperandsAndOutput(backend, inputs, inputsOwned, node.shape)
	switch lhs.shape.DType { //nolint:exhaustive
	case dtypes.Uint8:
		// For unsigned types, >> is already a logical shift; reuse the arithmetic shift function.
		shiftRightArithmeticOp(lhs.flat.([]uint8), rhs.flat.([]uint8), output.flat.([]uint8), lhs.shape, rhs.shape, output.shape)
	case dtypes.Uint16:
		shiftRightArithmeticOp(lhs.flat.([]uint16), rhs.flat.([]uint16), output.flat.([]uint16), lhs.shape, rhs.shape, output.shape)
	case dtypes.Uint32:
		shiftRightArithmeticOp(lhs.flat.([]uint32), rhs.flat.([]uint32), output.flat.([]uint32), lhs.shape, rhs.shape, output.shape)
	case dtypes.Uint64:
		shiftRightArithmeticOp(lhs.flat.([]uint64), rhs.flat.([]uint64), output.flat.([]uint64), lhs.shape, rhs.shape, output.shape)
	case dtypes.Int8:
		shiftRightLogicalSignedOp[int8, uint8](lhs.flat.([]int8), rhs.flat.([]int8), output.flat.([]int8), lhs.shape, rhs.shape, output.shape)
	case dtypes.Int16:
		shiftRightLogicalSignedOp[int16, uint16](lhs.flat.([]int16), rhs.flat.([]int16), output.flat.([]int16), lhs.shape, rhs.shape, output.shape)
	case dtypes.Int32:
		shiftRightLogicalSignedOp[int32, uint32](lhs.flat.([]int32), rhs.flat.([]int32), output.flat.([]int32), lhs.shape, rhs.shape, output.shape)
	case dtypes.Int64:
		shiftRightLogicalSignedOp[int64, uint64](lhs.flat.([]int64), rhs.flat.([]int64), output.flat.([]int64), lhs.shape, rhs.shape, output.shape)
	default:
		return nil, errors.Errorf("unsupported data type %s for %s", output.shape.DType, node.opType)
	}
	return output, nil
}

// shiftLeftOp performs lhs << rhs with broadcasting support.
// The operation is inlined to avoid per-element closure overhead.
func shiftLeftOp[T PODIntegerConstraints](lhs, rhs, output []T,
	lhsShape, rhsShape, outputShape shapes.Shape) {
	switch {
	case len(rhs) == 1:
		c := rhs[0]
		for i, v := range lhs {
			output[i] = v << uint(c)
		}
	case len(lhs) == 1:
		c := lhs[0]
		for i, v := range rhs {
			output[i] = c << uint(v)
		}
	case lhsShape.Equal(rhsShape):
		for i, v := range lhs {
			output[i] = v << uint(rhs[i])
		}
	default:
		lhsIter := newBroadcastIterator(lhsShape, outputShape)
		rhsIter := newBroadcastIterator(rhsShape, outputShape)
		for i := range output {
			output[i] = lhs[lhsIter.Next()] << uint(rhs[rhsIter.Next()])
		}
	}
}

// shiftRightArithmeticOp performs lhs >> rhs with broadcasting support.
// For signed types, Go's >> preserves the sign bit (arithmetic shift).
// For unsigned types, Go's >> is already a logical (zero-fill) shift, so this
// function is also used by execShiftRightLogical for unsigned dispatch.
func shiftRightArithmeticOp[T PODIntegerConstraints](lhs, rhs, output []T,
	lhsShape, rhsShape, outputShape shapes.Shape) {
	switch {
	case len(rhs) == 1:
		c := rhs[0]
		for i, v := range lhs {
			output[i] = v >> uint(c)
		}
	case len(lhs) == 1:
		c := lhs[0]
		for i, v := range rhs {
			output[i] = c >> uint(v)
		}
	case lhsShape.Equal(rhsShape):
		for i, v := range lhs {
			output[i] = v >> uint(rhs[i])
		}
	default:
		lhsIter := newBroadcastIterator(lhsShape, outputShape)
		rhsIter := newBroadcastIterator(rhsShape, outputShape)
		for i := range output {
			output[i] = lhs[lhsIter.Next()] >> uint(rhs[rhsIter.Next()])
		}
	}
}

// shiftRightLogicalSignedOp performs logical right shift for signed types by
// reinterpreting as unsigned, shifting, then converting back.
// T is the signed type, U is the corresponding unsigned type.
func shiftRightLogicalSignedOp[T ~int8 | ~int16 | ~int32 | ~int64, U ~uint8 | ~uint16 | ~uint32 | ~uint64](
	lhs, rhs, output []T, lhsShape, rhsShape, outputShape shapes.Shape) {
	switch {
	case len(rhs) == 1:
		c := rhs[0]
		for i, v := range lhs {
			output[i] = T(U(v) >> uint(c))
		}
	case len(lhs) == 1:
		c := lhs[0]
		for i, v := range rhs {
			output[i] = T(U(c) >> uint(v))
		}
	case lhsShape.Equal(rhsShape):
		for i, v := range lhs {
			output[i] = T(U(v) >> uint(rhs[i]))
		}
	default:
		lhsIter := newBroadcastIterator(lhsShape, outputShape)
		rhsIter := newBroadcastIterator(rhsShape, outputShape)
		for i := range output {
			output[i] = T(U(lhs[lhsIter.Next()]) >> uint(rhs[rhsIter.Next()]))
		}
	}
}
