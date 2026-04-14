// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"github.com/gomlx/gomlx/backends"
)

func init() {
	setNodeExecutor(backends.OpTypeShiftLeft, priorityGeneric, execShiftLeft)
	setNodeExecutor(backends.OpTypeShiftRightArithmetic, priorityGeneric, execShiftRightArithmetic)
	setNodeExecutor(backends.OpTypeShiftRightLogical, priorityGeneric, execShiftRightLogical)
}

var (
	shiftLeftDTypeMap            = NewDTypeMap("ShiftLeft")
	shiftRightArithmeticDTypeMap = NewDTypeMap("ShiftRightArithmetic")
	shiftRightLogicalDTypeMap    = NewDTypeMap("ShiftRightLogical")
)

// execShiftLeft executes lhs << rhs for integer types.
func execShiftLeft(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	lhs, rhs, output, _, _ := binaryOperandsAndOutput(backend, inputs, inputsOwned, node.shape)
	dtype := lhs.shape.DType
	fnAny, err := shiftLeftDTypeMap.Get(dtype) //nolint:errcheck
	if err != nil {
		return nil, err
	}
	fn := fnAny.(func(lhs, rhs, output *Buffer))
	fn(lhs, rhs, output)
	return output, nil
}

// execShiftRightArithmetic executes arithmetic right shift (preserves sign bit for signed types).
func execShiftRightArithmetic(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	lhs, rhs, output, _, _ := binaryOperandsAndOutput(backend, inputs, inputsOwned, node.shape)
	dtype := lhs.shape.DType
	fnAny, err := shiftRightArithmeticDTypeMap.Get(dtype) //nolint:errcheck
	if err != nil {
		return nil, err
	}
	fn := fnAny.(func(lhs, rhs, output *Buffer))
	fn(lhs, rhs, output)
	return output, nil
}

// execShiftRightLogical executes logical right shift (zero-fills from the left, ignoring sign).
// For signed types, we reinterpret as unsigned, shift, then reinterpret back.
func execShiftRightLogical(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	lhs, rhs, output, _, _ := binaryOperandsAndOutput(backend, inputs, inputsOwned, node.shape)
	dtype := lhs.shape.DType
	fnAny, err := shiftRightLogicalDTypeMap.Get(dtype) //nolint:errcheck
	if err != nil {
		return nil, err
	}
	fn := fnAny.(func(lhs, rhs, output *Buffer))
	fn(lhs, rhs, output)
	return output, nil
}

// shiftLeftGeneric performs lhs << rhs with broadcasting support.
// The operation is inlined to avoid per-element closure overhead.
func shiftLeftGeneric[T PODIntegerConstraints](lhsBuf, rhsBuf, outputBuf *Buffer) {
	lhs, rhs, output := lhsBuf.flat.([]T), rhsBuf.flat.([]T), outputBuf.flat.([]T)

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
	case lhsBuf.shape.Equal(rhsBuf.shape):
		for i, v := range lhs {
			output[i] = v << uint(rhs[i])
		}
	default:
		lhsIter := newBroadcastIterator(lhsBuf.shape, outputBuf.shape)
		rhsIter := newBroadcastIterator(rhsBuf.shape, outputBuf.shape)
		for i := range output {
			output[i] = lhs[lhsIter.Next()] << uint(rhs[rhsIter.Next()])
		}
	}
}

// shiftRightArithmeticGeneric performs lhs >> rhs with broadcasting support.
// For signed types, Go's >> preserves the sign bit (arithmetic shift).
// For unsigned types, Go's >> is already a logical (zero-fill) shift, so this
// function is also used by shiftRightLogicalGeneric for unsigned dispatch.
func shiftRightArithmeticGeneric[T PODIntegerConstraints](lhsBuf, rhsBuf, outputBuf *Buffer) {
	lhs, rhs, output := lhsBuf.flat.([]T), rhsBuf.flat.([]T), outputBuf.flat.([]T)
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
	case lhsBuf.shape.Equal(rhsBuf.shape):
		for i, v := range lhs {
			output[i] = v >> uint(rhs[i])
		}
	default:
		lhsIter := newBroadcastIterator(lhsBuf.shape, outputBuf.shape)
		rhsIter := newBroadcastIterator(rhsBuf.shape, outputBuf.shape)
		for i := range output {
			output[i] = lhs[lhsIter.Next()] >> uint(rhs[rhsIter.Next()])
		}
	}
}

// shiftRightLogicalGeneric performs logical right shift with broadcasting support.
func shiftRightLogicalGeneric[T PODIntegerConstraints](lhsBuf, rhsBuf, outputBuf *Buffer) {
	switch any(T(0)).(type) {
	case int8:
		shiftRightLogicalSignedGeneric[int8, uint8](lhsBuf, rhsBuf, outputBuf)
	case int16:
		shiftRightLogicalSignedGeneric[int16, uint16](lhsBuf, rhsBuf, outputBuf)
	case int32:
		shiftRightLogicalSignedGeneric[int32, uint32](lhsBuf, rhsBuf, outputBuf)
	case int64:
		shiftRightLogicalSignedGeneric[int64, uint64](lhsBuf, rhsBuf, outputBuf)
	default:
		// For unsigned types, >> is already a logical shift.
		shiftRightArithmeticGeneric[T](lhsBuf, rhsBuf, outputBuf)
	}
}

// shiftRightLogicalSignedGeneric performs logical right shift for signed types by
// reinterpreting as unsigned, shifting, then converting back.
// T is the signed type, U is the corresponding unsigned type.
func shiftRightLogicalSignedGeneric[T ~int8 | ~int16 | ~int32 | ~int64, U ~uint8 | ~uint16 | ~uint32 | ~uint64](
	lhsBuf, rhsBuf, outputBuf *Buffer) {
	lhs, rhs, output := lhsBuf.flat.([]T), rhsBuf.flat.([]T), outputBuf.flat.([]T)
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
	case lhsBuf.shape.Equal(rhsBuf.shape):
		for i, v := range lhs {
			output[i] = T(U(v) >> uint(rhs[i]))
		}
	default:
		lhsIter := newBroadcastIterator(lhsBuf.shape, outputBuf.shape)
		rhsIter := newBroadcastIterator(rhsBuf.shape, outputBuf.shape)
		for i := range output {
			output[i] = T(U(lhs[lhsIter.Next()]) >> uint(rhs[rhsIter.Next()]))
		}
	}
}
