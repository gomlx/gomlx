package simplego

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
)

// This file implements binary operations.
// One optimization supported is specially handling the cases where one of the operands is a scalar (or of size 1),
// in which case it becomes almost a unary operation with a constant value.
func init() {
	nodeExecutors[backends.OpTypeAdd] = execAdd
}

// binaryOperandsAndOutput is a convenience function to get the inputs and output -- which may be the reuse of the input.
func binaryOperandsAndOutput(backend *Backend, inputs []*Buffer, inputsOwned []bool, outputShape shapes.Shape) (
	lhs, rhs, output *Buffer, lhsIsScalarOr1, rhsIsScalarOr1 bool) {
	lhs, rhs = inputs[0], inputs[1]
	lhsIsScalarOr1, rhsIsScalarOr1 = lhs.shape.Size() == 1, rhs.shape.Size() == 1
	if inputsOwned[1] && rhs.shape.Equal(outputShape) {
		output = rhs
		inputs[1] = nil
	} else if inputsOwned[0] && lhs.shape.Equal(outputShape) {
		output = lhs
		inputs[0] = nil
	} else {
		output = backend.getBuffer(outputShape.DType, outputShape.Size())
		output.shape = outputShape
	}
	return
}

// broadcastIterator allows one to iterate over the flat indices of tensor that is being broadcast
// (some dimensions will grow)
type broadcastIterator struct {
	flatIdx     int
	perAxesIdx  []int
	targetDims  []int
	isBroadcast []bool
	strides     []int
}

func newBroadcastIterator(fromShape, toShape shapes.Shape) *broadcastIterator {
	rank := fromShape.Rank() // == toShape.Rank()
	bi := &broadcastIterator{
		perAxesIdx:  make([]int, rank),
		targetDims:  toShape.Dimensions,
		isBroadcast: make([]bool, rank),
		strides:     make([]int, rank),
	}
	stride := 1
	for axis := fromShape.Rank() - 1; axis >= 0; axis-- {
		bi.strides[axis] = stride
		stride *= fromShape.Dimensions[axis]
		bi.isBroadcast[axis] = fromShape.Dimensions[axis] != toShape.Dimensions[axis]
	}
	return bi
}

func (bi *broadcastIterator) Next() (flatIdx int) {
	flatIdx = bi.flatIdx
	bi.flatIdx++
	rank := len(bi.perAxesIdx)
	for axis := rank - 1; axis >= 0; axis-- {
		bi.perAxesIdx[axis]++
		if bi.perAxesIdx[axis] < bi.targetDims[axis] {
			if bi.isBroadcast[axis] {
				// If we are broadcasting on this axis, we need to go back and repeat the same slice of the tensor.
				bi.flatIdx -= bi.strides[axis]
			}
			break
		}
		bi.perAxesIdx[axis] = 0
	}
	return
}

// execAdd executes the binary op Add.
func execAdd(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) *Buffer {
	lhs, rhs, output, lhsIsScalarOr1, rhsIsScalarOr1 := binaryOperandsAndOutput(backend, inputs, inputsOwned, node.shape)

	// Add is commutative, so if any of the two is scalar, make the rhs the scalar one.
	if lhsIsScalarOr1 && !rhsIsScalarOr1 {
		lhs, rhs = rhs, lhs
		lhsIsScalarOr1, rhsIsScalarOr1 = rhsIsScalarOr1, lhsIsScalarOr1
	}

	switch output.shape.DType {
	case dtypes.Int8:
		execAddGeneric[int8](lhs.flat.([]int8), rhs.flat.([]int8), output.flat.([]int8),
			lhs.shape, rhs.shape, output.shape)
	case dtypes.Int16:
		execAddGeneric[int16](lhs.flat.([]int16), rhs.flat.([]int16), output.flat.([]int16),
			lhs.shape, rhs.shape, output.shape)
	case dtypes.Int32:
		execAddGeneric[int32](lhs.flat.([]int32), rhs.flat.([]int32), output.flat.([]int32),
			lhs.shape, rhs.shape, output.shape)
	case dtypes.Int64:
		execAddGeneric[int64](lhs.flat.([]int64), rhs.flat.([]int64), output.flat.([]int64),
			lhs.shape, rhs.shape, output.shape)
	case dtypes.Float32:
		execAddGeneric[float32](lhs.flat.([]float32), rhs.flat.([]float32), output.flat.([]float32),
			lhs.shape, rhs.shape, output.shape)
	case dtypes.Float64:
		execAddGeneric[float64](lhs.flat.([]float64), rhs.flat.([]float64), output.flat.([]float64),
			lhs.shape, rhs.shape, output.shape)
	case dtypes.BFloat16:
		execAddBF16(lhs.flat.([]bfloat16.BFloat16), rhs.flat.([]bfloat16.BFloat16), output.flat.([]bfloat16.BFloat16),
			lhs.shape, rhs.shape, output.shape)
	default:
		exceptions.Panicf("unsupported data type %s for %s", output.shape.DType, node.opType)
	}
	return output
}

func execAddGeneric[T signedNumericPODConstraints](lhs, rhs, output []T,
	lhsShape, rhsShape, outputShape shapes.Shape) {
	if len(rhs) == 1 {
		// Case 1: One side (rhs) is a scalar: only iterate over the lhs.
		c := rhs[0]
		for ii, input := range lhs {
			output[ii] = input + c
		}
		return

	} else if lhsShape.Equal(rhsShape) {
		// Case 2: Exact same shapes, no broadcasting.
		for ii, input := range lhs {
			output[ii] = input + rhs[ii]
		}
		return

	} else {
		// Case 3: with broadcasting non-scalar tensors:
		lhsIter := newBroadcastIterator(lhsShape, outputShape)
		rhsIter := newBroadcastIterator(rhsShape, outputShape)
		for outputIdx := range output {
			lhsIdx := lhsIter.Next()
			rhsIdx := rhsIter.Next()
			output[outputIdx] = lhs[lhsIdx] + rhs[rhsIdx]
		}
	}
	return
}

func execAddBF16(lhs, rhs, output []bfloat16.BFloat16,
	lhsShape, rhsShape, outputShape shapes.Shape) {
	if len(rhs) == 1 {
		// One side (rhs) is a scalar: only iterate over the lhs.
		c := rhs[0].Float32()
		for ii, input := range lhs {
			output[ii] = bfloat16.FromFloat32(input.Float32() + c)
		}
		return

	} else if lhsShape.Equal(rhsShape) {
		// Case 2: Exact same shapes, no broadcasting.
		for ii, input := range lhs {
			output[ii] = bfloat16.FromFloat32(input.Float32() + rhs[ii].Float32())
		}
		return

	} else {
		// Case 3: with broadcasting non-scalar tensors:
		lhsIter := newBroadcastIterator(lhsShape, outputShape)
		rhsIter := newBroadcastIterator(rhsShape, outputShape)
		for outputIdx := range output {
			lhsIdx := lhsIter.Next()
			rhsIdx := rhsIter.Next()
			output[outputIdx] = bfloat16.FromFloat32(lhs[lhsIdx].Float32() + rhs[rhsIdx].Float32())
		}
	}
	return
}

// execScalarPowIntGeneric is a O(num of bits) for Pow(base, exp) implementation for integers.
func execScalarPowIntGeneric[T integerPODConstraints](base, exp T) T {
	result := T(1)
	for exp > 0 {
		if exp%2 == 1 {
			result *= base
		}
		base *= base
		exp >>= 1 // exp /= 2
	}
	return result
}
