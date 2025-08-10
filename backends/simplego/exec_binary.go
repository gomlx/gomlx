package simplego

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
)

// This file implements binary operations.
// One optimization supported is specially handling the cases where one of the operands is a scalar (or of size 1),
// in which case it becomes almost a unary operation with a constant value.

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
//
// It is used by implicit broadcasting in binaryOps as well as by the the BroadcastInDim.
type broadcastIterator struct {
	flatIdx     int
	perAxesIdx  []int
	targetDims  []int
	isBroadcast []bool
	strides     []int
}

// newBroadcastIterator returns an iterator that allows one to iterate over the flat indices of a tensor that is being broadcast,
// where some dimensions will grow.
//
// Pre-requisite: fromShape.Rank() == toShape.Rank().
//
// It is used by implicit broadcasting in binaryOps as well as by the the execBroadcastInDim.
func newBroadcastIterator(fromShape, toShape shapes.Shape) *broadcastIterator {
	rank := fromShape.Rank() // == toShape.Rank()
	if rank != toShape.Rank() {
		exceptions.Panicf("broadcastIterator: rank mismatch fromShape=%s, toShape=%s", fromShape, toShape)
	}
	bi := &broadcastIterator{
		perAxesIdx:  make([]int, rank),
		targetDims:  toShape.Dimensions,
		isBroadcast: make([]bool, rank),
		strides:     make([]int, rank),
	}
	stride := 1
	for axis := rank - 1; axis >= 0; axis-- {
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

// execScalarPowIntGeneric is a O(num of bits) for Pow(base, exp) implementation for integers.
func execScalarPowIntGeneric[T PODIntegerConstraints](base, exp T) T {
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
