package stablehlo

// This file contains manually implemented operations.

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	stablehlotypes "github.com/gomlx/stablehlo/types"
	stablehloshapes "github.com/gomlx/stablehlo/types/shapes"
	"github.com/pkg/errors"
)

// Iota implements backends.Builder interface.
func (b *Builder) Iota(shape shapes.Shape, iotaAxis int) (backends.Op, error) {
	value, err := b.fn.Iota(ShapeToStableHLO(shape), iotaAxis)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil

}

// Reshape implements backends.Builder interface.
func (b *Builder) Reshape(x backends.Op, dimensions ...int) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("Reshape", x)
	if err != nil {
		return nil, err
	}
	xNode := nodes[0]
	dtype := xNode.shape.DType
	shape := stablehloshapes.Make(dtype, dimensions...)
	value, err := b.fn.Reshape(xNode.value, shape)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// comparison generic operation.
func (b *Builder) comparison(opType backends.OpType, lhs, rhs backends.Op) (backends.Op, error) {
	lhsNode, rhsNode, err := b.broadcastForBinaryOps(opType, lhs, rhs)
	if err != nil {
		return nil, err
	}

	// Find compareType
	dtype := lhsNode.shape.DType
	compareType := stablehlotypes.CompareFloat
	if !dtype.IsFloat() {
		if dtype.IsUnsigned() {
			compareType = stablehlotypes.CompareUnsigned
		} else {
			compareType = stablehlotypes.CompareSigned
		}
	}

	var direction stablehlotypes.ComparisonDirection
	switch opType {
	case backends.OpTypeEqual:
		direction = stablehlotypes.CompareEQ
	case backends.OpTypeNotEqual:
		direction = stablehlotypes.CompareNE
	case backends.OpTypeGreaterThan:
		direction = stablehlotypes.CompareGT
	case backends.OpTypeGreaterOrEqual:
		direction = stablehlotypes.CompareGE
	case backends.OpTypeLessThan:
		direction = stablehlotypes.CompareLT
	case backends.OpTypeLessOrEqual:
		direction = stablehlotypes.CompareLE

	case backends.OpTypeEqualTotalOrder:
		direction = stablehlotypes.CompareEQ
		compareType = stablehlotypes.CompareTotalOrder
	case backends.OpTypeNotEqualTotalOrder:
		direction = stablehlotypes.CompareNE
		compareType = stablehlotypes.CompareTotalOrder
	case backends.OpTypeGreaterThanTotalOrder:
		direction = stablehlotypes.CompareGT
		compareType = stablehlotypes.CompareTotalOrder
	case backends.OpTypeGreaterOrEqualTotalOrder:
		direction = stablehlotypes.CompareGE
		compareType = stablehlotypes.CompareTotalOrder
	case backends.OpTypeLessThanTotalOrder:
		direction = stablehlotypes.CompareLT
		compareType = stablehlotypes.CompareTotalOrder
	case backends.OpTypeLessOrEqualTotalOrder:
		direction = stablehlotypes.CompareLE
		compareType = stablehlotypes.CompareTotalOrder

	default:
		return nil, errors.Errorf("unsupported comparison operation %v", opType)
	}

	value, err := b.fn.Compare(lhsNode.value, rhsNode.value, direction, compareType)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// Equal returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (b *Builder) Equal(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeEqual, lhs, rhs)
}

// NotEqual returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (b *Builder) NotEqual(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeNotEqual, lhs, rhs)
}

// GreaterThan returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (b *Builder) GreaterThan(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeGreaterThan, lhs, rhs)
}

// GreaterOrEqual returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (b *Builder) GreaterOrEqual(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeGreaterOrEqual, lhs, rhs)
}

// LessThan returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (b *Builder) LessThan(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeLessThan, lhs, rhs)
}

// LessOrEqual returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (b *Builder) LessOrEqual(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeLessOrEqual, lhs, rhs)
}

// EqualTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func (b *Builder) EqualTotalOrder(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeEqualTotalOrder, lhs, rhs)
}

// NotEqualTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func (b *Builder) NotEqualTotalOrder(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeNotEqualTotalOrder, lhs, rhs)
}

// GreaterThanTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func (b *Builder) GreaterThanTotalOrder(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeGreaterThanTotalOrder, lhs, rhs)
}

// GreaterOrEqualTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func (b *Builder) GreaterOrEqualTotalOrder(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeGreaterOrEqualTotalOrder, lhs, rhs)
}

// LessThanTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func (b *Builder) LessThanTotalOrder(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeLessThanTotalOrder, lhs, rhs)
}

// LessOrEqualTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func (b *Builder) LessOrEqualTotalOrder(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeLessOrEqualTotalOrder, lhs, rhs)
}

// Conj returns the conjugate of a complex number. E.g: Conj(1+3i) = 1-3i
func (b *Builder) Conj(operand backends.Op) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("Conj", operand)
	if err != nil {
		return nil, err
	}
	operandNode := nodes[0]
	realValue, err := b.fn.Real(operandNode.value)
	if err != nil {
		return nil, err
	}
	imagValue, err := b.fn.Imag(operandNode.value)
	if err != nil {
		return nil, err
	}
	imagValue, err = b.fn.Negate(imagValue)
	if err != nil {
		return nil, err
	}
	value, err := b.fn.Complex(realValue, imagValue)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// Dot returns the "dot product" operation.
// The exact semantics of this operation depend on the ranks of the operands:
// | Input | Output | Semantics |
// | vector [n] dot vector [n] | scalar | vector dot product |
// | matrix [m x k] dot vector [k] | vector [m]	matrix-vector multiplication |
// | matrix [m x k] dot matrix [k x n] | matrix [m x n] | matrix-matrix multiplication |
// The operation performs sum of products over the second dimension of x0 (or the first if it has rank 1) and
// the first dimension of x1.
// These are the "contracted" dimensions.
// The contracted dimensions of x0 and x1 must be of the same size.
// In practice, it can be used to perform dot products between vectors, vector/matrix multiplications or
// matrix/matrix multiplications.
func (b *Builder) Dot(lhs, rhs backends.Op) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("Dot", lhs, rhs)
	if err != nil {
		return nil, err
	}
	lhsNode, rhsNode := nodes[0], nodes[1]
	var output backends.Op
	if lhsNode.shape.Rank() == 1 && rhsNode.shape.Rank() == 1 {
		// Contracting both vectors.
		output, err = b.DotGeneral(lhsNode, []int{0}, []int{}, rhsNode, []int{0}, []int{})
	} else if lhsNode.shape.Rank() == 2 && rhsNode.shape.Rank() == 1 {
		// Contract rhs vector.
		output, err = b.DotGeneral(lhsNode, []int{1}, []int{}, rhsNode, []int{0}, []int{})
	} else if lhsNode.shape.Rank() == 2 && rhsNode.shape.Rank() == 2 {
		// Traditional matrix multiplication:
		output, err = b.DotGeneral(lhsNode, []int{1}, []int{}, rhsNode, []int{0}, []int{})
	} else {
		return nil, errors.Errorf("Dot operands have invalid ranks: lhs=%v, rhs=%v", lhsNode.shape, rhsNode.shape)
	}
	if err != nil {
		return nil, errors.WithMessagef(err, "while building op Dot()")
	}
	return output, nil
}

// Clamp returns the element-wise clamping operation.
//
// The values max and min can either be a scalar or have the same shape as x.
func (b *Builder) Clamp(min, a backends.Op, max backends.Op) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("Clamp", min, a, max)
	if err != nil {
		return nil, err
	}
	minNode := nodes[0]
	aNode := nodes[1]
	maxNode := nodes[2]
	value, err := b.fn.Clamp(minNode.value, aNode.value, maxNode.value)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// Gather is a powerful but cumbersome Gather operation. See details in the backend.
//
// Notice GoMLX backend Gather operation doesn't support batching axes, which StableHLO does.
// For compatibility, we simply leave them empty.
func (b *Builder) Gather(operand, startIndices backends.Op, indexVectorAxis int, offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int, indicesAreSorted bool) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("Gather", operand, startIndices)
	if err != nil {
		return nil, err
	}
	operandNode := nodes[0]
	startIndicesNode := nodes[1]
	var operandBatchingAxes, startIndicesBatchingAxes []int
	value, err := b.fn.Gather(operandNode.value, startIndicesNode.value, indexVectorAxis,
		offsetOutputAxes, collapsedSliceAxes, operandBatchingAxes,
		startIndicesBatchingAxes, startIndexMap, sliceSizes, indicesAreSorted)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}
