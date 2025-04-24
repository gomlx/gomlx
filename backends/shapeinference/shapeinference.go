// Package shapeinference calculates the shape resulting from operations.
//
// This can be useful for new backends to test and help plan for buffer space for temporary or output buffers.
//
// It defines BinaryOp function for shape inference for the majority of binary functions, using the standard
// broadcasting rules.
//
// The majority of the unary functions don't change the shape, except those that explicitly say that in their name,
// like Reshape, etc.
//
// For the remainder ops, it defines one function per OpType.
//
// It also define some classes of operations that can be used.
package shapeinference

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"slices"
)

var (
	// BooleanOperations take booleans as input, aka. logical operations.
	BooleanOperations = types.SetWith(
		backends.OpTypeLogicalAnd,
		backends.OpTypeLogicalOr,
		backends.OpTypeLogicalXor,
		backends.OpTypeLogicalNot,
	)

	// BitwiseOperations operates only on integer (binary) numbers, and won't work on floats or complex numbers.
	BitwiseOperations = types.SetWith(
		backends.OpTypeBitwiseAnd,
		backends.OpTypeBitwiseOr,
		backends.OpTypeBitwiseXor,
		backends.OpTypeBitwiseNot,
		backends.OpTypeBitCount,
		backends.OpTypeShiftLeft,
		backends.OpTypeShiftRightArithmetic,
		backends.OpTypeShiftRightLogical,
		backends.OpTypeBitCount,
		backends.OpTypeClz,
	)

	// NumberOperations can take any type of number as input: integer, float or complex.
	NumberOperations = types.SetWith(
		backends.OpTypeAdd,
		backends.OpTypeSub,
		backends.OpTypeMul,
		backends.OpTypeDiv,
		backends.OpTypePow,
		backends.OpTypeRem,

		// Notice Abs and Sign works for unsigned ints: it's just a trivial implementation.
		backends.OpTypeAbs,
		backends.OpTypeSign,

		backends.OpTypeEqual,
		backends.OpTypeGreaterOrEqual,
		backends.OpTypeGreaterThan,
		backends.OpTypeLessOrEqual,
		backends.OpTypeLessThan,

		backends.OpTypeEqualTotalOrder,
		backends.OpTypeGreaterOrEqualTotalOrder,
		backends.OpTypeGreaterThanTotalOrder,
		backends.OpTypeLessOrEqualTotalOrder,
		backends.OpTypeLessThanTotalOrder,
	)

	SignedNumberOperations = types.SetWith(
		backends.OpTypeNeg,
	)

	// FloatOperations operates only on float (and not on complex numbers).
	FloatOperations = types.SetWith(
		backends.OpTypeLogistic,
		backends.OpTypeCos,
		backends.OpTypeSin,
		backends.OpTypeTanh,
	)

	// FloatOrComplexOperations operates only on float or complex numbers, and won't work on integer or boolean values.
	FloatOrComplexOperations = types.SetWith(
		backends.OpTypeExp,
		backends.OpTypeExpm1,
		backends.OpTypeLog,
		backends.OpTypeLog1p,
		backends.OpTypeCeil,
		backends.OpTypeFloor,
		backends.OpTypeRound,
		backends.OpTypeRsqrt,
		backends.OpTypeSqrt,
		backends.OpTypeIsFinite,
	)

	// ComplexOperations operates only on complex numbers.
	ComplexOperations = types.SetWith(
		backends.OpTypeImag,
		backends.OpTypeReal,
		backends.OpTypeConj,
	)

	// StandardBinaryOperations include all operations that have two operands (usually named lhs (left-hand-side) and
	// rhs (right-hand-side) and are usually commutative (invariant to order).
	StandardBinaryOperations = types.SetWith(
		backends.OpTypeAdd,
		backends.OpTypeSub,
		backends.OpTypeMul,
		backends.OpTypeDiv,
		backends.OpTypePow,
		backends.OpTypeRem,
		backends.OpTypeBitwiseAnd,
		backends.OpTypeBitwiseOr,
		backends.OpTypeBitwiseXor,
		backends.OpTypeLogicalAnd,
		backends.OpTypeLogicalOr,
		backends.OpTypeLogicalXor,
		backends.OpTypeMax,
		backends.OpTypeMin,
	)

	// ComparisonOperations include all operations that takes two inputs and returns booleans with the results of
	// a comparison.
	ComparisonOperations = types.SetWith(
		backends.OpTypeEqual,
		backends.OpTypeEqualTotalOrder,
		backends.OpTypeGreaterOrEqual,
		backends.OpTypeGreaterOrEqualTotalOrder,
		backends.OpTypeGreaterThan,
		backends.OpTypeGreaterThanTotalOrder,
		backends.OpTypeLessOrEqual,
		backends.OpTypeLessOrEqualTotalOrder,
		backends.OpTypeLessThan,
		backends.OpTypeLessThanTotalOrder,
	)

	// StandardUnaryOperations include all operations that have a single operand as input and the return shape is the
	// same as the input (so no reductions).
	StandardUnaryOperations = types.SetWith(
		backends.OpTypeLogicalNot,
		backends.OpTypeBitwiseNot,
		backends.OpTypeBitCount,
		backends.OpTypeClz,
		backends.OpTypeExp,
		backends.OpTypeExpm1,
		backends.OpTypeLog,
		backends.OpTypeLog1p,
		backends.OpTypeLogistic,
		backends.OpTypeCeil,
		backends.OpTypeFloor,
		backends.OpTypeRound,
		backends.OpTypeRsqrt,
		backends.OpTypeSqrt,
		backends.OpTypeImag,
		backends.OpTypeReal,
		backends.OpTypeConj,
		backends.OpTypeCos,
		backends.OpTypeSin,
		backends.OpTypeTanh,
		backends.OpTypeAbs,
		backends.OpTypeNeg,
		backends.OpTypeSign,
	)
)

// BinaryOp returns the expected output shape for ops in the StandardBinaryOperations set -- those include all
// operations that have two operands (usually named lhs (left-hand-side) and rhs (right-hand-side) and are usually
// commutative (invariant to order).
//
// It may throw (panic) an exception if the data type (shape.DType) is invalid for the operation -- e.g.: non-matching
// dtypes, or LogicalAnd not having booleans (dtype.Bool) as input.
func BinaryOp(opType backends.OpType, lhsShape, rhsShape shapes.Shape) shapes.Shape {
	if !StandardBinaryOperations.Has(opType) {
		exceptions.Panicf("operations %s is not in the StandardBinaryOperations set, cannot process it with BinaryOp", opType)
	}
	if lhsShape.DType == dtypes.InvalidDType || rhsShape.DType == dtypes.InvalidDType {
		exceptions.Panicf("invalid shape for %s or %s for BinaryOp %s", lhsShape, rhsShape, opType)
	}
	if lhsShape.DType != rhsShape.DType {
		exceptions.Panicf("data types (DType) for BinaryOp %s must match, got %s and %s", opType, lhsShape, rhsShape)
	}
	if BooleanOperations.Has(opType) && lhsShape.DType != dtypes.Bool {
		exceptions.Panicf("logical BinaryOp %s must have boolean (dtype.Bool) data types as input, got %s", opType, lhsShape)
	}
	if BitwiseOperations.Has(opType) && !lhsShape.DType.IsInt() {
		exceptions.Panicf("bitwise BinaryOp %s must have an integer (Int8, UInt8, Int32, ...) data type as input, got %s", opType, lhsShape)
	}
	if NumberOperations.Has(opType) && !(lhsShape.DType.IsInt() || lhsShape.DType.IsFloat() || lhsShape.DType.IsComplex()) {
		exceptions.Panicf("numeric BinaryOp %s must have a number (Int32, Float32, Complex64, ...) data type as input, got %s", opType, lhsShape)
	}
	if FloatOperations.Has(opType) && !lhsShape.DType.IsFloat() {
		exceptions.Panicf("float BinaryOp %s must have a float (Float32, Float64, ...) data type as input, got %s", opType, lhsShape)
	}
	if FloatOrComplexOperations.Has(opType) && !(lhsShape.DType.IsFloat() || lhsShape.DType.IsComplex()) {
		exceptions.Panicf("float/complex BinaryOp %s must have a float or complex (Float32, Complex64, ...) data type as input, got %s", opType, lhsShape)
	}
	if ComplexOperations.Has(opType) && !lhsShape.DType.IsComplex() {
		exceptions.Panicf("complex BinaryOp %s must have a complex (Complex64, Complex128) data type as input, got %s", opType, lhsShape)
	}

	return binaryOpImpl(opType, lhsShape, rhsShape)
}

func binaryOpImpl(opType backends.OpType, lhsShape, rhsShape shapes.Shape) shapes.Shape {
	// Trivial cases: if one of the sides is a scalar, return the other side shape.
	if lhsShape.IsScalar() {
		return rhsShape
	}
	if rhsShape.IsScalar() {
		return lhsShape
	}

	// Other cases, either the dimensions match or one of them is 1.
	if lhsShape.Rank() != rhsShape.Rank() {
		exceptions.Panicf("if operands are not scalars, their rank must match for BinaryOp (%s), got shapes %s and %s",
			opType, lhsShape, rhsShape)
	}
	shape := lhsShape.Clone()
	for axis := range shape.Rank() {
		lhsDim := lhsShape.Dimensions[axis]
		rhsDim := rhsShape.Dimensions[axis]
		if lhsDim != 1 && rhsDim != 1 && lhsDim != rhsDim {
			exceptions.Panicf("dimension of axis #%d doesn't match and cannot be broadcast for BinaryOp (%s), got shapes %s and %s",
				axis, opType, lhsShape, rhsShape)
		}
		shape.Dimensions[axis] = max(lhsDim, rhsDim)
	}
	return shape
}

// ComparisonOp returns the broadcast shape with dtype set to Bool, for comparison operations (Equal, LessThan, GreaterOrEqual, etc.)
func ComparisonOp(opType backends.OpType, lhsShape, rhsShape shapes.Shape) shapes.Shape {
	if !ComparisonOperations.Has(opType) {
		exceptions.Panicf("operation %s is not in the ComparisonOperations set, cannot process it with ComparisonOp", opType)
	}
	if lhsShape.DType == dtypes.InvalidDType || rhsShape.DType == dtypes.InvalidDType {
		exceptions.Panicf("invalid shape for %s or %s for ComparisonOp %s", lhsShape, rhsShape, opType)
	}
	if lhsShape.DType != rhsShape.DType {
		exceptions.Panicf("data types (DType) for ComparisonOp %s must match, got %s and %s", opType, lhsShape, rhsShape)
	}
	if !NumberOperations.Has(opType) {
		exceptions.Panicf("operation %s is not in the NumberOperations set, cannot process it with ComparisonOp", opType)
	}

	shape := binaryOpImpl(opType, lhsShape, rhsShape)
	shape.DType = dtypes.Bool
	return shape
}

// UnaryOp checks the validity of the data type for StandardUnaryOperations set, and throws an exception
// (panic) in case of mismatch.
//
// It returns the same shape as the operand -- there is no broadcast on standard unary operations.
func UnaryOp(opType backends.OpType, operand shapes.Shape) shapes.Shape {
	if !StandardUnaryOperations.Has(opType) {
		exceptions.Panicf("operation %s is not in the StandardUnaryOperations set, cannot process it with UnaryOp", opType)
	}
	if operand.DType == dtypes.InvalidDType {
		exceptions.Panicf("invalid shape %s for UnaryOp %s", operand, opType)
	}
	if BooleanOperations.Has(opType) && operand.DType != dtypes.Bool {
		exceptions.Panicf("logical UnaryOp %s must have boolean (dtype.Bool) data types as input, got %s", opType, operand)
	}
	if BitwiseOperations.Has(opType) && !operand.DType.IsInt() {
		exceptions.Panicf("bitwise UnaryOp %s must have an integer (Int8, UInt8, Int32, ...) data type as input, got %s", opType, operand)
	}
	if SignedNumberOperations.Has(opType) && (operand.DType.IsUnsigned() ||
		!(operand.DType.IsInt() || operand.DType.IsFloat() || operand.DType.IsComplex())) {
		exceptions.Panicf("signed UnaryOp %s must have a signed data type as input, got %s", opType, operand)
	}
	if NumberOperations.Has(opType) && !(operand.DType.IsInt() || operand.DType.IsFloat() || operand.DType.IsComplex()) {
		exceptions.Panicf("numeric UnaryOp %s must have a number (Int32, Float32, Complex64, ...) data type as input, got %s", opType, operand)
	}
	if FloatOperations.Has(opType) && !operand.DType.IsFloat() {
		exceptions.Panicf("float UnaryOp %s must have a float (Float32, Float64, ...) data type as input, got %s", opType, operand)
	}
	if FloatOrComplexOperations.Has(opType) && !(operand.DType.IsFloat() || operand.DType.IsComplex()) {
		exceptions.Panicf("float/complex UnaryOp %s must have a float or complex (Float32, Complex64, ...) data type as input, got %s", opType, operand)
	}
	if ComplexOperations.Has(opType) && !operand.DType.IsComplex() {
		exceptions.Panicf("complex UnaryOp %s must have a complex (Complex64, Complex128) data type as input, got %s", opType, operand)
	}
	return operand
}

// WhereOp returns the shape resulting from the Where operation.
//
// Shape constraints:
//  1. onTrue and onFalse must have the exact same shape.
//  2. condition must either be a scalar or match the shape of onTrue and onFalse, except for the DType that
//     must be Bool.
func WhereOp(condition, onTrue, onFalse shapes.Shape) shapes.Shape {
	if condition.DType != dtypes.Bool {
		exceptions.Panicf("condition for Where() must be a boolean, got %s instead", condition)
	}
	if !onTrue.IsScalar() && !onFalse.IsScalar() && !onTrue.Equal(onFalse) {
		exceptions.Panicf("onTrue (%s) and onFalse (%s) values for Where() must either be scalar or match each other's shape",
			onTrue, onFalse)
	}
	if !condition.IsScalar() && !(onTrue.IsScalar() && onFalse.IsScalar()) {
		if slices.Compare(condition.Dimensions, onTrue.Dimensions) != 0 {
			exceptions.Panicf("condition for Where() must either be a scalar or match the shape (not the DType) of the values (%s), but got %s",
				onTrue, condition)
		}
	}

	output := onTrue
	if output.IsScalar() {
		output = onFalse
		if output.IsScalar() && !condition.IsScalar() {
			output = condition.Clone()
			output.DType = onTrue.DType
		}
	}
	return output
}

// ReshapeOp to the given dimensions: trivial output shape, but this function also checks
// that the sizes are the same.
//
// Notice the backends.Reshape doesn't support auto-scaling dimensions (set to -1), as graph.Reshape does.
func ReshapeOp(operand shapes.Shape, dims []int) shapes.Shape {
	output := shapes.Make(operand.DType, dims...)
	if operand.Size() != output.Size() {
		exceptions.Panicf("Reshape() cannot reshape %s to dimensions %v, their size don't match",
			operand, dims)
	}
	return output
}

// TransposeOp all axes of the operand.
// There must be one value in permutations for each axis in the operand.
// The output will have: output.Shape.Dimension[ii] = operand.Shape.Dimension[permutations[i]].
func TransposeOp(operand shapes.Shape, permutations []int) shapes.Shape {
	rank := operand.Rank()
	if len(permutations) != rank {
		exceptions.Panicf("Transpose() requires all axes permutations to be defined, operand has shape %s, but %d permutations were given",
			operand, len(permutations))
	}
	if rank == 0 {
		return operand
	}

	// Check permutation axes are within range and unique.
	axesSet := slices.Clone(permutations)
	slices.Sort(axesSet)
	for ii, srcAxis := range axesSet {
		if srcAxis < 0 || srcAxis >= rank {
			exceptions.Panicf("invalid permutation axis %d given to Transpose(%s), it must be within the range of its rank",
				srcAxis, operand)
		}
		if ii > 0 && srcAxis == axesSet[ii-1] {
			exceptions.Panicf("invalid permutations given to Transpose(%s, %v), there cannot be any repeated axis, each must appear exactly once",
				operand, permutations)
		}
	}

	output := operand.Clone()
	for axis := range output.Dimensions {
		srcAxis := permutations[axis]
		output.Dimensions[axis] = operand.Dimensions[srcAxis]
	}
	return output
}

// ReduceOp works for the ReduceMax, ReduceMin, ReduceSum and ReduceProduct ops.
func ReduceOp(operand shapes.Shape, axes []int) shapes.Shape {
	if len(axes) == 0 {
		return operand
	}
	output := shapes.Make(operand.DType)
	outputRank := operand.Rank() - len(axes)
	if outputRank > 0 {
		// Copy over dimensions that will stay.
		output.Dimensions = make([]int, 0, outputRank)
		for _, axis := range axes {
			if axis < 0 || axis >= operand.Rank() {
				exceptions.Panicf("Reduce operation require each axis to be 0 <= axis < rank, but got invalid axis %d for shape %s", axis, operand)
			}
		}
		axesSet := types.SetWith(axes...)
		for axis, dim := range operand.Dimensions {
			if !axesSet.Has(axis) {
				output.Dimensions = append(output.Dimensions, dim)
			}
		}
	}
	return output
}
