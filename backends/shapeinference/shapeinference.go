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

	// FloatOperations operates only on float (and not on complex numbers).
	FloatOperations = types.SetWith(
		backends.OpTypeCos,
		backends.OpTypeSin,
		backends.OpTypeTanh,
	)

	// FloatOrComplexOperations operates only on float or complex numbers, and won't work on integer or boolean values.
	FloatOrComplexOperations = types.SetWith(
		backends.OpTypeExp,
		backends.OpTypeLog,
		backends.OpTypeLog1p,
		backends.OpTypeCeil,
		backends.OpTypeFloor,
		backends.OpTypeRound,
		backends.OpTypeRsqrt,
		backends.OpTypeSqrt,
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
		backends.OpTypeBitwiseAnd,
		backends.OpTypeBitwiseOr,
		backends.OpTypeBitwiseXor,
		backends.OpTypeLogicalAnd,
		backends.OpTypeLogicalOr,
		backends.OpTypeLogicalXor,
		backends.OpTypeAdd,
		backends.OpTypeSub,
		backends.OpTypeMul,
		backends.OpTypeDiv,
		backends.OpTypePow,
		backends.OpTypeComplex,
		backends.OpTypeMax,
		backends.OpTypeMin,
		backends.OpTypeRem,
	)

	// StandardUnaryOperations include all operations that have a single operand as input and the return shape is the
	// same as the input (so no reductions).
	StandardUnaryOperations = types.SetWith(
		backends.OpTypeLogicalNot,
		backends.OpTypeBitwiseNot,
		backends.OpTypeBitCount,
		backends.OpTypeBitCount,
		backends.OpTypeClz,
		backends.OpTypeExp,
		backends.OpTypeLog,
		backends.OpTypeLog1p,
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
		backends.OpTypeIsFinite,
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
func BinaryOp(opType backends.OpType, shape1, shape2 shapes.Shape) shapes.Shape {
	if !StandardBinaryOperations.Has(opType) {
		exceptions.Panicf("operations %s is not in the StandardBinaryOperations set, cannot process it with BinaryOp", opType)
	}
	if shape1.DType == dtypes.InvalidDType || shape2.DType == dtypes.InvalidDType {
		exceptions.Panicf("invalid shape for %s or %s for BinaryOp %s", shape1, shape2, opType)
	}
	if shape1.DType != shape2.DType {
		exceptions.Panicf("data types (DType) for BinaryOp %s must match, got %s and %s", opType, shape1, shape2)
	}
	if BooleanOperations.Has(opType) && shape1.DType != dtypes.Bool {
		exceptions.Panicf("logical BinaryOp %s must have boolean (dtype.Bool) data types as input, got %s", opType, shape1)
	}
	if BitwiseOperations.Has(opType) && !shape1.DType.IsInt() {
		exceptions.Panicf("bitwise BinaryOp %s must have an integer (Int8, UInt8, Int32, ...) data type as input, got %s", opType, shape1)
	}

	return shapes.Invalid()
}
