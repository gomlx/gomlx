// Package shapeinference calculates the shape resulting from operations, and validates its inputs.
//
// This can be useful for new backends to test and help plan for buffer space for temporary or output buffers.
//
// It defines a BinaryOp function for shape inference for the majority of binary functions, using the standard
// broadcasting rules.
//
// The majority of the unary functions don't change the shape, except those that explicitly say that in their name,
// like Reshape, etc.
//
// For the remainder ops, it defines one function per OpType.
package shapeinference

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
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

	// BitwiseOperations operates only on integer (binary) numbers and won't work on floats or complex numbers.
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

	// NumberOperations can take any type of number as input: integers, floats, or complex numbers.
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
		backends.OpTypeErf,
		backends.OpTypeLogistic,
		backends.OpTypeCos,
		backends.OpTypeSin,
		backends.OpTypeTanh,
	)

	// FloatOrComplexOperations operates only on float or complex numbers and won't work on integer or boolean values.
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

	// StandardBinaryOperations include all operations that have two operands usually named lhs (left-hand-side) and
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

	// ComparisonOperations include all operations that take two inputs and returns booleans with the results of
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

	// StandardUnaryOperations include all operations that have a single operand as input, and the return shape is the
	// same as the input (so no reductions).
	StandardUnaryOperations = types.SetWith(
		backends.OpTypeLogicalNot,
		backends.OpTypeBitwiseNot,
		backends.OpTypeBitCount,
		backends.OpTypeClz,
		backends.OpTypeErf,
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
// operations that have two operands usually named lhs (left-hand-side) and rhs (right-hand-side), and they are usually
// commutative (invariant to order).
//
// It returns an error if the data type (shape.DType) is invalid for the operation -- e.g.: non-matching
// dtypes, or LogicalAnd not having booleans (dtype.Bool) as input.
func BinaryOp(opType backends.OpType, lhsShape, rhsShape shapes.Shape) (output shapes.Shape, err error) {
	if !StandardBinaryOperations.Has(opType) {
		err = errors.Errorf("operations %s is not in the StandardBinaryOperations set, cannot process it with BinaryOp", opType)
		return
	}
	if lhsShape.DType == dtypes.InvalidDType || rhsShape.DType == dtypes.InvalidDType {
		err = errors.Errorf("invalid shape for %s or %s for BinaryOp %s", lhsShape, rhsShape, opType)
		return
	}
	if lhsShape.DType != rhsShape.DType {
		err = errors.Errorf("data types (DType) for BinaryOp %s must match, got %s and %s", opType, lhsShape, rhsShape)
		return
	}
	if BooleanOperations.Has(opType) && lhsShape.DType != dtypes.Bool {
		err = errors.Errorf("logical BinaryOp %s must have boolean (dtype.Bool) data types as input, got %s", opType, lhsShape)
		return
	}
	if BitwiseOperations.Has(opType) && !lhsShape.DType.IsInt() {
		err = errors.Errorf("bitwise BinaryOp %s must have an integer (Int8, UInt8, Int32, ...) data type as input, got %s", opType, lhsShape)
		return
	}
	if NumberOperations.Has(opType) && !(lhsShape.DType.IsInt() || lhsShape.DType.IsFloat() || lhsShape.DType.IsComplex()) {
		err = errors.Errorf("numeric BinaryOp %s must have a number (Int32, Float32, Complex64, ...) data type as input, got %s", opType, lhsShape)
		return
	}
	if FloatOperations.Has(opType) && !lhsShape.DType.IsFloat() {
		err = errors.Errorf("float BinaryOp %s must have a float (Float32, Float64, ...) data type as input, got %s", opType, lhsShape)
		return
	}
	if FloatOrComplexOperations.Has(opType) && !(lhsShape.DType.IsFloat() || lhsShape.DType.IsComplex()) {
		err = errors.Errorf("float/complex BinaryOp %s must have a float or complex (Float32, Complex64, ...) data type as input, got %s", opType, lhsShape)
		return
	}
	if ComplexOperations.Has(opType) && !lhsShape.DType.IsComplex() {
		err = errors.Errorf("complex BinaryOp %s must have a complex (Complex64, Complex128) data type as input, got %s", opType, lhsShape)
		return
	}

	return binaryOpImpl(opType, lhsShape, rhsShape)
}

func binaryOpImpl(opType backends.OpType, lhsShape, rhsShape shapes.Shape) (output shapes.Shape, err error) {
	// Trivial cases: if one of the sides is a scalar, return the other side shape.
	if lhsShape.IsScalar() {
		return rhsShape, nil
	}
	if rhsShape.IsScalar() {
		return lhsShape, nil
	}

	// Other cases, either the dimensions match or one of them is 1.
	if lhsShape.Rank() != rhsShape.Rank() {
		err = errors.Errorf("if operands are not scalars, their rank must match for BinaryOp (%s), got shapes %s and %s",
			opType, lhsShape, rhsShape)
	}
	output = lhsShape.Clone()
	for axis := range output.Rank() {
		lhsDim := lhsShape.Dimensions[axis]
		rhsDim := rhsShape.Dimensions[axis]
		if lhsDim != 1 && rhsDim != 1 && lhsDim != rhsDim {
			err = errors.Errorf("dimension of axis #%d doesn't match and cannot be broadcast for BinaryOp (%s), got shapes %s and %s",
				axis, opType, lhsShape, rhsShape)
			return
		}
		output.Dimensions[axis] = max(lhsDim, rhsDim)
	}
	return
}

// ComparisonOp returns the broadcast shape with dtype set to Bool, for comparison operations (Equal, LessThan, GreaterOrEqual, etc.)
func ComparisonOp(opType backends.OpType, lhsShape, rhsShape shapes.Shape) (output shapes.Shape, err error) {
	if !ComparisonOperations.Has(opType) {
		err = errors.Errorf("operation %s is not in the ComparisonOperations set, cannot process it with ComparisonOp", opType)
		return
	}
	if lhsShape.DType == dtypes.InvalidDType || rhsShape.DType == dtypes.InvalidDType {
		err = errors.Errorf("invalid shape for %s or %s for ComparisonOp %s", lhsShape, rhsShape, opType)
		return
	}
	if lhsShape.DType != rhsShape.DType {
		err = errors.Errorf("data types (DType) for ComparisonOp %s must match, got %s and %s", opType, lhsShape, rhsShape)
		return
	}
	if !NumberOperations.Has(opType) {
		err = errors.Errorf("operation %s is not in the NumberOperations set, cannot process it with ComparisonOp", opType)
		return
	}

	output, err = binaryOpImpl(opType, lhsShape, rhsShape)
	output.DType = dtypes.Bool
	return
}

// UnaryOp checks the validity of the data type for StandardUnaryOperations and returns either an error or
// the output shape, which is the same as the operand.
func UnaryOp(opType backends.OpType, operand shapes.Shape) (output shapes.Shape, err error) {
	if !StandardUnaryOperations.Has(opType) {
		err = errors.Errorf("operation %s is not in the StandardUnaryOperations set, cannot process it with UnaryOp", opType)
		return
	}
	if operand.DType == dtypes.InvalidDType {
		err = errors.Errorf("invalid shape %s for UnaryOp %s", operand, opType)
		return
	}
	if BooleanOperations.Has(opType) && operand.DType != dtypes.Bool {
		err = errors.Errorf("logical UnaryOp %s must have boolean (dtype.Bool) data types as input, got %s", opType, operand)
		return
	}
	if BitwiseOperations.Has(opType) && !operand.DType.IsInt() {
		err = errors.Errorf("bitwise UnaryOp %s must have an integer (Int8, UInt8, Int32, ...) data type as input, got %s", opType, operand)
		return
	}
	if SignedNumberOperations.Has(opType) && (operand.DType.IsUnsigned() ||
		!(operand.DType.IsInt() || operand.DType.IsFloat() || operand.DType.IsComplex())) {
		err = errors.Errorf("signed UnaryOp %s must have a signed data type as input, got %s", opType, operand)
		return
	}
	if NumberOperations.Has(opType) && !(operand.DType.IsInt() || operand.DType.IsFloat() || operand.DType.IsComplex()) {
		err = errors.Errorf("numeric UnaryOp %s must have a number (Int32, Float32, Complex64, ...) data type as input, got %s", opType, operand)
		return
	}
	if FloatOperations.Has(opType) && !operand.DType.IsFloat() {
		err = errors.Errorf("float UnaryOp %s must have a float (Float32, Float64, ...) data type as input, got %s", opType, operand)
		return
	}
	if FloatOrComplexOperations.Has(opType) && !(operand.DType.IsFloat() || operand.DType.IsComplex()) {
		err = errors.Errorf("float/complex UnaryOp %s must have a float or complex (Float32, Complex64, ...) data type as input, got %s", opType, operand)
		return
	}
	if ComplexOperations.Has(opType) && !operand.DType.IsComplex() {
		err = errors.Errorf("complex UnaryOp %s must have a complex (Complex64, Complex128) data type as input, got %s", opType, operand)
		return
	}
	output = operand
	return
}

// WhereOp returns the shape resulting from the Where operation.
//
// Shape constraints for the operation:
//
//  1. The onTrue and onFalse must have the exact same shape, or one can be a scalar.
//  2. The condition must either be a scalar or match the shape of onTrue or onFalse, except for the DType that
//     must be Bool.
func WhereOp(condition, onTrue, onFalse shapes.Shape) (output shapes.Shape, err error) {
	if condition.DType != dtypes.Bool {
		err = errors.Errorf("condition for Where() must be a boolean, got %s instead", condition)
		return
	}
	if !onTrue.IsScalar() && !onFalse.IsScalar() && !onTrue.Equal(onFalse) {
		err = errors.Errorf("onTrue (%s) and onFalse (%s) values for Where() must either be scalar or match each other's shape",
			onTrue, onFalse)
		return
	}

	output = onTrue
	if output.IsScalar() {
		output = onFalse
		if output.IsScalar() && !condition.IsScalar() {
			output = condition.Clone()
			output.DType = onTrue.DType
		}
	}

	if !condition.IsScalar() && slices.Compare(condition.Dimensions, output.Dimensions) != 0 {
		err = errors.Errorf("condition for Where() must either be a scalar or match the output shape (not the DType), instead got shapes condition=%s, onTrue=%s and onFalse=%s",
			condition, onTrue, onFalse)
		return
	}

	return
}

// ReshapeOp to the given dimensions: trivial output shape, but this function also checks
// that the sizes are the same.
//
// Notice the backends.Reshape doesn't support auto-scaling dimensions (set to -1), as graph.Reshape does.
func ReshapeOp(operand shapes.Shape, dims []int) (output shapes.Shape, err error) {
	output = shapes.Make(operand.DType, dims...)
	if operand.Size() != output.Size() {
		err = errors.Errorf("Reshape() cannot reshape %s to dimensions %v, their size don't match",
			operand, dims)
		return shapes.Invalid(), err
	}
	return
}

// TransposeOp all axes of the operand.
// There must be one value in permutations for each axis in the operand.
// The output will have: output.Shape.Dimension[ii] = operand.Shape.Dimension[permutations[i]].
func TransposeOp(operand shapes.Shape, permutations []int) (output shapes.Shape, err error) {
	rank := operand.Rank()
	if len(permutations) != rank {
		err = errors.Errorf("Transpose() requires all axes permutations to be defined, operand has shape %s, but %d permutations were given",
			operand, len(permutations))
		return
	}
	if rank == 0 {
		return operand, nil
	}

	// Check permutation axes are within range and unique.
	axesSet := slices.Clone(permutations)
	slices.Sort(axesSet)
	for ii, srcAxis := range axesSet {
		if srcAxis < 0 || srcAxis >= rank {
			err = errors.Errorf("invalid permutation axis %d given to Transpose(%s), it must be within the range of its rank",
				srcAxis, operand)
			return
		}
		if ii > 0 && srcAxis == axesSet[ii-1] {
			err = errors.Errorf("invalid permutations given to Transpose(%s, %v), there cannot be any repeated axis, each must appear exactly once",
				operand, permutations)
			return
		}
	}

	output = operand.Clone()
	for axis := range output.Dimensions {
		srcAxis := permutations[axis]
		output.Dimensions[axis] = operand.Dimensions[srcAxis]
	}
	return
}

// BroadcastOp adds the prefixDims to the start of the shape.
func BroadcastOp(operand shapes.Shape, prefixDims []int) (output shapes.Shape, err error) {
	if operand.DType == dtypes.InvalidDType {
		err = errors.Errorf("invalid shape %s for BroadcastOp", operand)
		return
	}
	if len(prefixDims) == 0 {
		return operand, nil
	}
	for _, dim := range prefixDims {
		if dim <= 0 {
			err = errors.Errorf("Invalid prefix dimensions %v for BroadcastOp, they must be positive", prefixDims)
			return
		}
	}
	output = shapes.Make(operand.DType)
	output.Dimensions = make([]int, len(prefixDims)+operand.Rank())
	copy(output.Dimensions, prefixDims)
	copy(output.Dimensions[len(prefixDims):], operand.Dimensions)
	return
}

// BroadcastInDimOp verifies that the arguments are valid. The output shape is already known, so nothing is returned.
func BroadcastInDimOp(operand, outputShape shapes.Shape, broadcastAxes []int) error {
	if len(broadcastAxes) != operand.Rank() {
		return errors.Errorf("there must be exactly one broadcastAxes (%v) per axis in the operand (%s)",
			broadcastAxes, operand)
	}

	// Verify that the values of expandedAxis and create a map of the expanded axis.
	preservedSet := types.MakeSet[int](len(broadcastAxes))
	for axisInOperand, axisInOutput := range broadcastAxes {
		if axisInOutput < 0 || axisInOutput >= outputShape.Rank() {
			return errors.Errorf("broadcastAxes (%v) defines a value out-of-range (%d-th value -> %d), they must be between 0 and outputShape.Rank()-1=%d",
				broadcastAxes, axisInOperand, axisInOutput, outputShape.Rank()-1)
		}
		if preservedSet.Has(axisInOutput) {
			return errors.Errorf("broadcastAxes (%v) repeats axis %d (broadcastAxes[%d]), they must be all unique and between 0 and outputShape.Rank()-1=%d",
				broadcastAxes, axisInOutput, axisInOperand, outputShape.Rank()-1)
		}
		preservedSet.Insert(axisInOutput)
		if operand.Dimensions[axisInOperand] != 1 && operand.Dimensions[axisInOperand] != outputShape.Dimensions[axisInOutput] {
			return errors.Errorf("the values of outputShape (%v) that are being broadcast (listed in broadcastAxes) "+
				"must match the corresponding value in the operand shape (%s) or be 1 (if broadcasting), "+
				"but the value of outputShape.Dimensions[%d]=%d does not match the value in operand.Shape().Dimensions[%d]=%d",
				outputShape, operand, axisInOutput, outputShape.Dimensions[axisInOutput], axisInOperand, operand.Dimensions[axisInOperand])
		}
	}
	return nil
}

// ReduceOp works for the ReduceMax, ReduceMin, ReduceSum and ReduceProduct ops.
func ReduceOp(operand shapes.Shape, axes []int) (output shapes.Shape, err error) {
	if len(axes) == 0 {
		return operand, nil
	}
	output = shapes.Make(operand.DType)
	outputRank := operand.Rank() - len(axes)
	if outputRank > 0 {
		// Copy over dimensions that will stay.
		output.Dimensions = make([]int, 0, outputRank)
		for _, axis := range axes {
			if axis < 0 || axis >= operand.Rank() {
				return shapes.Invalid(), errors.Errorf("Reduce operation require each axis to be 0 <= axis < rank, but got invalid axis %d for shape %s", axis, operand)
			}
		}
		axesSet := types.SetWith(axes...)
		for axis, dim := range operand.Dimensions {
			if !axesSet.Has(axis) {
				output.Dimensions = append(output.Dimensions, dim)
			}
		}
	}
	return
}

// GatherOp returns the output shape of a Gather operation.
func GatherOp(operand, startIndices shapes.Shape, indexVectorAxis int, offsetOutputAxes, collapsedSliceAxes,
	startIndexMap, sliceSizes []int, indicesAreSorted bool) (output shapes.Shape, err error) {
	//fmt.Printf("GatherOp parameters:\n"+
	//	"  operand: %v\n"+
	//	"  startIndices: %v\n"+
	//	"  indexVectorAxis: %d\n"+
	//	"  offsetOutputAxes: %v\n"+
	//	"  collapsedSliceAxes: %v\n"+
	//	"  startIndexMap: %v\n"+
	//	"  sliceSizes: %v\n"+
	//	"  indicesAreSorted: %v\n",
	//	operand, startIndices, indexVectorAxis, offsetOutputAxes, collapsedSliceAxes,
	//	startIndexMap, sliceSizes, indicesAreSorted)
	_ = indicesAreSorted // Not used for shape inference.

	if operand.IsScalar() {
		return output, errors.Errorf("Gather() requires a non-scalar operand, got %s", operand)
	}

	setCollapsedAxes := types.MakeSet[int]()
	for _, collapsedSliceAxis := range collapsedSliceAxes {
		if collapsedSliceAxis < 0 || collapsedSliceAxis >= operand.Rank() {
			return output, errors.Errorf("collapsed slice axis %d is out of range for operand %s", collapsedSliceAxis, operand)
		}
		if setCollapsedAxes.Has(collapsedSliceAxis) {
			return output, errors.Errorf("collapsed slice axis %d is defined more than once for operand %s", collapsedSliceAxis, operand)
		}
		setCollapsedAxes.Insert(collapsedSliceAxis)
	}

	// Check slice sizes.
	if len(sliceSizes) != operand.Rank() {
		return output, errors.Errorf("sliceSizes must have one value per operand axes, so it length (%d) must match operand rank (%d)", len(sliceSizes), operand.Rank())
	}
	for axis, sliceSize := range sliceSizes {
		if sliceSize < 0 {
			return output, errors.Errorf("sliceSize %d for axis %d is negative, it must be non-negative", sliceSize, axis)
		}
		if operand.Dimensions[axis] < sliceSize {
			return output, errors.Errorf("sliceSize %d for axis %d is larger than the corresponding operand dimension %d", sliceSize, axis, operand.Dimensions[axis])
		}
	}
	for collapseAxis := range setCollapsedAxes {
		if sliceSizes[collapseAxis] != 1 {
			return output, errors.Errorf("collapsed slice axis %d must have sliceSize 1, but got %d", collapseAxis, sliceSizes[collapseAxis])
		}
	}
	if operand.Rank() != len(collapsedSliceAxes)+len(offsetOutputAxes) {
		return output, errors.Errorf("the number of collapsedSliceAxes (%d) + the number of offsetOutputAxes (%d) must be equal to the number of axes in the operand (operand.Rank()=%d)",
			len(collapsedSliceAxes), len(offsetOutputAxes), operand.Rank())
	}

	// Check indexVectorAxis: it is ok if it is equal to startIndices.rank, in which case we assume implicit extra axes of dimension 1.
	if indexVectorAxis < 0 || indexVectorAxis > operand.Rank() {
		return output, errors.Errorf("indexVectorAxis=%d is out of range for operand %s", indexVectorAxis, operand)
	}

	// Check startIndexMap is set for the dimensions of indexVectorAxis in startIndices.
	if len(startIndexMap) != startIndices.Dimensions[indexVectorAxis] {
		return output, errors.Errorf("startIndexMap must have one value per dimension of indexVectorAxis, so it length (%d) must match startIndices.Dimensions[%d] (%d)",
			len(startIndexMap), indexVectorAxis, startIndices.Dimensions[indexVectorAxis])
	}
	for idx, operandAxis := range startIndexMap {
		if operandAxis < 0 || operandAxis >= operand.Rank() {
			return output, errors.Errorf("startIndexMap[%d]=%d is out of range for operand %s", idx, operandAxis, operand)
		}
	}

	// The number of batch axes is usually the number of startIndices - 1, except if indexVectorAxis==rank,
	// in which case we assume an extra one in the end.
	batchRank := startIndices.Rank() - 1
	if indexVectorAxis == startIndices.Rank() {
		batchRank++
	}

	// Build output shape: the order is defined as:
	//
	// - Axes in offsetOutputAxes are preset as offset, and their dimensions are taken sequentially from non-collapsed operand axes.
	// - Remaining axes are filled in order from the batch axes, taken from startIndices.
	output = shapes.Make(operand.DType)
	output.Dimensions = make([]int, batchRank+len(offsetOutputAxes))

	setOffsetOutputAxes := types.MakeSet[int]()
	for _, offsetOutputAxis := range offsetOutputAxes {
		if offsetOutputAxis < 0 || offsetOutputAxis >= output.Rank() {
			return shapes.Invalid(), errors.Errorf("offset output axis %d is out of range for output of rank %d", offsetOutputAxis, output.Rank())
		}
		if setOffsetOutputAxes.Has(offsetOutputAxis) {
			return shapes.Invalid(), errors.Errorf("offset output axis %d is defined more than once: offsetOutputAxes=%v", offsetOutputAxis, offsetOutputAxes)
		}
		setOffsetOutputAxes.Insert(offsetOutputAxis)
	}
	offsetDims := make([]int, 0, len(offsetOutputAxes))
	for axis, sliceSize := range sliceSizes {
		if setCollapsedAxes.Has(axis) {
			// This is a collapsed axis and not used as an offset.
			continue
		}
		offsetDims = append(offsetDims, sliceSize)
	}
	offsetDimsIdx := 0
	batchDimsIdx := 0
	for axis := range output.Dimensions {
		if setOffsetOutputAxes.Has(axis) {
			// Take an offset dimension from sliceSizes:
			output.Dimensions[axis] = offsetDims[offsetDimsIdx]
			offsetDimsIdx++
		} else {
			// Take a batch dimension:
			if batchDimsIdx == indexVectorAxis {
				batchDimsIdx++
			}
			output.Dimensions[axis] = startIndices.Dimensions[batchDimsIdx]
			batchDimsIdx++
		}
	}
	return output, nil
}

// ConcatenateOp calculates the output shape of a Concatenate operation.
// It takes a slice of input shapes and the dimension along which to concatenate.
func ConcatenateOp(inputs []shapes.Shape, axis int) (output shapes.Shape, err error) {
	if len(inputs) == 0 {
		return shapes.Invalid(), errors.Errorf("ConcatenateOp requires at least one input shape")
	}

	// Initialize output dimensions with the first shape.
	firstShape := inputs[0]
	dtype := firstShape.DType
	rank := firstShape.Rank()
	output = firstShape.Clone()
	if dtype == dtypes.InvalidDType {
		return shapes.Invalid(), errors.Errorf("invalid shape %s for first input of ConcatenateOp", firstShape)
	}
	if len(inputs) == 1 {
		return firstShape, nil
	}

	if axis < 0 || axis >= rank {
		return shapes.Invalid(), errors.Errorf("invalid concatenation axis %d for shapes with rank %d", axis, rank)
	}

	// Validate further inputs and accumulate the concatenation axis size.
	for i := 1; i < len(inputs); i++ {
		currentShape := inputs[i]
		if currentShape.DType == dtypes.InvalidDType {
			return shapes.Invalid(), errors.Errorf("invalid shape %s for input #%d of ConcatenateOp", currentShape, i)
		}
		if currentShape.DType != dtype {
			return shapes.Invalid(), errors.Errorf("mismatched DTypes for ConcatenateOp: input #0 has %s, input #%d has %s",
				dtype, i, currentShape.DType)
		}
		if currentShape.Rank() != rank {
			return shapes.Invalid(), errors.Errorf("mismatched ranks for ConcatenateOp: input #0 has rank %d, input #%d has rank %d",
				rank, i, currentShape.Rank())
		}

		for d := 0; d < rank; d++ {
			if d == axis {
				output.Dimensions[d] += currentShape.Dimensions[d]
			} else {
				if currentShape.Dimensions[d] != output.Dimensions[d] {
					return shapes.Invalid(), errors.Errorf("mismatched dimensions for ConcatenateOp at axis %d (non-concatenation axis): input #0 has %d, input #%d has %d",
						d, output.Dimensions[d], i, currentShape.Dimensions[d])
				}
			}
		}
	}
	return output, nil
}

// ScatterOp checks that the parameters are consistent. The output shape returned is the unchanged operand -- the scattered
// updates are applied to the operand, but its shape is unchanged.
//
// The Scatter operations indicesAreSorted and uniqueIndices don't play a role in this.
func ScatterOp(operand, indices, updates shapes.Shape, indexVectorAxis int, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int) (output shapes.Shape, err error) {
	if operand.DType == dtypes.InvalidDType || indices.DType == dtypes.InvalidDType || updates.DType == dtypes.InvalidDType {
		return shapes.Invalid(), errors.Errorf("invalid shape for operand (%s), indices (%s) or updates (%s) for ScatterOp", operand, indices, updates)
	}
	if operand.DType != updates.DType {
		return shapes.Invalid(), errors.Errorf("data types (DType) for ScatterOp operand (%s) and updates (%s) must match", operand, updates)
	}
	if !indices.DType.IsInt() {
		return shapes.Invalid(), errors.Errorf("indices DType (%s) must be an integer type", indices)
	}

	// Check indexVectorAxis and get scatter indices dimensions.
	if indexVectorAxis < 0 || indexVectorAxis > indices.Rank() {
		return shapes.Invalid(), errors.Errorf("indexVectorAxis=%d must be in range [0, indices.Rank()=%d]", indexVectorAxis, indices.Rank())
	}

	// Validate scatter axes mapping.
	numIndexedAxes := 1
	if indexVectorAxis < indices.Rank() {
		numIndexedAxes = indices.Dimensions[indexVectorAxis]
	}
	if len(scatterAxesToOperandAxes) != numIndexedAxes {
		return shapes.Invalid(), errors.Errorf("scatterAxesToOperandAxes length (%d) must match the size of indices's indexVectorAxis dimension (%d)",
			len(scatterAxesToOperandAxes), indices.Dimensions[indexVectorAxis])
	}
	for i, axis := range scatterAxesToOperandAxes {
		if axis < 0 || axis >= operand.Rank() {
			return shapes.Invalid(), errors.Errorf("scatterAxesToOperandAxes[%d]=%d must be in range [0, operand.Rank()=%d)", i, axis, operand.Rank())
		}
	}
	for i, axis := range updateWindowAxes {
		if axis < 0 || axis >= updates.Rank() {
			return shapes.Invalid(), errors.Errorf("updateWindowAxes[%d]=%d must be in range [0, updates.Rank()=%d)", i, axis, updates.Rank()-1)
		}
	}

	// Check that the batch axes of indices match the number of axes in the updates.
	numBatchAxes := indices.Rank() - 1
	if indexVectorAxis == indices.Rank() {
		numBatchAxes++
	}
	if len(updateWindowAxes)+numBatchAxes != updates.Rank() {
		return shapes.Invalid(), errors.Errorf("numBatchAxes (%d) + len(updateWindowAxes) (%d) must match updates.Rank() (%d), so it "+
			"can fully addressed -- where numBatchAxes=indices.Rank() - 1, or if indexVector == indices.Rank(), numBatchAxes=indices.Rank()",
			numBatchAxes, len(updateWindowAxes), updates.Rank())
	}

	// Validate update window dimensions.
	if len(updateWindowAxes)+len(insertedWindowAxes) != operand.Rank() {
		return shapes.Invalid(), errors.Errorf("operand.Rank() (%d) must match len(updateWindowAxes)(%d)+len(insertedWindowAxes)(%d), so operand indices can be fully defined",
			operand.Rank(), len(updateWindowAxes), len(insertedWindowAxes))
	}
	for i, axis := range insertedWindowAxes {
		if axis < 0 || axis >= operand.Rank() {
			return shapes.Invalid(), errors.Errorf("insertedWindowAxes[%d]=%d must be in range [0, operand.Rank()=%d)", i, axis, operand.Rank())
		}
	}

	// Validate that update dimensions fit into output dimensions.
	insertedWindowAxesSet := types.SetWith(insertedWindowAxes...)
	operandUpdatedWindowAxes := make([]int, 0, operand.Rank()-len(insertedWindowAxes))
	for axis := range operand.Rank() {
		if !insertedWindowAxesSet.Has(axis) {
			operandUpdatedWindowAxes = append(operandUpdatedWindowAxes, axis)
		}
	}
	for ii, updatesAxis := range updateWindowAxes {
		operandAxis := operandUpdatedWindowAxes[ii]
		if updates.Dimensions[updatesAxis] > operand.Dimensions[operandAxis] {
			return shapes.Invalid(), errors.Errorf("updates.Dimensions[axis=%d](%d) > operand.Dimensions[axis=%d](%d), updates won't fit into the operand",
				updatesAxis, updates.Dimensions[updatesAxis], operandAxis, operand.Dimensions[operandAxis])
		}
	}
	return operand, nil
}

// SliceOp calculates the output shape for a Slice operation.
// It checks that starts, limits, and strides have the correct length (matching operand rank),
// and that the slice parameters are valid for the operand's dimensions.
// Strides must be positive.
func SliceOp(operand shapes.Shape, starts, limits, strides []int) (output shapes.Shape, err error) {
	rank := operand.Rank()
	opName := "SliceOp"
	if operand.DType == dtypes.InvalidDType {
		return shapes.Invalid(), errors.Errorf("%s: invalid operand shape %s", opName, operand)
	}
	if len(starts) != rank {
		return shapes.Invalid(), errors.Errorf("%s: len(starts)=%d, but operand rank is %d", opName, len(starts), rank)
	}
	if len(limits) != rank {
		return shapes.Invalid(), errors.Errorf("%s: len(limits)=%d, but operand rank is %d", opName, len(limits), rank)
	}
	if len(strides) != rank {
		return shapes.Invalid(), errors.Errorf("%s: len(strides)=%d, but operand rank is %d", opName, len(strides), rank)
	}

	output = shapes.Shape{
		DType:      operand.DType,
		Dimensions: make([]int, rank),
	}

	for axis := 0; axis < rank; axis++ {
		start, limit, stride := starts[axis], limits[axis], strides[axis]
		dimSize := operand.Dimensions[axis]

		if stride <= 0 {
			return shapes.Invalid(), errors.Errorf("%s: stride must be positive, but got stride[%d]=%d for operand shape %s",
				opName, axis, stride, operand)
		}
		if start < 0 || start >= dimSize {
			return shapes.Invalid(), errors.Errorf("%s: start index %d is out of bounds for axis %d with size %d (operand shape %s)",
				opName, start, axis, dimSize, operand)
		}
		// Limit can be equal to dimSize.
		if limit < start || limit > dimSize {
			return shapes.Invalid(), errors.Errorf("%s: limit index %d is out of bounds for axis %d (start=%d, size=%d, operand shape %s)",
				opName, limit, axis, start, dimSize, operand)
		}

		// The first one is always taken, so we use the ceiling of the division.
		outputDimSize := (limit - start + (stride - 1)) / stride
		output.Dimensions[axis] = outputDimSize
	}

	return output, nil
}

// ArgMinMaxOp calculates the output shape for an ArgMinMax operation.
// It will be the shape of the operand minus the "reduce" axis.
func ArgMinMaxOp(operand shapes.Shape, axis int, outputDType dtypes.DType) (output shapes.Shape, err error) {
	if !outputDType.IsInt() {
		err = errors.Errorf("ArgMinMax outputDType must be an integer type, got %s", outputDType)
		return
	}
	if !operand.DType.IsFloat() && !operand.DType.IsInt() {
		err = errors.Errorf("ArgMinMax operand DType must be a floating point or integer type, got %s", operand)
		return
	}
	if operand.IsScalar() {
		err = errors.Errorf("ArgMinMax requires a non-scalar operand, got %s", operand)
		return
	}
	if axis < 0 || axis >= operand.Rank() {
		err = errors.Errorf("ArgMinMax axis %d is out of range for operand %s", axis, operand)
		return
	}
	newDims := slices.Clone(operand.Dimensions)
	newDims = slices.Delete(newDims, axis, axis+1)
	output = shapes.Make(outputDType, newDims...)
	return
}
