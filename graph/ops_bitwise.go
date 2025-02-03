package graph

import "github.com/gomlx/gopjrt/dtypes"

// ReduceLogicalAnd returns true if all values of x evaluate to true
// across the given axes.
// No gradients are defined.
//
// If reduceAxes is empty, it will reduce over all dimensions.
func ReduceLogicalAnd(x *Node, reduceAxes ...int) *Node {
	_ = validateBuildingGraphFromInputs(x)
	x = StopGradient(x) // No gradients defined to LogicalAll.
	axes := adjustAxesToRankAndSort(x.Rank(), reduceAxes, "x")
	return backendReduceLogicalAnd(x, axes...)
}

// ReduceLogicalOr returns true if any values of x evaluate to true
// across the given axes.
// No gradients are defined.
//
// If reduceAxes is empty, it will reduce over all dimensions.
func ReduceLogicalOr(x *Node, reduceAxes ...int) *Node {
	_ = validateBuildingGraphFromInputs(x)
	x = StopGradient(x) // No gradients defined to LogicalAll.
	axes := adjustAxesToRankAndSort(x.Rank(), reduceAxes, "x")
	return backendReduceLogicalOr(x, axes...)
}

// ReduceLogicalXor returns the xor of the values across the given axes.
// No gradients are defined.
//
// If reduceAxes is empty, it will reduce over all dimensions.
func ReduceLogicalXor(x *Node, reduceAxes ...int) *Node {
	_ = validateBuildingGraphFromInputs(x)
	x = StopGradient(x) // No gradients defined to LogicalAll.
	axes := adjustAxesToRankAndSort(x.Rank(), reduceAxes, "x")
	return backendReduceLogicalXor(x, axes...)
}

// LogicalAll returns true if all values of x (converted to boolean) evaluate to true.
// It's a "ReduceLogicalAnd" equivalent.
//
// If reduceAxes is empty, it will reduce over all dimensions.
func LogicalAll(x *Node, reduceAxes ...int) *Node {
	_ = validateBuildingGraphFromInputs(x)
	x = ConvertDType(x, dtypes.Bool) // No-op if already bool.
	x = StopGradient(x)              // No gradients defined to LogicalAll.
	axes := adjustAxesToRankAndSort(x.Rank(), reduceAxes, "x")
	return backendReduceLogicalAnd(x, axes...)
}

// LogicalAny returns true if any values of x (converted to boolean) evaluate to true.
// It's a "ReduceLogicalOr" equivalent.
//
// If reduceAxes is empty, it will reduce over all dimensions.
func LogicalAny(x *Node, reduceAxes ...int) *Node {
	_ = validateBuildingGraphFromInputs(x)
	x = ConvertDType(x, dtypes.Bool) // No-op if already bool.
	x = StopGradient(x)              // No gradients defined to LogicalAny.
	axes := adjustAxesToRankAndSort(x.Rank(), reduceAxes, "x")
	return backendReduceLogicalOr(x, axes...)
}

// ReduceBitwiseAnd returns the bitwise AND of the values across the given axes.
// Only defined for integer values.
// No gradients are defined.
//
// If reduceAxes is empty, it will reduce over all dimensions.
func ReduceBitwiseAnd(x *Node, reduceAxes ...int) *Node {
	_ = validateBuildingGraphFromInputs(x)
	x = StopGradient(x) // No gradients defined to BitwiseAll.
	axes := adjustAxesToRankAndSort(x.Rank(), reduceAxes, "x")
	return backendReduceBitwiseAnd(x, axes...)
}

// ReduceBitwiseOr returns the bitwise OR of the values across the given axes.
// Only defined for integer values.
// No gradients are defined.
//
// If reduceAxes is empty, it will reduce over all dimensions.
func ReduceBitwiseOr(x *Node, reduceAxes ...int) *Node {
	_ = validateBuildingGraphFromInputs(x)
	x = StopGradient(x) // No gradients defined to BitwiseAll.
	axes := adjustAxesToRankAndSort(x.Rank(), reduceAxes, "x")
	return backendReduceBitwiseOr(x, axes...)
}

// ReduceBitwiseXor returns the bitwise XOR of the values across the given axes.
// Only defined for integer values.
// No gradients are defined.
//
// If reduceAxes is empty, it will reduce over all dimensions.
func ReduceBitwiseXor(x *Node, reduceAxes ...int) *Node {
	_ = validateBuildingGraphFromInputs(x)
	x = StopGradient(x) // No gradients defined to BitwiseAll.
	axes := adjustAxesToRankAndSort(x.Rank(), reduceAxes, "x")
	return backendReduceBitwiseXor(x, axes...)
}

// And is an alias for LogicalAnd.
func And(lhs, rhs *Node) *Node {
	return LogicalAnd(lhs, rhs)
}

// Or is an alias for LogicalOr.
func Or(lhs, rhs *Node) *Node {
	return LogicalOr(lhs, rhs)
}

// Not is an alias for LogicalNot.
func Not(x *Node) *Node {
	return LogicalNot(x)
}

// BitwiseShiftLeft n bits of integer values.
// It implicitly preserves the sign bit, if there is no overflow. So BitwiseShiftLeft(-1, 1) = -2.
func BitwiseShiftLeft(x, n *Node) *Node {
	_ = validateBuildingGraphFromInputs(x, n)
	if n.DType() != x.DType() {
		n = ConvertDType(n, x.DType())
	}
	return backendShiftLeft(x, n)
}

// BitwiseShiftLeftScalar is an alias to BitwiseShiftLeft, but takes n as a scalar.
func BitwiseShiftLeftScalar[T dtypes.NumberNotComplex](x *Node, n T) *Node {
	g := validateBuildingGraphFromInputs(x)
	nNode := Scalar(g, x.DType(), n)
	return BitwiseShiftLeft(x, nNode)
}

// BitwiseShiftRightArithmetic n bits of integer values, preserving the sign bit. So ShiftRight(-2, 1) = -1.
// See also BitwiseShiftRightLogical for a version the ignores the sign bit.
func BitwiseShiftRightArithmetic(x, n *Node) *Node {
	_ = validateBuildingGraphFromInputs(x, n)
	if n.DType() != x.DType() {
		n = ConvertDType(n, x.DType())
	}
	return backendShiftRightArithmetic(x, n)
}

// BitwiseShiftRightArithmeticScalar is an alias to BitwiseShiftRightArithmetic, but takes n as a scalar.
// It shifts n bits of integer values, preserving the sign bit. So ShiftRight(-2, 1) = -1.
func BitwiseShiftRightArithmeticScalar[T dtypes.NumberNotComplex](x *Node, n T) *Node {
	g := validateBuildingGraphFromInputs(x)
	nNode := Scalar(g, x.DType(), n)
	return BitwiseShiftRightArithmetic(x, nNode)
}

// BitwiseShiftRightLogical n bits of integer values, ignoring the sign bit.
// See also BitwiseShiftRightArithmetic for a version that preserves the sign bit.
func BitwiseShiftRightLogical(x, n *Node) *Node {
	_ = validateBuildingGraphFromInputs(x, n)
	if n.DType() != x.DType() {
		n = ConvertDType(n, x.DType())
	}
	return backendShiftRightLogical(x, n)
}

// BitwiseShiftRightLogicalScalar is an alias to BitwiseShiftRightLogical, but takes n as a scalar.
// It shifts right n bits of integer values, ignoring the sign bit.
func BitwiseShiftRightLogicalScalar[T dtypes.NumberNotComplex](x *Node, n T) *Node {
	g := validateBuildingGraphFromInputs(x)
	nNode := Scalar(g, x.DType(), n)
	return BitwiseShiftRightLogical(x, nNode)
}
