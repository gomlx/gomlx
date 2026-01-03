package graph

import (
	. "github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
)

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

// UnpackInt2 unpacks a node with dtype dtypes.Uint8 (or dtypes.Int8) to dtypes.Int2 by
// shifting the bits accordingly.
// The final shape has one extra axis at the end of size 4 (4 Int2 values in an Uint8).
//
// Each Uint8 byte is unpacked into 4 Int2 values:
//   - Bits 0-1 become the first Int2 value
//   - Bits 2-3 become the second Int2 value
//   - Bits 4-5 become the third Int2 value
//   - Bits 6-7 become the fourth Int2 value
//
// The Int2 values are sign-extended from 2-bit signed integers.
// UnpackInt2 unpacks 4 2-bit integers from each byte of the input x.
//
// The input x is converted to Int8 (if it isn't already), and the returned tensor
// has one extra dimension of size 4 at the end, containing the unpacked values.
// The sign is correctly preserved (2-bit 2's complement).
//
// The unpacked order is from the least significant bits (0-1) to the most significant (6-7).
func UnpackInt2(x *Node) *Node {
	_ = validateBuildingGraphFromInputs(x)
	if x.DType() != dtypes.Uint8 && x.DType() != dtypes.Int8 {
		Panicf("UnpackInt2: input must be Uint8 or Int8, got %s", x.DType())
	}
	// Ensure input is Int8 so that the sign bit is at the 7th bit position (MSB of a byte).
	x = Bitcast(x, dtypes.Int8)
	var unpacked [4]*Node
	for i := range unpacked {
		// Rotate the 2 bits to the left (most significant bits) so we get the sign bit in the 7th bit position.
		// and then back again to the right, preserving the sign bit.
		unpacked[i] = BitwiseShiftRightArithmeticScalar(BitwiseShiftLeftScalar(x, 6-2*i), 6)
	}
	return Stack(unpacked[:], -1)
}

// UnpackInt4 unpacks a node with dtype dtypes.Uint8 (or dtypes.Int8) to dtypes.Int4 by
// shifting the bits accordingly.
// The final shape has one extra axis at the end of size 2 (2 Int4 values in an Uint8).
//
// Each Uint8 byte is unpacked into 2 Int4 values:
//   - Bits 0-3 become the first Int4 value
//   - Bits 4-7 become the second Int4 value
//
// The Int4 values are sign-extended from 4-bit signed integers.
// UnpackInt4 unpacks 2 4-bit integers from each byte of the input x.
//
// The input x is converted to Int8 (if it isn't already), and the returned tensor
// has one extra dimension of size 2 at the end, containing the unpacked values.
// The sign is correctly preserved (4-bit 2's complement).
//
// The unpacked order is from the least significant bits (0-3) to the most significant (4-7).
func UnpackInt4(x *Node) *Node {
	_ = validateBuildingGraphFromInputs(x)
	if x.DType() != dtypes.Uint8 && x.DType() != dtypes.Int8 {
		Panicf("UnpackInt4: input must be Uint8 or Int8, got %s", x.DType())
	}
	// Ensure input is Int8 so that the sign bit is at the 7th bit position (MSB of a byte).
	x = Bitcast(x, dtypes.Int8)
	var unpacked [2]*Node
	for i := range unpacked {
		// Rotate the 4 bits to the left (most significant bits) so we get the sign bit in the 7th bit position.
		// and then back again to the right, preserving the sign bit.
		unpacked[i] = BitwiseShiftRightArithmeticScalar(BitwiseShiftLeftScalar(x, 4-4*i), 4)
	}
	return Stack(unpacked[:], -1)
}

// UnpackUint2 unpacks a node with dtype dtypes.Uint8 (or dtypes.Int8) to dtypes.Uint2 by
// shifting the bits accordingly.
// The final shape has one extra axis at the end of size 4 (4 Uint2 values in an Uint8).
//
// Each Uint8 byte is unpacked into 4 Uint2 values:
//   - Bits 0-1 become the first Uint2 value
//   - Bits 2-3 become the second Uint2 value
//   - Bits 4-5 become the third Uint2 value
//   - Bits 6-7 become the fourth Uint2 value
//
// The Uint2 values are zero-extended (unsigned).
// UnpackUint2 unpacks 4 2-bit unsigned integers from each byte of the input x.
//
// The input x is converted to Uint8 (if it isn't already), and the returned tensor
// has one extra dimension of size 4 at the end, containing the unpacked values.
//
// The unpacked order is from the least significant bits (0-1) to the most significant (6-7).
func UnpackUint2(x *Node) *Node {
	_ = validateBuildingGraphFromInputs(x)
	if x.DType() != dtypes.Uint8 && x.DType() != dtypes.Int8 {
		Panicf("UnpackUint2: input must be Uint8 or Int8, got %s", x.DType())
	}
	// Ensure input is Uint8 for unsigned operations.
	x = Bitcast(x, dtypes.Uint8)
	g := x.Graph()
	mask := Scalar(g, dtypes.Uint8, uint8(0x03))
	var unpacked [4]*Node
	for i := range unpacked {
		// Shift right by 2*i bits and mask to get 2 bits.
		unpacked[i] = BitwiseAnd(BitwiseShiftRightLogicalScalar(x, 2*i), mask)
	}
	return Stack(unpacked[:], -1)
}

// UnpackUint4 unpacks a node with dtype dtypes.Uint8 (or dtypes.Int8) to dtypes.Uint4 by
// shifting the bits accordingly.
// The final shape has one extra axis at the end of size 2 (2 Uint4 values in an Uint8).
//
// Each Uint8 byte is unpacked into 2 Uint4 values:
//   - Bits 0-3 become the first Uint4 value
//   - Bits 4-7 become the second Uint4 value
//
// The Uint4 values are zero-extended (unsigned).
// UnpackUint4 unpacks 2 4-bit unsigned integers from each byte of the input x.
//
// The input x is converted to Uint8 (if it isn't already), and the returned tensor
// has one extra dimension of size 2 at the end, containing the unpacked values.
//
// The unpacked order is from the least significant bits (0-3) to the most significant (4-7).
func UnpackUint4(x *Node) *Node {
	_ = validateBuildingGraphFromInputs(x)
	if x.DType() != dtypes.Uint8 && x.DType() != dtypes.Int8 {
		Panicf("UnpackUint4: input must be Uint8 or Int8, got %s", x.DType())
	}
	// Ensure input is Uint8 for unsigned operations.
	x = Bitcast(x, dtypes.Uint8)
	g := x.Graph()
	mask := Scalar(g, dtypes.Uint8, uint8(0x0F))
	var unpacked [2]*Node
	for i := range unpacked {
		// Shift right by 4*i bits and mask to get 4 bits.
		unpacked[i] = BitwiseAnd(BitwiseShiftRightLogicalScalar(x, 4*i), mask)
	}
	return Stack(unpacked[:], -1)
}
