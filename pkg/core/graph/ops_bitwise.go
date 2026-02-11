// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"slices"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/support/exceptions"
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

// Unpack unpacks a node with dtype dtypes.Uint8 (or dtypes.Int8) to the specified dtype.
//
// The output shape is the same as the input shape, except the last dimension is multiplied by the
// packing ratio (2 for Int4 or Uint4 or 4 for Int2 or Uint2).
// The values in the lower bits come first (little-endian).
//
// Supported dtypes are:
//   - dtypes.Int2, dtypes.Int4: call UnpackInt2 or UnpackInt4 respectively.
//   - dtypes.Uint2, dtypes.Uint4: call UnpackUint2 or UnpackUint4 respectively.
//
// It panics if the dtype is not supported.
func Unpack(x *Node, dtype dtypes.DType) *Node {
	_ = validateBuildingGraphFromInputs(x)
	if x.DType() != dtypes.Uint8 && x.DType() != dtypes.Int8 {
		Panicf("UnpackUint4: input must be Uint8 or Int8, got %s", x.DType())
	}
	var output *Node
	switch dtype {
	case dtypes.Int2:
		output = unpackInt2(x)
	case dtypes.Int4:
		output = unpackInt4(x)
	case dtypes.Uint2:
		output = unpackUint2(x)
	case dtypes.Uint4:
		output = unpackUint4(x)
	default:
		Panicf("Unpack: unsupported dtype %s, only 2-bit and 4-bit integers are supported", dtype)
		return nil
	}
	if x.IsScalar() {
		return output
	}
	newDims := slices.Clone(x.Shape().Dimensions)
	newDims[len(newDims)-1] = -1 // auto-reshape.
	return Reshape(output, newDims...)
}

func unpackInt2(x *Node) *Node {
	x = Bitcast(x, dtypes.Int8)
	var unpacked [4]*Node
	for i := range unpacked {
		unpacked[i] = BitwiseShiftRightArithmeticScalar(BitwiseShiftLeftScalar(x, 6-2*i), 6)
	}
	return ConvertDType(Stack(unpacked[:], -1), dtypes.Int2)
}

func unpackInt4(x *Node) *Node {
	x = Bitcast(x, dtypes.Int8)
	var unpacked [2]*Node
	for i := range unpacked {
		unpacked[i] = BitwiseShiftRightArithmeticScalar(BitwiseShiftLeftScalar(x, 4-4*i), 4)
	}
	return ConvertDType(Stack(unpacked[:], -1), dtypes.Int4)
}

func unpackUint2(x *Node) *Node {
	x = Bitcast(x, dtypes.Uint8)
	g := x.Graph()
	mask := Scalar(g, dtypes.Uint8, uint8(0x03))
	var unpacked [4]*Node
	for i := range unpacked {
		unpacked[i] = BitwiseAnd(BitwiseShiftRightLogicalScalar(x, 2*i), mask)
	}
	return ConvertDType(Stack(unpacked[:], -1), dtypes.Uint2)
}

func unpackUint4(x *Node) *Node {
	x = Bitcast(x, dtypes.Uint8)
	g := x.Graph()
	mask := Scalar(g, dtypes.Uint8, uint8(0x0F))
	var unpacked [2]*Node
	for i := range unpacked {
		unpacked[i] = BitwiseAnd(BitwiseShiftRightLogicalScalar(x, 4*i), mask)
	}
	return ConvertDType(Stack(unpacked[:], -1), dtypes.Uint4)
}

// Pack packs a node with dtype dtypes.Int2, dtypes.Int4, dtypes.Uint2, or dtypes.Uint4
// into a packed Uint8 tensor.
//
// The last axis dimension of x must be divisible by the packing ratio (number of elements
// per byte, that is 2 for Int4 or Uint4 or 4 for Int2 or Uint2).
// The output will have the same rank as x, but the last dimension will be divided by the
// packing ratio.
//
// Sub-byte types don't transfer in a deterministic way in PJRT (that the author was able to find -- sometimes
// they come out packed, sometimes not), so GoMLX offers this functionality to properly pack them into bytes (Uint8)
// before transferring.
func Pack(x *Node) *Node {
	g := validateBuildingGraphFromInputs(x)
	if x.DType() != dtypes.Int2 {
		Panicf("Pack: input must be Int2, Int4, Uint2 or Uint4, got %s", x.DType())
	}
	shape := x.Shape()
	srcDtype := x.DType()
	var packingRatio, shiftCount int
	var maskValue uint8
	switch srcDtype {
	case dtypes.Int2, dtypes.Uint2:
		packingRatio = 4
		shiftCount = 2
		maskValue = 0x03
	case dtypes.Int4, dtypes.Uint4:
		packingRatio = 2
		shiftCount = 4
		maskValue = 0x0F
	default:
		Panicf("Pack: input must be Int2, Int4, Uint2 or Uint4, got %s", x.DType())
		return nil
	}

	// Reshape to [..., N, packingRatio] where N = lastDim / packingRatio
	lastDim := shape.Dim(-1)
	if lastDim%packingRatio != 0 {
		Panicf("Pack: input last dimension must be divisible by %d, got shape %s", packingRatio, shape)
	}
	newShapeDims := make([]int, shape.Rank())
	copy(newShapeDims, shape.Dimensions[:shape.Rank()-1])
	newShapeDims[shape.Rank()-1] = lastDim / packingRatio
	newShapeDims = append(newShapeDims, packingRatio)
	x = Reshape(x, newShapeDims...)

	// Int8 is enough to represent both signed and unsigned 2-bit and 4-bit integers.
	x = ConvertDType(x, dtypes.Int8)
	parts := Split(x, -1, packingRatio)
	mask := Const(g, maskValue)
	for i := range parts {
		// Convert to uint8 and mask the bits that matter.
		parts[i] = BitwiseAnd(Bitcast(parts[i], dtypes.Uint8), mask)
	}

	// Pack the bits by shifting and combining.
	// For Int4/Uint4: part[0] << 0 | part[1] << 2 | part[2] << 4 | part[3] << 6
	res := parts[0]
	for i := 1; i < packingRatio; i++ {
		shifted := BitwiseShiftLeftScalar(parts[i], uint8(i*shiftCount))
		res = BitwiseOr(res, shifted)
	}

	// Squeeze the last dimension (size 1) to get [..., N]
	return Squeeze(res, -1)
}
