package shapeinference

import (
	"fmt"
	. "github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
	"testing"
)

// Aliases
var (
	Bool = dtypes.Bool
	I8   = dtypes.Int8
	F32  = dtypes.Float32
	U64  = dtypes.Uint64

	MS = shapes.Make
)

func TestBinaryOp(t *testing.T) {
	// Invalid data types check.
	require.Panics(t, func() { BinaryOp(OpTypeLogicalAnd, MS(I8), MS(I8)) })
	require.Panics(t, func() { BinaryOp(OpTypeMul, MS(Bool, 1), MS(Bool, 1)) })
	require.Panics(t, func() { BinaryOp(OpTypeMul, MS(Bool, 1), MS(Bool, 1)) })
	require.Panics(t, func() { BinaryOp(OpTypeBitwiseXor, MS(F32, 1), MS(F32, 1)) })

	// Invalid operation type (not binary op).
	require.Panics(t, func() { BinaryOp(OpTypeExp, MS(F32), MS(F32)) })

	// The same shape should be ok.
	intMatrixShape := MS(I8, 3, 3)
	require.True(t, intMatrixShape.Equal(BinaryOp(OpTypeBitwiseOr, intMatrixShape, intMatrixShape)))

	// Scalar with matrix.
	scalarShape := MS(F32)
	matrixShape := MS(F32, 2, 3)
	expectedShape := MS(F32, 2, 3)
	require.True(t, scalarShape.Equal(BinaryOp(OpTypeAdd, scalarShape, scalarShape)))
	require.True(t, expectedShape.Equal(BinaryOp(OpTypeAdd, scalarShape, matrixShape)))

	// Broadcasting on both sides.
	shape1 := MS(F32, 2, 1, 3)
	shape2 := MS(F32, 1, 4, 3)
	expectedBroadcastShape := MS(F32, 2, 4, 3)
	require.True(t, expectedBroadcastShape.Equal(BinaryOp(OpTypeMul, shape1, shape2)))

	// Matrix with scalar.
	require.True(t, expectedShape.Equal(BinaryOp(OpTypeAdd, matrixShape, scalarShape)))

	// Invalid broadcasting shapes.
	invalidShape1 := MS(F32, 2, 3)
	invalidShape2 := MS(F32, 3, 2)
	require.Panics(t, func() { BinaryOp(OpTypeAdd, invalidShape1, invalidShape2) })
}

func TestUnaryOp(t *testing.T) {
	// Invalid data types check.
	require.Panics(t, func() { UnaryOp(OpTypeLogicalNot, MS(F32)) })
	require.Panics(t, func() { UnaryOp(OpTypeLogicalNot, MS(I8)) })
	require.Panics(t, func() { UnaryOp(OpTypeBitwiseNot, MS(F32)) })
	require.Panics(t, func() { UnaryOp(OpTypeNeg, MS(Bool)) })

	// Invalid operation type (not unary op).
	require.Panics(t, func() { UnaryOp(OpTypeAdd, MS(F32)) })
	require.Panics(t, func() { UnaryOp(OpTypeNeg, MS(U64)) })

	// Valid operations
	boolShape := MS(Bool, 2, 3)
	require.True(t, boolShape.Equal(UnaryOp(OpTypeLogicalNot, boolShape)))

	intShape := MS(I8, 3, 3)
	require.True(t, intShape.Equal(UnaryOp(OpTypeBitwiseNot, intShape)))

	floatShape := MS(F32, 2, 3)
	require.True(t, floatShape.Equal(UnaryOp(OpTypeExp, floatShape)))
	require.True(t, floatShape.Equal(UnaryOp(OpTypeNeg, floatShape)))
}

func TestGatherOp(t *testing.T) {
	// Test 1:
	operand := MS(F32, 4, 3, 2, 2)
	startIndices := MS(I8, 3, 3, 2)
	indexVectorAxis := 1
	offsetOutputAxes := []int{0, 3}
	collapsedSliceAxes := []int{0, 2}
	startIndexMap := []int{0, 2, 3}
	sliceSizes := []int{1, 3, 1, 1}
	output := GatherOp(operand, startIndices, indexVectorAxis,
		offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes, false)
	fmt.Printf("\tTest 1: outputShape=%s\n", output)
	require.NoError(t, output.Check(F32, 3, 3, 2, 1))

	// Test 2:
	operand = MS(F32, 3, 4, 5, 6)
	startIndices = MS(U64, 7, 3, 8)
	indexVectorAxis = 1
	offsetOutputAxes = []int{1, 2}
	collapsedSliceAxes = []int{1, 2}
	startIndexMap = []int{1, 2, 3}
	sliceSizes = []int{3, 1, 1, 1}
	output = GatherOp(operand, startIndices, indexVectorAxis, offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes, true)
	fmt.Printf("\tTest 2: outputShape=%s\n", output)
	require.NoError(t, output.Check(F32, 7, 3, 1, 8))

	// Test 3:
	operand = MS(F32, 8, 16)
	startIndices = MS(U64, 8, 1)
	indexVectorAxis = 1
	offsetOutputAxes = []int{1}
	collapsedSliceAxes = []int{0}
	startIndexMap = []int{0}
	sliceSizes = []int{1, 16}
	output = GatherOp(operand, startIndices, indexVectorAxis, offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes, true)
	fmt.Printf("\tTest 3: outputShape=%s\n", output)
	require.NoError(t, output.Check(F32, 8, 16))
}

func TestConcatenateOp(t *testing.T) {
	// Test Case 1: Concatenating vectors along dimension 0
	input1_1 := MS(F32, 3)
	input1_2 := MS(F32, 4)
	expected1 := MS(F32, 7)
	output1 := ConcatenateOp([]shapes.Shape{input1_1, input1_2}, 0)
	require.True(t, expected1.Equal(output1), "Test Case 1 Failed: Expected %s, got %s", expected1, output1)

	// Test Case 2: Concatenating matrices along dimension 0
	input2_1 := MS(I8, 2, 3)
	input2_2 := MS(I8, 5, 3)
	expected2 := MS(I8, 7, 3)
	output2 := ConcatenateOp([]shapes.Shape{input2_1, input2_2}, 0)
	require.True(t, expected2.Equal(output2), "Test Case 2 Failed: Expected %s, got %s", expected2, output2)

	// Test Case 3: Concatenating matrices along dimension 1
	input3_1 := MS(I8, 2, 3)
	input3_2 := MS(I8, 2, 4)
	expected3 := MS(I8, 2, 7)
	output3 := ConcatenateOp([]shapes.Shape{input3_1, input3_2}, 1)
	require.True(t, expected3.Equal(output3), "Test Case 3 Failed: Expected %s, got %s", expected3, output3)

	// Test Case 4: Concatenating 3 tensors along dimension 1
	input4_1 := MS(F32, 2, 3, 4)
	input4_2 := MS(F32, 2, 5, 4)
	input4_3 := MS(F32, 2, 1, 4)
	expected4 := MS(F32, 2, 9, 4)
	output4 := ConcatenateOp([]shapes.Shape{input4_1, input4_2, input4_3}, 1)
	require.True(t, expected4.Equal(output4), "Test Case 4 Failed: Expected %s, got %s", expected4, output4)

	// Error Case 1: Mismatched DTypes
	require.Panics(t, func() { ConcatenateOp([]shapes.Shape{MS(F32, 2), MS(I8, 3)}, 0) }, "Error Case 1 Failed: Mismatched DTypes")

	// Error Case 2: Mismatched Ranks
	require.Panics(t, func() { ConcatenateOp([]shapes.Shape{MS(F32, 2, 3), MS(F32, 4)}, 0) }, "Error Case 2 Failed: Mismatched Ranks")

	// Error Case 3: Invalid Dimension
	require.Panics(t, func() { ConcatenateOp([]shapes.Shape{MS(F32, 2, 3), MS(F32, 2, 3)}, 2) }, "Error Case 3 Failed: Invalid Dimension (too large)")
	require.Panics(t, func() { ConcatenateOp([]shapes.Shape{MS(F32, 2, 3), MS(F32, 2, 3)}, -1) }, "Error Case 3 Failed: Invalid Dimension (negative)")

	// Error Case 4: Mismatched non-concatenation dimensions
	require.Panics(t, func() { ConcatenateOp([]shapes.Shape{MS(F32, 2, 3), MS(F32, 2, 4)}, 0) }, "Error Case 4 Failed: Mismatched non-concatenation dimension")
	require.Panics(t, func() { ConcatenateOp([]shapes.Shape{MS(F32, 2, 3, 4), MS(F32, 2, 5, 5)}, 1) }, "Error Case 4 Failed: Mismatched non-concatenation dimension")

	// Error Case 5: No inputs
	require.Panics(t, func() { ConcatenateOp([]shapes.Shape{}, 0) }, "Error Case 5 Failed: No inputs")
}
