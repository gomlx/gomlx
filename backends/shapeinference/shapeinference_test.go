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

	// Same shape should be ok.
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
	operand := MS(F32, 2, 2, 2, 2)
	startIndices := MS(F32, 3, 3, 2)
	startVectorAxis := 0
	offsetAxes := []int{0}
	collapsedSliceAxes := []int{1, 2, 3}
	startIndexMap := []int{1, 2, 3}
	sliceSizes := []int{2, 1, 1, 1}
	outputShape := GatherOp(operand, startIndices, startVectorAxis, offsetAxes, collapsedSliceAxes, startIndexMap, sliceSizes, true)
	fmt.Printf("\toutputShape=%s\n", outputShape)
	require.NoError(t, outputShape.Check(F32, 2, 3, 2))
}
