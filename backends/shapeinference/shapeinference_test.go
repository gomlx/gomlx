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
	I32  = dtypes.Int32
	F32  = dtypes.Float32
	U64  = dtypes.Uint64

	MS = shapes.Make
)

// must1 panics if there is an error.
func must1[T any](value T, err error) T {
	if err != nil {
		panic(err)
	}
	return value
}

func TestBinaryOp(t *testing.T) {
	// Invalid data types check.
	var err error
	_, err = BinaryOp(OpTypeLogicalAnd, MS(I8), MS(I8))
	require.Error(t, err)
	_, err = BinaryOp(OpTypeMul, MS(Bool, 1), MS(Bool, 1))
	require.Error(t, err)
	_, err = BinaryOp(OpTypeMul, MS(Bool, 1), MS(Bool, 1))
	require.Error(t, err)
	_, err = BinaryOp(OpTypeBitwiseXor, MS(F32, 1), MS(F32, 1))
	require.Error(t, err)

	// Invalid operation type (not binary op).
	_, err = BinaryOp(OpTypeExp, MS(F32), MS(F32))
	require.Error(t, err)

	// The same shape should be ok.
	var output shapes.Shape
	intMatrixShape := MS(I8, 3, 3)
	output, err = BinaryOp(OpTypeBitwiseOr, intMatrixShape, intMatrixShape)
	require.NoError(t, err)
	require.True(t, intMatrixShape.Equal(output))

	// Scalar with matrix.
	scalarShape := MS(F32)
	matrixShape := MS(F32, 2, 3)
	expectedShape := MS(F32, 2, 3)
	output, err = BinaryOp(OpTypeAdd, scalarShape, scalarShape)
	require.NoError(t, err)
	require.True(t, scalarShape.Equal(output))
	output, err = BinaryOp(OpTypeAdd, scalarShape, matrixShape)
	require.NoError(t, err)
	require.True(t, expectedShape.Equal(output))

	// Broadcasting on both sides.
	shape1 := MS(F32, 2, 1, 3)
	shape2 := MS(F32, 1, 4, 3)
	expectedBroadcastShape := MS(F32, 2, 4, 3)
	require.True(t, expectedBroadcastShape.Equal(must1(BinaryOp(OpTypeMul, shape1, shape2))))

	// Matrix with scalar.
	require.True(t, expectedShape.Equal(must1(BinaryOp(OpTypeAdd, matrixShape, scalarShape))))

	// Invalid broadcasting shapes.
	invalidShape1 := MS(F32, 2, 3)
	invalidShape2 := MS(F32, 3, 2)
	_, err = BinaryOp(OpTypeAdd, invalidShape1, invalidShape2)
	require.Error(t, err)
}

func TestUnaryOp(t *testing.T) {
	// Invalid data types check.
	require.Panics(t, func() { must1(UnaryOp(OpTypeLogicalNot, MS(F32))) })
	require.Panics(t, func() { must1(UnaryOp(OpTypeLogicalNot, MS(I8))) })
	require.Panics(t, func() { must1(UnaryOp(OpTypeBitwiseNot, MS(F32))) })
	require.Panics(t, func() { must1(UnaryOp(OpTypeNeg, MS(Bool))) })

	// Invalid operation type (not unary op).
	require.Panics(t, func() { must1(UnaryOp(OpTypeAdd, MS(F32))) })
	require.Panics(t, func() { must1(UnaryOp(OpTypeNeg, MS(U64))) })

	// Valid operations
	boolShape := MS(Bool, 2, 3)
	require.True(t, boolShape.Equal(must1(UnaryOp(OpTypeLogicalNot, boolShape))))

	intShape := MS(I8, 3, 3)
	require.True(t, intShape.Equal(must1(UnaryOp(OpTypeBitwiseNot, intShape))))

	floatShape := MS(F32, 2, 3)
	require.True(t, floatShape.Equal(must1(UnaryOp(OpTypeExp, floatShape))))
	require.True(t, floatShape.Equal(must1(UnaryOp(OpTypeNeg, floatShape))))
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
	output, err := GatherOp(operand, startIndices, indexVectorAxis,
		offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes, false)
	require.NoError(t, err)
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
	output, err = GatherOp(operand, startIndices, indexVectorAxis, offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes, true)
	require.NoError(t, err)
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
	output, err = GatherOp(operand, startIndices, indexVectorAxis, offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes, true)
	require.NoError(t, err)
	fmt.Printf("\tTest 3: outputShape=%s\n", output)
	require.NoError(t, output.Check(F32, 8, 16))
}

func TestScatterOp(t *testing.T) {
	// --- Valid Cases ---

	// Case 1: Typical scatter (like TF ScatterNd)
	// Scatter 2 updates of shape [5] into operand [4, 5]
	// Indices shape [2, 1] indicates 2 indices, each pointing to 1 dimension (axis 0) of operand.
	operand1 := MS(F32, 4, 5)
	indices1 := MS(I8, 2, 1)  // Batch shape [2]
	updates1 := MS(F32, 2, 5) // Batch shape [2]
	indexVectorAxis1 := 1
	updateWindowAxes1 := []int{1}
	insertedWindowAxes1 := []int{0}
	scatterAxesToOperandAxes1 := []int{0} // Index coordinate vector element 0 maps to operand axis 0
	expected1 := operand1
	output1, err := ScatterOp(operand1, indices1, updates1, indexVectorAxis1, updateWindowAxes1, insertedWindowAxes1, scatterAxesToOperandAxes1)
	require.NoError(t, err)
	require.True(t, expected1.Equal(output1), "Valid Case 1 Failed: Expected %s, got %s", expected1, output1)

	// Case 2: Scattering into a higher-rank tensor
	// Scatter updates of shape [4] into operand[i, j, :], where [i, j] comes from indices.
	// Operand: [10, 9, 8] (Rank 3)
	// Indices: [2, 3, 2] (Rank 3) -> 2x3 batch, each index is a pointer to the first 2 axes of the operand
	// Updates: [2, 3, 8] (Rank 3) -> 2x3 batch, update window shape [8]
	operand2 := MS(F32, 10, 9, 8)
	indices2 := MS(I32, 2, 3, 2)             // 6 indices, each is a 2D coordinate
	updates2 := MS(F32, 2, 3, 8)             // 6 updates, window shape [8] matching operand's last dim
	indexVectorAxis2 := 2                    // Axis 2 of indices holds the coordinate vector [coord0, coord1]
	updateWindowAxes2 := []int{2}            // Axis 2 of updates corresponds to the window shape [8]
	insertedWindowAxes2 := []int{0, 1}       // Axis 0, 1 of operand are the dimensions determined by the indices[i,j,:]
	scatterAxesToOperandAxes2 := []int{0, 1} // index coord 0 -> operand axis 0, index coord 1 -> operand axis 1
	expected2 := operand2
	output2, err := ScatterOp(operand2, indices2, updates2, indexVectorAxis2, updateWindowAxes2, insertedWindowAxes2, scatterAxesToOperandAxes2)
	require.NoError(t, err)
	require.True(t, expected2.Equal(output2), "Valid Case 2 Failed: Expected %s, got %s", expected2, output2)

	// Case 3: Different indexVectorAxis
	// Same as case 2, but indices are [2, 2, 3] -> indexVectorAxis is 1 and different order of axes in the operand.
	operand3 := MS(F32, 10, 9, 8)
	indices3 := MS(I32, 2, 2, 3) // 2x3 batch, index vector size 2, indexVecAxis=1
	updates3 := MS(F32, 8, 2, 3) // Update axis [8] is "out-of-order", which should be fine.
	indexVectorAxis3 := 1        // Index vector is now axis 1
	updateWindowAxes3 := []int{0}
	insertedWindowAxes3 := []int{1, 2}
	scatterAxesToOperandAxes3 := []int{1, 2} // indices are used for different axes in the operand this time.
	expected3 := operand2                    // Still expect operand shape
	output3, err := ScatterOp(operand3, indices3, updates3, indexVectorAxis3, updateWindowAxes3, insertedWindowAxes3, scatterAxesToOperandAxes3)
	require.NoError(t, err)
	require.True(t, expected3.Equal(output3), "Valid Case 3 Failed (IndexVecAxis=1): Expected %s, got %s", expected3, output3)

	// Case 4: No insertedWindowAxes (scattering full slices)
	// Scatter updates of shape [9] into operand [10, 9]
	operand4 := MS(F32, 10, 9)
	indices4 := MS(I32, 6)                // 6 indices, coord size 1
	updates4 := MS(F32, 6, 9)             // 6 updates, window shape [] (scalar)
	indexVectorAxis4 := 1                 // == indices4.Rank() -> trigger extra virtual axes to indices4.
	updateWindowAxes4 := []int{1}         // No window axes in updates (updates are scalars matching batch dims)
	insertedWindowAxes4 := []int{0}       // No window axes in operand (index selects full slice - which is scalar here)
	scatterAxesToOperandAxes4 := []int{0} // Index coord 0 -> operand axis 0
	expected4 := operand4
	output4, err := ScatterOp(operand4, indices4, updates4, indexVectorAxis4, updateWindowAxes4, insertedWindowAxes4, scatterAxesToOperandAxes4)
	require.NoError(t, err)
	require.True(t, expected4.Equal(output4), "Valid Case 4 Failed (No Window): Expected %s, got %s", expected4, output4)

	// Case 5: rearranging the output axes:
	operand5 := MS(F32, 2, 5, 2)
	indices5 := MS(I32, 2, 2)
	updates5 := MS(F32, 5, 2)
	indexVectorAxis5 := 1
	updateWindowAxes5 := []int{0}
	insertedWindowAxes5 := []int{0, 2}
	scatterAxesToOperandAxes5 := []int{0, 2}
	output5, err := ScatterOp(operand5, indices5, updates5, indexVectorAxis5, updateWindowAxes5, insertedWindowAxes5, scatterAxesToOperandAxes5)
	require.NoError(t, err)
	require.True(t, operand5.Equal(output5), "Valid Case 5 Failed (No Window): Expected %s, got %s", operand5, output5)

	// --- Error Cases ---

	// Error Case 1: Mismatched DType (Operand vs Updates) - unchanged
	_, err = ScatterOp(MS(F32, 4, 5), MS(I8, 2, 1), MS(I8, 2, 5), 1, []int{1}, []int{1}, []int{0})
	require.Error(t, err, "Error Case 1 Failed: Mismatched operand/updates DType")

	// Error Case 2: Invalid DType for indices - unchanged
	_, err = ScatterOp(MS(F32, 4, 5), MS(F32, 2, 1), MS(F32, 2, 5), 1, []int{1}, []int{1}, []int{0})
	require.Error(t, err, "Error Case 2 Failed: Invalid indices DType")

	// Error Case 3: indexVectorAxis out of bounds
	_, err = ScatterOp(operand1, indices1, updates1, 2, updateWindowAxes1, insertedWindowAxes1, scatterAxesToOperandAxes1) // indices1 rank 2, axis 2 invalid
	require.Error(t, err, "Error Case 3 Failed: indexVectorAxis out of bounds")

	_, err = ScatterOp(operand1, indices1, updates1, -1, updateWindowAxes1, insertedWindowAxes1, scatterAxesToOperandAxes1) // Negative axis
	require.Error(t, err, "Error Case 3 Failed: negative indexVectorAxis")

	// Error Case 4: len(updateWindowAxes) != len(insertedWindowAxes)
	_, err = ScatterOp(operand1, indices1, updates1, indexVectorAxis1, []int{1}, []int{1, 0}, scatterAxesToOperandAxes1) // inserted has len 2, update has len 1
	require.Error(t, err, "Error Case 4 Failed: len(updateWindowAxes) != len(insertedWindowAxes)")

	// Error Case 5: len(scatterAxesToOperandAxes) != size of index vector dimension
	_, err = ScatterOp(operand1, indices1, updates1, indexVectorAxis1, updateWindowAxes1, insertedWindowAxes1, []int{0, 1}) // scatterAxes has len 2, expected 1
	require.Error(t, err, "Error Case 5 Failed: len(scatterAxesToOperandAxes) mismatch")
	_, err = ScatterOp(operand2, indices2, updates2, indexVectorAxis2, updateWindowAxes2, insertedWindowAxes2, []int{0}) // scatterAxes has len 1, expected 2
	require.Error(t, err, "Error Case 5 Failed: len(scatterAxesToOperandAxes) mismatch")

	// Error Case 6: Invalid axis index in updateWindowAxes
	_, err = ScatterOp(operand1, indices1, updates1, indexVectorAxis1, []int{2}, insertedWindowAxes1, scatterAxesToOperandAxes1) // axis 2 invalid for rank 2 updates
	require.Error(t, err, "Error Case 6 Failed: Invalid axis in updateWindowAxes")

	// Error Case 7: Invalid axis index in insertedWindowAxes
	_, err = ScatterOp(operand1, indices1, updates1, indexVectorAxis1, updateWindowAxes1, []int{2}, scatterAxesToOperandAxes1) // axis 2 invalid for rank 2 operand
	require.Error(t, err, "Error Case 7 Failed: Invalid axis in insertedWindowAxes")

	// Error Case 8: Invalid axis index in scatterAxesToOperandAxes
	_, err = ScatterOp(operand1, indices1, updates1, indexVectorAxis1, updateWindowAxes1, insertedWindowAxes1, []int{2}) // axis 2 invalid for rank 2 operand
	require.Error(t, err, "Error Case 8 Failed: Invalid axis in scatterAxesToOperandAxes")

	// Error Case 9: Update dimension is larger than the corresponding dimension in the operand:
	_, err = ScatterOp(operand5, indices5, updates5, indexVectorAxis5, updateWindowAxes5, []int{0, 1}, scatterAxesToOperandAxes5) // axis 2 invalid for rank 2 operand
	require.Error(t, err, "Error Case 9 Failed: Update dimension is larger than the corresponding dimension in the operand")
}

func TestSliceOp(t *testing.T) {
	opName := "SliceOp"

	// --- Valid Cases ---
	// Case 1: Simple 1D slice
	operand1 := MS(F32, 10)
	starts1 := []int{2}
	limits1 := []int{8}
	strides1 := []int{1}
	expected1 := MS(F32, 6)
	output1, err := SliceOp(operand1, starts1, limits1, strides1)
	require.NoError(t, err)
	require.True(t, expected1.Equal(output1), "%s Valid Case 1 Failed: Expected %s, got %s", opName, expected1, output1)

	// Case 2: 2D slice with stride 1
	operand2 := MS(I32, 5, 6)
	starts2 := []int{1, 2}
	limits2 := []int{4, 5}
	strides2 := []int{1, 1}
	expected2 := MS(I32, 3, 3)
	output2, err := SliceOp(operand2, starts2, limits2, strides2)
	require.NoError(t, err)
	require.True(t, expected2.Equal(output2), "%s Valid Case 2 Failed: Expected %s, got %s", opName, expected2, output2)

	// Case 3: 3D slice with different strides
	operand3 := MS(Bool, 10, 8, 6)
	starts3 := []int{1, 0, 1}
	limits3 := []int{10, 8, 6} // End index exclusive
	strides3 := []int{2, 3, 1}
	// Dim 0: (10-1)/2 = 4.5 -> 5 elements (indices 1, 3, 5, 7, 9)
	// Dim 1: (8-0)/3 = 2.66 -> 3 elements (indices 0, 3, 6)
	// Dim 2: (6-1)/1 = 5 -> 5 elements (indices 1, 2, 3, 4, 5)
	expected3 := MS(Bool, 5, 3, 5)
	output3, err := SliceOp(operand3, starts3, limits3, strides3)
	require.NoError(t, err)
	require.True(t, expected3.Equal(output3), "%s Valid Case 3 Failed: Expected %s, got %s", opName, expected3, output3)

	// Case 4: Slice resulting in size 1 dimension
	operand4 := MS(F32, 10)
	starts4 := []int{5}
	limits4 := []int{6}
	strides4 := []int{1}
	expected4 := MS(F32, 1)
	output4, err := SliceOp(operand4, starts4, limits4, strides4)
	require.NoError(t, err)
	require.True(t, expected4.Equal(output4), "%s Valid Case 4 Failed: Expected %s, got %s", opName, expected4, output4)

	// Case 5: Slice taking full dimension with stride > 1
	operand5 := MS(I8, 7)
	starts5 := []int{0}
	limits5 := []int{7}
	strides5 := []int{2}
	// Dim 0: (7-0)/2 = 3.5 -> 4 elements (indices 0, 2, 4, 6)
	expected5 := MS(I8, 4)
	output5, err := SliceOp(operand5, starts5, limits5, strides5)
	require.NoError(t, err)
	require.True(t, expected5.Equal(output5), "%s Valid Case 5 Failed: Expected %s, got %s", opName, expected5, output5)

	// --- Error Cases ---
	operand := MS(F32, 10, 5) // Rank 2
	validStarts := []int{1, 1}
	validLimits := []int{8, 4}
	validStrides := []int{1, 1}

	// Error 1: Invalid operand DType
	_, err = SliceOp(shapes.Shape{DType: dtypes.InvalidDType, Dimensions: []int{10}}, []int{0}, []int{5}, []int{1})
	require.Error(t, err, "%s Error Case 1 Failed: Invalid operand DType", opName)

	// Error 2: Incorrect length for starts
	_, err = SliceOp(operand, []int{1}, validLimits, validStrides)
	require.Error(t, err, "%s Error Case 2 Failed: len(starts) != rank", opName)

	// Error 3: Incorrect length for limits
	_, err = SliceOp(operand, validStarts, []int{8}, validStrides)
	require.Error(t, err, "%s Error Case 3 Failed: len(limits) != rank", opName)

	// Error 4: Incorrect length for strides
	_, err = SliceOp(operand, validStarts, validLimits, []int{1})
	require.Error(t, err, "%s Error Case 4 Failed: len(strides) != rank", opName)

	// Error 5: Zero stride
	_, err = SliceOp(operand, validStarts, validLimits, []int{1, 0})
	require.Error(t, err, "%s Error Case 5 Failed: Zero stride", opName)

	// Error 6: Negative stride
	_, err = SliceOp(operand, validStarts, validLimits, []int{-1, 1})
	require.Error(t, err, "%s Error Case 6 Failed: Negative stride", opName)

	// Error 7: Start index < 0
	_, err = SliceOp(operand, []int{-1, 1}, validLimits, validStrides)
	require.Error(t, err, "%s Error Case 7 Failed: Start < 0", opName)

	// Error 8: Start index >= dimSize
	_, err = SliceOp(operand, []int{10, 1}, validLimits, validStrides)
	require.Error(t, err, "%s Error Case 8 Failed: Start >= dimSize", opName)

	// Error 9: Limit index < start index
	_, err = SliceOp(operand, validStarts, []int{0, 4}, validStrides) // limit[0]=0 < start[0]=1
	require.Error(t, err, "%s Error Case 9 Failed: Limit < Start", opName)

	// Error 10: Limit index > dimSize
	_, err = SliceOp(operand, validStarts, []int{8, 6}, validStrides) // limit[1]=6 > dimSize[1]=5
	require.Error(t, err, "%s Error Case 10 Failed: Limit > dimSize", opName)
}

func TestArgMinMaxOp(t *testing.T) {
	// --- Valid Cases ---

	// Case 1: 1D tensor
	operand1 := MS(F32, 10)
	expected1 := MS(I32)
	output1 := must1(ArgMinMaxOp(operand1, 0, I32))
	require.True(t, expected1.Equal(output1), "Valid Case 1 Failed: Expected %s, got %s", expected1, output1)

	// Case 2: 2D tensor, single axis
	operand2 := MS(F32, 5, 6)
	expected2 := MS(I8, 5)
	output2 := must1(ArgMinMaxOp(operand2, 1, expected2.DType))
	require.True(t, expected2.Equal(output2), "Valid Case 2 Failed: Expected %s, got %s", expected2, output2)

	// Case 3: 3D tensor, multiple axes
	operand3 := MS(F32, 4, 5, 6)
	expected3 := MS(U64, 5, 6)
	output3 := must1(ArgMinMaxOp(operand3, 0, expected3.DType))
	require.True(t, expected3.Equal(output3), "Valid Case 3 Failed: Expected %s, got %s", expected3, output3)

	// --- Error Cases ---

	// Error 1: Invalid operand DType
	require.Panics(t, func() {
		must1(ArgMinMaxOp(shapes.Make(dtypes.InvalidDType, 10), 0, I32))
	}, "Error Case 1 Failed: Invalid operand DType")

	// Error 2: Invalid axis (out of bounds)
	require.Panics(t, func() {
		must1(ArgMinMaxOp(operand1, 1, I32)) // operand1 is rank 1, axis 1 invalid
	}, "Error Case 2 Failed: Invalid axis (out of bounds)")

	// Error 3: Negative axis
	require.Panics(t, func() {
		must1(ArgMinMaxOp(operand2, -1, I32))
	}, "Error Case 3 Failed: Negative axis")
}
