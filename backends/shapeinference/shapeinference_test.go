package shapeinference

import (
	"fmt"
	"testing"

	. "github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Aliases
var (
	Bool = dtypes.Bool
	I8   = dtypes.Int8
	I32  = dtypes.Int32
	F32  = dtypes.Float32
	U64  = dtypes.Uint64

	S = shapes.Make
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
	_, err = BinaryOp(OpTypeLogicalAnd, S(I8), S(I8))
	require.Error(t, err)
	_, err = BinaryOp(OpTypeMul, S(Bool, 1), S(Bool, 1))
	require.Error(t, err)
	_, err = BinaryOp(OpTypeMul, S(Bool, 1), S(Bool, 1))
	require.Error(t, err)
	_, err = BinaryOp(OpTypeBitwiseXor, S(F32, 1), S(F32, 1))
	require.Error(t, err)

	// Invalid operation type (not binary op).
	_, err = BinaryOp(OpTypeExp, S(F32), S(F32))
	require.Error(t, err)

	// The same shape should be ok.
	var output shapes.Shape
	intMatrixShape := S(I8, 3, 3)
	output, err = BinaryOp(OpTypeBitwiseOr, intMatrixShape, intMatrixShape)
	require.NoError(t, err)
	require.True(t, intMatrixShape.Equal(output))

	// Scalar with matrix.
	scalarShape := S(F32)
	matrixShape := S(F32, 2, 3)
	expectedShape := S(F32, 2, 3)
	output, err = BinaryOp(OpTypeAdd, scalarShape, scalarShape)
	require.NoError(t, err)
	require.True(t, scalarShape.Equal(output))
	output, err = BinaryOp(OpTypeAdd, scalarShape, matrixShape)
	require.NoError(t, err)
	require.True(t, expectedShape.Equal(output))

	// Broadcasting on both sides.
	shape1 := S(F32, 2, 1, 3)
	shape2 := S(F32, 1, 4, 3)
	expectedBroadcastShape := S(F32, 2, 4, 3)
	require.True(t, expectedBroadcastShape.Equal(must1(BinaryOp(OpTypeMul, shape1, shape2))))

	// Matrix with scalar.
	require.True(t, expectedShape.Equal(must1(BinaryOp(OpTypeAdd, matrixShape, scalarShape))))

	// Invalid broadcasting shapes.
	invalidShape1 := S(F32, 2, 3)
	invalidShape2 := S(F32, 3, 2)
	_, err = BinaryOp(OpTypeAdd, invalidShape1, invalidShape2)
	require.Error(t, err)
}

func TestUnaryOp(t *testing.T) {
	// Invalid data types check.
	require.Panics(t, func() { must1(UnaryOp(OpTypeLogicalNot, S(F32))) })
	require.Panics(t, func() { must1(UnaryOp(OpTypeLogicalNot, S(I8))) })
	require.Panics(t, func() { must1(UnaryOp(OpTypeBitwiseNot, S(F32))) })
	require.Panics(t, func() { must1(UnaryOp(OpTypeNeg, S(Bool))) })

	// Invalid operation type (not unary op).
	require.Panics(t, func() { must1(UnaryOp(OpTypeAdd, S(F32))) })
	require.Panics(t, func() { must1(UnaryOp(OpTypeNeg, S(U64))) })

	// Valid operations
	boolShape := S(Bool, 2, 3)
	require.True(t, boolShape.Equal(must1(UnaryOp(OpTypeLogicalNot, boolShape))))

	intShape := S(I8, 3, 3)
	require.True(t, intShape.Equal(must1(UnaryOp(OpTypeBitwiseNot, intShape))))

	floatShape := S(F32, 2, 3)
	require.True(t, floatShape.Equal(must1(UnaryOp(OpTypeExp, floatShape))))
	require.True(t, floatShape.Equal(must1(UnaryOp(OpTypeNeg, floatShape))))
}

func TestGatherOp(t *testing.T) {
	// Test 1:
	operand := S(F32, 4, 3, 2, 2)
	startIndices := S(I8, 3, 3, 2)
	indexVectorAxis := 1
	offsetOutputAxes := []int{0, 3}
	collapsedSliceAxes := []int{0, 2}
	startIndexMap := []int{0, 2, 3}
	sliceSizes := []int{1, 3, 1, 1}
	output, err := Gather(operand, startIndices, indexVectorAxis,
		offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes, false)
	require.NoError(t, err)
	fmt.Printf("\tTest 1: outputShape=%s\n", output)
	require.NoError(t, output.Check(F32, 3, 3, 2, 1))

	// Test 2:
	operand = S(F32, 3, 4, 5, 6)
	startIndices = S(U64, 7, 3, 8)
	indexVectorAxis = 1
	offsetOutputAxes = []int{1, 2}
	collapsedSliceAxes = []int{1, 2}
	startIndexMap = []int{1, 2, 3}
	sliceSizes = []int{3, 1, 1, 1}
	output, err = Gather(operand, startIndices, indexVectorAxis, offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes, true)
	require.NoError(t, err)
	fmt.Printf("\tTest 2: outputShape=%s\n", output)
	require.NoError(t, output.Check(F32, 7, 3, 1, 8))

	// Test 3:
	operand = S(F32, 8, 16)
	startIndices = S(U64, 8, 1)
	indexVectorAxis = 1
	offsetOutputAxes = []int{1}
	collapsedSliceAxes = []int{0}
	startIndexMap = []int{0}
	sliceSizes = []int{1, 16}
	output, err = Gather(operand, startIndices, indexVectorAxis, offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes, true)
	require.NoError(t, err)
	fmt.Printf("\tTest 3: outputShape=%s\n", output)
	require.NoError(t, output.Check(F32, 8, 16))
}

func TestScatterOp(t *testing.T) {
	// --- Valid Cases ---

	// Case 1: Typical scatter (like TF ScatterNd)
	// Scatter 2 updates of shape [5] into operand [4, 5]
	// Indices shape [2, 1] indicates 2 indices, each pointing to 1 dimension (axis 0) of operand.
	operand1 := S(F32, 4, 5)
	indices1 := S(I8, 2, 1)  // Batch shape [2]
	updates1 := S(F32, 2, 5) // Batch shape [2]
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
	operand2 := S(F32, 10, 9, 8)
	indices2 := S(I32, 2, 3, 2)              // 6 indices, each is a 2D coordinate
	updates2 := S(F32, 2, 3, 8)              // 6 updates, window shape [8] matching operand's last dim
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
	operand3 := S(F32, 10, 9, 8)
	indices3 := S(I32, 2, 2, 3) // 2x3 batch, index vector size 2, indexVecAxis=1
	updates3 := S(F32, 8, 2, 3) // Update axis [8] is "out-of-order", which should be fine.
	indexVectorAxis3 := 1       // Index vector is now axis 1
	updateWindowAxes3 := []int{0}
	insertedWindowAxes3 := []int{1, 2}
	scatterAxesToOperandAxes3 := []int{1, 2} // indices are used for different axes in the operand this time.
	expected3 := operand2                    // Still expect operand shape
	output3, err := ScatterOp(operand3, indices3, updates3, indexVectorAxis3, updateWindowAxes3, insertedWindowAxes3, scatterAxesToOperandAxes3)
	require.NoError(t, err)
	require.True(t, expected3.Equal(output3), "Valid Case 3 Failed (IndexVecAxis=1): Expected %s, got %s", expected3, output3)

	// Case 4: No insertedWindowAxes (scattering full slices)
	// Scatter updates of shape [9] into operand [10, 9]
	operand4 := S(F32, 10, 9)
	indices4 := S(I32, 6)                 // 6 indices, coord size 1
	updates4 := S(F32, 6, 9)              // 6 updates, window shape [] (scalar)
	indexVectorAxis4 := 1                 // == indices4.Rank() -> trigger extra virtual axes to indices4.
	updateWindowAxes4 := []int{1}         // No window axes in updates (updates are scalars matching batch dims)
	insertedWindowAxes4 := []int{0}       // No window axes in operand (index selects full slice - which is scalar here)
	scatterAxesToOperandAxes4 := []int{0} // Index coord 0 -> operand axis 0
	expected4 := operand4
	output4, err := ScatterOp(operand4, indices4, updates4, indexVectorAxis4, updateWindowAxes4, insertedWindowAxes4, scatterAxesToOperandAxes4)
	require.NoError(t, err)
	require.True(t, expected4.Equal(output4), "Valid Case 4 Failed (No Window): Expected %s, got %s", expected4, output4)

	// Case 5: rearranging the output axes:
	operand5 := S(F32, 2, 5, 2)
	indices5 := S(I32, 2, 2)
	updates5 := S(F32, 5, 2)
	indexVectorAxis5 := 1
	updateWindowAxes5 := []int{0}
	insertedWindowAxes5 := []int{0, 2}
	scatterAxesToOperandAxes5 := []int{0, 2}
	output5, err := ScatterOp(operand5, indices5, updates5, indexVectorAxis5, updateWindowAxes5, insertedWindowAxes5, scatterAxesToOperandAxes5)
	require.NoError(t, err)
	require.True(t, operand5.Equal(output5), "Valid Case 5 Failed (No Window): Expected %s, got %s", operand5, output5)

	// --- Error Cases ---

	// Error Case 1: Mismatched DType (Operand vs Updates) - unchanged
	_, err = ScatterOp(S(F32, 4, 5), S(I8, 2, 1), S(I8, 2, 5), 1, []int{1}, []int{1}, []int{0})
	require.Error(t, err, "Error Case 1 Failed: Mismatched operand/updates DType")

	// Error Case 2: Invalid DType for indices - unchanged
	_, err = ScatterOp(S(F32, 4, 5), S(F32, 2, 1), S(F32, 2, 5), 1, []int{1}, []int{1}, []int{0})
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
	operand1 := S(F32, 10)
	starts1 := []int{2}
	limits1 := []int{8}
	strides1 := []int{1}
	expected1 := S(F32, 6)
	output1, err := SliceOp(operand1, starts1, limits1, strides1)
	require.NoError(t, err)
	require.True(t, expected1.Equal(output1), "%s Valid Case 1 Failed: Expected %s, got %s", opName, expected1, output1)

	// Case 2: 2D slice with stride 1
	operand2 := S(I32, 5, 6)
	starts2 := []int{1, 2}
	limits2 := []int{4, 5}
	strides2 := []int{1, 1}
	expected2 := S(I32, 3, 3)
	output2, err := SliceOp(operand2, starts2, limits2, strides2)
	require.NoError(t, err)
	require.True(t, expected2.Equal(output2), "%s Valid Case 2 Failed: Expected %s, got %s", opName, expected2, output2)

	// Case 3: 3D slice with different strides
	operand3 := S(Bool, 10, 8, 6)
	starts3 := []int{1, 0, 1}
	limits3 := []int{10, 8, 6} // End index exclusive
	strides3 := []int{2, 3, 1}
	// Dim 0: (10-1)/2 = 4.5 -> 5 elements (indices 1, 3, 5, 7, 9)
	// Dim 1: (8-0)/3 = 2.66 -> 3 elements (indices 0, 3, 6)
	// Dim 2: (6-1)/1 = 5 -> 5 elements (indices 1, 2, 3, 4, 5)
	expected3 := S(Bool, 5, 3, 5)
	output3, err := SliceOp(operand3, starts3, limits3, strides3)
	require.NoError(t, err)
	require.True(t, expected3.Equal(output3), "%s Valid Case 3 Failed: Expected %s, got %s", opName, expected3, output3)

	// Case 4: Slice resulting in size 1 dimension
	operand4 := S(F32, 10)
	starts4 := []int{5}
	limits4 := []int{6}
	strides4 := []int{1}
	expected4 := S(F32, 1)
	output4, err := SliceOp(operand4, starts4, limits4, strides4)
	require.NoError(t, err)
	require.True(t, expected4.Equal(output4), "%s Valid Case 4 Failed: Expected %s, got %s", opName, expected4, output4)

	// Case 5: Slice taking full dimension with stride > 1
	operand5 := S(I8, 7)
	starts5 := []int{0}
	limits5 := []int{7}
	strides5 := []int{2}
	// Dim 0: (7-0)/2 = 3.5 -> 4 elements (indices 0, 2, 4, 6)
	expected5 := S(I8, 4)
	output5, err := SliceOp(operand5, starts5, limits5, strides5)
	require.NoError(t, err)
	require.True(t, expected5.Equal(output5), "%s Valid Case 5 Failed: Expected %s, got %s", opName, expected5, output5)

	// --- Error Cases ---
	operand := S(F32, 10, 5) // Rank 2
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
	operand1 := S(F32, 10)
	expected1 := S(I32)
	output1 := must1(ArgMinMaxOp(operand1, 0, I32))
	require.True(t, expected1.Equal(output1), "Valid Case 1 Failed: Expected %s, got %s", expected1, output1)

	// Case 2: 2D tensor, single axis
	operand2 := S(F32, 5, 6)
	expected2 := S(I8, 5)
	output2 := must1(ArgMinMaxOp(operand2, 1, expected2.DType))
	require.True(t, expected2.Equal(output2), "Valid Case 2 Failed: Expected %s, got %s", expected2, output2)

	// Case 3: 3D tensor, multiple axes
	operand3 := S(F32, 4, 5, 6)
	expected3 := S(U64, 5, 6)
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

func TestBinaryOpSymbolic(t *testing.T) {
	// Test symbolic dimension handling in binary operations

	// Same symbolic dimension with same axis name - should be preserved
	shape1 := shapes.MakeDynamic(F32, shapes.DynamicDim, 3, 4).WithAxisName(0, "batch")
	shape2 := shapes.MakeDynamic(F32, shapes.DynamicDim, 3, 4).WithAxisName(0, "batch")
	output, err := BinaryOp(OpTypeAdd, shape1, shape2)
	require.NoError(t, err)
	require.Equal(t, shapes.DynamicDim, output.Dimensions[0])
	require.Equal(t, "batch", output.AxisName(0)) // Same axis names preserved
	require.Equal(t, 3, output.Dimensions[1])
	require.Equal(t, 4, output.Dimensions[2])

	// Symbolic vs static broadcasting (with 1)
	shape3 := shapes.MakeDynamic(F32, shapes.DynamicDim, 1, 4).WithAxisName(0, "batch")
	shape4 := shapes.Make(F32, 1, 3, 4)
	output, err = BinaryOp(OpTypeMul, shape3, shape4)
	require.NoError(t, err)
	require.Equal(t, shapes.DynamicDim, output.Dimensions[0]) // Symbolic preserved when other is 1
	require.Equal(t, "batch", output.AxisName(0))             // Axis name preserved
	require.Equal(t, 3, output.Dimensions[1])
	require.Equal(t, 4, output.Dimensions[2])

	// Symbolic vs concrete > 1
	shape5 := shapes.MakeDynamic(F32, shapes.DynamicDim, 3, 4).WithAxisName(0, "batch")
	shape6 := shapes.Make(F32, 8, 3, 4)
	output, err = BinaryOp(OpTypeAdd, shape5, shape6)
	require.NoError(t, err)
	require.Equal(t, 8, output.Dimensions[0]) // Concrete > 1 wins
	require.Equal(t, 3, output.Dimensions[1])

	// Two different axis names -> axis name cleared
	shape7 := shapes.MakeDynamic(F32, shapes.DynamicDim, 3).WithAxisName(0, "batch")
	shape8 := shapes.MakeDynamic(F32, shapes.DynamicDim, 3).WithAxisName(0, "seq")
	output, err = BinaryOp(OpTypeAdd, shape7, shape8)
	require.NoError(t, err)
	require.Equal(t, shapes.DynamicDim, output.Dimensions[0]) // Stays dynamic
	require.Equal(t, "", output.AxisName(0))                   // Different names -> cleared

	// Broadcasting with symbolic dimensions
	shape9 := shapes.MakeDynamic(F32, shapes.DynamicDim, 1, 4).WithAxisName(0, "batch")
	shape10 := shapes.MakeDynamic(F32, 1, 3, 4)
	output, err = BinaryOp(OpTypeMul, shape9, shape10)
	require.NoError(t, err)
	require.Equal(t, shapes.DynamicDim, output.Dimensions[0]) // Symbolic wins over 1
	require.Equal(t, "batch", output.AxisName(0))             // Axis name preserved
	require.Equal(t, 3, output.Dimensions[1])
	require.Equal(t, 4, output.Dimensions[2])
}

func TestComparisonOpSymbolic(t *testing.T) {
	// Comparison operations should handle symbolic dimensions and return Bool dtype
	shape1 := shapes.MakeDynamic(F32, shapes.DynamicDim, 3, 4).WithAxisName(0, "batch")
	shape2 := shapes.MakeDynamic(F32, shapes.DynamicDim, 3, 4).WithAxisName(0, "batch")
	output, err := ComparisonOp(OpTypeGreaterThan, shape1, shape2)
	require.NoError(t, err)
	require.Equal(t, Bool, output.DType)
	require.Equal(t, shapes.DynamicDim, output.Dimensions[0])
	require.Equal(t, "batch", output.AxisName(0))
	require.Equal(t, 3, output.Dimensions[1])
	require.Equal(t, 4, output.Dimensions[2])
}

func TestReshapeOpSymbolic(t *testing.T) {
	// Reshape with symbolic dimensions should skip size validation
	shape1 := shapes.MakeDynamic(F32, shapes.DynamicDim, 10).WithAxisName(0, "batch")
	newDims := []int{shapes.DynamicDim, 5, 2}
	output, err := ReshapeOp(shape1, newDims)
	require.NoError(t, err)
	require.Equal(t, shapes.DynamicDim, output.Dimensions[0])
	require.Equal(t, 5, output.Dimensions[1])
	require.Equal(t, 2, output.Dimensions[2])

	// Reshape from static to symbolic
	shape2 := shapes.Make(F32, 8, 10)
	newDims2 := []int{shapes.DynamicDim, 10}
	output, err = ReshapeOp(shape2, newDims2)
	require.NoError(t, err)
	require.Equal(t, shapes.DynamicDim, output.Dimensions[0])
	require.Equal(t, 10, output.Dimensions[1])

	// Static reshape should still validate sizes
	shape3 := shapes.Make(F32, 4, 5)
	newDims3 := []int{10, 3}
	_, err = ReshapeOp(shape3, newDims3)
	require.Error(t, err) // Size mismatch: 20 != 30
}

func TestTransposeOpSymbolic(t *testing.T) {
	// Transpose should preserve symbolic dimensions in permuted positions
	shape := shapes.MakeDynamic(F32, shapes.DynamicDim, 3, 4).WithAxisName(0, "batch")
	permutations := []int{2, 0, 1}
	output, err := TransposeOp(shape, permutations)
	require.NoError(t, err)
	require.Equal(t, 4, output.Dimensions[0])
	require.Equal(t, shapes.DynamicDim, output.Dimensions[1])
	require.Equal(t, 3, output.Dimensions[2])
}

func TestReduceOpSymbolic(t *testing.T) {
	// Reduce operations should remove axes, including symbolic ones
	shape := shapes.MakeDynamic(F32, shapes.DynamicDim, 10, 20).WithAxisName(0, "batch")

	// Reduce along batch axis
	output, err := ReduceOp(shape, []int{0})
	require.NoError(t, err)
	require.Equal(t, 2, output.Rank())
	require.Equal(t, 10, output.Dimensions[0])
	require.Equal(t, 20, output.Dimensions[1])

	// Reduce along static axis
	output, err = ReduceOp(shape, []int{1})
	require.NoError(t, err)
	require.Equal(t, 2, output.Rank())
	require.Equal(t, shapes.DynamicDim, output.Dimensions[0])
	require.Equal(t, 20, output.Dimensions[1])
}

func TestConcatenateOpSymbolic(t *testing.T) {
	// Concatenate symbolic dimensions results in dynamic with no axis name
	shape1 := shapes.MakeDynamic(F32, shapes.DynamicDim, 3, 4).WithAxisName(0, "batch")
	shape2 := shapes.MakeDynamic(F32, shapes.DynamicDim, 3, 4).WithAxisName(0, "batch")
	output, err := ConcatenateOp([]shapes.Shape{shape1, shape2}, 0)
	require.NoError(t, err)
	require.Equal(t, shapes.DynamicDim, output.Dimensions[0]) // Still dynamic after concat
	require.Equal(t, 3, output.Dimensions[1])
	require.Equal(t, 4, output.Dimensions[2])

	// Concatenate static with symbolic
	shape3 := shapes.Make(F32, 5, 3, 4)
	shape4 := shapes.MakeDynamic(F32, shapes.DynamicDim, 3, 4).WithAxisName(0, "batch")
	output, err = ConcatenateOp([]shapes.Shape{shape3, shape4}, 0)
	require.NoError(t, err)
	require.Equal(t, shapes.DynamicDim, output.Dimensions[0]) // One symbolic -> dynamic

	// Concatenate along non-symbolic axis
	shape5 := shapes.MakeDynamic(F32, shapes.DynamicDim, 3, 4).WithAxisName(0, "batch")
	shape6 := shapes.MakeDynamic(F32, shapes.DynamicDim, 5, 4).WithAxisName(0, "batch")
	output, err = ConcatenateOp([]shapes.Shape{shape5, shape6}, 1)
	require.NoError(t, err)
	require.Equal(t, shapes.DynamicDim, output.Dimensions[0]) // Batch preserved
	require.Equal(t, "batch", output.AxisName(0))             // Same axis name preserved
	require.Equal(t, 8, output.Dimensions[1])                 // 3 + 5
	require.Equal(t, 4, output.Dimensions[2])

	// Different axis names on non-concat axis -> name cleared
	shape7 := shapes.MakeDynamic(F32, shapes.DynamicDim, 3).WithAxisName(0, "batch")
	shape8 := shapes.MakeDynamic(F32, shapes.DynamicDim, 3).WithAxisName(0, "seq")
	output, err = ConcatenateOp([]shapes.Shape{shape7, shape8}, 1)
	require.NoError(t, err)
	require.Equal(t, shapes.DynamicDim, output.Dimensions[0]) // Still dynamic
	require.Equal(t, "", output.AxisName(0))                  // Different names -> cleared
	require.Equal(t, 6, output.Dimensions[1])                 // 3 + 3

	// Prefer concrete over symbolic on non-concat axes
	shape9 := shapes.MakeDynamic(F32, shapes.DynamicDim, 3).WithAxisName(0, "batch")
	shape10 := shapes.Make(F32, 8, 3)
	output, err = ConcatenateOp([]shapes.Shape{shape9, shape10}, 1)
	require.NoError(t, err)
	require.Equal(t, 8, output.Dimensions[0]) // Concrete preferred
	require.Equal(t, 6, output.Dimensions[1]) // 3 + 3
}

func TestReduceWindowOp(t *testing.T) {
	type testCase struct {
		name                 string
		operandShape         shapes.Shape
		windowDimensions     []int
		strides              []int
		baseDilations        []int
		windowDilations      []int
		paddings             [][2]int
		expectedShape        shapes.Shape
		expectError          bool
		errorMessageContains string // Optional: for more specific error checking
	}

	testCases := []testCase{
		{
			name:             "ScalarInput_AllNilParams_Defaults",
			operandShape:     shapes.Make(dtypes.Float32), // Rank 0
			windowDimensions: nil,                         // Should be handled as empty for rank 0
			strides:          nil,                         // Should be handled as empty for rank 0
			baseDilations:    nil,
			windowDilations:  nil,
			paddings:         nil, // Should be handled as empty for rank 0
			expectedShape:    shapes.Make(dtypes.Float32),
			expectError:      false,
		},
		{
			name:             "1D_AllNilParams_Defaults",
			operandShape:     shapes.Make(dtypes.Float32, 10),
			windowDimensions: nil, // Defaults to {1}
			strides:          nil, // Defaults to windowDimensions, in this case {1}
			baseDilations:    nil, // Defaults to {1}
			windowDilations:  nil, // Defaults to {1}
			paddings:         nil, // Defaults to {{0,0}}
			// Calculation: EffIn=10, EffWin=1. PaddedEffIn=10. Num=10-1=9. Out=(9/1)+1=10.
			expectedShape: shapes.Make(dtypes.Float32, 10),
			expectError:   false,
		},
		{
			name:             "1D_NonDefault_WindowDimensions_NilOthers",
			operandShape:     shapes.Make(dtypes.Float32, 10),
			windowDimensions: []int{3}, // EffWin=3
			strides:          nil,      // Default to windowDimensions, in this case {3}
			baseDilations:    nil,      // Default {1}
			windowDilations:  nil,      // Default {1}
			paddings:         nil,      // Default {{0,0}}
			// Calculation: EffIn=10, EffWin=3. PaddedEffIn=10. Num=10-3=7. Out=(7/3)+1=3.
			expectedShape: shapes.Make(dtypes.Float32, 3),
			expectError:   false,
		},
		{
			name:             "1D_NonDefault_Strides_NilOthers",
			operandShape:     shapes.Make(dtypes.Float32, 10),
			windowDimensions: nil, // Default {1} => EffWin=1
			strides:          []int{2},
			baseDilations:    nil,
			windowDilations:  nil,
			paddings:         nil,
			// Calculation: EffIn=10, EffWin=1. PaddedEffIn=10. Num=10-1=9. Out=(9/2)+1=4+1=5.
			expectedShape: shapes.Make(dtypes.Float32, 5),
			expectError:   false,
		},
		{
			name:             "1D_NonDefault_Paddings_NilOthers",
			operandShape:     shapes.Make(dtypes.Float32, 10),
			windowDimensions: nil, // Default {1} => EffWin=1
			strides:          nil, // Default {1}
			baseDilations:    nil,
			windowDilations:  nil,
			paddings:         [][2]int{{1, 1}},
			// Calculation: EffIn=10, EffWin=1. PaddedEffIn=10+1+1=12. Num=12-1=11. Out=(11/1)+1=12.
			expectedShape: shapes.Make(dtypes.Float32, 12),
			expectError:   false,
		},
		{
			name:             "1D_NonDefault_BaseDilations",
			operandShape:     shapes.Make(dtypes.Float32, 5),
			windowDimensions: []int{3}, // EffWin=3
			strides:          []int{1},
			baseDilations:    []int{2}, // EffIn=(5-1)*2+1 = 9
			windowDilations:  nil,
			paddings:         nil,
			// Calculation: PaddedEffIn=9. Num=9-3=6. Out=(6/1)+1=7.
			expectedShape: shapes.Make(dtypes.Float32, 7),
			expectError:   false,
		},
		{
			name:             "1D_NonDefault_WindowDilations",
			operandShape:     shapes.Make(dtypes.Float32, 10),
			windowDimensions: []int{3},
			strides:          []int{1},
			baseDilations:    nil,
			windowDilations:  []int{2}, // EffWin=(3-1)*2+1=5
			paddings:         nil,
			// Calculation: EffIn=10. PaddedEffIn=10. Num=10-5=5. Out=(5/1)+1=6.
			expectedShape: shapes.Make(dtypes.Float32, 6),
			expectError:   false,
		},
		{
			name:             "2D_Comprehensive_AllNonDefault",
			operandShape:     shapes.Make(dtypes.Int32, 10, 12),
			windowDimensions: []int{3, 4},
			strides:          []int{2, 3},
			baseDilations:    []int{2, 1},
			windowDilations:  []int{1, 2},
			paddings:         [][2]int{{1, 1}, {0, 2}},
			// Dim0: In=10,Win=3,Str=2,Pad=[1,1],BD=2,WD=1. EffIn=(10-1)*2+1=19. EffWin=(3-1)*1+1=3. PaddedEffIn=19+1+1=21. Num=21-3=18. Out=18/2+1=10.
			// Dim1: In=12,Win=4,Str=3,Pad=[0,2],BD=1,WD=2. EffIn=(12-1)*1+1=12. EffWin=(4-1)*2+1=7. PaddedEffIn=12+0+2=14. Num=14-7=7. Out=7/3+1=2+1=3.
			expectedShape: shapes.Make(dtypes.Int32, 10, 3),
			expectError:   false,
		},
		{
			name:             "Rank4_Image_NHWC_Style_VariedParams",
			operandShape:     shapes.Make(dtypes.Float32, 1, 20, 22, 3), // N, H, W, C
			windowDimensions: []int{1, 3, 3, 1},                         // Window on H, W
			strides:          []int{1, 2, 2, 1},                         // Stride on H, W
			baseDilations:    nil,                                       // Default {1,1,1,1}
			windowDilations:  nil,                                       // Default {1,1,1,1}
			paddings:         [][2]int{{0, 0}, {1, 0}, {0, 1}, {0, 0}},  // Padding H (low), W (high)
			// Dim0(N): In=1,Win=1,Str=1,Pad0,BD1,WD1. EffIn=1,EffWin=1.Padded=1.Num=0.Out=1.
			// Dim1(H): In=20,Win=3,Str=2,PadL=1,PadH=0,BD1,WD1. EffIn=20,EffWin=3.Padded=20+1+0=21.Num=21-3=18.Out=18/2+1=10.
			// Dim2(W): In=22,Win=3,Str=2,PadL=0,PadH=1,BD1,WD1. EffIn=22,EffWin=3.Padded=22+0+1=23.Num=23-3=20.Out=20/2+1=11.
			// Dim3(C): In=3,Win=1,Str=1,Pad0,BD1,WD1. EffIn=3,EffWin=1.Padded=3.Num=2.Out=3.
			expectedShape: shapes.Make(dtypes.Float32, 1, 10, 11, 3),
			expectError:   false,
		},
		{
			name:                 "Error_WindowTooLarge_NoPadding",
			operandShape:         shapes.Make(dtypes.Float32, 5),
			windowDimensions:     []int{6}, // EffWin=6
			strides:              []int{1},
			paddings:             nil, // PaddedEffIn=5
			expectError:          true,
			errorMessageContains: "effective window dimension 6 for axis 0 is larger than padded effective input dimension 5",
		},
		{
			name:                 "Error_InvalidStrideZero",
			operandShape:         shapes.Make(dtypes.Float32, 5),
			windowDimensions:     []int{2},
			strides:              []int{0},
			expectError:          true,
			errorMessageContains: "strides[0]=0 must be >= 1",
		},
		{
			name:                 "Error_InvalidWindowDimZero_FromNonNil",
			operandShape:         shapes.Make(dtypes.Float32, 5),
			windowDimensions:     []int{0},
			strides:              []int{1},
			expectError:          true,
			errorMessageContains: "windowDimensions[0]=0 must be >= 1",
		},
		{
			name:                 "Error_NegativePadding_FromNonNil",
			operandShape:         shapes.Make(dtypes.Float32, 5),
			windowDimensions:     []int{2},
			strides:              []int{1},
			paddings:             [][2]int{{-1, 0}},
			expectError:          true,
			errorMessageContains: "paddings[0]=[-1, 0] must be non-negative",
		},
		{
			name:                 "Error_InvalidBaseDilationZero",
			operandShape:         shapes.Make(dtypes.Float32, 5),
			windowDimensions:     []int{2},
			strides:              []int{1},
			baseDilations:        []int{0},
			expectError:          true,
			errorMessageContains: "baseDilations[0]=0 must be >= 1",
		},
		{
			name:                 "Error_InvalidWindowDilationZero",
			operandShape:         shapes.Make(dtypes.Float32, 5),
			windowDimensions:     []int{2},
			strides:              []int{1},
			windowDilations:      []int{0},
			expectError:          true,
			errorMessageContains: "windowDilations[0]=0 must be >= 1",
		},
		{
			name:                 "Error_StridesNotNil_WrongLengthForRank",
			operandShape:         shapes.Make(dtypes.Float32, 5, 5), // Rank 2
			windowDimensions:     nil,                               // Defaults to {1,1}
			strides:              []int{1},                          // Error: len 1, rank 2
			expectError:          true,
			errorMessageContains: "len(strides)=1, but operand rank is 2",
		},
		{
			name:                 "Error_WindowDimensionsNotNil_WrongLengthForRank",
			operandShape:         shapes.Make(dtypes.Float32, 5, 5), // Rank 2
			windowDimensions:     []int{1},                          // Error: len 1, rank 2
			strides:              nil,
			expectError:          true,
			errorMessageContains: "len(windowDimensions)=1, but operand rank is 2",
		},
		{
			name:                 "Error_PaddingsNotNil_WrongLengthForRank",
			operandShape:         shapes.Make(dtypes.Float32, 5, 5), // Rank 2
			windowDimensions:     nil,
			strides:              nil,
			paddings:             [][2]int{{0, 0}}, // Error: len 1, rank 2
			expectError:          true,
			errorMessageContains: "len(paddings)=1, but operand rank is 2",
		},
		{
			name:                 "Error_BaseDilationsNotNil_WrongLength",
			operandShape:         shapes.Make(dtypes.Float32, 5, 5), // Rank 2
			baseDilations:        []int{1},                          // Error: len 1, rank 2
			expectError:          true,
			errorMessageContains: "baseDilations is not nil and len(baseDilations)=1, but operand rank is 2",
		},
		{
			name:                 "Error_WindowDilationsNotNil_WrongLength",
			operandShape:         shapes.Make(dtypes.Float32, 5, 5), // Rank 2
			windowDilations:      []int{1},                          // Error: len 1, rank 2
			expectError:          true,
			errorMessageContains: "windowDilations is not nil and len(windowDilations)=1, but operand rank is 2",
		},
		{
			// This case would lead to outputDim 0 if the formula `(Num/Stride)+1` was used naively with Num < 0.
			// The pre-check `effectiveWindowDim > paddedEffectiveInputDim` should catch this.
			name:                 "NearZeroOutputDim_HandledByPreCheck",
			operandShape:         shapes.Make(dtypes.Float32, 2), // InputDim=2
			windowDimensions:     []int{3},                       // EffWin=3
			strides:              []int{1},
			paddings:             nil, // PaddedEffIn=2
			expectError:          true,
			errorMessageContains: "effective window dimension 3 for axis 0 is larger than padded effective input dimension 2",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			outputShape, err := ReduceWindowOp(
				tc.operandShape,
				tc.windowDimensions,
				tc.strides,
				tc.baseDilations,
				tc.windowDilations,
				tc.paddings,
			)

			if tc.expectError {
				require.Error(t, err, "Expected an error for test case: %s", tc.name)
				if tc.errorMessageContains != "" {
					assert.Contains(t, err.Error(), tc.errorMessageContains, "Error message mismatch for: %s", tc.name)
				}
			} else {
				require.NoError(t, err, "Did not expect an error for test case: %s (error was: %v)", tc.name, err)
				assert.True(t, tc.expectedShape.Equal(outputShape),
					"Mismatch in output shape for test case: %s. Expected %s, Got %s",
					tc.name, tc.expectedShape, outputShape)
			}
		})
	}
}
