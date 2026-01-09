//go:build darwin && cgo

package coreml

import (
	"math"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// TestDynamicUpdateSlice1D tests updating a 1D slice at a dynamic position.
func TestDynamicUpdateSlice1D(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_dynamic_update_slice_1d")
	mainFn := builder.Main()

	// Create operand: [0, 1, 2, 3, 4]
	operandShape := shapes.Make(dtypes.Float32, 5)
	operand, err := mainFn.Parameter("operand", operandShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Create update: [10, 11]
	updateShape := shapes.Make(dtypes.Float32, 2)
	update, err := mainFn.Parameter("update", updateShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Create start index: 1-element int32 tensor (CoreML doesn't support scalar inputs)
	startIndexShape := shapes.Make(dtypes.Int32, 1)
	startIndex, err := mainFn.Parameter("start_index", startIndexShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// DynamicUpdateSlice
	result, err := mainFn.DynamicUpdateSlice(operand, update, []backends.Value{startIndex})
	if err != nil {
		t.Fatalf("DynamicUpdateSlice() failed: %v", err)
	}

	err = mainFn.Return([]backends.Value{result}, nil)
	if err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Create input buffers
	operandData := []float32{0, 1, 2, 3, 4}
	updateData := []float32{10, 11}
	startIndexData := []int32{2} // Update at position 2

	operandBuf, err := backend.BufferFromFlatData(0, operandData, operandShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(operand) failed: %v", err)
	}

	updateBuf, err := backend.BufferFromFlatData(0, updateData, updateShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(update) failed: %v", err)
	}

	startIndexBuf, err := backend.BufferFromFlatData(0, startIndexData, startIndexShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(startIndex) failed: %v", err)
	}

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{operandBuf, updateBuf, startIndexBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	// Verify output
	outputData := make([]float32, 5)
	err = backend.BufferToFlatData(outputs[0], outputData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// Expected: [0, 1, 10, 11, 4]
	expected := []float32{0, 1, 10, 11, 4}
	for i := range expected {
		if math.Abs(float64(outputData[i]-expected[i])) > 1e-5 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], expected[i])
		}
	}
}

// TestDynamicUpdateSlice2D tests updating a 2D slice at a dynamic position.
func TestDynamicUpdateSlice2D(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_dynamic_update_slice_2d")
	mainFn := builder.Main()

	// Create operand: 4x4 matrix
	operandShape := shapes.Make(dtypes.Float32, 4, 4)
	operand, err := mainFn.Parameter("operand", operandShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Create update: 2x2 matrix
	updateShape := shapes.Make(dtypes.Float32, 2, 2)
	update, err := mainFn.Parameter("update", updateShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Create start indices: two 1-element int32 tensors (CoreML doesn't support scalar inputs)
	startIndexShape := shapes.Make(dtypes.Int32, 1)
	startIndex0, err := mainFn.Parameter("start_index_0", startIndexShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}
	startIndex1, err := mainFn.Parameter("start_index_1", startIndexShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// DynamicUpdateSlice
	result, err := mainFn.DynamicUpdateSlice(operand, update, []backends.Value{startIndex0, startIndex1})
	if err != nil {
		t.Fatalf("DynamicUpdateSlice() failed: %v", err)
	}

	err = mainFn.Return([]backends.Value{result}, nil)
	if err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Create input buffers
	// Operand: 4x4 matrix with values 0-15
	operandData := make([]float32, 16)
	for i := range operandData {
		operandData[i] = float32(i)
	}

	// Update: 2x2 matrix with values 100, 101, 102, 103
	updateData := []float32{100, 101, 102, 103}

	// Start at position (1, 1)
	startIndex0Data := []int32{1}
	startIndex1Data := []int32{1}

	operandBuf, err := backend.BufferFromFlatData(0, operandData, operandShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(operand) failed: %v", err)
	}

	updateBuf, err := backend.BufferFromFlatData(0, updateData, updateShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(update) failed: %v", err)
	}

	startIndex0Buf, err := backend.BufferFromFlatData(0, startIndex0Data, startIndexShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(startIndex0) failed: %v", err)
	}

	startIndex1Buf, err := backend.BufferFromFlatData(0, startIndex1Data, startIndexShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(startIndex1) failed: %v", err)
	}

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{operandBuf, updateBuf, startIndex0Buf, startIndex1Buf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	// Verify output
	outputData := make([]float32, 16)
	err = backend.BufferToFlatData(outputs[0], outputData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// Original:           Expected after update at (1,1):
	// [[ 0,  1,  2,  3],  [[ 0,   1,   2,   3],
	//  [ 4,  5,  6,  7],   [ 4, 100, 101,   7],
	//  [ 8,  9, 10, 11],   [ 8, 102, 103,  11],
	//  [12, 13, 14, 15]]   [12,  13,  14,  15]]
	expected := []float32{0, 1, 2, 3, 4, 100, 101, 7, 8, 102, 103, 11, 12, 13, 14, 15}
	for i := range expected {
		if math.Abs(float64(outputData[i]-expected[i])) > 1e-5 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], expected[i])
		}
	}
}

// TestDynamicUpdateSlice3D tests updating a 3D slice.
func TestDynamicUpdateSlice3D(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_dynamic_update_slice_3d")
	mainFn := builder.Main()

	// Create operand: 2x3x4 tensor
	operandShape := shapes.Make(dtypes.Float32, 2, 3, 4)
	operand, err := mainFn.Parameter("operand", operandShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Create update: 1x2x2 tensor
	updateShape := shapes.Make(dtypes.Float32, 1, 2, 2)
	update, err := mainFn.Parameter("update", updateShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Create start indices (1-element tensors since CoreML doesn't support scalar inputs)
	startIndexShape := shapes.Make(dtypes.Int32, 1)
	startIndex0, _ := mainFn.Parameter("start_index_0", startIndexShape, nil)
	startIndex1, _ := mainFn.Parameter("start_index_1", startIndexShape, nil)
	startIndex2, _ := mainFn.Parameter("start_index_2", startIndexShape, nil)

	// DynamicUpdateSlice
	result, err := mainFn.DynamicUpdateSlice(operand, update, []backends.Value{startIndex0, startIndex1, startIndex2})
	if err != nil {
		t.Fatalf("DynamicUpdateSlice() failed: %v", err)
	}

	err = mainFn.Return([]backends.Value{result}, nil)
	if err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Create input buffers
	operandData := make([]float32, 24)
	for i := range operandData {
		operandData[i] = float32(i)
	}

	updateData := []float32{100, 101, 102, 103}

	// Start at position (1, 0, 1)
	startIndex0Data := []int32{1}
	startIndex1Data := []int32{0}
	startIndex2Data := []int32{1}

	operandBuf, _ := backend.BufferFromFlatData(0, operandData, operandShape)
	updateBuf, _ := backend.BufferFromFlatData(0, updateData, updateShape)
	startIndex0Buf, _ := backend.BufferFromFlatData(0, startIndex0Data, startIndexShape)
	startIndex1Buf, _ := backend.BufferFromFlatData(0, startIndex1Data, startIndexShape)
	startIndex2Buf, _ := backend.BufferFromFlatData(0, startIndex2Data, startIndexShape)

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{operandBuf, updateBuf, startIndex0Buf, startIndex1Buf, startIndex2Buf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	// Verify output shape is correct
	outputShape, err := backend.BufferShape(outputs[0])
	if err != nil {
		t.Fatalf("BufferShape() failed: %v", err)
	}
	if !outputShape.Equal(operandShape) {
		t.Errorf("Output shape = %v, want %v", outputShape, operandShape)
	}

	// Verify specific updated positions
	outputData := make([]float32, 24)
	err = backend.BufferToFlatData(outputs[0], outputData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// The update of shape [1,2,2] at position [1,0,1] should update:
	// [1,0,1], [1,0,2], [1,1,1], [1,1,2]
	// In flat indices for shape [2,3,4]: index = i*12 + j*4 + k
	// [1,0,1] = 1*12 + 0*4 + 1 = 13
	// [1,0,2] = 1*12 + 0*4 + 2 = 14
	// [1,1,1] = 1*12 + 1*4 + 1 = 17
	// [1,1,2] = 1*12 + 1*4 + 2 = 18

	if math.Abs(float64(outputData[13]-100)) > 1e-5 {
		t.Errorf("outputData[13] = %f, want 100", outputData[13])
	}
	if math.Abs(float64(outputData[14]-101)) > 1e-5 {
		t.Errorf("outputData[14] = %f, want 101", outputData[14])
	}
	if math.Abs(float64(outputData[17]-102)) > 1e-5 {
		t.Errorf("outputData[17] = %f, want 102", outputData[17])
	}
	if math.Abs(float64(outputData[18]-103)) > 1e-5 {
		t.Errorf("outputData[18] = %f, want 103", outputData[18])
	}
}

// TestDynamicUpdateSliceValidation tests error cases.
func TestDynamicUpdateSliceValidation(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_dynamic_update_slice_validation")
	mainFn := builder.Main()

	// Create operand and update with mismatched ranks
	operandShape := shapes.Make(dtypes.Float32, 4, 4)
	operand, _ := mainFn.Parameter("operand", operandShape, nil)

	// Update with wrong rank
	updateShape := shapes.Make(dtypes.Float32, 2) // Should be 2D
	update, _ := mainFn.Parameter("update", updateShape, nil)

	startIndexShape := shapes.Make(dtypes.Int32, 1)
	startIndex0, _ := mainFn.Parameter("start_index_0", startIndexShape, nil)
	startIndex1, _ := mainFn.Parameter("start_index_1", startIndexShape, nil)

	// Should fail because update rank doesn't match operand rank
	_, err = mainFn.DynamicUpdateSlice(operand, update, []backends.Value{startIndex0, startIndex1})
	if err == nil {
		t.Error("DynamicUpdateSlice() should fail with mismatched ranks")
	}
}
