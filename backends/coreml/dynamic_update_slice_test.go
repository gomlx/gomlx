//go:build darwin

package coreml

import (
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// TestDynamicUpdateSlice1D tests updating a 1D slice at a dynamic position
func TestDynamicUpdateSlice1D(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_dynamic_update_slice_1d")

	// Operand: [10] tensor with values [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	operandData := []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	operand, err := builder.Constant(operandData, 10)
	if err != nil {
		t.Fatalf("Constant() for operand failed: %v", err)
	}

	// Update: [3] tensor with values [100, 101, 102]
	updateData := []float32{100, 101, 102}
	update, err := builder.Constant(updateData, 3)
	if err != nil {
		t.Fatalf("Constant() for update failed: %v", err)
	}

	// Start index: 2 (dynamic)
	startIdx, err := builder.Constant([]int32{2})
	if err != nil {
		t.Fatalf("Constant() for startIdx failed: %v", err)
	}

	// Perform dynamic update slice
	result, err := builder.DynamicUpdateSlice(operand, update, []backends.Op{startIdx})
	if err != nil {
		t.Fatalf("DynamicUpdateSlice() failed: %v", err)
	}

	// Build and execute
	exec, err := builder.Compile([]backends.Op{result}, nil)
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	outputs, err := exec.Execute([]backends.Buffer{}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	// Expected: [0, 1, 100, 101, 102, 5, 6, 7, 8, 9]
	expected := []float32{0, 1, 100, 101, 102, 5, 6, 7, 8, 9}

	// Read result from buffer
	resultData := make([]float32, 10)
	err = backend.BufferToFlatData(outputs[0], resultData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	if len(resultData) != len(expected) {
		t.Fatalf("Result length mismatch: got %d, want %d", len(resultData), len(expected))
	}

	for i := range expected {
		if resultData[i] != expected[i] {
			t.Errorf("Result[%d] = %f, want %f", i, resultData[i], expected[i])
		}
	}
}

// TestDynamicUpdateSlice2D tests updating a 2D slice at a dynamic position
func TestDynamicUpdateSlice2D(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_dynamic_update_slice_2d")

	// Operand: [5, 6] tensor filled with 0s
	operandData := make([]float32, 5*6)
	for i := range operandData {
		operandData[i] = 0
	}
	operand, err := builder.Constant(operandData, 5, 6)
	if err != nil {
		t.Fatalf("Constant() for operand failed: %v", err)
	}

	// Update: [2, 3] tensor with values [[1, 2, 3], [4, 5, 6]]
	updateData := []float32{1, 2, 3, 4, 5, 6}
	update, err := builder.Constant(updateData, 2, 3)
	if err != nil {
		t.Fatalf("Constant() for update failed: %v", err)
	}

	// Start indices: [1, 2] (dynamic)
	startIdx0, err := builder.Constant([]int32{1})
	if err != nil {
		t.Fatalf("Constant() for startIdx0 failed: %v", err)
	}
	startIdx1, err := builder.Constant([]int32{2})
	if err != nil {
		t.Fatalf("Constant() for startIdx1 failed: %v", err)
	}

	// Perform dynamic update slice
	result, err := builder.DynamicUpdateSlice(operand, update, []backends.Op{startIdx0, startIdx1})
	if err != nil {
		t.Fatalf("DynamicUpdateSlice() failed: %v", err)
	}

	// Build and execute
	exec, err := builder.Compile([]backends.Op{result}, nil)
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	outputs, err := exec.Execute([]backends.Buffer{}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	// Expected: First row all 0s, second row: [0, 0, 1, 2, 3, 0], third row: [0, 0, 4, 5, 6, 0], etc.
	expected := []float32{
		0, 0, 0, 0, 0, 0,
		0, 0, 1, 2, 3, 0,
		0, 0, 4, 5, 6, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
	}

	// Read result from buffer
	resultData := make([]float32, 5*6)
	err = backend.BufferToFlatData(outputs[0], resultData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	if len(resultData) != len(expected) {
		t.Fatalf("Result length mismatch: got %d, want %d", len(resultData), len(expected))
	}

	for i := range expected {
		if resultData[i] != expected[i] {
			t.Errorf("Result[%d] = %f, want %f", i, resultData[i], expected[i])
		}
	}
}

// TestDynamicUpdateSlice3D tests updating a 3D slice
func TestDynamicUpdateSlice3D(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_dynamic_update_slice_3d")

	// Operand: [3, 4, 5] tensor filled with 0s
	operandData := make([]float32, 3*4*5)
	operand, err := builder.Constant(operandData, 3, 4, 5)
	if err != nil {
		t.Fatalf("Constant() for operand failed: %v", err)
	}

	// Update: [2, 2, 2] tensor with sequential values
	updateData := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	update, err := builder.Constant(updateData, 2, 2, 2)
	if err != nil {
		t.Fatalf("Constant() for update failed: %v", err)
	}

	// Start indices: [0, 1, 1] (dynamic)
	startIdx0, err := builder.Constant([]int32{0})
	if err != nil {
		t.Fatalf("Constant() for startIdx0 failed: %v", err)
	}
	startIdx1, err := builder.Constant([]int32{1})
	if err != nil {
		t.Fatalf("Constant() for startIdx1 failed: %v", err)
	}
	startIdx2, err := builder.Constant([]int32{1})
	if err != nil {
		t.Fatalf("Constant() for startIdx2 failed: %v", err)
	}

	// Perform dynamic update slice
	result, err := builder.DynamicUpdateSlice(operand, update, []backends.Op{startIdx0, startIdx1, startIdx2})
	if err != nil {
		t.Fatalf("DynamicUpdateSlice() failed: %v", err)
	}

	// Build and execute
	exec, err := builder.Compile([]backends.Op{result}, nil)
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	outputs, err := exec.Execute([]backends.Buffer{}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	// Read result from buffer
	resultData := make([]float32, 3*4*5)
	err = backend.BufferToFlatData(outputs[0], resultData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	resultShape, err := backend.BufferShape(outputs[0])
	if err != nil {
		t.Fatalf("BufferShape() failed: %v", err)
	}

	// Verify shape
	expectedShape := shapes.Make(dtypes.Float32, 3, 4, 5)
	if !resultShape.Equal(expectedShape) {
		t.Errorf("Result shape = %v, want [3, 4, 5]", resultShape)
	}

	// Verify that the update region has the correct values
	// The update starts at [0, 1, 1] and has size [2, 2, 2]
	// Check a few key positions
	getValue := func(i, j, k int) float32 {
		return resultData[i*20+j*5+k]
	}

	// Position [0, 1, 1] should be 1 (first element of update)
	if val := getValue(0, 1, 1); val != 1 {
		t.Errorf("Position [0, 1, 1] = %f, want 1", val)
	}

	// Position [1, 2, 2] should be 8 (last element of update)
	if val := getValue(1, 2, 2); val != 8 {
		t.Errorf("Position [1, 2, 2] = %f, want 8", val)
	}

	// Position [2, 0, 0] should be 0 (outside update region)
	if val := getValue(2, 0, 0); val != 0 {
		t.Errorf("Position [2, 0, 0] = %f, want 0", val)
	}
}

// TestDynamicUpdateSliceValidation tests error cases
func TestDynamicUpdateSliceValidation(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_dynamic_update_slice_validation")

	// Operand: [10] tensor
	operandData := make([]float32, 10)
	operand, err := builder.Constant(operandData, 10)
	if err != nil {
		t.Fatalf("Constant() for operand failed: %v", err)
	}

	// Update: [3] tensor
	updateData := []float32{1, 2, 3}
	update, err := builder.Constant(updateData, 3)
	if err != nil {
		t.Fatalf("Constant() for update failed: %v", err)
	}

	startIdx, err := builder.Constant([]int32{0})
	if err != nil {
		t.Fatalf("Constant() for startIdx failed: %v", err)
	}

	// Test: Wrong number of start indices
	t.Run("WrongNumIndices", func(t *testing.T) {
		_, err := builder.DynamicUpdateSlice(operand, update, []backends.Op{startIdx, startIdx})
		if err == nil {
			t.Error("Expected error for wrong number of indices, got nil")
		}
	})

	// Test: Update larger than operand
	t.Run("UpdateTooLarge", func(t *testing.T) {
		largeUpdate, err := builder.Constant(make([]float32, 15), 15)
		if err != nil {
			t.Fatalf("Constant() for largeUpdate failed: %v", err)
		}
		_, err = builder.DynamicUpdateSlice(operand, largeUpdate, []backends.Op{startIdx})
		if err == nil {
			t.Error("Expected error for update larger than operand, got nil")
		}
	})
}
