//go:build darwin && cgo

package coreml

import (
	"math"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// TestIota tests the Iota operation.
func TestIota(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	testCases := []struct {
		name     string
		shape    shapes.Shape
		iotaAxis int
		expected []float32
	}{
		{
			name:     "1D",
			shape:    shapes.Make(dtypes.Float32, 5),
			iotaAxis: 0,
			expected: []float32{0, 1, 2, 3, 4},
		},
		{
			name:     "2D_axis0",
			shape:    shapes.Make(dtypes.Float32, 2, 3),
			iotaAxis: 0,
			// [[0, 0, 0], [1, 1, 1]]
			expected: []float32{0, 0, 0, 1, 1, 1},
		},
		{
			name:     "2D_axis1",
			shape:    shapes.Make(dtypes.Float32, 2, 3),
			iotaAxis: 1,
			// [[0, 1, 2], [0, 1, 2]]
			expected: []float32{0, 1, 2, 0, 1, 2},
		},
		{
			name:     "3D_axis0",
			shape:    shapes.Make(dtypes.Float32, 2, 2, 2),
			iotaAxis: 0,
			// [[[0, 0], [0, 0]], [[1, 1], [1, 1]]]
			expected: []float32{0, 0, 0, 0, 1, 1, 1, 1},
		},
		{
			name:     "3D_axis1",
			shape:    shapes.Make(dtypes.Float32, 2, 2, 2),
			iotaAxis: 1,
			// [[[0, 0], [1, 1]], [[0, 0], [1, 1]]]
			expected: []float32{0, 0, 1, 1, 0, 0, 1, 1},
		},
		{
			name:     "3D_axis2",
			shape:    shapes.Make(dtypes.Float32, 2, 2, 2),
			iotaAxis: 2,
			// [[[0, 1], [0, 1]], [[0, 1], [0, 1]]]
			expected: []float32{0, 1, 0, 1, 0, 1, 0, 1},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			builder := backend.Builder("test_iota_" + tc.name)
			mainFn := builder.Main()

			// Create Iota
			y, err := mainFn.Iota(tc.shape, tc.iotaAxis)
			if err != nil {
				t.Fatalf("Iota() failed: %v", err)
			}

			err = mainFn.Return([]backends.Value{y}, nil)
			if err != nil {
				t.Fatalf("Return() failed: %v", err)
			}

			exec, err := builder.Compile()
			if err != nil {
				t.Fatalf("Compile() failed: %v", err)
			}
			defer exec.Finalize()

			// Verify output shape
			outputShapes := exec.Outputs()
			if len(outputShapes) != 1 {
				t.Fatalf("Expected 1 output, got %d", len(outputShapes))
			}
			if !outputShapes[0].Equal(tc.shape) {
				t.Errorf("Output shape = %v, want %v", outputShapes[0], tc.shape)
			}

			// Execute (no inputs needed for Iota)
			outputs, err := exec.Execute(nil, nil, 0)
			if err != nil {
				t.Fatalf("Execute() failed: %v", err)
			}

			outputData := make([]float32, tc.shape.Size())
			err = backend.BufferToFlatData(outputs[0], outputData)
			if err != nil {
				t.Fatalf("BufferToFlatData() failed: %v", err)
			}

			for i := range tc.expected {
				if math.Abs(float64(outputData[i]-tc.expected[i])) > 1e-5 {
					t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], tc.expected[i])
				}
			}
		})
	}
}

// TestIotaInt32 tests Iota with Int32 dtype.
func TestIotaInt32(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_iota_int32")
	mainFn := builder.Main()

	shape := shapes.Make(dtypes.Int32, 4)
	y, err := mainFn.Iota(shape, 0)
	if err != nil {
		t.Fatalf("Iota() failed: %v", err)
	}

	err = mainFn.Return([]backends.Value{y}, nil)
	if err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	outputs, err := exec.Execute(nil, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	outputData := make([]int32, 4)
	err = backend.BufferToFlatData(outputs[0], outputData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	expected := []int32{0, 1, 2, 3}
	for i := range expected {
		if outputData[i] != expected[i] {
			t.Errorf("outputData[%d] = %d, want %d", i, outputData[i], expected[i])
		}
	}
}

// TestIotaValidation tests Iota validation errors.
func TestIotaValidation(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_iota_validation")
	mainFn := builder.Main()

	// Test invalid axis
	shape := shapes.Make(dtypes.Float32, 2, 3)
	_, err = mainFn.Iota(shape, 5) // axis 5 is out of bounds for rank 2
	if err == nil {
		t.Error("Iota() should fail with invalid axis")
	}

	// Test negative axis
	_, err = mainFn.Iota(shape, -1)
	if err == nil {
		t.Error("Iota() should fail with negative axis")
	}

	// Test scalar shape
	scalarShape := shapes.Make(dtypes.Float32)
	_, err = mainFn.Iota(scalarShape, 0)
	if err == nil {
		t.Error("Iota() should fail with scalar shape")
	}
}
