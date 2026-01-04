//go:build darwin

package coreml

import (
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
		iotaDim  int
		expected []int32
	}{
		{
			name:     "Iota_1D",
			shape:    shapes.Make(dtypes.Int32, 5),
			iotaDim:  0,
			expected: []int32{0, 1, 2, 3, 4},
		},
		{
			name:    "Iota_2D_Dim0",
			shape:   shapes.Make(dtypes.Int32, 3, 4),
			iotaDim: 0,
			expected: []int32{
				0, 0, 0, 0,
				1, 1, 1, 1,
				2, 2, 2, 2,
			},
		},
		{
			name:    "Iota_2D_Dim1",
			shape:   shapes.Make(dtypes.Int32, 3, 4),
			iotaDim: 1,
			expected: []int32{
				0, 1, 2, 3,
				0, 1, 2, 3,
				0, 1, 2, 3,
			},
		},
		{
			name:    "Iota_3D_Dim2",
			shape:   shapes.Make(dtypes.Int32, 2, 2, 3),
			iotaDim: 2,
			expected: []int32{
				0, 1, 2, // [0,0,:]
				0, 1, 2, // [0,1,:]
				0, 1, 2, // [1,0,:]
				0, 1, 2, // [1,1,:]
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			builder := backend.Builder("test_" + tc.name)

			// Create Iota tensor
			iota, err := builder.Iota(tc.shape, tc.iotaDim)
			if err != nil {
				t.Fatalf("Iota() failed: %v", err)
			}

			// Verify shape
			iotaShape, err := builder.OpShape(iota)
			if err != nil {
				t.Fatalf("OpShape() failed: %v", err)
			}
			if !iotaShape.Equal(tc.shape) {
				t.Errorf("OpShape(iota) = %v, want %v", iotaShape, tc.shape)
			}

			// Compile
			exec, err := builder.Compile([]backends.Op{iota}, nil)
			if err != nil {
				t.Fatalf("Compile() failed: %v", err)
			}
			defer exec.Finalize()

			// Execute
			outputs, err := exec.Execute([]backends.Buffer{}, nil, 0)
			if err != nil {
				t.Fatalf("Execute() failed: %v", err)
			}

			// Verify output
			outputData := make([]int32, tc.shape.Size())
			err = backend.BufferToFlatData(outputs[0], outputData)
			if err != nil {
				t.Fatalf("BufferToFlatData() failed: %v", err)
			}

			for i := range tc.expected {
				if outputData[i] != tc.expected[i] {
					t.Errorf("%s: outputData[%d] = %d, want %d", tc.name, i, outputData[i], tc.expected[i])
				}
			}
		})
	}
}
