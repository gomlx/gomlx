/*
 *	Copyright 2025 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package train

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/bucketing"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

func TestBucketShapes(t *testing.T) {
	tests := []struct {
		name           string
		config         BucketConfig
		inputShapes    []shapes.Shape
		expectedShapes []shapes.Shape
	}{
		{
			name: "Pow2 bucketing on first axis",
			config: BucketConfig{
				Strategy:    bucketing.Pow2(),
				DynamicAxes: []int{0},
			},
			inputShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 3, 10),
				shapes.Make(dtypes.Float32, 5, 20),
			},
			expectedShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 4, 10),  // 3 -> 4
				shapes.Make(dtypes.Float32, 8, 20),  // 5 -> 8
			},
		},
		{
			name: "Linear bucketing with step 8",
			config: BucketConfig{
				Strategy:    bucketing.Linear(8),
				DynamicAxes: []int{0},
			},
			inputShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 5, 10),
				shapes.Make(dtypes.Int32, 12, 20),
			},
			expectedShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 8, 10),   // 5 -> 8
				shapes.Make(dtypes.Int32, 16, 20),    // 12 -> 16
			},
		},
		{
			name: "No bucketing",
			config: BucketConfig{
				Strategy:    bucketing.None(),
				DynamicAxes: []int{0},
			},
			inputShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 3, 10),
			},
			expectedShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 3, 10),  // unchanged
			},
		},
		{
			name: "Bucket multiple axes",
			config: BucketConfig{
				Strategy:    bucketing.Pow2(),
				DynamicAxes: []int{0, 1},
			},
			inputShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 3, 5),
			},
			expectedShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 4, 8),  // 3 -> 4, 5 -> 8
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bucketedShapes, bucketKey := BucketShapes(tt.config, tt.inputShapes...)

			// Check number of shapes
			if len(bucketedShapes) != len(tt.expectedShapes) {
				t.Fatalf("expected %d shapes, got %d", len(tt.expectedShapes), len(bucketedShapes))
			}

			// Check each shape
			for i, expected := range tt.expectedShapes {
				got := bucketedShapes[i]
				if !got.Equal(expected) {
					t.Errorf("shape %d: expected %s, got %s", i, expected, got)
				}
			}

			// Check that bucket key is not empty
			if bucketKey == "" {
				t.Error("bucket key should not be empty")
			}
		})
	}
}

func TestPadTensor(t *testing.T) {
	tests := []struct {
		name         string
		originalData []float32
		originalDims []int
		targetDims   []int
		padValue     any
		expectedData []float32
	}{
		{
			name:         "Pad first dimension with zeros",
			originalData: []float32{1, 2, 3, 4},
			originalDims: []int{2, 2},
			targetDims:   []int{4, 2},
			padValue:     nil, // zero padding
			expectedData: []float32{1, 2, 3, 4, 0, 0, 0, 0},
		},
		{
			name:         "Pad with custom value",
			originalData: []float32{1, 2, 3, 4},
			originalDims: []int{2, 2},
			targetDims:   []int{4, 2},
			padValue:     float32(-1.0),
			expectedData: []float32{1, 2, 3, 4, -1, -1, -1, -1},
		},
		{
			name:         "No padding needed",
			originalData: []float32{1, 2, 3, 4},
			originalDims: []int{2, 2},
			targetDims:   []int{2, 2},
			padValue:     nil,
			expectedData: []float32{1, 2, 3, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create original tensor
			originalShape := shapes.Make(dtypes.Float32, tt.originalDims...)
			original := tensors.FromShape(originalShape)
			original.MustMutableFlatData(func(flat any) {
				data := flat.([]float32)
				copy(data, tt.originalData)
			})

			// Pad tensor
			targetShape := shapes.Make(dtypes.Float32, tt.targetDims...)
			padded, err := PadTensor(original, targetShape, tt.padValue)
			if err != nil {
				t.Fatalf("PadTensor failed: %v", err)
			}

			// Verify shape
			if !padded.Shape().Equal(targetShape) {
				t.Errorf("expected shape %s, got %s", targetShape, padded.Shape())
			}

			// Verify data
			padded.ConstFlatData(func(flat any) {
				data := flat.([]float32)
				if len(data) != len(tt.expectedData) {
					t.Fatalf("expected %d elements, got %d", len(tt.expectedData), len(data))
				}
				for i, expected := range tt.expectedData {
					if data[i] != expected {
						t.Errorf("element %d: expected %f, got %f", i, expected, data[i])
					}
				}
			})
		})
	}
}

func TestBucketTensors(t *testing.T) {
	config := BucketConfig{
		Strategy:    bucketing.Pow2(),
		DynamicAxes: []int{0},
		PadValue:    float32(-1.0),
	}

	// Create test tensors
	tensor1 := tensors.FromShape(shapes.Make(dtypes.Float32, 3, 2))
	tensor1.MustMutableFlatData(func(flat any) {
		data := flat.([]float32)
		for i := range data {
			data[i] = float32(i)
		}
	})

	tensor2 := tensors.FromShape(shapes.Make(dtypes.Float32, 5, 3))
	tensor2.MustMutableFlatData(func(flat any) {
		data := flat.([]float32)
		for i := range data {
			data[i] = float32(i + 10)
		}
	})

	// Apply bucketing
	bucketed, info, err := BucketTensors(config, tensor1, tensor2)
	if err != nil {
		t.Fatalf("BucketTensors failed: %v", err)
	}

	// Verify we got the right number of tensors
	if len(bucketed) != 2 {
		t.Fatalf("expected 2 tensors, got %d", len(bucketed))
	}

	// Verify info
	if len(info.OriginalShapes) != 2 {
		t.Errorf("expected 2 original shapes, got %d", len(info.OriginalShapes))
	}
	if len(info.BucketedShapes) != 2 {
		t.Errorf("expected 2 bucketed shapes, got %d", len(info.BucketedShapes))
	}
	if info.BucketKey == "" {
		t.Error("bucket key should not be empty")
	}

	// Verify first tensor (3, 2) -> (4, 2)
	expectedShape1 := shapes.Make(dtypes.Float32, 4, 2)
	if !bucketed[0].Shape().Equal(expectedShape1) {
		t.Errorf("tensor1: expected shape %s, got %s", expectedShape1, bucketed[0].Shape())
	}

	// Verify second tensor (5, 3) -> (8, 3)
	expectedShape2 := shapes.Make(dtypes.Float32, 8, 3)
	if !bucketed[1].Shape().Equal(expectedShape2) {
		t.Errorf("tensor2: expected shape %s, got %s", expectedShape2, bucketed[1].Shape())
	}

	// Verify that padding used the custom value
	bucketed[0].ConstFlatData(func(flat any) {
		data := flat.([]float32)
		// Original data should be preserved
		for i := 0; i < 6; i++ {
			if data[i] != float32(i) {
				t.Errorf("tensor1 element %d: expected %f, got %f", i, float32(i), data[i])
			}
		}
		// Padded data should be -1.0
		for i := 6; i < len(data); i++ {
			if data[i] != -1.0 {
				t.Errorf("tensor1 padded element %d: expected -1.0, got %f", i, data[i])
			}
		}
	})
}

func TestBucketKey(t *testing.T) {
	config := BucketConfig{
		Strategy:    bucketing.Pow2(),
		DynamicAxes: []int{0},
	}

	shapes1 := []shapes.Shape{
		shapes.Make(dtypes.Float32, 3, 10),
		shapes.Make(dtypes.Float32, 5, 20),
	}

	shapes2 := []shapes.Shape{
		shapes.Make(dtypes.Float32, 4, 10),
		shapes.Make(dtypes.Float32, 8, 20),
	}

	// Get bucket keys
	_, key1 := BucketShapes(config, shapes1...)
	_, key2 := BucketShapes(config, shapes2...)

	// Keys should be the same because 3->4 and 4->4, 5->8 and 8->8
	if key1 != key2 {
		t.Errorf("expected same bucket key, got %q and %q", key1, key2)
	}

	// Different shapes should produce different keys
	shapes3 := []shapes.Shape{
		shapes.Make(dtypes.Float32, 2, 10),
	}
	_, key3 := BucketShapes(config, shapes3...)
	if key1 == key3 {
		t.Error("expected different bucket keys for different shapes")
	}
}

func TestPadTensorErrors(t *testing.T) {
	t.Run("dtype mismatch", func(t *testing.T) {
		original := tensors.FromShape(shapes.Make(dtypes.Float32, 2, 2))
		target := shapes.Make(dtypes.Int32, 4, 2)
		_, err := PadTensor(original, target, nil)
		if err == nil {
			t.Error("expected error for dtype mismatch")
		}
	})

	t.Run("rank mismatch", func(t *testing.T) {
		original := tensors.FromShape(shapes.Make(dtypes.Float32, 2, 2))
		target := shapes.Make(dtypes.Float32, 4, 2, 2)
		_, err := PadTensor(original, target, nil)
		if err == nil {
			t.Error("expected error for rank mismatch")
		}
	})

	t.Run("target smaller than original", func(t *testing.T) {
		original := tensors.FromShape(shapes.Make(dtypes.Float32, 4, 2))
		target := shapes.Make(dtypes.Float32, 2, 2)
		_, err := PadTensor(original, target, nil)
		if err == nil {
			t.Error("expected error when target is smaller than original")
		}
	})

	t.Run("wrong pad value type", func(t *testing.T) {
		original := tensors.FromShape(shapes.Make(dtypes.Float32, 2, 2))
		target := shapes.Make(dtypes.Float32, 4, 2)
		_, err := PadTensor(original, target, int32(0)) // Wrong type
		if err == nil {
			t.Error("expected error for wrong pad value type")
		}
	})
}
