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

	"github.com/gomlx/gomlx/pkg/core/tensors/bucketing"
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

func TestGeneratePadMask(t *testing.T) {
	tests := []struct {
		name            string
		originalShapes  []shapes.Shape
		bucketedShapes  []shapes.Shape
		wantErr         bool
		validateMask    func(t *testing.T, masks []*tensors.Tensor)
	}{
		{
			name: "1D mask - batch dimension",
			originalShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 3),
			},
			bucketedShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 4),
			},
			wantErr: false,
			validateMask: func(t *testing.T, masks []*tensors.Tensor) {
				if len(masks) != 1 {
					t.Fatalf("expected 1 mask, got %d", len(masks))
				}
				mask := masks[0]
				expectedShape := shapes.Make(dtypes.Bool, 4)
				if !mask.Shape().Equal(expectedShape) {
					t.Errorf("expected shape %s, got %s", expectedShape, mask.Shape())
				}
				// Verify mask values: [true, true, true, false]
				mask.ConstFlatData(func(flat any) {
					data := flat.([]bool)
					expected := []bool{true, true, true, false}
					for i, exp := range expected {
						if data[i] != exp {
							t.Errorf("mask[%d]: expected %v, got %v", i, exp, data[i])
						}
					}
				})
			},
		},
		{
			name: "2D mask - batch and sequence",
			originalShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 3, 10),
			},
			bucketedShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 4, 16),
			},
			wantErr: false,
			validateMask: func(t *testing.T, masks []*tensors.Tensor) {
				mask := masks[0]
				expectedShape := shapes.Make(dtypes.Bool, 4, 16)
				if !mask.Shape().Equal(expectedShape) {
					t.Errorf("expected shape %s, got %s", expectedShape, mask.Shape())
				}
				// Verify mask values
				mask.ConstFlatData(func(flat any) {
					data := flat.([]bool)
					// Check first 3 rows (valid batch elements)
					for batch := 0; batch < 3; batch++ {
						for seq := 0; seq < 16; seq++ {
							idx := batch*16 + seq
							expected := seq < 10 // true for seq < 10, false for seq >= 10
							if data[idx] != expected {
								t.Errorf("mask[%d,%d]: expected %v, got %v", batch, seq, expected, data[idx])
							}
						}
					}
					// Check 4th row (padded batch element) - all false
					for seq := 0; seq < 16; seq++ {
						idx := 3*16 + seq
						if data[idx] != false {
							t.Errorf("mask[3,%d]: expected false (padded batch), got %v", seq, data[idx])
						}
					}
				})
			},
		},
		{
			name: "No padding needed - shapes equal",
			originalShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 4, 8),
			},
			bucketedShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 4, 8),
			},
			wantErr: false,
			validateMask: func(t *testing.T, masks []*tensors.Tensor) {
				mask := masks[0]
				// All values should be true when no padding
				mask.ConstFlatData(func(flat any) {
					data := flat.([]bool)
					for i, val := range data {
						if !val {
							t.Errorf("mask[%d]: expected true (no padding), got false", i)
						}
					}
				})
			},
		},
		{
			name: "Multiple tensors with different padding",
			originalShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 3, 10),
				shapes.Make(dtypes.Int32, 5, 7),
			},
			bucketedShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 4, 16),
				shapes.Make(dtypes.Int32, 8, 8),
			},
			wantErr: false,
			validateMask: func(t *testing.T, masks []*tensors.Tensor) {
				if len(masks) != 2 {
					t.Fatalf("expected 2 masks, got %d", len(masks))
				}
				// Check first mask shape
				if !masks[0].Shape().Equal(shapes.Make(dtypes.Bool, 4, 16)) {
					t.Errorf("mask 0: unexpected shape %s", masks[0].Shape())
				}
				// Check second mask shape
				if !masks[1].Shape().Equal(shapes.Make(dtypes.Bool, 8, 8)) {
					t.Errorf("mask 1: unexpected shape %s", masks[1].Shape())
				}
			},
		},
		{
			name: "3D tensor",
			originalShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 2, 3, 4),
			},
			bucketedShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 4, 4, 8),
			},
			wantErr: false,
			validateMask: func(t *testing.T, masks []*tensors.Tensor) {
				mask := masks[0]
				expectedShape := shapes.Make(dtypes.Bool, 4, 4, 8)
				if !mask.Shape().Equal(expectedShape) {
					t.Errorf("expected shape %s, got %s", expectedShape, mask.Shape())
				}
				// Sample checks: valid region should be true
				mask.ConstFlatData(func(flat any) {
					data := flat.([]bool)
					// Check position [0,0,0] - should be true
					if !data[0] {
						t.Error("position [0,0,0] should be true")
					}
					// Check position [1,2,3] - should be true (within 2,3,4)
					idx := 1*4*8 + 2*8 + 3
					if !data[idx] {
						t.Error("position [1,2,3] should be true")
					}
					// Check position [3,0,0] - should be false (batch padding)
					idx = 3*4*8
					if data[idx] {
						t.Error("position [3,0,0] should be false (padded batch)")
					}
				})
			},
		},
		{
			name: "Error: shape count mismatch",
			originalShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 3, 10),
			},
			bucketedShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 4, 16),
				shapes.Make(dtypes.Float32, 8, 8),
			},
			wantErr: true,
		},
		{
			name: "Error: rank mismatch",
			originalShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 3, 10),
			},
			bucketedShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 4, 16, 1),
			},
			wantErr: true,
		},
		{
			name: "Error: bucketed smaller than original",
			originalShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 10, 10),
			},
			bucketedShapes: []shapes.Shape{
				shapes.Make(dtypes.Float32, 8, 10),
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			masks, err := GeneratePadMask(tt.originalShapes, tt.bucketedShapes)

			if tt.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("GeneratePadMask failed: %v", err)
			}

			if tt.validateMask != nil {
				tt.validateMask(t, masks)
			}
		})
	}
}

func TestBucketTensorsWithMask(t *testing.T) {
	config := BucketConfig{
		Strategy:     bucketing.Pow2(),
		DynamicAxes:  []int{0, 1},
		PadValue:     float32(-1.0),
		GenerateMask: true,
	}

	// Create test tensor with shape [3, 5]
	tensor := tensors.FromShape(shapes.Make(dtypes.Float32, 3, 5))
	tensor.MustMutableFlatData(func(flat any) {
		data := flat.([]float32)
		for i := range data {
			data[i] = float32(i)
		}
	})

	// Apply bucketing with mask generation
	bucketed, info, err := BucketTensors(config, tensor)
	if err != nil {
		t.Fatalf("BucketTensors failed: %v", err)
	}

	// Verify bucketed tensor shape [3,5] -> [4,8]
	expectedShape := shapes.Make(dtypes.Float32, 4, 8)
	if !bucketed[0].Shape().Equal(expectedShape) {
		t.Errorf("expected shape %s, got %s", expectedShape, bucketed[0].Shape())
	}

	// Verify mask was generated
	if info.PaddingMasks == nil {
		t.Fatal("PaddingMasks should not be nil when GenerateMask=true")
	}
	if len(info.PaddingMasks) != 1 {
		t.Fatalf("expected 1 mask, got %d", len(info.PaddingMasks))
	}

	// Verify mask shape matches bucketed shape
	mask := info.PaddingMasks[0]
	expectedMaskShape := shapes.Make(dtypes.Bool, 4, 8)
	if !mask.Shape().Equal(expectedMaskShape) {
		t.Errorf("expected mask shape %s, got %s", expectedMaskShape, mask.Shape())
	}

	// Verify mask values
	mask.ConstFlatData(func(flat any) {
		data := flat.([]bool)
		// Check valid region [0:3, 0:5] - should be true
		for batch := 0; batch < 3; batch++ {
			for seq := 0; seq < 5; seq++ {
				idx := batch*8 + seq
				if !data[idx] {
					t.Errorf("mask[%d,%d]: expected true (valid region), got false", batch, seq)
				}
			}
		}
		// Check sequence padding [0:3, 5:8] - should be false
		for batch := 0; batch < 3; batch++ {
			for seq := 5; seq < 8; seq++ {
				idx := batch*8 + seq
				if data[idx] {
					t.Errorf("mask[%d,%d]: expected false (sequence padding), got true", batch, seq)
				}
			}
		}
		// Check batch padding [3:4, :] - should be false
		for seq := 0; seq < 8; seq++ {
			idx := 3*8 + seq
			if data[idx] {
				t.Errorf("mask[3,%d]: expected false (batch padding), got true", seq)
			}
		}
	})
}

func TestBucketTensorsWithoutMask(t *testing.T) {
	config := BucketConfig{
		Strategy:     bucketing.Pow2(),
		DynamicAxes:  []int{0},
		GenerateMask: false, // explicitly false
	}

	tensor := tensors.FromShape(shapes.Make(dtypes.Float32, 3, 5))

	bucketed, info, err := BucketTensors(config, tensor)
	if err != nil {
		t.Fatalf("BucketTensors failed: %v", err)
	}

	// Verify mask was NOT generated
	if info.PaddingMasks != nil {
		t.Error("PaddingMasks should be nil when GenerateMask=false")
	}

	// Bucketing should still work
	if len(bucketed) != 1 {
		t.Errorf("expected 1 bucketed tensor, got %d", len(bucketed))
	}
}
