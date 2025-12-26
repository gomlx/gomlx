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
	"fmt"
	"reflect"
	"strings"

	"github.com/gomlx/gomlx/pkg/core/bucketing"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
)

// BucketConfig configures how to bucket and pad tensors.
// This is useful for applying bucketing in data pipelines before passing tensors to Exec.
type BucketConfig struct {
	// Strategy determines how dimensions are bucketed (e.g., Pow2, Linear, Exponential).
	Strategy bucketing.Strategy

	// DynamicAxes specifies which axes to bucket. Default is []int{0} (first axis only).
	DynamicAxes []int

	// PadValue is the value to use for padding. Default is nil (zero padding).
	// Must be type-compatible with the tensor's dtype.
	PadValue any

	// GenerateMask indicates whether to generate padding masks.
	// When true, BucketTensors will return boolean masks indicating valid (non-padded) positions.
	GenerateMask bool
}

// BucketInfo contains information about the bucketing applied to tensors.
type BucketInfo struct {
	// OriginalShapes are the shapes of the input tensors before bucketing.
	OriginalShapes []shapes.Shape

	// BucketedShapes are the shapes after applying the bucketing strategy.
	BucketedShapes []shapes.Shape

	// BucketKey is a string representation of the bucketed shapes that can be used
	// as a map key for grouping batches with the same bucketed dimensions.
	BucketKey string

	// PaddingMasks contains boolean tensors indicating valid (non-padded) positions.
	// Each mask has the same shape as the corresponding bucketed tensor.
	// Values are true for original data positions, false for padded positions.
	// Only populated when BucketConfig.GenerateMask is true.
	PaddingMasks []*tensors.Tensor
}

// BucketTensors applies the bucketing strategy to tensors and pads them if necessary.
// Returns the padded tensors and bucketing information.
//
// Example without masks:
//
//	config := BucketConfig{
//	    Strategy: bucketing.Pow2(),
//	    DynamicAxes: []int{0, 1},
//	    PadValue: float32(-1.0),
//	}
//	paddedTensors, info, err := BucketTensors(config, tensor1, tensor2)
//	if err != nil {
//	    return err
//	}
//	// Use info.BucketKey to group batches
//
// Example with padding masks (for attention):
//
//	config := BucketConfig{
//	    Strategy: bucketing.Pow2(),
//	    DynamicAxes: []int{0, 1},  // bucket batch and sequence dims
//	    PadValue: float32(0.0),
//	    GenerateMask: true,  // generate masks for padded positions
//	}
//	paddedTensors, info, err := BucketTensors(config, sequences)
//	if err != nil {
//	    return err
//	}
//	// info.PaddingMasks[0] contains bool tensor: true=valid, false=padded
//	// Use with attention: attention.SetKeyMask(info.PaddingMasks[0])
func BucketTensors(config BucketConfig, inputTensors ...*tensors.Tensor) ([]*tensors.Tensor, *BucketInfo, error) {
	if len(inputTensors) == 0 {
		return nil, nil, errors.New("no tensors provided")
	}

	// Extract original shapes
	originalShapes := make([]shapes.Shape, len(inputTensors))
	for i, t := range inputTensors {
		if t == nil {
			return nil, nil, errors.Errorf("tensor %d is nil", i)
		}
		originalShapes[i] = t.Shape()
	}

	// Apply bucketing to get target shapes
	bucketedShapes, bucketKey := BucketShapes(config, originalShapes...)

	// Pad tensors to bucketed shapes
	result := make([]*tensors.Tensor, len(inputTensors))
	for i, t := range inputTensors {
		if originalShapes[i].Equal(bucketedShapes[i]) {
			// No padding needed, use original
			result[i] = t
		} else {
			// Pad tensor to bucketed shape
			padded, err := PadTensor(t, bucketedShapes[i], config.PadValue)
			if err != nil {
				return nil, nil, errors.WithMessagef(err, "failed to pad tensor %d", i)
			}
			result[i] = padded
		}
	}

	info := &BucketInfo{
		OriginalShapes: originalShapes,
		BucketedShapes: bucketedShapes,
		BucketKey:      bucketKey,
	}

	// Generate padding masks if requested
	if config.GenerateMask {
		masks, err := GeneratePadMask(originalShapes, bucketedShapes)
		if err != nil {
			return nil, nil, errors.WithMessage(err, "failed to generate padding masks")
		}
		info.PaddingMasks = masks
	}

	return result, info, nil
}

// BucketShapes applies bucketing to shapes without creating tensors.
// Useful for pre-computing bucket assignments.
// Returns the bucketed shapes and a string key for grouping same-bucket batches.
//
// Example:
//
//	config := BucketConfig{Strategy: bucketing.Pow2(), DynamicAxes: []int{0}}
//	bucketedShapes, key := BucketShapes(config, shape1, shape2)
func BucketShapes(config BucketConfig, inputShapes ...shapes.Shape) ([]shapes.Shape, string) {
	if config.Strategy == nil {
		config.Strategy = bucketing.None()
	}

	dynamicAxes := config.DynamicAxes
	if len(dynamicAxes) == 0 {
		dynamicAxes = []int{0} // Default: first axis
	}

	result := make([]shapes.Shape, len(inputShapes))
	for i, shape := range inputShapes {
		result[i] = shape.Clone()
		for _, axis := range dynamicAxes {
			adjustedAxis := axis
			if adjustedAxis < 0 {
				adjustedAxis = shape.Rank() + adjustedAxis
			}
			if adjustedAxis >= 0 && adjustedAxis < shape.Rank() {
				result[i].Dimensions[adjustedAxis] = config.Strategy.Bucket(shape.Dimensions[adjustedAxis])
			}
		}
	}

	// Generate bucket key from bucketed shapes
	bucketKey := makeBucketKey(result)

	return result, bucketKey
}

// makeBucketKey creates a string key from shapes for grouping same-bucket batches.
func makeBucketKey(shapes []shapes.Shape) string {
	parts := make([]string, len(shapes))
	for i, shape := range shapes {
		// Format: dtype[dim1,dim2,...]
		dims := make([]string, len(shape.Dimensions))
		for j, dim := range shape.Dimensions {
			dims[j] = fmt.Sprintf("%d", dim)
		}
		parts[i] = fmt.Sprintf("%s[%s]", shape.DType, strings.Join(dims, ","))
	}
	return strings.Join(parts, ";")
}

// PadTensor pads a single tensor to the target shape with the given pad value.
// The pad value can be nil (for zero padding) or a type-compatible value.
//
// Example:
//
//	padded, err := PadTensor(tensor, targetShape, float32(-1.0))
func PadTensor(t *tensors.Tensor, targetShape shapes.Shape, padValue any) (*tensors.Tensor, error) {
	if t == nil {
		return nil, errors.New("tensor is nil")
	}

	originalShape := t.Shape()

	// Validate shapes are compatible
	if originalShape.DType != targetShape.DType {
		return nil, errors.Errorf("dtype mismatch: original=%s, target=%s",
			originalShape.DType, targetShape.DType)
	}
	if originalShape.Rank() != targetShape.Rank() {
		return nil, errors.Errorf("rank mismatch: original=%d, target=%d",
			originalShape.Rank(), targetShape.Rank())
	}

	// Check that target shape is >= original in all dimensions
	for i := 0; i < originalShape.Rank(); i++ {
		if targetShape.Dimensions[i] < originalShape.Dimensions[i] {
			return nil, errors.Errorf("target shape dimension %d (%d) is smaller than original (%d)",
				i, targetShape.Dimensions[i], originalShape.Dimensions[i])
		}
	}

	// If shapes are equal, return clone
	if originalShape.Equal(targetShape) {
		return t.LocalClone()
	}

	// Create new tensor with target shape
	padded := tensors.FromShape(targetShape)

	// Initialize with pad value if provided
	if padValue != nil {
		if err := initTensorWithValue(padded, padValue); err != nil {
			return nil, errors.WithMessage(err, "failed to initialize padded tensor")
		}
	}

	// Copy original data to padded tensor
	if err := copyTensorData(t, padded); err != nil {
		return nil, errors.WithMessage(err, "failed to copy tensor data")
	}

	return padded, nil
}

// initTensorWithValue initializes all elements of a tensor with the given value.
func initTensorWithValue(t *tensors.Tensor, value any) error {
	var initErr error
	t.MustMutableFlatData(func(flat any) {
		switch data := flat.(type) {
		case []float32:
			v, ok := value.(float32)
			if !ok {
				initErr = errors.Errorf("pad value type %T does not match tensor type float32", value)
				return
			}
			for i := range data {
				data[i] = v
			}
		case []float64:
			v, ok := value.(float64)
			if !ok {
				initErr = errors.Errorf("pad value type %T does not match tensor type float64", value)
				return
			}
			for i := range data {
				data[i] = v
			}
		case []int32:
			v, ok := value.(int32)
			if !ok {
				initErr = errors.Errorf("pad value type %T does not match tensor type int32", value)
				return
			}
			for i := range data {
				data[i] = v
			}
		case []int64:
			v, ok := value.(int64)
			if !ok {
				initErr = errors.Errorf("pad value type %T does not match tensor type int64", value)
				return
			}
			for i := range data {
				data[i] = v
			}
		case []uint8:
			v, ok := value.(uint8)
			if !ok {
				initErr = errors.Errorf("pad value type %T does not match tensor type uint8", value)
				return
			}
			for i := range data {
				data[i] = v
			}
		case []bool:
			v, ok := value.(bool)
			if !ok {
				initErr = errors.Errorf("pad value type %T does not match tensor type bool", value)
				return
			}
			for i := range data {
				data[i] = v
			}
		default:
			initErr = errors.Errorf("unsupported tensor dtype for padding initialization: %T", flat)
		}
	})
	return initErr
}

// copyTensorData copies data from src to dst tensor, assuming dst is larger or equal in all dimensions.
func copyTensorData(src, dst *tensors.Tensor) error {
	// For now, we copy the flat data assuming dst is zero-initialized or value-initialized
	// The padding is always at the end of each axis
	var copyErr error
	dst.MustMutableFlatData(func(dstFlat any) {
		src.ConstFlatData(func(srcFlat any) {
			// Copy srcFlat to the beginning of dstFlat
			switch srcData := srcFlat.(type) {
			case []float32:
				dstData, ok := dstFlat.([]float32)
				if !ok {
					copyErr = errors.New("type mismatch in copyTensorData for float32")
					return
				}
				copy(dstData, srcData)
			case []float64:
				dstData, ok := dstFlat.([]float64)
				if !ok {
					copyErr = errors.New("type mismatch in copyTensorData for float64")
					return
				}
				copy(dstData, srcData)
			case []int32:
				dstData, ok := dstFlat.([]int32)
				if !ok {
					copyErr = errors.New("type mismatch in copyTensorData for int32")
					return
				}
				copy(dstData, srcData)
			case []int64:
				dstData, ok := dstFlat.([]int64)
				if !ok {
					copyErr = errors.New("type mismatch in copyTensorData for int64")
					return
				}
				copy(dstData, srcData)
			case []uint8:
				dstData, ok := dstFlat.([]uint8)
				if !ok {
					copyErr = errors.New("type mismatch in copyTensorData for uint8")
					return
				}
				copy(dstData, srcData)
			case []bool:
				dstData, ok := dstFlat.([]bool)
				if !ok {
					copyErr = errors.New("type mismatch in copyTensorData for bool")
					return
				}
				copy(dstData, srcData)
			default:
				// Use reflection for other types
				srcV := reflect.ValueOf(srcFlat)
				dstV := reflect.ValueOf(dstFlat)
				if srcV.Type() != dstV.Type() {
					copyErr = errors.Errorf("type mismatch: src=%T, dst=%T", srcFlat, dstFlat)
					return
				}
				reflect.Copy(dstV, srcV)
			}
		})
	})
	return copyErr
}

// GeneratePadMask creates boolean mask tensors indicating valid (non-padded) positions.
// For each pair of original and bucketed shapes, creates a mask where:
//   - true = original data position (valid)
//   - false = padded position (should be masked in attention)
//
// The masks can be used directly with attention layers:
//
//	attention.SetKeyMask(masks[0])
//
// Example:
//
//	originalShapes := []shapes.Shape{shapes.Make(dtypes.Float32, 3, 10)}
//	bucketedShapes := []shapes.Shape{shapes.Make(dtypes.Float32, 4, 16)}
//	masks, err := GeneratePadMask(originalShapes, bucketedShapes)
//	// masks[0] has shape [4, 16] with true for positions [0:3, 0:10], false elsewhere
func GeneratePadMask(originalShapes, bucketedShapes []shapes.Shape) ([]*tensors.Tensor, error) {
	if len(originalShapes) != len(bucketedShapes) {
		return nil, errors.Errorf("shape count mismatch: %d original vs %d bucketed",
			len(originalShapes), len(bucketedShapes))
	}

	masks := make([]*tensors.Tensor, len(originalShapes))

	for i := 0; i < len(originalShapes); i++ {
		original := originalShapes[i]
		bucketed := bucketedShapes[i]

		// Validate shapes are compatible
		if original.Rank() != bucketed.Rank() {
			return nil, errors.Errorf("shape %d: rank mismatch: original=%d, bucketed=%d",
				i, original.Rank(), bucketed.Rank())
		}

		// Check that bucketed shape is >= original in all dimensions
		for j := 0; j < original.Rank(); j++ {
			if bucketed.Dimensions[j] < original.Dimensions[j] {
				return nil, errors.Errorf("shape %d: bucketed dimension %d (%d) is smaller than original (%d)",
					i, j, bucketed.Dimensions[j], original.Dimensions[j])
			}
		}

		// Create mask with bucketed shape (using Bool dtype)
		maskShape := shapes.Make(dtypes.Bool, bucketed.Dimensions...)
		mask := tensors.FromShape(maskShape)

		// Initialize mask values
		var maskErr error
		mask.MustMutableFlatData(func(flat any) {
			data, ok := flat.([]bool)
			if !ok {
				maskErr = errors.New("mask tensor is not bool type")
				return
			}

			// Calculate strides for efficient indexing
			strides := make([]int, bucketed.Rank())
			stride := 1
			for j := bucketed.Rank() - 1; j >= 0; j-- {
				strides[j] = stride
				stride *= bucketed.Dimensions[j]
			}

			// Set mask values: true for valid positions, false for padded positions
			for flatIdx := 0; flatIdx < len(data); flatIdx++ {
				// Convert flat index to multi-dimensional index
				isValid := true
				remaining := flatIdx
				for j := 0; j < bucketed.Rank(); j++ {
					dimIdx := remaining / strides[j]
					remaining %= strides[j]

					// Check if this position is within original bounds
					if dimIdx >= original.Dimensions[j] {
						isValid = false
						break
					}
				}
				data[flatIdx] = isValid
			}
		})

		if maskErr != nil {
			return nil, errors.WithMessagef(maskErr, "failed to initialize mask %d", i)
		}

		masks[i] = mask
	}

	return masks, nil
}
