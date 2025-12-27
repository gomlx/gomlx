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

package graph

import (
	"github.com/gomlx/gomlx/backends"
)

// PadToBucketSize pads the tensor along specified axes to match the bucketed size.
// This is useful when using pattern caching with bucketing strategies to ensure
// that inputs match the expected bucketed dimensions.
//
// Parameters:
//   - x: Input tensor to pad
//   - fillValue: Value to use for padding (typically 0)
//   - strategy: Bucketing strategy to determine target sizes (Pow2Bucketing{}, LinearBucketing{Step: n}, or NoBucketing{})
//   - axes: Which axes to pad (default: all axes if not specified)
//
// Example with pattern caching:
//
//	// Create an executor with power-of-2 bucketing on the batch dimension
//	exec := MustNewExec(backend, func(x *Node) *Node {
//	    // Pad input to match bucketed size
//	    x = PadToPow2(x, Const(x.Graph(), float32(0)), 0)
//	    // ... rest of computation
//	    return result
//	}).WithPow2Bucketing()
//
// Simple example:
//
//	// Pad batch dimension to next power of 2
//	padded := PadToBucketSize(input, Const(g, float32(0)), Pow2Bucketing{}, 0)
//	// input [7, 512] -> padded [8, 512]
func PadToBucketSize(x *Node, fillValue *Node, strategy BucketingStrategy, axes ...int) *Node {
	shape := x.Shape()

	if len(axes) == 0 {
		// Default: pad all axes
		axes = make([]int, shape.Rank())
		for i := range axes {
			axes[i] = i
		}
	}

	// Calculate padding needed for each axis
	padConfigs := make([]backends.PadAxis, shape.Rank())
	needsPadding := false

	for axis := 0; axis < shape.Rank(); axis++ {
		currentDim := shape.Dimensions[axis]

		// Check if this axis should be padded
		shouldPad := false
		for _, a := range axes {
			normalizedAxis := a
			if normalizedAxis < 0 {
				normalizedAxis = shape.Rank() + normalizedAxis
			}
			if normalizedAxis == axis {
				shouldPad = true
				break
			}
		}

		if shouldPad && currentDim > 0 {
			// Only bucket positive (non-symbolic) dimensions
			bucketedDim := strategy.Bucket(currentDim)
			padAmount := bucketedDim - currentDim
			if padAmount > 0 {
				padConfigs[axis] = backends.PadAxis{End: padAmount}
				needsPadding = true
			}
		}
	}

	if !needsPadding {
		return x // No padding needed
	}

	return Pad(x, fillValue, padConfigs...)
}

// PadToPow2 is a convenience function that pads specified axes to the next power of 2.
//
// Example:
//
//	// Pad first axis to power of 2
//	padded := PadToPow2(input, Const(g, float32(0)), 0)
//	// input [7, 512] -> padded [8, 512]
func PadToPow2(x *Node, fillValue *Node, axes ...int) *Node {
	return PadToBucketSize(x, fillValue, Pow2Bucketing{}, axes...)
}

// PadToMultiple pads specified axes to be multiples of the given step size.
//
// Example:
//
//	// Pad first axis to multiple of 8
//	padded := PadToMultiple(input, Const(g, float32(0)), 8, 0)
//	// input [7, 512] -> padded [8, 512]
//	// input [9, 512] -> padded [16, 512]
func PadToMultiple(x *Node, fillValue *Node, step int, axes ...int) *Node {
	return PadToBucketSize(x, fillValue, LinearBucketing{Step: step}, axes...)
}
