package graph

import (
	. "github.com/gomlx/gomlx/internal/exceptions"
)

// ops_dynamic.go implements dynamic shape operations that work with runtime-computed shapes.
// These operations enable full dynamic shapes support in GoMLX.

// GetDimensionSize returns a scalar node containing the runtime size of the specified dimension.
//
// The dimension parameter can be negative, in which case it's adjusted relative to the rank
// (e.g., -1 refers to the last dimension).
//
// The returned node is always a scalar of type dtypes.Int32.
//
// Example:
//
//	x := Iota(g, MakeShape(dtypes.Float32, 3, 4), 0)
//	size := GetDimensionSize(x, 1) // Returns scalar containing 4
func GetDimensionSize(operand *Node, dimension int) *Node {
	rank := operand.Rank()
	if dimension < 0 {
		dimension += rank
	}
	if dimension < 0 || dimension >= rank {
		Panicf("GetDimensionSize: dimension %d out of bounds for rank %d", dimension, rank)
	}
	return backendGetDimensionSize(operand, dimension)
}

// DynamicBroadcastInDim broadcasts operand to the shape specified by outputDimensions tensor.
//
// This is similar to BroadcastInDim but uses a runtime-computed shape instead of a static shape.
//
// Parameters:
//   - operand: The tensor to broadcast.
//   - outputDimensions: A 1D integer tensor containing the target shape dimensions.
//   - broadcastDimensions: Specifies which axes of the output correspond to which axes of the input.
//     The i-th axis of operand is mapped to the broadcastDimensions[i]-th dimension of the output.
//     Must have len(broadcastDimensions) == operand.Rank().
//
// The outputDimensions must be a 1D integer tensor. The number of elements in outputDimensions
// determines the rank of the output.
//
// Example:
//
//	x := Const(g, []float32{1.0, 2.0})          // shape [2]
//	targetShape := Const(g, []int32{3, 2})      // shape to broadcast to: [3, 2]
//	y := DynamicBroadcastInDim(x, targetShape, []int{1})
//	// y has shape [3, 2] with values:
//	// [[1.0, 2.0],
//	//  [1.0, 2.0],
//	//  [1.0, 2.0]]
func DynamicBroadcastInDim(operand *Node, outputDimensions *Node, broadcastDimensions []int) *Node {
	// Validate outputDimensions
	if !outputDimensions.DType().IsInt() {
		Panicf("DynamicBroadcastInDim: outputDimensions must be integer type, got %s",
			outputDimensions.DType())
	}
	if outputDimensions.Rank() != 1 {
		Panicf("DynamicBroadcastInDim: outputDimensions must be 1D, got rank %d",
			outputDimensions.Rank())
	}

	// Validate broadcastDimensions
	if len(broadcastDimensions) != operand.Rank() {
		Panicf("DynamicBroadcastInDim: len(broadcastDimensions)=%d must equal operand.Rank()=%d",
			len(broadcastDimensions), operand.Rank())
	}

	return backendDynamicBroadcastInDim(operand, outputDimensions, broadcastDimensions)
}

// DynamicReshape reshapes operand to the shape specified by outputShape tensor.
//
// This is similar to Reshape but uses a runtime-computed shape instead of static dimensions.
//
// Parameters:
//   - operand: The tensor to reshape.
//   - outputShape: A 1D integer tensor containing the target shape dimensions.
//
// The outputShape must be a 1D integer tensor. The total number of elements must match
// between the input and output shapes (or the output shape can contain at most one -1,
// which will be inferred).
//
// Example:
//
//	x := Iota(g, MakeShape(dtypes.Float32, 6), 0)  // shape [6]
//	newShape := Const(g, []int32{2, 3})             // shape to reshape to: [2, 3]
//	y := DynamicReshape(x, newShape)
//	// y has shape [2, 3] with values:
//	// [[0, 1, 2],
//	//  [3, 4, 5]]
func DynamicReshape(operand *Node, outputShape *Node) *Node {
	// Validate outputShape
	if !outputShape.DType().IsInt() {
		Panicf("DynamicReshape: outputShape must be integer type, got %s",
			outputShape.DType())
	}
	if outputShape.Rank() != 1 {
		Panicf("DynamicReshape: outputShape must be 1D, got rank %d",
			outputShape.Rank())
	}

	return backendDynamicReshape(operand, outputShape)
}

// DynamicReshapeWithBounds reshapes operand to the shape specified by outputShape tensor,
// using explicit dimension bounds for XLA compilation and buffer allocation.
//
// This is useful for data-dependent shapes (e.g., NonZero output) where the shape
// cannot be determined at compile time but the caller knows upper bounds.
//
// Parameters:
//   - operand: The tensor to reshape.
//   - outputShape: A 1D integer tensor containing the target shape dimensions.
//   - bounds: Upper bounds for each dimension. Must have len(bounds) == len(outputShape).
//     Each bound value specifies the maximum possible size for that dimension.
//
// The bounds are used by XLA for buffer allocation and shape validation at compile time.
//
// Example:
//
//	// Reshape to a data-dependent shape with known upper bounds
//	x := SomeTensor(g)                              // e.g., from NonZero
//	targetShape := Const(g, []int32{numNonZeros, 64})  // computed at runtime
//	bounds := []int{1000, 64}                       // max 1000 non-zeros
//	y := DynamicReshapeWithBounds(x, targetShape, bounds)
func DynamicReshapeWithBounds(operand *Node, outputShape *Node, bounds []int) *Node {
	// Validate outputShape
	if !outputShape.DType().IsInt() {
		Panicf("DynamicReshapeWithBounds: outputShape must be integer type, got %s",
			outputShape.DType())
	}
	if outputShape.Rank() != 1 {
		Panicf("DynamicReshapeWithBounds: outputShape must be 1D, got rank %d",
			outputShape.Rank())
	}

	// Validate bounds
	outputRank := outputShape.Shape().Dimensions[0]
	if len(bounds) != outputRank {
		Panicf("DynamicReshapeWithBounds: len(bounds)=%d must equal output rank=%d",
			len(bounds), outputRank)
	}
	for i, b := range bounds {
		if b <= 0 {
			Panicf("DynamicReshapeWithBounds: bounds[%d]=%d must be positive", i, b)
		}
	}

	return backendDynamicReshapeWithBounds(operand, outputShape, bounds)
}

// DynamicBroadcastInDimWithBounds broadcasts operand to the shape specified by outputDimensions tensor,
// using explicit dimension bounds for XLA compilation and buffer allocation.
//
// This is useful for data-dependent shapes where bounds are known but exact dimensions are not.
//
// Parameters:
//   - operand: The tensor to broadcast.
//   - outputDimensions: A 1D integer tensor containing the target shape dimensions.
//   - broadcastDimensions: Specifies which axes of the output correspond to which axes of the input.
//   - bounds: Upper bounds for each dimension. Must have len(bounds) == len(outputDimensions).
//
// Example:
//
//	x := Const(g, []float32{1.0, 2.0})           // shape [2]
//	targetShape := SomeDataDependentShape(g)     // e.g., from NonZero
//	bounds := []int{100, 2}                      // max sizes
//	y := DynamicBroadcastInDimWithBounds(x, targetShape, []int{1}, bounds)
func DynamicBroadcastInDimWithBounds(operand *Node, outputDimensions *Node, broadcastDimensions []int, bounds []int) *Node {
	// Validate outputDimensions
	if !outputDimensions.DType().IsInt() {
		Panicf("DynamicBroadcastInDimWithBounds: outputDimensions must be integer type, got %s",
			outputDimensions.DType())
	}
	if outputDimensions.Rank() != 1 {
		Panicf("DynamicBroadcastInDimWithBounds: outputDimensions must be 1D, got rank %d",
			outputDimensions.Rank())
	}

	// Validate broadcastDimensions
	if len(broadcastDimensions) != operand.Rank() {
		Panicf("DynamicBroadcastInDimWithBounds: len(broadcastDimensions)=%d must equal operand.Rank()=%d",
			len(broadcastDimensions), operand.Rank())
	}

	// Validate bounds
	outputRank := outputDimensions.Shape().Dimensions[0]
	if len(bounds) != outputRank {
		Panicf("DynamicBroadcastInDimWithBounds: len(bounds)=%d must equal output rank=%d",
			len(bounds), outputRank)
	}
	for i, b := range bounds {
		if b <= 0 {
			Panicf("DynamicBroadcastInDimWithBounds: bounds[%d]=%d must be positive", i, b)
		}
	}

	return backendDynamicBroadcastInDimWithBounds(operand, outputDimensions, broadcastDimensions, bounds)
}
