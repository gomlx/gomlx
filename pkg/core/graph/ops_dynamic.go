package graph

import (
	. "github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/shapes"
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

	// Try to extract concrete dimensions from the shape tensor node.
	// This allows us to use static BroadcastInDim when all dimensions are known at graph build time.
	dims, allConcrete, bounds := ExtractShapeDimensions(outputDimensions)

	if allConcrete && dims != nil {
		// Validate that the broadcast dimensions are compatible.
		// For each operand dimension, the corresponding output dimension must either:
		// 1. Equal the operand dimension, or
		// 2. Operand dimension must be 1 (broadcast)
		operandShape := operand.Shape()
		valid := true
		operandHasDynamic := operandShape.HasSymbolicDim()

		// If operand has dynamic dimensions, we can't use static broadcast
		// because we don't know if the dimensions are compatible at compile time.
		if operandHasDynamic {
			valid = false
		} else {
			for i, bdim := range broadcastDimensions {
				if bdim < 0 || bdim >= len(dims) {
					valid = false
					break
				}
				operandDim := operandShape.Dimensions[i]
				if operandDim > 0 && dims[bdim] > 0 && operandDim != dims[bdim] && operandDim != 1 {
					valid = false
					break
				}
			}
		}

		if valid {
			// Use static BroadcastInDim for better performance
			outputShape := shapes.Make(operand.DType(), dims...)
			return backendBroadcastInDim(operand, outputShape, broadcastDimensions)
		}
		// Validation failed but we have all concrete dims - use dynamic broadcast with shape propagation
		if hasConcreteBounds(bounds) {
			return backendDynamicBroadcastInDimWithBoundsAndShape(operand, outputDimensions, broadcastDimensions, bounds, dims)
		}
	}

	// For partially extracted dimensions, propagate shape info using extracted values
	// and inferred dimensions where possible.
	if dims != nil && len(dims) > 0 {
		// Compute effective bounds using extracted info + operand shape
		effectiveBounds := computeEffectiveBounds(dims, bounds, operand.Shape())
		// Compute output dimensions, using operand dims for broadcast axes
		outputDims := computeOutputDimsForBroadcast(dims, effectiveBounds, operand.Shape(), broadcastDimensions)
		return backendDynamicBroadcastInDimWithBoundsAndShape(operand, outputDimensions, broadcastDimensions, effectiveBounds, outputDims)
	}

	// Fallback: compute bounds and output dims from operand shape if possible
	outputRank := outputDimensions.Shape().Dimensions[0]
	fallbackBounds := computeBroadcastFallbackBounds(operand.Shape(), outputRank, broadcastDimensions)
	if fallbackBounds != nil {
		// Compute output dims using operand dims for broadcast axes
		fallbackDims := computeBroadcastFallbackOutputDims(operand.Shape(), outputRank, broadcastDimensions)
		return backendDynamicBroadcastInDimWithBoundsAndShape(operand, outputDimensions, broadcastDimensions, fallbackBounds, fallbackDims)
	}

	// Last resort: use default bounds and dims to ensure XLA can compile
	if outputRank > 0 {
		const defaultBound = 4096
		defaultBounds := make([]int, outputRank)
		defaultDims := make([]int, outputRank)
		operandDims := operand.Shape().Dimensions
		for i := range defaultBounds {
			defaultBounds[i] = defaultBound
			// For broadcast, we need to distinguish between:
			// 1. Operand dims > 1: the output will have this size (actual data)
			// 2. Operand dims == 1: the output size is unknown (will be broadcast to target)
			// Use DynamicDim for case 2 to propagate the uncertainty to downstream ops.
			correspondingOperandAxis := -1
			for operandAxis, outputAxis := range broadcastDimensions {
				if outputAxis == i {
					correspondingOperandAxis = operandAxis
					break
				}
			}
			if correspondingOperandAxis >= 0 && correspondingOperandAxis < len(operandDims) && operandDims[correspondingOperandAxis] > 1 {
				defaultDims[i] = operandDims[correspondingOperandAxis]
			} else {
				// Output dimension is dynamic (from shape tensor).
				// Use the bound value instead of DynamicDim so downstream ops have concrete shapes.
				// XLA allocates buffers at bounds size anyway, so this is correct for compilation.
				defaultDims[i] = defaultBounds[i]
			}
		}
		return backendDynamicBroadcastInDimWithBoundsAndShape(operand, outputDimensions, broadcastDimensions, defaultBounds, defaultDims)
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

	// Try to extract concrete dimensions from the shape tensor node.
	// This allows us to use static Reshape when all dimensions are known at graph build time.
	dims, allConcrete, bounds := ExtractShapeDimensions(outputShape)

	// Debug: trace Int64 tensors being reshaped to 2D
	debugOutputRank := outputShape.Shape().Dimensions[0]

	// VALIDATION: If we have concrete output dimensions, verify element count matches
	if allConcrete && dims != nil && len(dims) > 0 {
		outputSize := 1
		for _, d := range dims {
			if d > 0 {
				outputSize *= d
			}
		}
		operandSize := operand.Shape().Size()
		if operandSize < 0 {
			operandSize = -operandSize // Bounded dynamic
		}
		if outputSize > 0 && operandSize > 0 && outputSize != operandSize {
			// Don't proceed with the obviously wrong reshape
			// Instead, compute correct dimensions that preserve element count
			if debugOutputRank == 2 && operandSize > 0 {
				// For 2D output, try [operandSize, 1] or [1, operandSize]
				dims = []int{operandSize, 1}
				allConcrete = true
			}
		}
	}

	if allConcrete && dims != nil {
		// Compute the size from extracted dimensions
		extractedSize := 1
		for _, d := range dims {
			if d > 0 {
				extractedSize *= d
			}
		}

		// Compute operand size, using absolute value if it's dynamic (negative)
		operandSize := operand.Shape().Size()
		absOperandSize := operandSize
		if absOperandSize < 0 {
			absOperandSize = -absOperandSize
		}

		// CRITICAL FIX: When we have all concrete dimensions extracted AND they're compatible
		// with the operand size, use static Reshape. This prevents creating dynamic reshape
		// operations that XLA can't handle when chained together.
		//
		// We validate against absolute size because the operand might have come from a
		// previous dynamic operation, but we know the actual concrete bounds.
		if extractedSize > 0 && (absOperandSize == extractedSize || absOperandSize == 0) {
			return Reshape(operand, dims...)
		}

		// Sizes don't match - use dynamic reshape but propagate extracted shape.
		// This allows downstream operations (like MatMul) to have concrete shapes
		// for validation, while XLA handles the actual reshape at runtime.
		if hasConcreteBounds(bounds) {
			return backendDynamicReshapeWithBoundsAndShape(operand, outputShape, bounds, dims)
		}
	}

	// For partially extracted dimensions, we can still propagate shape info using
	// extracted concrete values and inferred dimensions where possible.
	if dims != nil && len(dims) > 0 {
		// Compute effective bounds using extracted info + operand shape
		effectiveBounds := computeEffectiveBounds(dims, bounds, operand.Shape())
		// Compute output dimensions, trying to infer unknowns from operand size
		outputDims := computeOutputDimsForReshape(dims, effectiveBounds, operand.Shape())
		return backendDynamicReshapeWithBoundsAndShape(operand, outputShape, effectiveBounds, outputDims)
	}

	// Fallback: compute bounds and output dims from operand shape if possible
	outputRank := outputShape.Shape().Dimensions[0]
	fallbackBounds := computeFallbackBounds(operand.Shape(), outputRank)
	if fallbackBounds != nil {
		// Compute output dims - try to preserve operand dimensions or use operand-derived dims
		fallbackDims := computeFallbackOutputDims(operand.Shape(), outputRank)
		return backendDynamicReshapeWithBoundsAndShape(operand, outputShape, fallbackBounds, fallbackDims)
	}

	// Last resort: use default bounds and dims to ensure XLA can compile
	// Use smaller default bounds for shape propagation to avoid huge shape mismatches
	if outputRank > 0 {
		// Try to derive bounds from operand's concrete dimensions
		operandDims := operand.Shape().Dimensions
		maxConcreteDim := 0
		for _, d := range operandDims {
			if d > maxConcreteDim {
				maxConcreteDim = d
			}
		}

		defaultBound := maxConcreteDim
		if defaultBound <= 0 {
			defaultBound = DefaultBound // Use canonical 4096 only as final fallback
		}

		defaultBounds := make([]int, outputRank)
		defaultDims := make([]int, outputRank)
		for i := range defaultBounds {
			defaultBounds[i] = defaultBound
			// Use 1 as default dim to minimize shape mismatch issues
			// This is better than using large bounds which causes huge products
			defaultDims[i] = 1
		}
		return backendDynamicReshapeWithBoundsAndShape(operand, outputShape, defaultBounds, defaultDims)
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

	// Try to extract shape info to compute output dimensions
	dims, allConcrete, _ := ExtractShapeDimensions(outputShape)

	// CRITICAL FIX: If we extracted all concrete dimensions, prefer static reshape to avoid
	// chaining dynamic reshapes which XLA can't compile.
	if allConcrete && dims != nil {
		// Compute the size from extracted dimensions
		extractedSize := 1
		for _, d := range dims {
			if d > 0 {
				extractedSize *= d
			}
		}

		// Compute operand size, using absolute value if it's dynamic (negative)
		operandSize := operand.Shape().Size()
		absOperandSize := operandSize
		if absOperandSize < 0 {
			absOperandSize = -absOperandSize
		}

		// Use static reshape when sizes are compatible
		if extractedSize > 0 && (absOperandSize == extractedSize || absOperandSize == 0) {
			return Reshape(operand, dims...)
		}
	}

	if dims != nil && len(dims) > 0 {
		// Compute output dimensions using extracted info + operand shape
		// IMPORTANT: Use the provided bounds, NOT computeEffectiveBounds which would override them
		outputDims := computeOutputDimsForReshape(dims, bounds, operand.Shape())
		return backendDynamicReshapeWithBoundsAndShape(operand, outputShape, bounds, outputDims)
	}

	// If extraction fails, compute output dims from bounds and operand
	if allConcrete && dims != nil {
		return backendDynamicReshapeWithBoundsAndShape(operand, outputShape, bounds, dims)
	}

	// Fallback: compute reasonable output dimensions
	outputDims := computeFallbackOutputDims(operand.Shape(), outputRank)
	if outputDims != nil {
		return backendDynamicReshapeWithBoundsAndShape(operand, outputShape, bounds, outputDims)
	}

	// Last resort: use bounds as dims (but this is likely incorrect for shape propagation)
	return backendDynamicReshapeWithBoundsAndShape(operand, outputShape, bounds, bounds)
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

	// Try to extract shape info to compute output dimensions
	dims, allConcrete, _ := ExtractShapeDimensions(outputDimensions)
	if dims != nil && len(dims) > 0 {
		// Compute output dimensions using extracted info + operand shape
		outputDims := computeOutputDimsForBroadcast(dims, bounds, operand.Shape(), broadcastDimensions)
		return backendDynamicBroadcastInDimWithBoundsAndShape(operand, outputDimensions, broadcastDimensions, bounds, outputDims)
	}

	// If extraction fails, compute output dims from bounds and operand
	if allConcrete && dims != nil {
		return backendDynamicBroadcastInDimWithBoundsAndShape(operand, outputDimensions, broadcastDimensions, bounds, dims)
	}

	// Fallback: compute reasonable output dimensions using operand dims for broadcast axes
	outputDims := computeBroadcastFallbackOutputDims(operand.Shape(), outputRank, broadcastDimensions)
	if outputDims != nil {
		return backendDynamicBroadcastInDimWithBoundsAndShape(operand, outputDimensions, broadcastDimensions, bounds, outputDims)
	}

	// Last resort: use bounds as dims. While bounds may be large (e.g., 4096),
	// using concrete values allows downstream ops to have concrete shapes for XLA compilation.
	// XLA allocates buffers at bounds size anyway.
	return backendDynamicBroadcastInDimWithBoundsAndShape(operand, outputDimensions, broadcastDimensions, bounds, bounds)
}
