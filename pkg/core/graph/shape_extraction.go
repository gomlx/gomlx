package graph

import (
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// shape_extraction.go implements shape tensor analysis at the GoMLX graph level.
// This allows us to extract concrete dimensions from shape tensors built from
// Const, GetDimensionSize, Stack, Concatenate, and other operations.
//
// The extraction happens BEFORE calling backend methods, so we have full access
// to the GoMLX node graph structure and can use introspection APIs.

// ExtractShapeDimensions attempts to extract concrete integer dimensions from a shape tensor node.
// It traverses the computation graph to resolve constant expressions.
//
// Returns:
//   - dims: the extracted dimensions (shapes.DynamicDim for unknown/dynamic dimensions)
//   - allConcrete: true if all dimensions were successfully extracted as concrete values
//   - bounds: upper bounds for each dimension (same as dims when concrete, or estimated bounds)
//
// The function handles common patterns used to build shape tensors:
//   - Constant tensors
//   - Concatenation of dimension values
//   - GetDimensionSize operations
//   - Arithmetic operations (Mul, Div)
//   - Type conversions
//   - Reshape and Slice operations
func ExtractShapeDimensions(shapeTensor *Node) (dims []int, allConcrete bool, bounds []int) {
	if shapeTensor == nil {
		return nil, false, nil
	}

	switch shapeTensor.Type() {
	case NodeTypeConstant:
		return extractFromConstant(shapeTensor)
	case NodeTypeConcatenate:
		return extractFromConcatenate(shapeTensor)
	case NodeTypeGetDimensionSize:
		return extractFromGetDimensionSize(shapeTensor)
	case NodeTypeReshape:
		return extractFromReshape(shapeTensor)
	case NodeTypeSlice:
		return extractFromSlice(shapeTensor)
	case NodeTypeMul:
		return extractFromMultiply(shapeTensor)
	case NodeTypeDiv:
		return extractFromDivide(shapeTensor)
	case NodeTypeConvertDType:
		return extractFromConvert(shapeTensor)
	case NodeTypeGather:
		return extractFromGather(shapeTensor)
	case NodeTypeAdd:
		return extractFromAdd(shapeTensor)
	case NodeTypeSub:
		return extractFromSub(shapeTensor)
	case NodeTypeMax:
		return extractFromMax(shapeTensor)
	case NodeTypeMin:
		return extractFromMin(shapeTensor)
	case NodeTypeWhere:
		return extractFromWhere(shapeTensor)
	case NodeTypeReduceMax:
		return extractFromReduceMax(shapeTensor)
	default:
		// Unknown node type - return dynamic result based on shape
		return makeDynamicResult(shapeTensor)
	}
}

// extractFromConstant extracts integer values from a Constant node.
func extractFromConstant(node *Node) (dims []int, allConcrete bool, bounds []int) {
	tensor := node.ConstantValue()
	if tensor == nil {
		return makeDynamicResult(node)
	}

	dims = extractIntegersFromTensor(tensor)
	if dims == nil {
		return makeDynamicResult(node)
	}

	// Check for ONNX-style infer markers (-1) or other negative values
	// These are placeholders meaning "infer at runtime", not concrete dimensions
	bounds = make([]int, len(dims))
	allConcrete = true
	for i, d := range dims {
		if d < 0 {
			// -1 is ONNX's "infer this dimension" marker
			// Treat as dynamic with fallback bound
			allConcrete = false
			bounds[i] = DefaultBound
		} else {
			bounds[i] = d
		}
	}
	return dims, allConcrete, bounds
}

// extractFromConcatenate extracts dimensions from a Concatenate node.
// Shape tensors are typically built by concatenating individual dimension values along axis 0.
func extractFromConcatenate(node *Node) (dims []int, allConcrete bool, bounds []int) {
	inputs := node.inputs.(*nodeInputsConcatenate)

	// Only handle axis 0 concatenation (1D shape tensors)
	if inputs.axis != 0 {
		return makeDynamicResult(node)
	}

	var result []int
	var resultBounds []int
	allConcrete = true

	for _, operand := range inputs.operands {
		operandDims, operandConcrete, operandBounds := ExtractShapeDimensions(operand)
		if operandDims == nil {
			// Couldn't extract - determine size from operand shape and fill with DynamicDim
			operandShape := operand.Shape()
			if operandShape.Rank() == 1 && operandShape.Dimensions[0] > 0 {
				size := operandShape.Dimensions[0]
				for i := 0; i < size; i++ {
					result = append(result, shapes.DynamicDim)
					resultBounds = append(resultBounds, DefaultBound) // Use default bound
				}
			} else if operandShape.IsScalar() {
				// Single unknown dimension
				result = append(result, shapes.DynamicDim)
				resultBounds = append(resultBounds, DefaultBound) // Use default bound
			}
			allConcrete = false
		} else {
			result = append(result, operandDims...)
			resultBounds = append(resultBounds, operandBounds...)
			if !operandConcrete {
				allConcrete = false
			}
		}
	}

	return result, allConcrete, resultBounds
}

// extractFromGetDimensionSize extracts the dimension value from a GetDimensionSize node.
// If the operand's shape has a known dimension at the specified index, returns that value.
func extractFromGetDimensionSize(node *Node) (dims []int, allConcrete bool, bounds []int) {
	inputs := node.inputs.(*nodeInputsGetDimensionSize)
	operandShape := inputs.operand.Shape()
	dimension := inputs.dimension

	if dimension < 0 || dimension >= operandShape.Rank() {
		return makeDynamicResult(node)
	}

	dimValue := operandShape.Dimensions[dimension]
	if dimValue < 0 {
		// Dynamic dimension - we don't know the value at compile time
		// Use DefaultBound as the bound since we can't know the actual size
		return []int{shapes.DynamicDim}, false, []int{DefaultBound}
	}

	return []int{dimValue}, true, []int{dimValue}
}

// extractFromReshape passes through to the input node for extraction.
// A reshaped shape tensor still contains the same values.
func extractFromReshape(node *Node) (dims []int, allConcrete bool, bounds []int) {
	inputs := node.inputs.(*nodeInputsReshape)
	dims, allConcrete, bounds = ExtractShapeDimensions(inputs.x)
	return dims, allConcrete, bounds
}

// extractFromSlice extracts a slice of dimensions from a shape tensor.
func extractFromSlice(node *Node) (dims []int, allConcrete bool, bounds []int) {
	inputs := node.inputs.(*nodeInputsSlice)

	// Extract from the sliced operand
	operandDims, operandConcrete, operandBounds := ExtractShapeDimensions(inputs.x)
	if operandDims == nil {
		return makeDynamicResult(node)
	}

	// Only handle 1D slices (shape tensors)
	if len(inputs.starts) != 1 || len(inputs.limits) != 1 {
		return makeDynamicResult(node)
	}

	start := inputs.starts[0]
	limit := inputs.limits[0]
	stride := 1
	if len(inputs.strides) > 0 {
		stride = inputs.strides[0]
	}

	if start < 0 || limit > len(operandDims) || start >= limit || stride != 1 {
		return makeDynamicResult(node)
	}

	slicedDims := operandDims[start:limit]
	slicedBounds := operandBounds[start:limit]

	// Check if all sliced dimensions are concrete
	allConcrete = operandConcrete
	for _, d := range slicedDims {
		if d < 0 {
			allConcrete = false
			break
		}
	}

	return slicedDims, allConcrete, slicedBounds
}

// extractFromMultiply extracts and multiplies dimension values.
// Handles scalar * tensor and element-wise multiplication.
// Also handles partial extraction where one operand is dynamic but has bounds,
// or where one operand is completely unknown (nil) but the other has useful info.
func extractFromMultiply(node *Node) (dims []int, allConcrete bool, bounds []int) {
	inputs := node.inputs.(*nodeInputsMul)

	lhsDims, lhsConcrete, lhsBounds := ExtractShapeDimensions(inputs.lhs)
	rhsDims, rhsConcrete, rhsBounds := ExtractShapeDimensions(inputs.rhs)

	// Handle case where one operand is completely unknown (nil) but the other has info.
	// This is common when multiplying a dynamic value (e.g., repeat count from Slice)
	// with a concrete dimension size. We can still provide useful bounds.
	if lhsDims == nil && rhsDims != nil {
		// LHS is unknown, use RHS concrete values as bounds
		result := make([]int, len(rhsDims))
		resultBounds := make([]int, len(rhsDims))
		for i, d := range rhsDims {
			result[i] = shapes.DynamicDim // Can't know exact value
			if d > 0 {
				// Concrete value from rhs - use as bound (assumes lhs >= 1)
				resultBounds[i] = d
			} else if rhsBounds != nil && i < len(rhsBounds) && rhsBounds[i] > 0 {
				resultBounds[i] = rhsBounds[i]
			}
		}
		return result, false, resultBounds
	}
	if rhsDims == nil && lhsDims != nil {
		// RHS is unknown, use LHS concrete values as bounds
		result := make([]int, len(lhsDims))
		resultBounds := make([]int, len(lhsDims))
		for i, d := range lhsDims {
			result[i] = shapes.DynamicDim // Can't know exact value
			if d > 0 {
				// Concrete value from lhs - use as bound (assumes rhs >= 1)
				resultBounds[i] = d
			} else if lhsBounds != nil && i < len(lhsBounds) && lhsBounds[i] > 0 {
				resultBounds[i] = lhsBounds[i]
			}
		}
		return result, false, resultBounds
	}
	if lhsDims == nil || rhsDims == nil {
		return makeDynamicResult(node)
	}

	var result, resultBounds []int
	allConcrete = lhsConcrete && rhsConcrete

	// Helper to multiply values, handling DynamicDim
	mulValues := func(a, b int) int {
		if a <= 0 || b <= 0 {
			return shapes.DynamicDim
		}
		return a * b
	}

	// Helper to multiply bounds, using a conservative fallback when bound is unknown
	mulBounds := func(aBound, bBound, aVal, bVal int) int {
		// If we have concrete values, use them as bounds
		if aVal > 0 && bVal > 0 {
			return aVal * bVal
		}
		// If one is concrete and the other has a bound, multiply them
		if aVal > 0 && bBound > 0 {
			return aVal * bBound
		}
		if bVal > 0 && aBound > 0 {
			return bVal * aBound
		}
		// Both have bounds
		if aBound > 0 && bBound > 0 {
			return aBound * bBound
		}
		// At least one has a bound
		if aBound > 0 {
			return aBound
		}
		if bBound > 0 {
			return bBound
		}
		return 0 // Unknown bound
	}

	if len(lhsDims) == 1 && len(rhsDims) == 1 {
		// Scalar multiplication
		result = []int{mulValues(lhsDims[0], rhsDims[0])}
		resultBounds = []int{mulBounds(lhsBounds[0], rhsBounds[0], lhsDims[0], rhsDims[0])}
	} else if len(lhsDims) == 1 {
		// Broadcast lhs scalar
		result = make([]int, len(rhsDims))
		resultBounds = make([]int, len(rhsDims))
		for i := range rhsDims {
			result[i] = mulValues(lhsDims[0], rhsDims[i])
			resultBounds[i] = mulBounds(lhsBounds[0], rhsBounds[i], lhsDims[0], rhsDims[i])
		}
	} else if len(rhsDims) == 1 {
		// Broadcast rhs scalar
		result = make([]int, len(lhsDims))
		resultBounds = make([]int, len(lhsDims))
		for i := range lhsDims {
			result[i] = mulValues(lhsDims[i], rhsDims[0])
			resultBounds[i] = mulBounds(lhsBounds[i], rhsBounds[0], lhsDims[i], rhsDims[0])
		}
	} else if len(lhsDims) == len(rhsDims) {
		// Element-wise multiplication
		result = make([]int, len(lhsDims))
		resultBounds = make([]int, len(lhsDims))
		for i := range lhsDims {
			result[i] = mulValues(lhsDims[i], rhsDims[i])
			resultBounds[i] = mulBounds(lhsBounds[i], rhsBounds[i], lhsDims[i], rhsDims[i])
		}
	} else {
		return makeDynamicResult(node)
	}

	// Check if all results are concrete
	for _, d := range result {
		if d <= 0 {
			allConcrete = false
			break
		}
	}

	return result, allConcrete, resultBounds
}

// extractFromDivide extracts and divides dimension values.
// Handles integer division (common for splitting dimensions).
func extractFromDivide(node *Node) (dims []int, allConcrete bool, bounds []int) {
	inputs := node.inputs.(*nodeInputsDiv)

	lhsDims, lhsConcrete, _ := ExtractShapeDimensions(inputs.lhs)
	rhsDims, rhsConcrete, _ := ExtractShapeDimensions(inputs.rhs)

	if lhsDims == nil || rhsDims == nil || !lhsConcrete || !rhsConcrete {
		return makeDynamicResult(node)
	}

	// Only handle scalar division for now
	if len(lhsDims) != 1 || len(rhsDims) != 1 {
		return makeDynamicResult(node)
	}

	if rhsDims[0] == 0 {
		return makeDynamicResult(node) // Division by zero
	}

	result := lhsDims[0] / rhsDims[0]
	return []int{result}, true, []int{result}
}

// extractFromConvert passes through to the input node for extraction.
// Type conversion doesn't change the values for shape tensor purposes.
func extractFromConvert(node *Node) (dims []int, allConcrete bool, bounds []int) {
	inputs := node.inputs.(*nodeInputsConvertDType)
	return ExtractShapeDimensions(inputs.x)
}

// extractFromGather extracts values when Gather is used to select elements from a shape tensor.
// This is common in ONNX models for extracting specific dimensions.
func extractFromGather(node *Node) (dims []int, allConcrete bool, bounds []int) {
	inputs := node.inputs.(*nodeInputsGather)

	// Extract from the operand (the shape tensor being gathered from)
	operandDims, operandConcrete, operandBounds := ExtractShapeDimensions(inputs.operand)
	if operandDims == nil {
		return makeDynamicResult(node)
	}

	// Try to get the indices as constants
	// First, try directly
	indicesTensor := inputs.startIndices.ConstantValue()
	if indicesTensor == nil {
		// If not directly a constant, try to extract from the indices node
		// This handles cases where indices come from Reshape, Cast, etc.
		indicesDims, indicesConcrete, _ := ExtractShapeDimensions(inputs.startIndices)
		if indicesDims != nil && indicesConcrete {
			// Use the extracted indices directly
			indices := indicesDims
			// Continue with the gather logic
			if len(inputs.sliceSizes) == 1 && inputs.sliceSizes[0] == 1 {
				result := make([]int, len(indices))
				resultBounds := make([]int, len(indices))
				allConcrete = operandConcrete
				for i, idx := range indices {
					if idx < 0 || idx >= len(operandDims) {
						return makeDynamicResult(node)
					}
					result[i] = operandDims[idx]
					if operandBounds != nil && idx < len(operandBounds) {
						resultBounds[i] = operandBounds[idx]
					}
					if result[i] < 0 {
						allConcrete = false
					}
				}
				return result, allConcrete, resultBounds
			}
		}
		return makeDynamicResult(node)
	}

	indices := extractIntegersFromTensor(indicesTensor)
	if indices == nil {
		return makeDynamicResult(node)
	}

	// For simple 1D gather with sliceSizes=[1], extract the selected elements
	if len(inputs.sliceSizes) == 1 && inputs.sliceSizes[0] == 1 {
		result := make([]int, len(indices))
		resultBounds := make([]int, len(indices))
		allConcrete = operandConcrete
		for i, idx := range indices {
			if idx < 0 || idx >= len(operandDims) {
				return makeDynamicResult(node)
			}
			result[i] = operandDims[idx]
			if operandBounds != nil && idx < len(operandBounds) {
				resultBounds[i] = operandBounds[idx]
			}
			if result[i] < 0 {
				allConcrete = false
			}
		}
		return result, allConcrete, resultBounds
	}

	return makeDynamicResult(node)
}

// extractFromAdd extracts and adds dimension values.
func extractFromAdd(node *Node) (dims []int, allConcrete bool, bounds []int) {
	inputs := node.inputs.(*nodeInputsAdd)

	lhsDims, lhsConcrete, lhsBounds := ExtractShapeDimensions(inputs.lhs)
	rhsDims, rhsConcrete, rhsBounds := ExtractShapeDimensions(inputs.rhs)

	if lhsDims == nil || rhsDims == nil || !lhsConcrete || !rhsConcrete {
		return makeDynamicResult(node)
	}

	var result, resultBounds []int

	if len(lhsDims) == 1 && len(rhsDims) == 1 {
		// Scalar addition
		result = []int{lhsDims[0] + rhsDims[0]}
		resultBounds = []int{lhsBounds[0] + rhsBounds[0]}
	} else if len(lhsDims) == 1 {
		// Broadcast lhs scalar
		result = make([]int, len(rhsDims))
		resultBounds = make([]int, len(rhsDims))
		for i := range rhsDims {
			result[i] = lhsDims[0] + rhsDims[i]
			resultBounds[i] = lhsBounds[0] + rhsBounds[i]
		}
	} else if len(rhsDims) == 1 {
		// Broadcast rhs scalar
		result = make([]int, len(lhsDims))
		resultBounds = make([]int, len(lhsDims))
		for i := range lhsDims {
			result[i] = lhsDims[i] + rhsDims[0]
			resultBounds[i] = lhsBounds[i] + rhsBounds[0]
		}
	} else if len(lhsDims) == len(rhsDims) {
		// Element-wise addition
		result = make([]int, len(lhsDims))
		resultBounds = make([]int, len(lhsDims))
		for i := range lhsDims {
			result[i] = lhsDims[i] + rhsDims[i]
			resultBounds[i] = lhsBounds[i] + rhsBounds[i]
		}
	} else {
		return makeDynamicResult(node)
	}

	return result, true, resultBounds
}

// extractFromSub extracts and subtracts dimension values.
func extractFromSub(node *Node) (dims []int, allConcrete bool, bounds []int) {
	inputs := node.inputs.(*nodeInputsSub)

	lhsDims, lhsConcrete, _ := ExtractShapeDimensions(inputs.lhs)
	rhsDims, rhsConcrete, _ := ExtractShapeDimensions(inputs.rhs)

	if lhsDims == nil || rhsDims == nil || !lhsConcrete || !rhsConcrete {
		return makeDynamicResult(node)
	}

	// Only handle scalar subtraction for now
	if len(lhsDims) != 1 || len(rhsDims) != 1 {
		return makeDynamicResult(node)
	}

	result := lhsDims[0] - rhsDims[0]
	if result < 0 {
		result = 0 // Clamp to 0 for shape purposes
	}
	return []int{result}, true, []int{result}
}

// extractFromMax extracts element-wise maximum of dimension values.
// This is useful for computing bounds when shapes depend on max of two paths.
func extractFromMax(node *Node) (dims []int, allConcrete bool, bounds []int) {
	inputs := node.inputs.(*nodeInputsMax)

	lhsDims, lhsConcrete, lhsBounds := ExtractShapeDimensions(inputs.lhs)
	rhsDims, rhsConcrete, rhsBounds := ExtractShapeDimensions(inputs.rhs)

	if lhsDims == nil || rhsDims == nil {
		return makeDynamicResult(node)
	}

	// For bounds computation, we can compute max of bounds even if values aren't concrete
	var result, resultBounds []int

	if len(lhsDims) == 1 && len(rhsDims) == 1 {
		// Scalar max
		if lhsConcrete && rhsConcrete {
			maxVal := lhsDims[0]
			if rhsDims[0] > maxVal {
				maxVal = rhsDims[0]
			}
			result = []int{maxVal}
			resultBounds = []int{maxVal}
		} else {
			// Use max of bounds for conservative bound
			maxBound := lhsBounds[0]
			if rhsBounds[0] > maxBound {
				maxBound = rhsBounds[0]
			}
			result = []int{shapes.DynamicDim}
			resultBounds = []int{maxBound}
		}
	} else if len(lhsDims) == len(rhsDims) {
		// Element-wise max
		result = make([]int, len(lhsDims))
		resultBounds = make([]int, len(lhsDims))
		allConcrete = lhsConcrete && rhsConcrete
		for i := range lhsDims {
			if lhsDims[i] > 0 && rhsDims[i] > 0 {
				maxVal := lhsDims[i]
				if rhsDims[i] > maxVal {
					maxVal = rhsDims[i]
				}
				result[i] = maxVal
				resultBounds[i] = maxVal
			} else {
				result[i] = shapes.DynamicDim
				allConcrete = false
				// Use max of bounds
				maxBound := lhsBounds[i]
				if rhsBounds[i] > maxBound {
					maxBound = rhsBounds[i]
				}
				resultBounds[i] = maxBound
			}
		}
		return result, allConcrete, resultBounds
	} else {
		return makeDynamicResult(node)
	}

	return result, lhsConcrete && rhsConcrete, resultBounds
}

// extractFromMin extracts element-wise minimum of dimension values.
// This is useful for computing bounds when shapes depend on min of two paths.
func extractFromMin(node *Node) (dims []int, allConcrete bool, bounds []int) {
	inputs := node.inputs.(*nodeInputsMin)

	lhsDims, lhsConcrete, lhsBounds := ExtractShapeDimensions(inputs.lhs)
	rhsDims, rhsConcrete, rhsBounds := ExtractShapeDimensions(inputs.rhs)

	if lhsDims == nil || rhsDims == nil {
		return makeDynamicResult(node)
	}

	var result, resultBounds []int

	if len(lhsDims) == 1 && len(rhsDims) == 1 {
		// Scalar min
		if lhsConcrete && rhsConcrete {
			minVal := lhsDims[0]
			if rhsDims[0] < minVal {
				minVal = rhsDims[0]
			}
			result = []int{minVal}
			resultBounds = []int{minVal}
		} else {
			// For min, use min of bounds as conservative bound (can't exceed smaller bound)
			minBound := lhsBounds[0]
			if rhsBounds[0] > 0 && (minBound <= 0 || rhsBounds[0] < minBound) {
				minBound = rhsBounds[0]
			}
			result = []int{shapes.DynamicDim}
			resultBounds = []int{minBound}
		}
	} else if len(lhsDims) == len(rhsDims) {
		// Element-wise min
		result = make([]int, len(lhsDims))
		resultBounds = make([]int, len(lhsDims))
		allConcrete = lhsConcrete && rhsConcrete
		for i := range lhsDims {
			if lhsDims[i] > 0 && rhsDims[i] > 0 {
				minVal := lhsDims[i]
				if rhsDims[i] < minVal {
					minVal = rhsDims[i]
				}
				result[i] = minVal
				resultBounds[i] = minVal
			} else {
				result[i] = shapes.DynamicDim
				allConcrete = false
				// Use min of bounds
				minBound := lhsBounds[i]
				if rhsBounds[i] > 0 && (minBound <= 0 || rhsBounds[i] < minBound) {
					minBound = rhsBounds[i]
				}
				resultBounds[i] = minBound
			}
		}
		return result, allConcrete, resultBounds
	} else {
		return makeDynamicResult(node)
	}

	return result, lhsConcrete && rhsConcrete, resultBounds
}

// extractFromWhere extracts dimensions from a Where (select) operation.
// Returns the extracted dimensions from either branch if they match,
// or combines bounds from both branches.
func extractFromWhere(node *Node) (dims []int, allConcrete bool, bounds []int) {
	inputs := node.inputs.(*nodeInputsWhere)

	// Extract from both branches
	trueDims, trueConcrete, trueBounds := ExtractShapeDimensions(inputs.onTrue)
	falseDims, falseConcrete, falseBounds := ExtractShapeDimensions(inputs.onFalse)

	if trueDims == nil || falseDims == nil {
		return makeDynamicResult(node)
	}

	if len(trueDims) != len(falseDims) {
		return makeDynamicResult(node)
	}

	// If both branches have the same concrete values, use those
	if trueConcrete && falseConcrete {
		sameValues := true
		for i := range trueDims {
			if trueDims[i] != falseDims[i] {
				sameValues = false
				break
			}
		}
		if sameValues {
			return trueDims, true, trueBounds
		}
	}

	// Branches differ - return dynamic dims with max bounds from either branch
	result := make([]int, len(trueDims))
	resultBounds := make([]int, len(trueDims))
	for i := range trueDims {
		if trueDims[i] > 0 && trueDims[i] == falseDims[i] {
			result[i] = trueDims[i]
			resultBounds[i] = trueDims[i]
		} else {
			result[i] = shapes.DynamicDim
			// Use max of bounds from both branches
			maxBound := trueBounds[i]
			if falseBounds[i] > maxBound {
				maxBound = falseBounds[i]
			}
			resultBounds[i] = maxBound
		}
	}

	return result, false, resultBounds
}

// extractFromReduceMax attempts to extract the maximum value when ReduceMax is used
// to compute a shape dimension (e.g., finding the maximum sequence length).
func extractFromReduceMax(node *Node) (dims []int, allConcrete bool, bounds []int) {
	inputs := node.inputs.(*nodeInputsReduceMax)

	// If the operand is a constant tensor, try to extract the maximum value directly
	if inputs.x.Type() == NodeTypeConstant {
		tensor := inputs.x.ConstantValue()
		if tensor != nil {

			// Extract all integer values and find the maximum
			var maxVal int
			var found bool
			tensor.MustConstFlatData(func(flat any) {
				switch data := flat.(type) {
				case []int32:
					if len(data) > 0 {
						maxVal = int(data[0])
						for _, v := range data[1:] {
							if int(v) > maxVal {
								maxVal = int(v)
							}
						}
						found = true
					}
				case []int64:
					if len(data) > 0 {
						maxVal = int(data[0])
						for _, v := range data[1:] {
							if int(v) > maxVal {
								maxVal = int(v)
							}
						}
						found = true
					}
				case []int:
					if len(data) > 0 {
						maxVal = data[0]
						for _, v := range data[1:] {
							if v > maxVal {
								maxVal = v
							}
						}
						found = true
					}
				}
			})

			if found {
				return []int{maxVal}, true, []int{maxVal}
			}
		}
	}

	// Try to extract dimensions from the operand (for 1D shape tensors)
	operandDims, operandConcrete, operandBounds := ExtractShapeDimensions(inputs.x)

	if operandDims != nil && operandConcrete {
		// If all operand dimensions are concrete, compute the maximum
		if len(operandDims) > 0 {
			maxVal := operandDims[0]
			for _, d := range operandDims[1:] {
				if d > maxVal {
					maxVal = d
				}
			}
			return []int{maxVal}, true, []int{maxVal}
		}
	}

	// If operand has bounds, use the maximum bound as an upper bound
	if operandBounds != nil && len(operandBounds) > 0 {
		maxBound := operandBounds[0]
		for _, b := range operandBounds[1:] {
			if b > maxBound {
				maxBound = b
			}
		}
		return []int{shapes.DynamicDim}, false, []int{maxBound}
	}

	return []int{shapes.DynamicDim}, false, []int{DefaultBound}
}

// makeDynamicResult creates a result for when extraction fails.
// Returns DynamicDim for each dimension in the node's output shape.
func makeDynamicResult(node *Node) (dims []int, allConcrete bool, bounds []int) {
	shape := node.Shape()
	if shape.Rank() == 0 {
		// Scalar - single unknown dimension, use DefaultBound
		return []int{shapes.DynamicDim}, false, []int{DefaultBound}
	}
	if shape.Rank() != 1 {
		// Not a 1D shape tensor
		return nil, false, nil
	}

	size := shape.Dimensions[0]
	if size < 0 {
		// Dynamic size - can't determine number of dimensions
		return nil, false, nil
	}

	dims = make([]int, size)
	bounds = make([]int, size)
	for i := range dims {
		dims[i] = shapes.DynamicDim
		bounds[i] = DefaultBound // Use default bound for unknown dimensions
	}
	return dims, false, bounds
}

// extractIntegersFromTensor extracts integer values from a 1D tensor.
// Supports int32, int64, and int types.
func extractIntegersFromTensor(t *tensors.Tensor) []int {
	if t == nil {
		return nil
	}

	shape := t.Shape()
	if shape.Rank() == 0 {
		// Handle scalar case
		var result []int
		t.MustConstFlatData(func(flat any) {
			switch data := flat.(type) {
			case []int32:
				if len(data) > 0 {
					result = []int{int(data[0])}
				}
			case []int64:
				if len(data) > 0 {
					result = []int{int(data[0])}
				}
			case []int:
				if len(data) > 0 {
					result = []int{data[0]}
				}
			}
		})
		return result
	}

	if shape.Rank() != 1 {
		return nil // Shape tensors are 1D
	}

	size := shape.Size()
	result := make([]int, size)

	var extractErr bool
	t.MustConstFlatData(func(flat any) {
		switch data := flat.(type) {
		case []int32:
			for i, v := range data {
				result[i] = int(v)
			}
		case []int64:
			for i, v := range data {
				result[i] = int(v)
			}
		case []int:
			copy(result, data)
		default:
			extractErr = true
		}
	})

	if extractErr {
		return nil
	}
	return result
}

// hasConcreteBounds returns true if all bounds are positive (concrete).
func hasConcreteBounds(bounds []int) bool {
	if len(bounds) == 0 {
		return false
	}
	for _, b := range bounds {
		if b <= 0 {
			return false
		}
	}
	return true
}

// computeReshapeBounds computes bounds for DynamicReshapeWithBounds when the
// extracted dimensions don't match the operand size. It uses the maximum of
// extracted dimensions and distributes the operand's total size.
func computeReshapeBounds(operandShape shapes.Shape, extractedDims []int) []int {
	operandSize := operandShape.Size()
	if operandSize <= 0 {
		// Operand has dynamic dimensions, use extracted dims as bounds
		return extractedDims
	}

	bounds := make([]int, len(extractedDims))

	// Use the operand size and extracted rank to compute reasonable bounds
	// Strategy: for each dimension, use the max of extracted and a fair share
	extractedSize := 1
	for _, d := range extractedDims {
		if d > 0 {
			extractedSize *= d
		}
	}

	// If extracted is smaller, scale up proportionally
	if extractedSize > 0 && extractedSize < operandSize {
		scale := (operandSize + extractedSize - 1) / extractedSize // ceiling division
		for i, d := range extractedDims {
			if d > 0 {
				bounds[i] = d * scale
			} else {
				bounds[i] = operandSize // conservative
			}
		}
	} else {
		// Use extracted dims directly, they're already large enough
		copy(bounds, extractedDims)
	}

	// Ensure bounds are at least as large as operand dimensions where applicable
	operandDims := operandShape.Dimensions
	for i := range bounds {
		if i < len(operandDims) && operandDims[i] > bounds[i] {
			bounds[i] = operandDims[i]
		}
		// Ensure minimum bound of 1
		if bounds[i] <= 0 {
			bounds[i] = operandSize // conservative fallback
		}
	}

	return bounds
}

// computeFallbackBounds computes bounds when extraction fails completely.
// Uses the operand's total size as a conservative upper bound for each dimension.
func computeFallbackBounds(operandShape shapes.Shape, outputRank int) []int {
	if outputRank <= 0 {
		return nil
	}

	operandSize := operandShape.Size()
	if operandSize <= 0 {
		// Can't compute bounds for dynamic operand
		return nil
	}

	bounds := make([]int, outputRank)
	for i := range bounds {
		// Use operand's total size as conservative bound
		// This ensures the buffer is large enough for any reshape
		bounds[i] = operandSize
	}

	return bounds
}

// computeFallbackOutputDims computes reasonable output dimensions when extraction fails.
// Tries to preserve operand dimensions where possible.
// IMPORTANT: Avoids using operandSize for each dimension, as that would create huge
// shape mismatches (e.g., [98304, 98304] instead of [1536, 64]).
func computeFallbackOutputDims(operandShape shapes.Shape, outputRank int) []int {
	if outputRank <= 0 {
		return nil
	}

	operandDims := operandShape.Dimensions
	outputDims := make([]int, outputRank)

	// Strategy 1: If output rank matches operand rank, use operand dimensions directly
	// This is the most common case for reshapes that just compute the same shape dynamically
	if outputRank == len(operandDims) {
		for i := range outputDims {
			if operandDims[i] > 0 {
				outputDims[i] = operandDims[i]
			} else {
				outputDims[i] = 1
			}
		}
		return outputDims
	}

	// Strategy 2: For different ranks, try to map operand dimensions to output
	// Use operand dims for matching positions, 1 for the rest
	for i := range outputDims {
		if i < len(operandDims) && operandDims[i] > 0 {
			outputDims[i] = operandDims[i]
		} else {
			outputDims[i] = 1
		}
	}

	return outputDims
}

// DefaultBound is the fallback bound value used when no better bound can be determined.
// This is used as a sentinel to distinguish between extracted bounds and fallback bounds.
const DefaultBound = 4096

// computeEffectiveBounds computes effective bounds for each dimension using extracted
// dimensions, extracted bounds, and operand shape information.
// This provides better bounds than using a fixed default when partial information is available.
//
// Key insight: Bounds must respect element count constraints. For a reshape, the product
// of output dimensions must equal the product of input dimensions. Using the same bound
// for all unknown dimensions can lead to bounds whose product far exceeds the element count.
func computeEffectiveBounds(dims []int, extractedBounds []int, operandShape shapes.Shape) []int {
	bounds := make([]int, len(dims))

	// Check if operand has any dynamic dimensions
	operandHasDynamic := operandShape.HasSymbolicDim()
	operandSize := 0
	if !operandHasDynamic {
		operandSize = operandShape.Size()
	}

	// For scalar operands, use a small default
	operandIsScalar := operandShape.Rank() == 0

	// First pass: fill in known dimensions and identify unknowns
	knownProduct := 1
	unknownCount := 0
	unknownIndices := make([]int, 0, len(dims))

	for i, d := range dims {
		if d > 0 {
			// Concrete dimension - use as bound
			bounds[i] = d
			knownProduct *= d
		} else if extractedBounds != nil && i < len(extractedBounds) && extractedBounds[i] > 0 && extractedBounds[i] != DefaultBound {
			// Have extracted bound that's not the fallback DefaultBound
			bounds[i] = extractedBounds[i]
			knownProduct *= extractedBounds[i]
		} else {
			// Unknown - will fill in later
			bounds[i] = 0
			unknownCount++
			unknownIndices = append(unknownIndices, i)
		}
	}

	// Second pass: fill in unknown dimensions with element-count-aware bounds
	// For reshape operations, the output element count must equal input element count.
	// XLA requires: bounds_product == operand_size exactly.
	if unknownCount > 0 && operandSize > 0 && knownProduct > 0 {
		// Calculate remaining elements for unknown dimensions
		// The product of unknown dimension bounds must equal this value
		remainingElements := operandSize / knownProduct

		if unknownCount == 1 {
			// Single unknown - use exact value
			bounds[unknownIndices[0]] = remainingElements
		} else {
			// Multiple unknowns - assign bounds such that:
			// 1. Product of bounds == remainingElements (XLA requirement for reshape)
			// 2. Bounds are reasonably distributed to accommodate various factorizations
			//
			// Strategy: Use 1 for the first (n-1) unknowns and put everything in the last.
			// This is the safest approach because:
			// - Product is exactly remainingElements
			// - The large bound on the last dim can accommodate any distribution
			// - Works for matmuls where last dim often matches weights
			for i := 0; i < len(unknownIndices)-1; i++ {
				bounds[unknownIndices[i]] = 1
			}
			bounds[unknownIndices[len(unknownIndices)-1]] = remainingElements
		}
	} else if unknownCount > 0 {
		// Fallback for dynamic operand or unknown sizes
		perDimBound := DefaultBound
		if !operandHasDynamic && operandSize > 0 {
			perDimBound = operandSize
		}
		for _, idx := range unknownIndices {
			bounds[idx] = perDimBound
		}
	}

	// Handle any remaining zeros (shouldn't happen, but defensive)
	for i := range bounds {
		if bounds[i] <= 0 {
			if operandIsScalar {
				bounds[i] = 1
			} else {
				bounds[i] = DefaultBound
			}
		}
	}

	return bounds
}

// computeOutputDimsForReshape computes reasonable output dimensions for shape propagation.
// It tries to infer unknown dimensions from the operand size when possible.
// This is critical for downstream operations like MatMul that need concrete dimensions
// for batch size validation.
func computeOutputDimsForReshape(dims []int, bounds []int, operandShape shapes.Shape) []int {
	outputDims := make([]int, len(dims))

	// Check if operand has any dynamic dimensions
	// Note: Size() can give misleading results for dynamic shapes since -1 * -1 = 1
	operandHasDynamic := operandShape.HasSymbolicDim()
	operandSize := 0
	if !operandHasDynamic {
		operandSize = operandShape.Size()
	}

	operandDims := operandShape.Dimensions

	// Count known and unknown dimensions
	knownProduct := 1
	unknownCount := 0
	unknownIdx := -1

	for i, d := range dims {
		if d > 0 {
			outputDims[i] = d
			knownProduct *= d
		} else {
			unknownCount++
			unknownIdx = i
		}
	}

	// If only one unknown and we know operand size, we can infer it
	if unknownCount == 1 && operandSize > 0 && knownProduct > 0 && operandSize%knownProduct == 0 {
		inferredDim := operandSize / knownProduct
		outputDims[unknownIdx] = inferredDim
		return outputDims
	}

	// Strategy: Fill unknowns using operand dims, then infer the remaining one
	// This handles common reshape patterns like [B, S, D] -> [B, S, H, D/H]
	filledProduct := knownProduct
	filledCount := len(dims) - unknownCount
	lastUnknownIdx := -1

	for i := range dims {
		if outputDims[i] <= 0 {
			// Try to use operand dimension at same position
			if i < len(operandDims) && operandDims[i] > 0 {
				outputDims[i] = operandDims[i]
				filledProduct *= operandDims[i]
				filledCount++
			} else {
				lastUnknownIdx = i
			}
		}
	}

	// If we've filled all but one, and we know operand size, infer the last one
	if filledCount == len(dims)-1 && lastUnknownIdx >= 0 && operandSize > 0 && filledProduct > 0 && operandSize%filledProduct == 0 {
		inferredDim := operandSize / filledProduct
		outputDims[lastUnknownIdx] = inferredDim
		return outputDims
	}

	// For remaining unknowns, use bounds for XLA compilation.
	// XLA cannot handle dynamic dimensions, so we use the bound as the shape.
	for i := range dims {
		if outputDims[i] <= 0 {
			if bounds[i] > 0 {
				outputDims[i] = bounds[i]
			} else {
				// Fallback to default bound
				outputDims[i] = DefaultBound
			}
		}
	}

	return outputDims
}

// computeOutputDimsForBroadcast computes output dimensions for broadcast shape propagation.
// Uses extracted dimensions where available, and operand dimensions for broadcast axes.
func computeOutputDimsForBroadcast(dims []int, bounds []int, operandShape shapes.Shape, broadcastDimensions []int) []int {
	outputDims := make([]int, len(dims))
	operandDims := operandShape.Dimensions

	for i, d := range dims {
		if d > 0 {
			outputDims[i] = d
		} else {
			// Check if this dimension corresponds to a broadcast axis
			// Only use the operand dimension if it's > 1 (not being broadcast from 1).
			// When operandDim == 1, the operand is being broadcast, so the output
			// dimension should remain symbolic (determined by another operand at runtime).
			for operandAxis, outputAxis := range broadcastDimensions {
				if outputAxis == i && operandAxis < len(operandDims) && operandDims[operandAxis] > 1 {
					outputDims[i] = operandDims[operandAxis]
					break
				}
			}
			// If still unknown, use bound value for XLA compilation.
			// XLA cannot handle dynamic dimensions, so we use the bound as the shape.
			if outputDims[i] <= 0 {
				if bounds[i] > 0 {
					outputDims[i] = bounds[i]
				} else {
					// Fallback to default bound
					outputDims[i] = DefaultBound
				}
			}
		}
	}

	return outputDims
}

// computeBroadcastFallbackOutputDims computes reasonable output dimensions for broadcast
// when extraction fails. Uses operand dimensions for broadcast axes when available.
// For unknown dimensions, defaults to DefaultBound since XLA cannot handle dynamic dimensions.
func computeBroadcastFallbackOutputDims(operandShape shapes.Shape, outputRank int, broadcastDimensions []int) []int {
	if outputRank <= 0 {
		return nil
	}

	operandDims := operandShape.Dimensions
	outputDims := make([]int, outputRank)

	// Default to DefaultBound for all dimensions since XLA cannot handle dynamic dimensions.
	// The actual runtime size may be smaller, but XLA allocates buffers at the bound size.
	for i := range outputDims {
		outputDims[i] = DefaultBound
	}

	// Use operand dimensions for the corresponding broadcast axes when available.
	for operandAxis, outputAxis := range broadcastDimensions {
		if outputAxis >= 0 && outputAxis < outputRank && operandAxis < len(operandDims) {
			dim := operandDims[operandAxis]
			if dim > 0 {
				// Use the operand dimension (whether it's 1 or larger)
				outputDims[outputAxis] = dim
			}
		}
	}

	return outputDims
}

// computeBroadcastFallbackBounds computes bounds for DynamicBroadcastInDimWithBounds
// when extraction fails. Uses operand dimensions where applicable, but only when they're
// not being broadcast (i.e., dim > 1). When operand dim is 1, it means that dimension is
// being broadcast to a larger size, so we should use a conservative default bound.
// Returns nil if the operand has dynamic dimensions (can't compute reliable bounds).
func computeBroadcastFallbackBounds(operandShape shapes.Shape, outputRank int, broadcastDimensions []int) []int {
	if outputRank <= 0 {
		return nil
	}

	// Check if operand has any dynamic dimensions - if so, we can't compute reliable bounds
	operandDims := operandShape.Dimensions
	for _, d := range operandDims {
		if d <= 0 {
			// Can't compute bounds for dynamic operand
			return nil
		}
	}

	bounds := make([]int, outputRank)

	// Default bound - use a reasonable maximum
	const defaultBound = 4096

	for i := range bounds {
		bounds[i] = defaultBound
	}

	// Use operand dimensions for the corresponding broadcast axes, but ONLY if dim > 1.
	// When operandDim == 1, the operand is being broadcast to a larger size, so we should
	// use the default bound (which represents the maximum possible output size).
	for operandAxis, outputAxis := range broadcastDimensions {
		if outputAxis >= 0 && outputAxis < outputRank && operandAxis < len(operandDims) {
			dim := operandDims[operandAxis]
			if dim > 1 {
				// Only use dims > 1 - these are actual data dimensions, not broadcast dims
				bounds[outputAxis] = dim
			}
			// When dim == 1, leave as defaultBound since it will be broadcast to a larger size
		}
	}

	return bounds
}
