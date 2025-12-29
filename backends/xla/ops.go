package xla

// This file contains manually implemented operations.

import (
	"github.com/gomlx/go-xla/pkg/stablehlo"
	stablehlotypes "github.com/gomlx/go-xla/pkg/types"
	stablehloshapes "github.com/gomlx/go-xla/pkg/types/shapes"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// adjustAxisToRank returns a positive axis, adjusting negative numbers to the correct rank.
func adjustAxisToRank(axis, rank int) (int, error) {
	if axis < -rank || axis >= rank {
		return -1, errors.Errorf("axis %d is out of range for the rank %d", axis, rank)
	}
	if axis < 0 {
		axis += rank
	}
	return axis, nil
}

// Identity returns an Op whose output is the same as its input.
// It's a no-op that can serve as a place-holder.
func (b *Builder) Identity(x backends.Op) (backends.Op, error) {
	if err := b.CheckValid(); err != nil {
		return nil, err
	}
	nodes, err := b.verifyAndCastValues("OpShape", x)
	if err != nil {
		return nil, err
	}
	return nodes[0], nil
}

// BroadcastInDim broadcasts x to an output with the given shape.
// broadcastAxes has an output axes value for each x axes (len(broadcastAxes) == x.Shape.Rank()).
// The i-th axis of x is mapped to the broadcastAxes[i]-th dimension of the output.
// broadcastAxes must be also increasing: this operation cannot be used to transpose axes, it will only
// broadcast and introduce new axes in-between.
// This also requires that the i-th input axis is either 1 or is the same as the
// output dimension it's broadcasting into.
// For example, say operand `x = (s32)[2]{1, 2}`; outputShape = `(s32)[2,2]`:
//   - Specifying []int{1} as broadcastAxes will generate output
//     {{1, 2},
//     {1, 2}}
//   - On the other hand, specifying []int{0} as broadcastAxes
//     will generate output
//     {{1 , 1},
//     {2 , 2}}
func (b *Builder) BroadcastInDim(x backends.Op, outputShape shapes.Shape, broadcastAxes []int) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("BroadcastInDim", x)
	if err != nil {
		return nil, err
	}
	xNode := nodes[0]

	// Fast path: if both operand and output are fully concrete, use static broadcast
	if xNode.shape.IsFullyConcrete() && outputShape.IsFullyConcrete() {
		value, err := stablehlo.BroadcastInDim(xNode.value, ShapeToXLA(outputShape), broadcastAxes)
		if err != nil {
			return nil, err
		}
		return b.newNode(value), nil
	}

	// If output has dynamic dimensions but operand is concrete, try to resolve them
	if outputShape.IsDynamic() && xNode.shape.IsFullyConcrete() {
		concreteOutputDims := make([]int, len(outputShape.Dimensions))
		allResolved := true

		for i, d := range outputShape.Dimensions {
			if d >= 0 {
				// Already concrete
				concreteOutputDims[i] = d
			} else {
				// Dynamic - find which operand dimension maps to this output dimension
				operandDim := -1
				for j, bd := range broadcastAxes {
					if bd == i {
						operandDim = j
						break
					}
				}

				if operandDim >= 0 && operandDim < xNode.shape.Rank() {
					// Use the concrete operand dimension
					concreteOutputDims[i] = xNode.shape.Dimensions[operandDim]
				} else {
					// No mapping - this is a broadcast dimension, default to 1
					concreteOutputDims[i] = 1
				}

				// Verify we got a concrete value
				if concreteOutputDims[i] < 0 {
					allResolved = false
					break
				}
			}
		}

		if allResolved {
			// Use static broadcast with resolved dimensions
			resolvedShape := shapes.Make(outputShape.DType, concreteOutputDims...)
			value, err := stablehlo.BroadcastInDim(xNode.value, ShapeToXLA(resolvedShape), broadcastAxes)
			if err != nil {
				return nil, err
			}
			return b.newNode(value), nil
		}
	}

	// Dynamic path: need DynamicBroadcastInDim
	// Build a tensor with the output dimensions:
	// - Static dims (>= 0): use constant
	// - Dynamic dims (< 0): get from operand at corresponding broadcast dimension
	dimOps := make([]backends.Op, len(outputShape.Dimensions))
	for i, d := range outputShape.Dimensions {
		var dimOp backends.Op
		if d < 0 {
			// Dynamic dimension - find source from operand via broadcast mapping
			operandDim := -1
			for j, bd := range broadcastAxes {
				if bd == i {
					operandDim = j
					break
				}
			}

			if operandDim >= 0 && operandDim < xNode.shape.Rank() {
				// Get runtime size from operand
				dimOp, err = b.GetDimensionSize(x, operandDim)
				if err != nil {
					return nil, errors.Wrapf(err, "getting dimension %d size for DynamicBroadcastInDim", operandDim)
				}
				// Reshape scalar to rank 1 for concatenation
				dimOp, err = b.Reshape(dimOp, 1)
				if err != nil {
					return nil, errors.Wrapf(err, "reshaping dimension %d size", i)
				}
			} else {
				// No mapping - broadcast dimension, use 1
				dimOp, err = b.Constant([]int32{1}, 1)
				if err != nil {
					return nil, errors.Wrapf(err, "creating constant 1 for dimension %d", i)
				}
			}
		} else {
			// Static dimension
			dimOp, err = b.Constant([]int32{int32(d)}, 1)
			if err != nil {
				return nil, errors.Wrapf(err, "creating constant %d for dimension %d", d, i)
			}
		}
		dimOps[i] = dimOp
	}

	// Concatenate all dimension sizes into a 1D tensor
	outputDimsTensor, err := b.Concatenate(0, dimOps...)
	if err != nil {
		return nil, errors.WithMessage(err, "concatenating output dimensions for DynamicBroadcastInDim")
	}
	return b.DynamicBroadcastInDim(x, outputDimsTensor, broadcastAxes)
}

// Iota implements backends.Builder interface.
func (b *Builder) Iota(shape shapes.Shape, iotaAxis int) (backends.Op, error) {
	value, err := b.fn.Iota(ShapeToXLA(shape), iotaAxis)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil

}

// Reshape implements backends.Builder interface.
//
// For shapes with dynamic dimensions (negative values), this uses placeholder values
// at the StableHLO level while preserving dynamic shape information at the GoMLX level.
// For truly dynamic reshapes where the target shape is computed at runtime, use
// DynamicReshape with a shape tensor instead.
func (b *Builder) Reshape(x backends.Op, dimensions ...int) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("Reshape", x)
	if err != nil {
		return nil, err
	}
	xNode := nodes[0]
	dtype := xNode.shape.DType

	// Check if all dimensions are concrete (fast path)
	targetShape := shapes.Make(dtype, dimensions...)
	if targetShape.IsFullyConcrete() {
		shape := stablehloshapes.Make(DTypeToXLA(dtype), dimensions...)
		value, err := stablehlo.Reshape(xNode.value, shape)
		if err != nil {
			return nil, err
		}
		return b.newNode(value), nil
	}

	// Dynamic dimensions: convert to placeholder values (1) for StableHLO
	// The actual dynamic dimensions are tracked at the GoMLX level
	concreteDims := make([]int, len(dimensions))
	for i, d := range dimensions {
		if d < 0 {
			concreteDims[i] = 1 // Placeholder for dynamic dimension
		} else {
			concreteDims[i] = d
		}
	}

	shape := stablehloshapes.Make(DTypeToXLA(dtype), concreteDims...)
	value, err := stablehlo.Reshape(xNode.value, shape)
	if err != nil {
		return nil, err
	}

	// Preserve dynamic shape information at GoMLX level
	dynamicShape := shapes.MakeDynamic(dtype, dimensions...)
	return &Node{
		value:   value,
		shape:   dynamicShape,
		builder: b,
	}, nil
}

// Slice implements backends.Builder interface.
func (b *Builder) Slice(x backends.Op, starts, limits, strides []int) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("Slice", x)
	if err != nil {
		return nil, err
	}
	xNode := nodes[0]
	value, err := stablehlo.Slice(xNode.value, starts, limits, strides)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

func compareTypeForDType(dtype dtypes.DType) stablehlotypes.ComparisonType {
	// Find compareType
	compareType := stablehlotypes.CompareFloat
	if !dtype.IsFloat() && !dtype.IsComplex() {
		if dtype.IsUnsigned() || dtype == dtypes.Bool {
			compareType = stablehlotypes.CompareUnsigned
		} else {
			compareType = stablehlotypes.CompareSigned
		}
	}
	return compareType
}

// comparison generic operation.
func (b *Builder) comparison(opType backends.OpType, lhs, rhs backends.Op) (backends.Op, error) {
	lhsNode, rhsNode, err := b.broadcastForBinaryOps(opType, lhs, rhs)
	if err != nil {
		return nil, err
	}

	compareType := compareTypeForDType(lhsNode.shape.DType)
	var direction stablehlotypes.ComparisonDirection
	switch opType {
	case backends.OpTypeEqual:
		direction = stablehlotypes.CompareEQ
	case backends.OpTypeNotEqual:
		direction = stablehlotypes.CompareNE
	case backends.OpTypeGreaterThan:
		direction = stablehlotypes.CompareGT
	case backends.OpTypeGreaterOrEqual:
		direction = stablehlotypes.CompareGE
	case backends.OpTypeLessThan:
		direction = stablehlotypes.CompareLT
	case backends.OpTypeLessOrEqual:
		direction = stablehlotypes.CompareLE

	case backends.OpTypeEqualTotalOrder:
		direction = stablehlotypes.CompareEQ
		compareType = stablehlotypes.CompareTotalOrder
	case backends.OpTypeNotEqualTotalOrder:
		direction = stablehlotypes.CompareNE
		compareType = stablehlotypes.CompareTotalOrder
	case backends.OpTypeGreaterThanTotalOrder:
		direction = stablehlotypes.CompareGT
		compareType = stablehlotypes.CompareTotalOrder
	case backends.OpTypeGreaterOrEqualTotalOrder:
		direction = stablehlotypes.CompareGE
		compareType = stablehlotypes.CompareTotalOrder
	case backends.OpTypeLessThanTotalOrder:
		direction = stablehlotypes.CompareLT
		compareType = stablehlotypes.CompareTotalOrder
	case backends.OpTypeLessOrEqualTotalOrder:
		direction = stablehlotypes.CompareLE
		compareType = stablehlotypes.CompareTotalOrder

	default:
		return nil, errors.Errorf("unsupported comparison operation %v", opType)
	}

	value, err := stablehlo.Compare(lhsNode.value, rhsNode.value, direction, compareType)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// Equal returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (b *Builder) Equal(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeEqual, lhs, rhs)
}

// NotEqual returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (b *Builder) NotEqual(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeNotEqual, lhs, rhs)
}

// GreaterThan returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (b *Builder) GreaterThan(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeGreaterThan, lhs, rhs)
}

// GreaterOrEqual returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (b *Builder) GreaterOrEqual(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeGreaterOrEqual, lhs, rhs)
}

// LessThan returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (b *Builder) LessThan(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeLessThan, lhs, rhs)
}

// LessOrEqual returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (b *Builder) LessOrEqual(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeLessOrEqual, lhs, rhs)
}

// EqualTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func (b *Builder) EqualTotalOrder(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeEqualTotalOrder, lhs, rhs)
}

// NotEqualTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func (b *Builder) NotEqualTotalOrder(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeNotEqualTotalOrder, lhs, rhs)
}

// GreaterThanTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func (b *Builder) GreaterThanTotalOrder(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeGreaterThanTotalOrder, lhs, rhs)
}

// GreaterOrEqualTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func (b *Builder) GreaterOrEqualTotalOrder(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeGreaterOrEqualTotalOrder, lhs, rhs)
}

// LessThanTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func (b *Builder) LessThanTotalOrder(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeLessThanTotalOrder, lhs, rhs)
}

// LessOrEqualTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func (b *Builder) LessOrEqualTotalOrder(lhs, rhs backends.Op) (backends.Op, error) {
	return b.comparison(backends.OpTypeLessOrEqualTotalOrder, lhs, rhs)
}

// Abs returns the Op that represents the output of the corresponding operation.
//
// It is special-cased here because StableHLO doesn't define the Abs() of complex numbers.
func (b *Builder) Abs(operand backends.Op) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("Abs", operand)
	if err != nil {
		return nil, err
	}
	if nodes[0].shape.DType.IsComplex() {
		realV, err := b.Real(operand)
		if err != nil {
			return nil, err
		}
		imagV, err := b.Imag(operand)
		if err != nil {
			return nil, err
		}
		realV2, err := b.Mul(realV, realV)
		if err != nil {
			return nil, err
		}
		imagV2, err := b.Mul(imagV, imagV)
		if err != nil {
			return nil, err
		}
		lenV2, err := b.Add(realV2, imagV2)
		if err != nil {
			return nil, err
		}
		lenV, err := b.Sqrt(lenV2)
		if err != nil {
			return nil, err
		}
		return lenV, nil
	}

	// Normal absolute value.
	value, err := stablehlo.Abs(nodes[0].value)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// Conj returns the conjugate of a complex number. E.g: Conj(1+3i) = 1-3i
func (b *Builder) Conj(operand backends.Op) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("Conj", operand)
	if err != nil {
		return nil, err
	}
	operandNode := nodes[0]
	realValue, err := stablehlo.Real(operandNode.value)
	if err != nil {
		return nil, err
	}
	imagValue, err := stablehlo.Imag(operandNode.value)
	if err != nil {
		return nil, err
	}
	imagValue, err = stablehlo.Negate(imagValue)
	if err != nil {
		return nil, err
	}
	value, err := stablehlo.Complex(realValue, imagValue)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// Dot returns the "dot product" operation.
// The exact semantics of this operation depend on the ranks of the operands:
// | Input | Output | Semantics |
// | vector [n] dot vector [n] | scalar | vector dot product |
// | matrix [m x k] dot vector [k] | vector [m]	matrix-vector multiplication |
// | matrix [m x k] dot matrix [k x n] | matrix [m x n] | matrix-matrix multiplication |
// The operation performs sum of products over the second dimension of x0 (or the first if it has rank 1) and
// the first dimension of x1.
// These are the "contracted" dimensions.
// The contracted dimensions of x0 and x1 must be of the same size.
// In practice, it can be used to perform dot products between vectors, vector/matrix multiplications, or
// matrix/matrix multiplications.
func (b *Builder) Dot(lhs, rhs backends.Op) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("Dot", lhs, rhs)
	if err != nil {
		return nil, err
	}
	lhsNode, rhsNode := nodes[0], nodes[1]
	var output backends.Op
	if lhsNode.shape.Rank() == 1 && rhsNode.shape.Rank() == 1 {
		// Contracting both vectors.
		output, err = b.DotGeneral(lhsNode, []int{0}, []int{}, rhsNode, []int{0}, []int{})
	} else if lhsNode.shape.Rank() == 2 && rhsNode.shape.Rank() == 1 {
		// Contract rhs vector.
		output, err = b.DotGeneral(lhsNode, []int{1}, []int{}, rhsNode, []int{0}, []int{})
	} else if lhsNode.shape.Rank() == 1 && rhsNode.shape.Rank() == 2 {
		// Contract lhs vector.
		output, err = b.DotGeneral(lhsNode, []int{0}, []int{}, rhsNode, []int{0}, []int{})
	} else if lhsNode.shape.Rank() == 2 && rhsNode.shape.Rank() == 2 {
		// Traditional matrix multiplication:
		output, err = b.DotGeneral(lhsNode, []int{1}, []int{}, rhsNode, []int{0}, []int{})
	} else {
		return nil, errors.Errorf("Dot operands have invalid ranks: lhs=%v, rhs=%v", lhsNode.shape, rhsNode.shape)
	}
	if err != nil {
		return nil, errors.WithMessagef(err, "while building op Dot()")
	}
	return output, nil
}

// Clamp returns the element-wise clamping operation.
//
// The values max and min can either be a scalar or have the same shape as x.
func (b *Builder) Clamp(min, a backends.Op, max backends.Op) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("Clamp", min, a, max)
	if err != nil {
		return nil, err
	}
	minNode := nodes[0]
	aNode := nodes[1]
	maxNode := nodes[2]
	value, err := stablehlo.Clamp(minNode.value, aNode.value, maxNode.value)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// Gather is a powerful but cumbersome Gather operation. See details in the backend.
//
// Notice GoMLX backend Gather operation doesn't support batching axes, which StableHLO does.
// For compatibility, we simply leave them empty.
func (b *Builder) Gather(operand, startIndices backends.Op, indexVectorAxis int, offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int, indicesAreSorted bool) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("Gather", operand, startIndices)
	if err != nil {
		return nil, err
	}
	operandNode := nodes[0]
	startIndicesNode := nodes[1]
	var operandBatchingAxes, startIndicesBatchingAxes []int
	value, err := stablehlo.Gather(operandNode.value, startIndicesNode.value, indexVectorAxis,
		offsetOutputAxes, collapsedSliceAxes, operandBatchingAxes,
		startIndicesBatchingAxes, startIndexMap, sliceSizes, indicesAreSorted)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// Concatenate operands on the given axis.
//
// All axes that are not being concatenated must match dimensions, except on the axes being concatenated.
// It doesn't work with scalars -- use ExpandAxes.
// If there is only one operand, it is returned and this is a no-op.
func (b *Builder) Concatenate(axis int, operands ...backends.Op) (backends.Op, error) {
	operandsNodes, err := b.verifyAndCastValues("Concatenate", operands...)
	if err != nil {
		return nil, err
	}

	// Check if we need to broadcast operands to make their non-concat dimensions compatible
	// This happens when we have dynamic dimensions that were concretized to 1
	rank := operandsNodes[0].shape.Rank()
	adjustedAxis := axis
	if adjustedAxis < 0 {
		adjustedAxis += rank
	}

	// Find the maximum dimension for each non-concat axis
	maxDims := make([]int, rank)
	for d := 0; d < rank; d++ {
		maxDims[d] = 1
		for _, node := range operandsNodes {
			// Use GoMLX shape (which has dynamic dims) to find max
			if dim := node.shape.Dimensions[d]; dim > maxDims[d] {
				maxDims[d] = dim
			}
		}
	}

	// Broadcast operands if needed
	operandsValues := make([]*stablehlo.Value, len(operandsNodes))
	for i, node := range operandsNodes {
		needsBroadcast := false
		targetDims := make([]int, rank)
		for d := 0; d < rank; d++ {
			if d == adjustedAxis {
				// Concat axis: keep original dimension
				targetDims[d] = node.value.Shape().Dimensions[d]
			} else {
				// Non-concat axis: broadcast to max dimension if different
				targetDims[d] = maxDims[d]
				if node.value.Shape().Dimensions[d] != maxDims[d] {
					needsBroadcast = true
				}
			}
		}

		if needsBroadcast {
			// Create broadcast dimensions (all axes map to themselves)
			broadcastDims := make([]int, rank)
			for d := 0; d < rank; d++ {
				broadcastDims[d] = d
			}
			// Use our wrapper which handles dynamic shapes properly
			// Check if any dimension is dynamic (< 0)
			hasDynamic := false
			for _, d := range targetDims {
				if d < 0 {
					hasDynamic = true
					break
				}
			}
			var targetShape shapes.Shape
			if hasDynamic {
				targetShape = shapes.MakeDynamic(node.shape.DType, targetDims...)
			} else {
				targetShape = shapes.Make(node.shape.DType, targetDims...)
			}
			broadcastedOp, err := b.BroadcastInDim(node, targetShape, broadcastDims)
			if err != nil {
				return nil, errors.Wrapf(err, "broadcasting operand %d for concatenate", i)
			}
			operandsValues[i] = broadcastedOp.(*Node).value
		} else {
			operandsValues[i] = node.value
		}
	}

	value, err := stablehlo.Concatenate(axis, operandsValues...)
	if err != nil {
		return nil, err
	}

	// Check if any operand has dynamic dimensions
	// If so, we need to compute the output shape using the GoMLX shapes, not the StableHLO shapes
	hasDynamicDims := false
	for _, node := range operandsNodes {
		if node.shape.IsDynamic() {
			hasDynamicDims = true
			break
		}
	}

	if hasDynamicDims {
		// Compute output shape using GoMLX shapes (which preserve dynamic dimensions)
		outputShape := computeConcatenateShape(operandsNodes, axis)
		return &Node{
			value:   value,
			shape:   outputShape,
			builder: b,
		}, nil
	}

	return b.newNode(value), nil
}

// computeConcatenateShape computes the output shape for concatenation using GoMLX shapes
func computeConcatenateShape(operands []*Node, axis int) shapes.Shape {
	if len(operands) == 0 {
		return shapes.Shape{}
	}

	firstShape := operands[0].shape
	rank := firstShape.Rank()

	// Adjust negative axis
	if axis < 0 {
		axis += rank
	}

	// Start with a copy of the first shape
	outputDims := make([]int, rank)
	copy(outputDims, firstShape.Dimensions)

	// For the concatenation axis, sum up dimensions or use -3 if any are dynamic
	concatDim := firstShape.Dimensions[axis]
	for i := 1; i < len(operands); i++ {
		currentDim := operands[i].shape.Dimensions[axis]
		if concatDim >= 0 && currentDim >= 0 {
			concatDim += currentDim
		} else {
			concatDim = -3 // Dynamic
		}
	}
	outputDims[axis] = concatDim

	// For non-concatenation axes, if one is dynamic and another is concrete, use concrete
	for d := 0; d < rank; d++ {
		if d == axis {
			continue
		}
		for i := 1; i < len(operands); i++ {
			currentDim := operands[i].shape.Dimensions[d]
			if outputDims[d] < 0 && currentDim >= 0 {
				outputDims[d] = currentDim
			}
		}
	}

	return shapes.MakeDynamic(firstShape.DType, outputDims...)
}

// Where implements backends.Builder interface.
func (b *Builder) Where(condition, onTrue, onFalse backends.Op) (backends.Op, error) {
	operandsNodes, err := b.verifyAndCastValues("Where", condition, onTrue, onFalse)
	if err != nil {
		return nil, err
	}
	conditionN, onTrueN, onFalseN := operandsNodes[0], operandsNodes[1], operandsNodes[2]

	// Where allows onTrue and onFalse to be broadcast automatically if they are scalars, while stablehlo.Select doesn't.
	// We perform their broadcasting here but leave the condition broadcasting to be handled by stablehlo.Select.

	outputDims := conditionN.shape.Dimensions
	if !onTrueN.shape.IsScalar() {
		outputDims = onTrueN.shape.Dimensions
	}
	if !onFalseN.shape.IsScalar() {
		outputDims = onFalseN.shape.Dimensions
	}

	// Helper to create shape (static or dynamic)
	makeOutputShape := func(dtype dtypes.DType, dims []int) shapes.Shape {
		// Check if any dimension is dynamic (negative)
		for _, d := range dims {
			if d < 0 {
				return shapes.MakeDynamic(dtype, dims...)
			}
		}
		return shapes.Make(dtype, dims...)
	}

	// Check if shapes match (considering dynamic dimensions)
	shapesMatch := func(dims1, dims2 []int) bool {
		if len(dims1) != len(dims2) {
			return false
		}
		for i := range dims1 {
			d1, d2 := dims1[i], dims2[i]
			// Both dynamic - they match
			if d1 < 0 && d2 < 0 {
				continue
			}
			// One dynamic, one static - don't match (need to broadcast)
			if (d1 < 0 && d2 >= 0) || (d1 >= 0 && d2 < 0) {
				return false
			}
			// Both static - must be equal
			if d1 != d2 {
				return false
			}
		}
		return true
	}

	// Broadcast onTrue if needed
	if len(outputDims) > 0 && !shapesMatch(onTrueN.shape.Dimensions, outputDims) {
		outputShape := makeOutputShape(onTrueN.shape.DType, outputDims)
		var broadcastDims []int
		if onTrueN.shape.IsScalar() {
			broadcastDims = nil
		} else if onTrueN.shape.Rank() == len(outputDims) {
			// Same rank - identity mapping
			broadcastDims = make([]int, onTrueN.shape.Rank())
			for i := range onTrueN.shape.Rank() {
				broadcastDims[i] = i
			}
		}

		// Check if operand or output has dynamic dimensions - if so, use DynamicBroadcastInDim
		hasDynamic := false
		for _, d := range onTrueN.shape.Dimensions {
			if d < 0 {
				hasDynamic = true
				break
			}
		}
		if !hasDynamic {
			for _, d := range outputDims {
				if d < 0 {
					hasDynamic = true
					break
				}
			}
		}

		if hasDynamic {
			// Use DynamicBroadcastInDim for dynamic shapes
			// We need to compute the output shape at runtime by getting each dimension size
			// from the condition tensor (which has the target output dimensions)
			dimOps := make([]backends.Op, len(outputDims))
			for i := range outputDims {
				dimSize, err := b.GetDimensionSize(condition, i)
				if err != nil {
					return nil, errors.Wrapf(err, "while getting dimension %d size for DynamicBroadcastInDim", i)
				}
				// GetDimensionSize returns a scalar, we need to reshape it to rank 1 for concatenation
				dimSize, err = b.Reshape(dimSize, 1)
				if err != nil {
					return nil, errors.Wrapf(err, "while reshaping dimension %d size for concatenation", i)
				}
				dimOps[i] = dimSize
			}
			// Concatenate all dimension sizes into a 1D tensor
			outputDimsTensor, err := b.Concatenate(0, dimOps...)
			if err != nil {
				return nil, errors.WithMessage(err, "while concatenating output dimensions for DynamicBroadcastInDim")
			}
			onTrue, err = b.DynamicBroadcastInDim(onTrue, outputDimsTensor, broadcastDims)
			if err != nil {
				return nil, errors.WithMessage(err, "while dynamically broadcasting onTrue for op Where()")
			}
		} else {
			// Use static BroadcastInDim for static shapes
			onTrue, err = b.BroadcastInDim(onTrue, outputShape, broadcastDims)
			if err != nil {
				return nil, errors.WithMessage(err, "while broadcasting onTrue for op Where()")
			}
		}
		onTrueN = onTrue.(*Node)
		// Fix shape to preserve dynamic dimensions (BroadcastInDim converts them to 1)
		// Create a new node with the correct dynamic shape
		onTrueN = &Node{
			value:   onTrueN.value,
			shape:   outputShape,
			builder: b,
		}
		onTrue = onTrueN
	}

	// Broadcast onFalse if needed
	if len(outputDims) > 0 && !shapesMatch(onFalseN.shape.Dimensions, outputDims) {
		outputShape := makeOutputShape(onFalseN.shape.DType, outputDims)
		var broadcastDims []int
		if onFalseN.shape.IsScalar() {
			broadcastDims = nil
		} else if onFalseN.shape.Rank() == len(outputDims) {
			// Same rank - identity mapping
			broadcastDims = make([]int, onFalseN.shape.Rank())
			for i := range onFalseN.shape.Rank() {
				broadcastDims[i] = i
			}
		}

		// Check if operand or output has dynamic dimensions - if so, use DynamicBroadcastInDim
		hasDynamic := false
		for _, d := range onFalseN.shape.Dimensions {
			if d < 0 {
				hasDynamic = true
				break
			}
		}
		if !hasDynamic {
			for _, d := range outputDims {
				if d < 0 {
					hasDynamic = true
					break
				}
			}
		}

		if hasDynamic {
			// Use DynamicBroadcastInDim for dynamic shapes
			// We need to compute the output shape at runtime by getting each dimension size
			// from the condition tensor (which has the target output dimensions)
			dimOps := make([]backends.Op, len(outputDims))
			for i := range outputDims {
				dimSize, err := b.GetDimensionSize(condition, i)
				if err != nil {
					return nil, errors.Wrapf(err, "while getting dimension %d size for DynamicBroadcastInDim", i)
				}
				// GetDimensionSize returns a scalar, we need to reshape it to rank 1 for concatenation
				dimSize, err = b.Reshape(dimSize, 1)
				if err != nil {
					return nil, errors.Wrapf(err, "while reshaping dimension %d size for concatenation", i)
				}
				dimOps[i] = dimSize
			}
			// Concatenate all dimension sizes into a 1D tensor
			outputDimsTensor, err := b.Concatenate(0, dimOps...)
			if err != nil {
				return nil, errors.WithMessage(err, "while concatenating output dimensions for DynamicBroadcastInDim")
			}
			onFalse, err = b.DynamicBroadcastInDim(onFalse, outputDimsTensor, broadcastDims)
			if err != nil {
				return nil, errors.WithMessage(err, "while dynamically broadcasting onFalse for op Where()")
			}
		} else {
			// Use static BroadcastInDim for static shapes
			onFalse, err = b.BroadcastInDim(onFalse, outputShape, broadcastDims)
			if err != nil {
				return nil, errors.WithMessage(err, "while broadcasting onFalse for op Where()")
			}
		}
		onFalseN = onFalse.(*Node)
		// Fix shape to preserve dynamic dimensions (BroadcastInDim converts them to 1)
		// Create a new node with the correct dynamic shape
		onFalseN = &Node{
			value:   onFalseN.value,
			shape:   outputShape,
			builder: b,
		}
		onFalse = onFalseN
	}

	// Where operation is called Select in stablehlo.
	// TODO: There's a known issue with shape inference in go-xla when dynamic dimensions are involved.
	// The shapeinference.Select function doesn't properly handle the case where operands have
	// different representations of the same logical shape (e.g., [1,1,1] static vs [-3,-3,-3] dynamic).
	// This needs to be fixed in go-xla/internal/shapeinference/shapeinference.go
	value, err := stablehlo.Select(conditionN.value, onTrueN.value, onFalseN.value)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// IsNaN implements backends.Builder interface.
func (b *Builder) IsNaN(x backends.Op) (backends.Op, error) {
	result, err := b.NotEqual(x, x)
	if err != nil {
		return nil, errors.WithMessage(err, "while building op IsNaN")
	}
	return result, nil
}

// Bitcast implements backends.Builder interface.
func (b *Builder) Bitcast(x backends.Op, targetDType dtypes.DType) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("Bitcast", x)
	if err != nil {
		return nil, err
	}
	value, err := stablehlo.BitcastConvert(nodes[0].value, DTypeToXLA(targetDType))
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// Transpose implements backends.Builder interface.
// It transposes input tensor x according to the given permutation axes.
func (b *Builder) Transpose(x backends.Op, permutation ...int) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("Transpose", x)
	if err != nil {
		return nil, err
	}
	value, err := stablehlo.Transpose(nodes[0].value, permutation...)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// RNGBitGenerator generates the given shape filled with random bits.
//
// It takes as input a state (usually [3]uint64) and returns the updated state and the generated values (with random bits).
//
// Currently, the backend only supports the Philox algorithm. See https://dl.acm.org/doi/10.1145/2063384.2063405
func (b *Builder) RNGBitGenerator(state backends.Op, shape shapes.Shape) (newState backends.Op, values backends.Op, err error) {
	nodes, err := b.verifyAndCastValues("RNGBitGenerator", state)
	if err != nil {
		return nil, nil, err
	}
	shloShape := ShapeToXLA(shape)
	if !shloShape.Ok() {
		return nil, nil, errors.Errorf("RNGBitGenerator: invalid shape: %s", shape)
	}
	newStateV, valueV, err := stablehlo.RNGBitGenerator(nodes[0].value, shloShape, stablehlotypes.RNGPhilox)
	if err != nil {
		return nil, nil, err
	}
	return b.newNode(newStateV), b.newNode(valueV), nil
}

// ConvertDType implements backends.Builder interface.
func (b *Builder) ConvertDType(x backends.Op, dtype dtypes.DType) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("ConvertDType", x)
	if err != nil {
		return nil, err
	}
	output, err := stablehlo.Convert(nodes[0].value, DTypeToXLA(dtype))
	if err != nil {
		return nil, err
	}
	return b.newNode(output), nil
}

// Pad injects padding on the start, end, or interior (in between each element) of the given operand.
// There must be at most `operand.Rank()` axesConfig values. Missing PadAxis are assumed to be zeros,
// that is, no padding for those axes.
func (b *Builder) Pad(x, fillValue backends.Op, axesConfig ...backends.PadAxis) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("ConvertDType", x, fillValue)
	if err != nil {
		return nil, err
	}
	xN := nodes[0]
	if xN.shape.IsScalar() {
		// No axes to be padded.
		return x, nil
	}
	rank := xN.shape.Rank()
	if len(axesConfig) > rank {
		return nil, errors.Errorf("Pad: too many axesConfig values: %d > x.Rank()=%d", len(axesConfig), rank)
	}
	padStart, padEnd, padInterior := make([]int, rank), make([]int, rank), make([]int, rank)
	for i, axisConfig := range axesConfig {
		padStart[i] = axisConfig.Start
		padEnd[i] = axisConfig.End
		padInterior[i] = axisConfig.Interior
	}
	output, err := stablehlo.Pad(xN.value, nodes[1].value, padStart, padEnd, padInterior)
	if err != nil {
		return nil, err
	}
	return b.newNode(output), nil
}

// ConvGeneral implements the backends.Builder interface.
func (b *Builder) ConvGeneral(input, kernel backends.Op,
	axes backends.ConvolveAxesConfig, strides []int, paddings [][2]int,
	inputDilations, kernelDilations []int, channelGroupCount, batchGroupCount int) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("ConvGeneral", input, kernel)
	if err != nil {
		return nil, err
	}
	inputN, kernelN := nodes[0], nodes[1]
	output, err := stablehlo.Convolution(inputN.value, kernelN.value,
		strides, paddings, inputDilations, kernelDilations,
		axes.InputBatch, axes.InputChannels, axes.InputSpatial,
		axes.KernelInputChannels, axes.KernelOutputChannels, axes.KernelSpatial,
		axes.OutputBatch, axes.OutputChannels, axes.OutputSpatial,
		channelGroupCount, batchGroupCount,
		stablehlotypes.DotGeneralPrecisionDefault, stablehlotypes.DotGeneralPrecisionDefault)
	if err != nil {
		return nil, err
	}
	return b.newNode(output), nil
}

// Reverse implements the backends.Builder interface.
func (b *Builder) Reverse(x backends.Op, axes ...int) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("Reverse", x)
	if err != nil {
		return nil, err
	}
	xN := nodes[0]
	output, err := stablehlo.Reverse(xN.value, axes...)
	if err != nil {
		return nil, err
	}
	return b.newNode(output), nil
}

// FFT implements the Fast Fourier Transform operation.
// fftType specifies the type of FFT operation to perform.
// fftLength specifies the length of the transform for each axis.
func (b *Builder) FFT(x backends.Op, fftType backends.FFTType, fftLength []int) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("FFT", x)
	if err != nil {
		return nil, err
	}
	xNode := nodes[0]
	value, err := stablehlo.FFT(xNode.value, stablehlotypes.FFTType(fftType), fftLength...)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// extractStartIndexValues extracts start index values from nodes, handling both 1D tensor and scalar cases
func (b *Builder) extractStartIndexValues(startIndexNodes []*Node, rank int) ([]*stablehlo.Value, error) {
	var startIndexValues []*stablehlo.Value
	if len(startIndexNodes) == 1 && !startIndexNodes[0].shape.IsScalar() && startIndexNodes[0].shape.Rank() == 1 {
		// Special case: single 1D start indices tensor
		for i := range rank {
			sliced, err := stablehlo.Slice(startIndexNodes[0].value, []int{i}, []int{i + 1}, []int{1})
			if err != nil {
				return nil, err
			}
			reshaped, err := stablehlo.Reshape(sliced, stablehloshapes.Make(DTypeToXLA(startIndexNodes[0].shape.DType)))
			if err != nil {
				return nil, err
			}
			startIndexValues = append(startIndexValues, reshaped)
		}
	} else {
		// Normal case: one scalar tensor per axis
		startIndexValues = make([]*stablehlo.Value, len(startIndexNodes))
		for i, node := range startIndexNodes {
			startIndexValues[i] = node.value
		}
	}
	return startIndexValues, nil
}

// DynamicSlice extracts a slice from the operand at the startIndices position and the given sliceSizes.
//
// - operand: tensor from where to take the slice.
// - startIndices: scalar tensors, one per axis of operand: len(startIndices) == operand.Rank().
// - sliceSizes: static values and fixed to keep the shape of the output static.
//
// The startIndices are adjusted as follows:
//
//	adjustedStartIndices[i] = clamp(0, StartIndices[i], operand.Dimensions[i] - sliceSizes[i])
//
// See description in https://openxla.org/xla/operation_semantics#dynamicslice
func (b *Builder) DynamicSlice(operand backends.Op, startIndices []backends.Op, sliceDims []int) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("DynamicSlice", append([]backends.Op{operand}, startIndices...)...)
	if err != nil {
		return nil, err
	}
	operandNode := nodes[0]
	startIndexValues, err := b.extractStartIndexValues(nodes[1:], operandNode.shape.Rank())
	if err != nil {
		return nil, err
	}
	value, err := stablehlo.DynamicSlice(operandNode.value, startIndexValues, sliceDims)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// DynamicUpdateSlice updates the operand with the values given in update, at the position given by startIndices.
//
// - operand: original value that to be updated.
// - update: values to "paste" on top of operand, at position startIndices.
// - startIndices: scalar tensors, one per axis of operand: len(startIndices) == operand.Rank().
// - sliceSizes: static values and fixed to keep the shape of the output static.
//
// It returns a value with the same shape as the operand, with the values updated.
//
// The startIndices are adjusted as follows:
//
//	adjustedStartIndices[i] = clamp(0, StartIndices[i], operand.Dimensions[i] - update.Dimensions[i])
func (b *Builder) DynamicUpdateSlice(operand, update backends.Op, startIndices []backends.Op) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("DynamicUpdateSlice", append([]backends.Op{operand, update}, startIndices...)...)
	if err != nil {
		return nil, err
	}
	operandNode := nodes[0]
	updateNode := nodes[1]
	startIndexValues, err := b.extractStartIndexValues(nodes[2:], operandNode.shape.Rank())
	if err != nil {
		return nil, err
	}
	value, err := stablehlo.DynamicUpdateSlice(operandNode.value, updateNode.value, startIndexValues)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// BatchNormForInference implements backends.Builder interface.
func (b *Builder) BatchNormForInference(input, scale, offset, mean, variance backends.Op, epsilon float32, featureAxis int) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("BatchNormForInference", input, scale, offset, mean, variance)
	if err != nil {
		return nil, err
	}
	inputN, scaleN, offsetN, meanN, varN := nodes[0], nodes[1], nodes[2], nodes[3], nodes[4]
	value, err := stablehlo.BatchNormInference(inputN.value, scaleN.value, offsetN.value, meanN.value, varN.value, epsilon, featureAxis)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// BatchNormForTraining implements backends.Builder interface.
func (b *Builder) BatchNormForTraining(input, scale, offset backends.Op, epsilon float32, featureAxis int) (output, batchMean, batchVar backends.Op, err error) {
	nodes, err := b.verifyAndCastValues("BatchNormForTraining", input, scale, offset)
	if err != nil {
		return nil, nil, nil, err
	}
	inputN, scaleN, offsetN := nodes[0], nodes[1], nodes[2]
	outputV, batchMeanV, batchVarV, err := stablehlo.BatchNormTraining(inputN.value, scaleN.value, offsetN.value, epsilon, featureAxis)
	if err != nil {
		return nil, nil, nil, err
	}
	return b.newNode(outputV), b.newNode(batchMeanV), b.newNode(batchVarV), nil
}

// BatchNormGradient implements backends.Builder interface.
func (b *Builder) BatchNormGradient(gradOutput, input, scale, mean, variance backends.Op, epsilon float32, featureAxis int) (gradInput, gradScale, gradOffset backends.Op, err error) {
	nodes, err := b.verifyAndCastValues("BatchNormGradient", gradOutput, input, scale, mean, variance)
	if err != nil {
		return nil, nil, nil, err
	}
	gradOutputN, inputN, scaleN, meanN, varN := nodes[0], nodes[1], nodes[2], nodes[3], nodes[4]
	gradInputV, gradScaleV, gradOffsetV, err := stablehlo.BatchNormGradient(gradOutputN.value, inputN.value, scaleN.value, meanN.value, varN.value, epsilon, featureAxis)
	if err != nil {
		return nil, nil, nil, err
	}
	return b.newNode(gradInputV), b.newNode(gradScaleV), b.newNode(gradOffsetV), nil
}

// GetDimensionSize implements backends.Builder interface.
// Returns a scalar i32 containing the size of the specified dimension.
func (b *Builder) GetDimensionSize(operand backends.Op, dimension int) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("GetDimensionSize", operand)
	if err != nil {
		return nil, err
	}
	operandNode := nodes[0]
	dimension, err = adjustAxisToRank(dimension, operandNode.shape.Rank())
	if err != nil {
		return nil, err
	}
	value, err := stablehlo.GetDimensionSize(operandNode.value, dimension)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// DynamicBroadcastInDim implements backends.Builder interface.
// Broadcasts operand to the shape specified by outputDimensions (provided as a tensor).
func (b *Builder) DynamicBroadcastInDim(operand backends.Op, outputDimensions backends.Op, broadcastDimensions []int) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("DynamicBroadcastInDim", operand, outputDimensions)
	if err != nil {
		return nil, err
	}
	operandNode := nodes[0]
	outputDimensionsNode := nodes[1]
	value, err := stablehlo.DynamicBroadcastInDim(operandNode.value, outputDimensionsNode.value, broadcastDimensions)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// DynamicReshape implements backends.Builder interface.
// Reshapes operand to the shape specified by outputShape tensor.
func (b *Builder) DynamicReshape(operand backends.Op, outputShape backends.Op) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("DynamicReshape", operand, outputShape)
	if err != nil {
		return nil, err
	}
	operandNode := nodes[0]
	outputShapeNode := nodes[1]
	value, err := stablehlo.DynamicReshape(operandNode.value, outputShapeNode.value)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// DynamicReshapeWithBounds reshapes operand to the shape specified by outputShape tensor,
// using explicit dimension bounds for XLA compilation.
//
// This is useful for data-dependent shapes (e.g., NonZero output) where the shape
// cannot be determined at compile time but the caller knows upper bounds.
//
// Parameters:
//   - operand: the tensor to reshape.
//   - outputShape: a 1D tensor specifying the target shape dimensions.
//   - bounds: upper bounds for each output dimension. Must have length equal to output rank.
//
// The bounds are used by XLA to allocate buffers. At runtime, the actual dimensions
// from outputShape are used, but they must not exceed the specified bounds.
func (b *Builder) DynamicReshapeWithBounds(operand backends.Op, outputShape backends.Op, bounds []int) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("DynamicReshapeWithBounds", operand, outputShape)
	if err != nil {
		return nil, err
	}
	operandNode := nodes[0]
	outputShapeNode := nodes[1]
	value, err := stablehlo.SimpleDynamicReshape(operandNode.value, outputShapeNode.value, bounds)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// DynamicBroadcastInDimWithBounds broadcasts operand to a shape specified by outputDimensions tensor,
// using explicit dimension bounds for XLA compilation.
//
// This is useful for data-dependent shapes where the caller knows upper bounds.
//
// Parameters:
//   - operand: the tensor to broadcast.
//   - outputDimensions: a 1D tensor specifying the target shape.
//   - broadcastDimensions: maps operand axes to output axes.
//   - bounds: upper bounds for each output dimension.
func (b *Builder) DynamicBroadcastInDimWithBounds(operand backends.Op, outputDimensions backends.Op, broadcastDimensions []int, bounds []int) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("DynamicBroadcastInDimWithBounds", operand, outputDimensions)
	if err != nil {
		return nil, err
	}
	operandNode := nodes[0]
	outputDimensionsNode := nodes[1]
	value, err := stablehlo.SimpleDynamicBroadcastInDim(operandNode.value, outputDimensionsNode.value, broadcastDimensions, bounds)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// While implements backends.Builder interface.
// Executes bodyFn repeatedly while condFn returns true.
func (b *Builder) While(condFn, bodyFn any, initialStates ...backends.Op) ([]backends.Op, error) {
	if err := b.CheckValid(); err != nil {
		return nil, err
	}

	// Cast condFn and bodyFn to *stablehlo.Function
	condFunction, ok := condFn.(*stablehlo.Function)
	if !ok {
		return nil, errors.Errorf("While: condFn must be a *stablehlo.Function, got %T", condFn)
	}
	bodyFunction, ok := bodyFn.(*stablehlo.Function)
	if !ok {
		return nil, errors.Errorf("While: bodyFn must be a *stablehlo.Function, got %T", bodyFn)
	}

	// Verify and convert initial states
	nodes, err := b.verifyAndCastValues("While", initialStates...)
	if err != nil {
		return nil, err
	}

	// Extract stablehlo.Value from nodes
	initialValues := make([]*stablehlo.Value, len(nodes))
	for i, node := range nodes {
		initialValues[i] = node.value
	}

	// Call stablehlo.While
	resultValues, err := stablehlo.While(condFunction, bodyFunction, initialValues...)
	if err != nil {
		return nil, errors.WithMessage(err, "while building While operation")
	}

	// Convert results back to backends.Op
	results := make([]backends.Op, len(resultValues))
	for i, value := range resultValues {
		results[i] = b.newNode(value)
	}

	return results, nil
}
