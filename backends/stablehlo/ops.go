package stablehlo

// This file contains manually implemented operations.

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/stablehlo"
	stablehlotypes "github.com/gomlx/stablehlo/types"
	stablehloshapes "github.com/gomlx/stablehlo/types/shapes"
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
	value, err := stablehlo.BroadcastInDim(nodes[0].value, ShapeToStableHLO(outputShape), broadcastAxes)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// Broadcast implements the backends.Builder interface.
func (b *Builder) Broadcast(x backends.Op, prefixDims ...int) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("Broadcast", x)
	if err != nil {
		return nil, err
	}
	if len(prefixDims) == 0 {
		return x, nil
	}

	xNode := nodes[0]
	shape := xNode.shape
	newDims := make([]int, shape.Rank()+len(prefixDims))
	copy(newDims, prefixDims)
	copy(newDims[len(prefixDims):], shape.Dimensions)
	outputShape := shapes.Make(shape.DType, newDims...)
	broadcastAxes := make([]int, shape.Rank())
	for i := range shape.Rank() {
		broadcastAxes[i] = i + len(prefixDims)
	}
	value, err := stablehlo.BroadcastInDim(xNode.value, ShapeToStableHLO(outputShape), broadcastAxes)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// Iota implements backends.Builder interface.
func (b *Builder) Iota(shape shapes.Shape, iotaAxis int) (backends.Op, error) {
	value, err := b.fn.Iota(ShapeToStableHLO(shape), iotaAxis)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil

}

// Reshape implements backends.Builder interface.
func (b *Builder) Reshape(x backends.Op, dimensions ...int) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("Reshape", x)
	if err != nil {
		return nil, err
	}
	xNode := nodes[0]
	dtype := xNode.shape.DType
	shape := stablehloshapes.Make(dtype, dimensions...)
	value, err := stablehlo.Reshape(xNode.value, shape)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
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

// comparison generic operation.
func (b *Builder) comparison(opType backends.OpType, lhs, rhs backends.Op) (backends.Op, error) {
	lhsNode, rhsNode, err := b.broadcastForBinaryOps(opType, lhs, rhs)
	if err != nil {
		return nil, err
	}

	// Find compareType
	dtype := lhsNode.shape.DType
	compareType := stablehlotypes.CompareFloat
	if !dtype.IsFloat() {
		if dtype.IsUnsigned() {
			compareType = stablehlotypes.CompareUnsigned
		} else {
			compareType = stablehlotypes.CompareSigned
		}
	}

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
// In practice, it can be used to perform dot products between vectors, vector/matrix multiplications or
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
	operandsValues := make([]*stablehlo.Value, len(operandsNodes))
	for i, node := range operandsNodes {
		operandsValues[i] = node.value
	}
	value, err := stablehlo.Concatenate(axis, operandsValues...)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
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
	if onTrueN.shape.IsScalar() && len(outputDims) > 0 {
		onTrue, err = b.Broadcast(onTrue, outputDims...)
		if err != nil {
			return nil, errors.WithMessage(err, "while broadcasting onTrue for op Where()")
		}
		onTrueN = onTrue.(*Node)
	}
	if onFalseN.shape.IsScalar() && len(outputDims) > 0 {
		onFalse, err = b.Broadcast(onFalse, outputDims...)
		if err != nil {
			return nil, errors.WithMessage(err, "while broadcasting onFalse for op Where()")
		}
		onFalseN = onFalse.(*Node)
	}

	// Where operation is called Select in stablehlo.
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
	value, err := stablehlo.BitcastConvert(nodes[0].value, targetDType)
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

// RngBitGenerator generates the given shape filled with random bits.
//
// It takes as input a state (usually [3]uint64) and returns the updated state and the generated values (with random bits).
//
// Currently, the backend only supports the Philox algorithm. See https://dl.acm.org/doi/10.1145/2063384.2063405
func (b *Builder) RngBitGenerator(state backends.Op, shape shapes.Shape) (newState backends.Op, values backends.Op, err error) {
	nodes, err := b.verifyAndCastValues("RngBitGenerator", state)
	if err != nil {
		return nil, nil, err
	}
	shloShape := ShapeToStableHLO(shape)
	if !shloShape.Ok() {
		return nil, nil, errors.Errorf("RngBitGenerator: invalid shape: %s", shape)
	}
	newStateV, valueV, err := stablehlo.RngBitGenerator(nodes[0].value, shloShape, stablehlotypes.RngPhilox)
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
	output, err := stablehlo.Convert(nodes[0].value, dtype)
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
