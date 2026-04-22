// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xla

// This file contains manually implemented operations.

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/go-xla/pkg/stablehlo"
	stablehlotypes "github.com/gomlx/go-xla/pkg/types"
	stablehloshapes "github.com/gomlx/go-xla/pkg/types/shapes"
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
func (f *Function) Identity(x compute.Value) (compute.Value, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}
	nodes, err := f.verifyAndCastValues("OpShape", x)
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
func (f *Function) BroadcastInDim(x compute.Value, outputShape shapes.Shape, broadcastAxes []int) (compute.Value, error) {
	nodes, err := f.verifyAndCastValues("BroadcastInDim", x)
	if err != nil {
		return nil, err
	}
	value, err := stablehlo.BroadcastInDim(nodes[0].value, ShapeToXLA(outputShape), broadcastAxes)
	if err != nil {
		return nil, err
	}
	return f.newNode(value), nil
}

// Iota implements compute.Function interface.
func (f *Function) Iota(shape shapes.Shape, iotaAxis int) (compute.Value, error) {
	value, err := f.fn.Iota(ShapeToXLA(shape), iotaAxis)
	if err != nil {
		return nil, err
	}
	return f.newNode(value), nil

}

// Reshape implements compute.Function interface.
func (f *Function) Reshape(x compute.Value, dimensions ...int) (compute.Value, error) {
	nodes, err := f.verifyAndCastValues("Reshape", x)
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
	return f.newNode(value), nil
}

// Slice implements compute.Function interface.
func (f *Function) Slice(x compute.Value, starts, limits, strides []int) (compute.Value, error) {
	nodes, err := f.verifyAndCastValues("Slice", x)
	if err != nil {
		return nil, err
	}
	xNode := nodes[0]
	value, err := stablehlo.Slice(xNode.value, starts, limits, strides)
	if err != nil {
		return nil, err
	}
	return f.newNode(value), nil
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
func (f *Function) comparison(opType compute.OpType, lhs, rhs compute.Value) (compute.Value, error) {
	lhsNode, rhsNode, err := f.builder.broadcastForBinaryOps(opType, lhs, rhs)
	if err != nil {
		return nil, err
	}

	compareType := compareTypeForDType(lhsNode.shape.DType)
	var direction stablehlotypes.ComparisonDirection
	switch opType {
	case compute.OpTypeEqual:
		direction = stablehlotypes.CompareEQ
	case compute.OpTypeNotEqual:
		direction = stablehlotypes.CompareNE
	case compute.OpTypeGreaterThan:
		direction = stablehlotypes.CompareGT
	case compute.OpTypeGreaterOrEqual:
		direction = stablehlotypes.CompareGE
	case compute.OpTypeLessThan:
		direction = stablehlotypes.CompareLT
	case compute.OpTypeLessOrEqual:
		direction = stablehlotypes.CompareLE

	case compute.OpTypeEqualTotalOrder:
		direction = stablehlotypes.CompareEQ
		compareType = stablehlotypes.CompareTotalOrder
	case compute.OpTypeNotEqualTotalOrder:
		direction = stablehlotypes.CompareNE
		compareType = stablehlotypes.CompareTotalOrder
	case compute.OpTypeGreaterThanTotalOrder:
		direction = stablehlotypes.CompareGT
		compareType = stablehlotypes.CompareTotalOrder
	case compute.OpTypeGreaterOrEqualTotalOrder:
		direction = stablehlotypes.CompareGE
		compareType = stablehlotypes.CompareTotalOrder
	case compute.OpTypeLessThanTotalOrder:
		direction = stablehlotypes.CompareLT
		compareType = stablehlotypes.CompareTotalOrder
	case compute.OpTypeLessOrEqualTotalOrder:
		direction = stablehlotypes.CompareLE
		compareType = stablehlotypes.CompareTotalOrder

	default:
		return nil, errors.Errorf("unsupported comparison operation %v", opType)
	}

	value, err := stablehlo.Compare(lhsNode.value, rhsNode.value, direction, compareType)
	if err != nil {
		return nil, err
	}
	return f.newNode(value), nil
}

// Equal returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (f *Function) Equal(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparison(compute.OpTypeEqual, lhs, rhs)
}

// NotEqual returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (f *Function) NotEqual(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparison(compute.OpTypeNotEqual, lhs, rhs)
}

// GreaterThan returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (f *Function) GreaterThan(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparison(compute.OpTypeGreaterThan, lhs, rhs)
}

// GreaterOrEqual returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (f *Function) GreaterOrEqual(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparison(compute.OpTypeGreaterOrEqual, lhs, rhs)
}

// LessThan returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (f *Function) LessThan(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparison(compute.OpTypeLessThan, lhs, rhs)
}

// LessOrEqual returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (f *Function) LessOrEqual(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparison(compute.OpTypeLessOrEqual, lhs, rhs)
}

// EqualTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func (f *Function) EqualTotalOrder(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparison(compute.OpTypeEqualTotalOrder, lhs, rhs)
}

// NotEqualTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func (f *Function) NotEqualTotalOrder(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparison(compute.OpTypeNotEqualTotalOrder, lhs, rhs)
}

// GreaterThanTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func (f *Function) GreaterThanTotalOrder(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparison(compute.OpTypeGreaterThanTotalOrder, lhs, rhs)
}

// GreaterOrEqualTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func (f *Function) GreaterOrEqualTotalOrder(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparison(compute.OpTypeGreaterOrEqualTotalOrder, lhs, rhs)
}

// LessThanTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func (f *Function) LessThanTotalOrder(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparison(compute.OpTypeLessThanTotalOrder, lhs, rhs)
}

// LessOrEqualTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func (f *Function) LessOrEqualTotalOrder(lhs, rhs compute.Value) (compute.Value, error) {
	return f.comparison(compute.OpTypeLessOrEqualTotalOrder, lhs, rhs)
}

// Abs returns the Op that represents the output of the corresponding operation.
//
// It is special-cased here because StableHLO doesn't define the Abs() of complex numbers.
func (f *Function) Abs(operand compute.Value) (compute.Value, error) {
	nodes, err := f.verifyAndCastValues("Abs", operand)
	if err != nil {
		return nil, err
	}
	if nodes[0].shape.DType.IsComplex() {
		realV, err := f.Real(operand)
		if err != nil {
			return nil, err
		}
		imagV, err := f.Imag(operand)
		if err != nil {
			return nil, err
		}
		realV2, err := f.Mul(realV, realV)
		if err != nil {
			return nil, err
		}
		imagV2, err := f.Mul(imagV, imagV)
		if err != nil {
			return nil, err
		}
		lenV2, err := f.Add(realV2, imagV2)
		if err != nil {
			return nil, err
		}
		lenV, err := f.Sqrt(lenV2)
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
	return f.newNode(value), nil
}

// Conj returns the conjugate of a complex number. E.g: Conj(1+3i) = 1-3i
func (f *Function) Conj(operand compute.Value) (compute.Value, error) {
	nodes, err := f.verifyAndCastValues("Conj", operand)
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
	return f.newNode(value), nil
}

// Clamp returns the element-wise clamping operation.
//
// The values max and min can either be a scalar or have the same shape as x.
func (f *Function) Clamp(min, a compute.Value, max compute.Value) (compute.Value, error) {
	nodes, err := f.verifyAndCastValues("Clamp", min, a, max)
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
	return f.newNode(value), nil
}

// Gather is a powerful but cumbersome Gather operation. See details in the backend.
//
// Notice GoMLX backend Gather operation doesn't support batching axes, which StableHLO does.
// For compatibility, we simply leave them empty.
func (f *Function) Gather(operand, startIndices compute.Value, indexVectorAxis int, offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int, indicesAreSorted bool) (compute.Value, error) {
	nodes, err := f.verifyAndCastValues("Gather", operand, startIndices)
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
	return f.newNode(value), nil
}

// Concatenate operands on the given axis.
//
// All axes that are not being concatenated must match dimensions, except on the axes being concatenated.
// It doesn't work with scalars -- use ExpandAxes.
// If there is only one operand, it is returned and this is a no-op.
func (f *Function) Concatenate(axis int, operands ...compute.Value) (compute.Value, error) {
	operandsNodes, err := f.verifyAndCastValues("Concatenate", operands...)
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
	return f.newNode(value), nil
}

// Where implements compute.Function interface.
func (f *Function) Where(condition, onTrue, onFalse compute.Value) (compute.Value, error) {
	operandsNodes, err := f.verifyAndCastValues("Where", condition, onTrue, onFalse)
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
		onTrue, err = f.BroadcastInDim(onTrue, shapes.Make(onTrueN.shape.DType, outputDims...), nil)
		if err != nil {
			return nil, errors.WithMessage(err, "while broadcasting onTrue for op Where()")
		}
		onTrueN = onTrue.(*Node)
	}
	if onFalseN.shape.IsScalar() && len(outputDims) > 0 {
		onFalse, err = f.BroadcastInDim(onFalse, shapes.Make(onTrueN.shape.DType, outputDims...), nil)
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
	return f.newNode(value), nil
}

// IsNaN implements compute.Function interface.
func (f *Function) IsNaN(x compute.Value) (compute.Value, error) {
	result, err := f.NotEqual(x, x)
	if err != nil {
		return nil, errors.WithMessage(err, "while building op IsNaN")
	}
	return result, nil
}

const stableHLOConstantOp = "Constant"

// Bitcast implements compute.Function interface.
func (f *Function) Bitcast(x compute.Value, targetDType dtypes.DType) (compute.Value, error) {
	nodes, err := f.verifyAndCastValues("Bitcast", x)
	if err != nil {
		return nil, err
	}
	xValue := nodes[0].value
	if targetDType.IsPacked() && xValue.OpName() == stableHLOConstantOp {
		return nil, errors.Errorf("Cannot bitcast constant value to packed sub-byte type %s (x.Shape is %s): see details in "+
			"https://github.com/openxla/xla/issues/38964", targetDType, xValue.Shape())
	}
	value, err := stablehlo.BitcastConvert(nodes[0].value, targetDType)
	if err != nil {
		return nil, err
	}
	return f.newNode(value), nil
}

// Transpose implements compute.Function interface.
// It transposes input tensor x according to the given permutation axes.
func (f *Function) Transpose(x compute.Value, permutation ...int) (compute.Value, error) {
	nodes, err := f.verifyAndCastValues("Transpose", x)
	if err != nil {
		return nil, err
	}
	value, err := stablehlo.Transpose(nodes[0].value, permutation...)
	if err != nil {
		return nil, err
	}
	return f.newNode(value), nil
}

// RNGBitGenerator generates the given shape filled with random bits.
//
// It takes as input a state (usually [3]uint64) and returns the updated state and the generated values (with random bits).
//
// Currently, the backend only supports the Philox algorithm. See https://dl.acm.org/doi/10.1145/2063384.2063405
func (f *Function) RNGBitGenerator(state compute.Value, shape shapes.Shape) (newState compute.Value, values compute.Value, err error) {
	nodes, err := f.verifyAndCastValues("RNGBitGenerator", state)
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
	return f.newNode(newStateV), f.newNode(valueV), nil
}

// ConvertDType implements compute.Function interface.
func (f *Function) ConvertDType(x compute.Value, dtype dtypes.DType) (compute.Value, error) {
	nodes, err := f.verifyAndCastValues("ConvertDType", x)
	if err != nil {
		return nil, err
	}
	output, err := stablehlo.Convert(nodes[0].value, dtype)
	if err != nil {
		return nil, err
	}
	return f.newNode(output), nil
}

// Pad injects padding on the start, end, or interior (in between each element) of the given operand.
// There must be at most `operand.Rank()` axesConfig values. Missing PadAxis are assumed to be zeros,
// that is, no padding for those axes.
func (f *Function) Pad(x, fillValue compute.Value, axesConfig ...compute.PadAxis) (compute.Value, error) {
	nodes, err := f.verifyAndCastValues("ConvertDType", x, fillValue)
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
	return f.newNode(output), nil
}

// ConvGeneral implements the compute.Function interface.
func (f *Function) ConvGeneral(input, kernel compute.Value,
	axes compute.ConvolveAxesConfig, strides []int, paddings [][2]int,
	inputDilations, kernelDilations []int, channelGroupCount, batchGroupCount int) (compute.Value, error) {
	nodes, err := f.verifyAndCastValues("ConvGeneral", input, kernel)
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
	return f.newNode(output), nil
}

// Reverse implements the compute.Function interface.
func (f *Function) Reverse(x compute.Value, axes ...int) (compute.Value, error) {
	nodes, err := f.verifyAndCastValues("Reverse", x)
	if err != nil {
		return nil, err
	}
	xN := nodes[0]
	output, err := stablehlo.Reverse(xN.value, axes...)
	if err != nil {
		return nil, err
	}
	return f.newNode(output), nil
}

// FFT implements the Fast Fourier Transform operation.
// fftType specifies the type of FFT operation to perform.
// fftLength specifies the length of the transform for each axis.
func (f *Function) FFT(x compute.Value, fftType compute.FFTType, fftLength []int) (compute.Value, error) {
	nodes, err := f.verifyAndCastValues("FFT", x)
	if err != nil {
		return nil, err
	}
	xNode := nodes[0]
	value, err := stablehlo.FFT(xNode.value, stablehlotypes.FFTType(fftType), fftLength...)
	if err != nil {
		return nil, err
	}
	return f.newNode(value), nil
}

// extractStartIndexValues extracts start index values from nodes, handling both 1D tensor and scalar cases
func (f *Function) extractStartIndexValues(startIndexNodes []*Node, rank int) ([]*stablehlo.Value, error) {
	var startIndexValues []*stablehlo.Value
	if len(startIndexNodes) == 1 && !startIndexNodes[0].shape.IsScalar() && startIndexNodes[0].shape.Rank() == 1 {
		// Special case: single 1D start indices tensor
		for i := range rank {
			sliced, err := stablehlo.Slice(startIndexNodes[0].value, []int{i}, []int{i + 1}, []int{1})
			if err != nil {
				return nil, err
			}
			reshaped, err := stablehlo.Reshape(sliced, stablehloshapes.Make(startIndexNodes[0].shape.DType))
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
func (f *Function) DynamicSlice(operand compute.Value, startIndices []compute.Value, sliceDims []int) (compute.Value, error) {
	nodes, err := f.verifyAndCastValues("DynamicSlice", append([]compute.Value{operand}, startIndices...)...)
	if err != nil {
		return nil, err
	}
	operandNode := nodes[0]
	startIndexValues, err := f.extractStartIndexValues(nodes[1:], operandNode.shape.Rank())
	if err != nil {
		return nil, err
	}
	value, err := stablehlo.DynamicSlice(operandNode.value, startIndexValues, sliceDims)
	if err != nil {
		return nil, err
	}
	return f.newNode(value), nil
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
func (f *Function) DynamicUpdateSlice(operand, update compute.Value, startIndices []compute.Value) (compute.Value, error) {
	nodes, err := f.verifyAndCastValues("DynamicUpdateSlice", append([]compute.Value{operand, update}, startIndices...)...)
	if err != nil {
		return nil, err
	}
	operandNode := nodes[0]
	updateNode := nodes[1]
	startIndexValues, err := f.extractStartIndexValues(nodes[2:], operandNode.shape.Rank())
	if err != nil {
		return nil, err
	}
	value, err := stablehlo.DynamicUpdateSlice(operandNode.value, updateNode.value, startIndexValues)
	if err != nil {
		return nil, err
	}
	return f.newNode(value), nil
}

// BatchNormForInference implements compute.Function interface.
func (f *Function) BatchNormForInference(input, scale, offset, mean, variance compute.Value, epsilon float32, featureAxis int) (compute.Value, error) {
	nodes, err := f.verifyAndCastValues("BatchNormForInference", input, scale, offset, mean, variance)
	if err != nil {
		return nil, err
	}
	inputN, scaleN, offsetN, meanN, varN := nodes[0], nodes[1], nodes[2], nodes[3], nodes[4]
	value, err := stablehlo.BatchNormInference(inputN.value, scaleN.value, offsetN.value, meanN.value, varN.value, epsilon, featureAxis)
	if err != nil {
		return nil, err
	}
	return f.newNode(value), nil
}

// BatchNormForTraining implements compute.Function interface.
func (f *Function) BatchNormForTraining(input, scale, offset compute.Value, epsilon float32, featureAxis int) (output, batchMean, batchVar compute.Value, err error) {
	nodes, err := f.verifyAndCastValues("BatchNormForTraining", input, scale, offset)
	if err != nil {
		return nil, nil, nil, err
	}
	inputN, scaleN, offsetN := nodes[0], nodes[1], nodes[2]
	outputV, batchMeanV, batchVarV, err := stablehlo.BatchNormTraining(inputN.value, scaleN.value, offsetN.value, epsilon, featureAxis)
	if err != nil {
		return nil, nil, nil, err
	}
	return f.newNode(outputV), f.newNode(batchMeanV), f.newNode(batchVarV), nil
}

// BatchNormGradient implements compute.Function interface.
func (f *Function) BatchNormGradient(gradOutput, input, scale, mean, variance compute.Value, epsilon float32, featureAxis int) (gradInput, gradScale, gradOffset compute.Value, err error) {
	nodes, err := f.verifyAndCastValues("BatchNormGradient", gradOutput, input, scale, mean, variance)
	if err != nil {
		return nil, nil, nil, err
	}
	gradOutputN, inputN, scaleN, meanN, varN := nodes[0], nodes[1], nodes[2], nodes[3], nodes[4]
	gradInputV, gradScaleV, gradOffsetV, err := stablehlo.BatchNormGradient(gradOutputN.value, inputN.value, scaleN.value, meanN.value, varN.value, epsilon, featureAxis)
	if err != nil {
		return nil, nil, nil, err
	}
	return f.newNode(gradInputV), f.newNode(gradScaleV), f.newNode(gradOffsetV), nil
}

// Call implements the compute.Function interface.
func (f *Function) Call(target compute.Function, inputs ...compute.Value) ([]compute.Value, error) {
	nodes, err := f.verifyAndCastValues("Call", inputs...)
	if err != nil {
		return nil, err
	}
	targetF, ok := target.(*Function)
	if !ok {
		return nil, errors.Errorf("Call target function must be of type *xla.Function, but got %T", target)
	}
	if targetF.builder != f.builder {
		return nil, errors.Errorf("Call target function must be from the same builder")
	}

	inputValues := make([]*stablehlo.Value, len(nodes))
	for i, n := range nodes {
		inputValues[i] = n.value
	}
	outputValues, err := stablehlo.Call(targetF.fn, inputValues...)
	if err != nil {
		return nil, err
	}
	outputNodes := make([]compute.Value, len(outputValues))
	for i, v := range outputValues {
		outputNodes[i] = f.newNode(v)
	}
	return outputNodes, nil
}

// OptimizationBarrier returned values are identity to the operands, but they become only available
// in compilation time once all the inptus are calculated.
func (f *Function) OptimizationBarrier(operands ...compute.Value) ([]compute.Value, error) {
	if len(operands) == 0 {
		return nil, errors.New("OptimizationBarrier requires at least one operand")
	}
	if len(operands) == 1 {
		// No-op: a value already depends on itself.
		return operands, nil
	}
	nodes, err := f.verifyAndCastValues("OptimizationBarrier", operands...)
	if err != nil {
		return nil, err
	}
	operandValues := xslices.Map(nodes, func(n *Node) *stablehlo.Value { return n.value })
	values, err := stablehlo.OptimizationBarrier(operandValues...)
	if err != nil {
		return nil, err
	}
	outputNodes := xslices.Map(values, func(v *stablehlo.Value) compute.Value { return f.newNode(v) })
	return outputNodes, nil
}
