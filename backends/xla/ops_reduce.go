package xla

import (
	"reflect"

	"github.com/gomlx/go-xla/pkg/stablehlo"
	stablehlotypes "github.com/gomlx/go-xla/pkg/types"
	stablehloshapes "github.com/gomlx/go-xla/pkg/types/shapes"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/pkg/errors"
	"github.com/x448/float16"
)

func (f *Function) getReductionOp(reductionType backends.ReduceOpType) (backends.OpType, error) {
	var opType backends.OpType
	switch reductionType {
	case backends.ReduceOpMax:
		opType = backends.OpTypeReduceMax
	case backends.ReduceOpMin:
		opType = backends.OpTypeReduceMin
	case backends.ReduceOpSum:
		opType = backends.OpTypeReduceSum
	case backends.ReduceOpProduct:
		opType = backends.OpTypeReduceProduct
	default:
		return backends.OpTypeInvalid, errors.Errorf("unsupported reduction type %s", reductionType)
	}
	return opType, nil
}

func (f *Function) getReductionFn(dtype dtypes.DType, opType backends.OpType) (*stablehlo.Function, error) {
	// Create the reduction function for this dtype/op, use cache if possible.
	rKey := reductionKey{
		dtype:  dtype,
		opType: opType,
	}
	reductionFn, ok := f.builder.cacheReductions[rKey]
	if ok {
		return reductionFn, nil
	}
	reductionFn = f.fn.Closure()
	var lhs, rhs *stablehlo.Value
	var err error
	lhs, err = reductionFn.NamedInput("lhs", stablehloshapes.Make(DTypeToXLA(dtype)))
	if err != nil {
		return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
	}
	rhs, err = reductionFn.NamedInput("rhs", stablehloshapes.Make(DTypeToXLA(dtype)))
	if err != nil {
		return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
	}
	var result *stablehlo.Value
	switch opType {
	case backends.OpTypeReduceSum:
		result, err = stablehlo.Add(lhs, rhs)
	case backends.OpTypeReduceProduct:
		result, err = stablehlo.Multiply(lhs, rhs)
	case backends.OpTypeReduceMax:
		result, err = stablehlo.Maximum(lhs, rhs)
	case backends.OpTypeReduceMin:
		result, err = stablehlo.Minimum(lhs, rhs)
	case backends.OpTypeReduceBitwiseAnd:
		result, err = stablehlo.And(lhs, rhs)
	case backends.OpTypeReduceBitwiseOr:
		result, err = stablehlo.Or(lhs, rhs)
	case backends.OpTypeReduceBitwiseXor:
		result, err = stablehlo.Xor(lhs, rhs)
	case backends.OpTypeReduceLogicalAnd:
		result, err = stablehlo.And(lhs, rhs)
	case backends.OpTypeReduceLogicalOr:
		result, err = stablehlo.Or(lhs, rhs)
	case backends.OpTypeReduceLogicalXor:
		result, err = stablehlo.Xor(lhs, rhs)
	default:
		return nil, errors.Errorf("unsupported op type %s", opType)
	}
	if err != nil {
		return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
	}
	err = reductionFn.Return(result)
	if err != nil {
		return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
	}
	f.builder.cacheReductions[rKey] = reductionFn
	return reductionFn, nil
}

func (f *Function) getInitialValue(dtype dtypes.DType, opType backends.OpType) (*stablehlo.Value, error) {
	switch opType {
	case backends.OpTypeReduceSum:
		flat := scalarToFlat(0, dtype)
		if flat == nil {
			return nil, errors.Errorf("unsupported scalar for dtype %s", dtype)
		}
		initialValue, err := f.Constant(flat)
		if err != nil {
			return nil, err
		}
		return initialValue.(*Node).value, nil
	case backends.OpTypeReduceProduct:
		flat := scalarToFlat(1, dtype)
		if flat == nil {
			return nil, errors.Errorf("unsupported scalar for dtype %s", dtype)
		}
		initialValue, err := f.Constant(flat)
		if err != nil {
			return nil, err
		}
		return initialValue.(*Node).value, nil
	case backends.OpTypeReduceMax:
		flat := scalarAnyToFlat(dtype.LowestValue())
		if flat == nil {
			return nil, errors.Errorf("unsupported scalar for dtype %s", dtype)
		}
		initialValue, err := f.Constant(flat)
		if err != nil {
			return nil, err
		}
		return initialValue.(*Node).value, nil
	case backends.OpTypeReduceMin:
		flat := scalarAnyToFlat(dtype.HighestValue())
		if flat == nil {
			return nil, errors.Errorf("unsupported scalar for dtype %s", dtype)
		}
		initialValue, err := f.Constant(flat)
		if err != nil {
			return nil, err
		}
		return initialValue.(*Node).value, nil
	case backends.OpTypeReduceBitwiseAnd:
		flat := scalarToFlat(0, dtype)
		if flat == nil {
			return nil, errors.Errorf("unsupported scalar for dtype %s", dtype)
		}
		initialValue, err := f.Constant(flat)
		if err != nil {
			return nil, err
		}
		initialValue, err = f.BitwiseNot(initialValue)
		if err != nil {
			return nil, err
		}
		return initialValue.(*Node).value, nil
	case backends.OpTypeReduceBitwiseOr:
		flat := scalarToFlat(0, dtype)
		if flat == nil {
			return nil, errors.Errorf("unsupported scalar for dtype %s", dtype)
		}
		initialValue, err := f.Constant(flat)
		if err != nil {
			return nil, err
		}
		return initialValue.(*Node).value, nil
	case backends.OpTypeReduceBitwiseXor:
		flat := scalarToFlat(0, dtype)
		if flat == nil {
			return nil, errors.Errorf("unsupported scalar for dtype %s", dtype)
		}
		initialValue, err := f.Constant(flat)
		if err != nil {
			return nil, err
		}
		return initialValue.(*Node).value, nil
	case backends.OpTypeReduceLogicalAnd:
		flat := scalarToFlat(1, dtype)
		if flat == nil {
			return nil, errors.Errorf("unsupported scalar for dtype %s", dtype)
		}
		initialValue, err := f.Constant(flat)
		if err != nil {
			return nil, err
		}
		return initialValue.(*Node).value, nil
	case backends.OpTypeReduceLogicalOr:
		flat := scalarToFlat(0, dtype)
		if flat == nil {
			return nil, errors.Errorf("unsupported scalar for dtype %s", dtype)
		}
		initialValue, err := f.Constant(flat)
		if err != nil {
			return nil, err
		}
		return initialValue.(*Node).value, nil
	case backends.OpTypeReduceLogicalXor:
		flat := scalarToFlat(0, dtype)
		if flat == nil {
			return nil, errors.Errorf("unsupported scalar for dtype %s", dtype)
		}
		initialValue, err := f.Constant(flat)
		if err != nil {
			return nil, err
		}
		return initialValue.(*Node).value, nil
	default:
		return nil, errors.Errorf("unsupported (getInitialValue) reduce operation type %s", opType)
	}
}

// reduce helper for all Reduce* methods.
func (f *Function) reduce(opType backends.OpType, x backends.Value, axes ...int) (backends.Value, error) {
	nodes, err := f.verifyAndCastValues(opType.String(), x)
	if err != nil {
		return nil, err
	}
	xNode := nodes[0]
	dtype := xNode.shape.DType
	rank := xNode.shape.Rank()

	reductionFn, err := f.getReductionFn(dtype, opType)
	if err != nil {
		return nil, err
	}
	initialValue, err := f.getInitialValue(dtype, opType)
	if err != nil {
		return nil, err
	}

	// If no axes are given, reduce over all axes.
	if len(axes) == 0 {
		axes = xslices.Iota(0, rank)
	}

	value, err := stablehlo.Reduce(xNode.value, initialValue, reductionFn, axes...)
	if err != nil {
		return nil, err
	}
	return f.newNode(value), nil
}

// scalarToFlat converts a scalar value to a flat array with one element of the given dtype.
func scalarToFlat[T interface{ float64 | int | int64 }](value T, dtype dtypes.DType) any {
	switch dtype {
	case dtypes.Float32:
		return []float32{float32(value)}
	case dtypes.Float64:
		return []float64{float64(value)}
	case dtypes.Int8:
		return []int8{int8(value)}
	case dtypes.Int16:
		return []int16{int16(value)}
	case dtypes.Int32:
		return []int32{int32(value)}
	case dtypes.Int64:
		return []int64{int64(value)}
	case dtypes.Uint8:
		return []uint8{uint8(value)}
	case dtypes.Uint16:
		return []uint16{uint16(value)}
	case dtypes.Uint32:
		return []uint32{uint32(value)}
	case dtypes.Uint64:
		return []uint64{uint64(value)}
	case dtypes.Complex64:
		return []complex64{complex(float32(value), 0)}
	case dtypes.Complex128:
		return []complex128{complex(float64(value), 0)}
	case dtypes.BFloat16:
		return []bfloat16.BFloat16{bfloat16.FromFloat32(float32(value))}
	case dtypes.Float16:
		return []float16.Float16{float16.Fromfloat32(float32(value))}
	case dtypes.Bool:
		return []bool{value != 0}
	default:
		return nil
	}
}

// scalarAnyToFlat converts a scalar value to a flat array with one element of the given dtype.
func scalarAnyToFlat(valueAny any) any {
	if valueAny == nil {
		return nil
	}
	valueR := reflect.ValueOf(valueAny)
	sliceT := reflect.SliceOf(valueR.Type())
	sliceR := reflect.MakeSlice(sliceT, 1, 1)
	sliceR.Index(0).Set(valueR)
	return sliceR.Interface()
}

// ReduceSum implements the corresponding method of the backends.Function interface.
func (f *Function) ReduceSum(x backends.Value, axes ...int) (backends.Value, error) {
	opType := backends.OpTypeReduceSum
	return f.reduce(opType, x, axes...)
}

// ReduceProduct implements the corresponding method of the backends.Function interface.
func (f *Function) ReduceProduct(x backends.Value, axes ...int) (backends.Value, error) {
	opType := backends.OpTypeReduceProduct
	return f.reduce(opType, x, axes...)
}

// ReduceMax implements the corresponding method of the backends.Function interface.
func (f *Function) ReduceMax(x backends.Value, axes ...int) (backends.Value, error) {
	opType := backends.OpTypeReduceMax
	return f.reduce(opType, x, axes...)
}

// ReduceMin implements the corresponding method of the backends.Function interface.
func (f *Function) ReduceMin(x backends.Value, axes ...int) (backends.Value, error) {
	opType := backends.OpTypeReduceMin
	return f.reduce(opType, x, axes...)
}

// ReduceBitwiseAnd implements the corresponding method of the backends.Function interface.
func (f *Function) ReduceBitwiseAnd(x backends.Value, axes ...int) (backends.Value, error) {
	opType := backends.OpTypeReduceBitwiseAnd
	return f.reduce(opType, x, axes...)
}

// ReduceBitwiseOr implements the corresponding method of the backends.Function interface.
func (f *Function) ReduceBitwiseOr(x backends.Value, axes ...int) (backends.Value, error) {
	opType := backends.OpTypeReduceBitwiseOr
	return f.reduce(opType, x, axes...)
}

// ReduceBitwiseXor implements the corresponding method of the backends.Function interface.
func (f *Function) ReduceBitwiseXor(x backends.Value, axes ...int) (backends.Value, error) {
	opType := backends.OpTypeReduceBitwiseXor
	return f.reduce(opType, x, axes...)
}

// ReduceLogicalAnd implements the corresponding method of the backends.Function interface.
func (f *Function) ReduceLogicalAnd(x backends.Value, axes ...int) (backends.Value, error) {
	opType := backends.OpTypeReduceLogicalAnd
	return f.reduce(opType, x, axes...)
}

// ReduceLogicalOr implements the corresponding method of the backends.Function interface.
func (f *Function) ReduceLogicalOr(x backends.Value, axes ...int) (backends.Value, error) {
	opType := backends.OpTypeReduceLogicalOr
	return f.reduce(opType, x, axes...)
}

// ReduceLogicalXor implements the corresponding method of the backends.Function interface.
func (f *Function) ReduceLogicalXor(x backends.Value, axes ...int) (backends.Value, error) {
	opType := backends.OpTypeReduceLogicalXor
	return f.reduce(opType, x, axes...)
}

// ArgMinMax calculates the "argmin" or "argmax" across an axis of the given input array x.
//
// outputDType defines the output of the argmin/argmax, it doesn't need to be the same as the input.
// It's a form of reduction on the given axis, and that axis goes away.
// So the rank of the result is one less than the rank of x.
//
// Examples:
//
//	ArgMinMax(x={{2, 0, 7}, {-3, 4, 2}}, axis=1, isMin=true) -> {1, 0}  // (it chooses the 0 and the -3)
//	ArgMinMax(x={{2, 0, 7}, {-3, 4, 2}}, axis=0, isMin=false) -> {0, 1, 0} // (it choose the 2, 4 and 7)
func (f *Function) ArgMinMax(x backends.Value, axis int, outputDType dtypes.DType, isMin bool) (backends.Value, error) {
	opType := backends.OpTypeArgMinMax
	nodes, err := f.verifyAndCastValues(opType.String(), x)
	if err != nil {
		return nil, err
	}
	xNode := nodes[0]
	valuesDType := xNode.shape.DType
	rank := xNode.shape.Rank()
	adjustedAxis, err := adjustAxisToRank(axis, rank)
	if err != nil {
		return nil, err
	}

	// Create the reduction function for this valuesDType/op, use cache if possible.
	cacheKey := argMinMaxKey{
		valuesDType: valuesDType,
		outputDType: outputDType,
		isMin:       isMin,
	}
	reduceFn, ok := f.builder.cacheArgMinMax[cacheKey]
	if !ok {
		compareType := compareTypeForDType(valuesDType)

		// Create a new reduction function for this valuesDType/op.
		reduceFn = f.fn.Closure()
		var lhsIndex, lhsValue, rhsIndex, rhsValue *stablehlo.Value
		lhsIndex, err = reduceFn.NamedInput("lhs_idx", stablehloshapes.Make(DTypeToXLA(outputDType)))
		if err != nil {
			return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
		}
		lhsValue, err = reduceFn.NamedInput("lhs_v", stablehloshapes.Make(DTypeToXLA(valuesDType)))
		if err != nil {
			return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
		}
		rhsIndex, err = reduceFn.NamedInput("rhs_idx", stablehloshapes.Make(DTypeToXLA(outputDType)))
		if err != nil {
			return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
		}
		rhsValue, err = reduceFn.NamedInput("rhs_v", stablehloshapes.Make(DTypeToXLA(valuesDType)))
		if err != nil {
			return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
		}
		var isLeft *stablehlo.Value
		if isMin {
			isLeft, err = stablehlo.Compare(lhsValue, rhsValue, stablehlotypes.CompareLT, compareType)
		} else {
			isLeft, err = stablehlo.Compare(lhsValue, rhsValue, stablehlotypes.CompareGT, compareType)
		}
		if err != nil {
			return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
		}

		if valuesDType.IsFloat() {
			// We want to make sure to select NaNs if either side is NaN.
			// - if Rhs is NaN, then isLeft is false already, so we don't need to do anything.
			// - If Lhs is NaN, we must set it to true.
			lhsIsNan, err := stablehlo.Compare(lhsValue, lhsValue, stablehlotypes.CompareNE, stablehlotypes.CompareFloat)
			if err != nil {
				return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
			}
			isLeft, err = stablehlo.Or(isLeft, lhsIsNan)
			if err != nil {
				return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
			}
		}

		// Ties breaking if values are the same.
		isEqual, err := stablehlo.Compare(lhsValue, rhsValue, stablehlotypes.CompareEQ, compareType)
		if err != nil {
			return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
		}

		idxIfEqual, err := stablehlo.Minimum(lhsIndex, rhsIndex)
		if err != nil {
			return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
		}

		// NEW: Nested select to implement the full logic:
		// If isLeft is true, lhs wins.
		// Otherwise, check if it's a tie. If so, use the minimum index.
		// Otherwise, rhs must be the winner.
		indexIfRightOrTie, err := stablehlo.Select(isEqual, idxIfEqual, rhsIndex)
		if err != nil {
			return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
		}
		index, err := stablehlo.Select(isLeft, lhsIndex, indexIfRightOrTie)
		if err != nil {
			return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
		}
		var value *stablehlo.Value
		if isMin {
			value, err = stablehlo.Minimum(lhsValue, rhsValue)
		} else {
			value, err = stablehlo.Maximum(lhsValue, rhsValue)
		}
		if err != nil {
			return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
		}

		err = reduceFn.Return(index, value)
		if err != nil {
			return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
		}
	}
	f.builder.cacheArgMinMax[cacheKey] = reduceFn

	// Create indices and its initial value.
	indicesShape := xNode.shape.Clone()
	indicesShape.DType = outputDType
	indices, err := f.fn.Iota(ShapeToXLA(indicesShape), adjustedAxis)
	if err != nil {
		return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
	}
	flat := scalarToFlat(0, outputDType)
	initialIndex, err := f.fn.ConstantFromFlatAndDimensions(flat)
	if err != nil {
		return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
	}

	var initialValue *stablehlo.Value
	if isMin {
		flat := scalarAnyToFlat(valuesDType.HighestValue())
		if flat == nil {
			return nil, errors.Errorf("unsupported scalar for dtype %s", outputDType)
		}
		initialValue, err = f.fn.ConstantFromFlatAndDimensions(flat)
		if err != nil {
			return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
		}
	} else {
		flat := scalarAnyToFlat(valuesDType.LowestValue())
		if flat == nil {
			return nil, errors.Errorf("unsupported scalar for dtype %s", outputDType)
		}
		initialValue, err = f.fn.ConstantFromFlatAndDimensions(flat)
		if err != nil {
			return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
		}
	}

	// Create the indices and its initial value.
	results, err := stablehlo.MultiReduce(
		[]*stablehlo.Value{indices, xNode.value},
		[]*stablehlo.Value{initialIndex, initialValue},
		reduceFn, adjustedAxis)
	if err != nil {
		return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
	}
	return f.newNode(results[0]), nil
}

// ReduceWindow runs a reduction function of the type given by reduceType.
// It can be either ReduceMaxNode, ReduceSumNode or ReduceMultiplyNode.
//
// The parameter windowDimensions must be set and have a value for each axis.
// If strides is nil, it's assumed to be the same as windowDimensions -- that is, the strides jump a window at a time.
// If baseDilations, windowDilations are nil, they are assumed to be 1 (no dilation).
// If paddings is nil, they are assumed to be 0.
func (f *Function) ReduceWindow(x backends.Value, reductionType backends.ReduceOpType, windowDimensions, strides, baseDilations, windowDilations []int, paddings [][2]int) (backends.Value, error) {
	opType := backends.OpTypeReduceWindow
	nodes, err := f.verifyAndCastValues(opType.String(), x)
	if err != nil {
		return nil, err
	}
	xNode := nodes[0]
	dtype := xNode.shape.DType

	reduceOpType, err := f.getReductionOp(reductionType)
	if err != nil {
		return nil, err
	}
	reductionFn, err := f.getReductionFn(dtype, reduceOpType)
	if err != nil {
		return nil, err
	}
	initialValue, err := f.getInitialValue(dtype, reduceOpType)
	if err != nil {
		return nil, err
	}

	value, err := stablehlo.ReduceWindow(xNode.value, initialValue, reductionFn,
		windowDimensions, strides, baseDilations, windowDilations, paddings)
	if err != nil {
		return nil, err
	}
	return f.newNode(value), nil
}

func (f *Function) getSelectFn(dtype dtypes.DType, opType backends.OpType) (*stablehlo.Function, error) {
	// Create the reduction function for this dtype/op, use cache if possible.
	rKey := reductionKey{
		dtype:  dtype,
		opType: opType,
	}
	selectionFn, ok := f.builder.cacheSelections[rKey]
	if ok {
		return selectionFn, nil
	}
	selectionFn = f.fn.Closure()
	var lhs, rhs *stablehlo.Value
	var err error
	lhs, err = selectionFn.NamedInput("lhs", stablehloshapes.Make(DTypeToXLA(dtype)))
	if err != nil {
		return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
	}
	rhs, err = selectionFn.NamedInput("rhs", stablehloshapes.Make(DTypeToXLA(dtype)))
	if err != nil {
		return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
	}
	var result *stablehlo.Value
	compareType := compareTypeForDType(dtype)
	switch opType {
	case backends.OpTypeSelectAndScatterMax:
		result, err = stablehlo.Compare(lhs, rhs, stablehlotypes.CompareGE, compareType)
	case backends.OpTypeSelectAndScatterMin:
		result, err = stablehlo.Compare(lhs, rhs, stablehlotypes.CompareLE, compareType)
	default:
		return nil, errors.Errorf("unsupported op type %s for selection function", opType)
	}
	if err != nil {
		return nil, errors.WithMessagef(err, "while building selection function for %s", opType)
	}
	err = selectionFn.Return(result)
	if err != nil {
		return nil, errors.WithMessagef(err, "while building selection function for %s", opType)
	}
	f.builder.cacheSelections[rKey] = selectionFn
	return selectionFn, nil
}

// SelectAndScatterMax runs windows (similar to ReduceWindow) over the operand and
// selects the lowest values to update the output (like ScatterSum)
//
// It selects the values in the window such that it works as reverse for a PoolMax operation.
//
// Note: "Max" refers to the selection. After selected, the values are added into the output position.
//
// See details in https://openxla.org/xla/operation_semantics#selectandscatter
func (f *Function) SelectAndScatterMax(operand, source backends.Value, windowDimensions, windowStrides []int, paddings [][2]int) (backends.Value, error) {
	return f.selectAndScatterImpl(backends.OpTypeSelectAndScatterMax,
		operand, source, windowDimensions, windowStrides, paddings)
}

// SelectAndScatterMin runs windows (similar to ReduceWindow) over the operand and
// selects the lowest values to update the output (like ScatterSum)
//
// It selects the values in the window such that it works as reverse for a PoolMax operation.
//
// Note: "Min" refers to the selection. After selected, values are added into the output position.
//
// See details in https://openxla.org/xla/operation_semantics#selectandscatter
func (f *Function) SelectAndScatterMin(operand, source backends.Value, windowDimensions, windowStrides []int, paddings [][2]int) (backends.Value, error) {
	return f.selectAndScatterImpl(backends.OpTypeSelectAndScatterMin,
		operand, source, windowDimensions, windowStrides, paddings)
}

// selectAndScatterImpl implements SelectAndScatterMax and SelectAndScatterMin.
//
// See details in https://openxla.org/xla/operation_semantics#selectandscatter
func (f *Function) selectAndScatterImpl(opType backends.OpType, operand, source backends.Value, windowDimensions, windowStrides []int, paddings [][2]int) (backends.Value, error) {
	nodes, err := f.verifyAndCastValues(opType.String(), operand, source)
	if err != nil {
		return nil, err
	}
	operandN, sourceN := nodes[0], nodes[1]
	dtype := operandN.shape.DType

	selectFn, err := f.getSelectFn(dtype, opType)
	if err != nil {
		return nil, err
	}
	reductionFn, err := f.getReductionFn(dtype, backends.OpTypeReduceSum)
	if err != nil {
		return nil, err
	}

	initialValue, err := f.fn.ConstantFromFlatAndDimensions(scalarToFlat(0, dtype))
	if err != nil {
		return nil, errors.WithMessagef(err, "while building zero constant for %q", opType)
	}

	value, err := stablehlo.SelectAndScatter(operandN.value, sourceN.value, initialValue,
		selectFn, reductionFn,
		windowDimensions, windowStrides, paddings)
	if err != nil {
		return nil, err
	}
	return f.newNode(value), nil
}
