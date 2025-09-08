package stablehlo

import (
	"reflect"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/gomlx/stablehlo"
	stablehloshapes "github.com/gomlx/stablehlo/types/shapes"
	"github.com/pkg/errors"
	"github.com/x448/float16"
)

// reduce helper for all Reduce* methods.
func (b *Builder) reduce(opType backends.OpType,
	reductionOpFn func(*stablehlo.Value, *stablehlo.Value) (*stablehlo.Value, error),
	initialValue *Node, x backends.Op, axes ...int) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues(opType.String(), x)
	if err != nil {
		return nil, err
	}
	xNode := nodes[0]
	dtype := xNode.shape.DType
	rank := xNode.shape.Rank()

	// Create the reduction function for this dtype/op, use cache if possible.
	rKey := reductionKey{
		dtype:  dtype,
		opType: opType,
	}
	reductionFn, ok := b.cacheReductions[rKey]
	if !ok {
		// Create a new reduction function for this dtype/op.
		reductionFn = b.fn.Closure()
		lhs := reductionFn.NamedInput("lhs", stablehloshapes.Make(dtype))
		rhs := reductionFn.NamedInput("rhs", stablehloshapes.Make(dtype))
		result, err := reductionOpFn(lhs, rhs)
		if err != nil {
			return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
		}
		err = reductionFn.Return(result)
		if err != nil {
			return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
		}
		b.cacheReductions[rKey] = reductionFn
	}

	// If no axes are given, reduce over all axes.
	if len(axes) == 0 {
		axes = xslices.Iota(0, rank)
	}

	value, err := stablehlo.Reduce(xNode.value, initialValue.value, reductionFn, axes...)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
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

// ReduceSum implements the corresponding method of the backends.Builder interface.
func (b *Builder) ReduceSum(x backends.Op, axes ...int) (backends.Op, error) {
	opType := backends.OpTypeReduceSum
	dtype := x.(*Node).shape.DType
	flat := scalarToFlat(0, dtype)
	if flat == nil {
		return nil, errors.Errorf("unsupported scalar for dtype %s", dtype)
	}
	initialValue, err := b.Constant(flat)
	if err != nil {
		return nil, err
	}
	return b.reduce(opType, stablehlo.Add, initialValue.(*Node), x, axes...)
}

// ReduceProduct implements the corresponding method of the backends.Builder interface.
func (b *Builder) ReduceProduct(x backends.Op, axes ...int) (backends.Op, error) {
	opType := backends.OpTypeReduceProduct
	dtype := x.(*Node).shape.DType
	flat := scalarToFlat(1, dtype)
	if flat == nil {
		return nil, errors.Errorf("unsupported scalar for dtype %s", dtype)
	}
	initialValue, err := b.Constant(flat)
	if err != nil {
		return nil, err
	}
	return b.reduce(opType, stablehlo.Multiply, initialValue.(*Node), x, axes...)
}

// ReduceMax implements the corresponding method of the backends.Builder interface.
func (b *Builder) ReduceMax(x backends.Op, axes ...int) (backends.Op, error) {
	opType := backends.OpTypeReduceMax
	dtype := x.(*Node).shape.DType
	flat := scalarAnyToFlat(dtype.LowestValue())
	if flat == nil {
		return nil, errors.Errorf("unsupported scalar for dtype %s", dtype)
	}
	initialValue, err := b.Constant(flat)
	if err != nil {
		return nil, err
	}
	return b.reduce(opType, stablehlo.Maximum, initialValue.(*Node), x, axes...)
}

// ReduceMin implements the corresponding method of the backends.Builder interface.
func (b *Builder) ReduceMin(x backends.Op, axes ...int) (backends.Op, error) {
	opType := backends.OpTypeReduceMin
	dtype := x.(*Node).shape.DType
	flat := scalarAnyToFlat(dtype.HighestValue())
	if flat == nil {
		return nil, errors.Errorf("unsupported scalar for dtype %s", dtype)
	}
	initialValue, err := b.Constant(flat)
	if err != nil {
		return nil, err
	}
	return b.reduce(opType, stablehlo.Minimum, initialValue.(*Node), x, axes...)
}

// ReduceBitwiseAnd implements the corresponding method of the backends.Builder interface.
func (b *Builder) ReduceBitwiseAnd(x backends.Op, axes ...int) (backends.Op, error) {
	opType := backends.OpTypeReduceBitwiseAnd
	dtype := x.(*Node).shape.DType
	flat := scalarToFlat(0, dtype)
	if flat == nil {
		return nil, errors.Errorf("unsupported scalar for dtype %s", dtype)
	}
	initialValue, err := b.Constant(flat)
	if err != nil {
		return nil, err
	}
	initialValue, err = b.BitwiseNot(initialValue)
	return b.reduce(opType, stablehlo.And, initialValue.(*Node), x, axes...)
}

// ReduceBitwiseOr implements the corresponding method of the backends.Builder interface.
func (b *Builder) ReduceBitwiseOr(x backends.Op, axes ...int) (backends.Op, error) {
	opType := backends.OpTypeReduceBitwiseOr
	dtype := x.(*Node).shape.DType
	flat := scalarToFlat(0, dtype)
	if flat == nil {
		return nil, errors.Errorf("unsupported scalar for dtype %s", dtype)
	}
	initialValue, err := b.Constant(flat)
	if err != nil {
		return nil, err
	}
	return b.reduce(opType, stablehlo.Or, initialValue.(*Node), x, axes...)
}

// ReduceBitwiseXor implements the corresponding method of the backends.Builder interface.
func (b *Builder) ReduceBitwiseXor(x backends.Op, axes ...int) (backends.Op, error) {
	opType := backends.OpTypeReduceBitwiseXor
	dtype := x.(*Node).shape.DType
	flat := scalarToFlat(0, dtype)
	if flat == nil {
		return nil, errors.Errorf("unsupported scalar for dtype %s", dtype)
	}
	initialValue, err := b.Constant(flat)
	if err != nil {
		return nil, err
	}
	return b.reduce(opType, stablehlo.Xor, initialValue.(*Node), x, axes...)
}

// ReduceLogicalAnd implements the corresponding method of the backends.Builder interface.
func (b *Builder) ReduceLogicalAnd(x backends.Op, axes ...int) (backends.Op, error) {
	opType := backends.OpTypeReduceLogicalAnd
	dtype := x.(*Node).shape.DType
	flat := scalarToFlat(1, dtype)
	if flat == nil {
		return nil, errors.Errorf("unsupported scalar for dtype %s", dtype)
	}
	initialValue, err := b.Constant(flat)
	if err != nil {
		return nil, err
	}
	return b.reduce(opType, stablehlo.And, initialValue.(*Node), x, axes...)
}

// ReduceLogicalOr implements the corresponding method of the backends.Builder interface.
func (b *Builder) ReduceLogicalOr(x backends.Op, axes ...int) (backends.Op, error) {
	opType := backends.OpTypeReduceLogicalOr
	dtype := x.(*Node).shape.DType
	flat := scalarToFlat(0, dtype)
	if flat == nil {
		return nil, errors.Errorf("unsupported scalar for dtype %s", dtype)
	}
	initialValue, err := b.Constant(flat)
	if err != nil {
		return nil, err
	}
	return b.reduce(opType, stablehlo.Or, initialValue.(*Node), x, axes...)
}

// ReduceLogicalXor implements the corresponding method of the backends.Builder interface.
func (b *Builder) ReduceLogicalXor(x backends.Op, axes ...int) (backends.Op, error) {
	opType := backends.OpTypeReduceLogicalXor
	dtype := x.(*Node).shape.DType
	flat := scalarToFlat(0, dtype)
	if flat == nil {
		return nil, errors.Errorf("unsupported scalar for dtype %s", dtype)
	}
	initialValue, err := b.Constant(flat)
	if err != nil {
		return nil, err
	}
	return b.reduce(opType, stablehlo.Xor, initialValue.(*Node), x, axes...)
}

/*
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
func (b *Builder) ArgMinMax(x backends.Op, axis int, outputDType dtypes.DType, isMin bool) (backends.Op, error) {
	opType := backends.OpTypeArgMinMax
	nodes, err := b.verifyAndCastValues(opType.String(), x)
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
	reduceFn, ok := b.cacheArgMinMax[cacheKey]
	if !ok {
		// Create a new reduction function for this valuesDType/op.
		reduceFn = b.fn.Closure()
		lhsIndex := reduceFn.NamedInput("lhs_idx", stablehloshapes.Make(outputDType))
		lhsValue := reduceFn.NamedInput("lhs_v", stablehloshapes.Make(valuesDType))
		rhsIndex := reduceFn.NamedInput("rhs_idx", stablehloshapes.Make(outputDType))
		rhsValue := reduceFn.NamedInput("rhs_v", stablehloshapes.Make(valuesDType))
		if isMin {

		}
		result, err := (reduceFn, lhsIndex, lhsValue, rhsIndex, rhsValue)
		if err != nil {
			return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
		}
		reduceFn.Return(result)
		b.cacheReductions[cacheKey] = reduceFn
	}

	// If no axes are given, reduce over all axes.
	if len(axes) == 0 {
		axes = xslices.Iota(0, rank)
	}

	value, err := b.fn.Reduce(xNode.value, initialValue.value, reduceFn, axes...)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}
*/
