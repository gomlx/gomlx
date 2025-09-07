package stablehlo

import (
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
	method func(*stablehlo.Function, *stablehlo.Value, *stablehlo.Value) (*stablehlo.Value, error),
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
		reductionFn = b.fn.Builder.NewInlineFunction()
		lhs := reductionFn.NewNamedInput("lhs", stablehloshapes.Make(dtype))
		rhs := reductionFn.NewNamedInput("rhs", stablehloshapes.Make(dtype))
		result, err := method(reductionFn, lhs, rhs)
		if err != nil {
			return nil, errors.WithMessagef(err, "while building reduction function for %s", opType)
		}
		reductionFn.Return(result)
		b.cacheReductions[rKey] = reductionFn
	}

	// If no axes are given, reduce over all axes.
	if len(axes) == 0 {
		axes = xslices.Iota(0, rank)
	}

	value, err := b.fn.Reduce(xNode.value, initialValue.value, reductionFn, axes...)
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
	default:
		return nil
	}
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
	return b.reduce(opType, (*stablehlo.Function).Add, initialValue.(*Node), x, axes...)
}
