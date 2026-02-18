// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package packgemm

import (
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/gomlx/gomlx/internal/workerspool"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/x448/float16"
)

// highwayToDType converts a go-highway type to a dtypes.DType.
func highwayToDType[T hwy.Floats]() dtypes.DType {
	t := any(T(0))
	switch t.(type) {
	case hwy.Float16:
		return dtypes.Float16
	case hwy.BFloat16:
		return dtypes.BFloat16
	default:
		return dtypes.FromAny(t)
	}
}

// HalfPrecision constraint (interface) that includes all half-precision types.
type HalfPrecision interface {
	~uint16 | ~int16
}

// castHalfPrecisionSlice types a slice of type Src to a slice of type Dst, sharing the same underlying memory.
//
// It is intended for casting between compatible types, e.g. different definitions of Float16
// (hwy.Float16 vs float16.Float16) or BFloat16.
//
// It panics if the size of Src and Dst (unsafe.Sizeof) differ.
func castHalfPrecisionSlice[Dst HalfPrecision, Src HalfPrecision](src []Src) []Dst {
	return unsafe.Slice((*Dst)(unsafe.Pointer(unsafe.SliceData(src))), len(src))
}

// castBufAllocFn casts a BufAllocFn for half-precision type Src to a BufAllocFn for type Dst.
func castBufAllocFn[Dst HalfPrecision, Src HalfPrecision](src BufAllocFn[Src]) BufAllocFn[Dst] {
	return func(size int) (ref any, data []Dst) {
		ref, dataSrc := src(size)
		return ref, castHalfPrecisionSlice[Dst](dataSrc)
	}
}

// Half-Precision wrappers for the BasicSymmetric algorithms.

// TODO: the underlying implementation is not yet working.
func basicSymmetricFloat16(alpha, beta float16.Float16, lhsFlat, rhsFlat []float16.Float16,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	outputFlat []float16.Float16,
	bufAllocFn BufAllocFn[float16.Float16], bufReleaseFn BufReleaseFn, pool *workerspool.Pool) error {
	return basicSymmetricGeneric(hwy.Float16(alpha), hwy.Float16(beta),
		castHalfPrecisionSlice[hwy.Float16](lhsFlat),
		castHalfPrecisionSlice[hwy.Float16](rhsFlat),
		batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
		castHalfPrecisionSlice[hwy.Float16](outputFlat),
		castBufAllocFn[hwy.Float16](bufAllocFn), bufReleaseFn, pool)
}

func basicSymmetricBFloat16(alpha, beta bfloat16.BFloat16, lhsFlat, rhsFlat []bfloat16.BFloat16,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	outputFlat []bfloat16.BFloat16,
	bufAllocFn BufAllocFn[bfloat16.BFloat16], bufReleaseFn BufReleaseFn, pool *workerspool.Pool) error {
	return basicSymmetricGeneric(hwy.BFloat16(alpha), hwy.BFloat16(beta),
		castHalfPrecisionSlice[hwy.BFloat16](lhsFlat),
		castHalfPrecisionSlice[hwy.BFloat16](rhsFlat),
		batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
		castHalfPrecisionSlice[hwy.BFloat16](outputFlat),
		castBufAllocFn[hwy.BFloat16](bufAllocFn), bufReleaseFn, pool)
}
