// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package packgemm

import (
	"unsafe"
)

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
