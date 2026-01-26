// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"reflect"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
)

// copyFlatReflect is the old reflection-based implementation for comparison.
func copyFlatReflect(flatDst, flatSrc any) {
	reflect.Copy(reflect.ValueOf(flatDst), reflect.ValueOf(flatSrc))
}

// makeSliceReflect is the old reflection-based implementation for comparison.
func makeSliceReflect(dtype dtypes.DType, length int) any {
	return reflect.MakeSlice(reflect.SliceOf(dtype.GoType()), length, length).Interface()
}

func BenchmarkCopyFlat_Float32_Small(b *testing.B) {
	src := make([]float32, 64)
	dst := make([]float32, 64)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copyFlat(dst, src)
	}
}

func BenchmarkCopyFlat_Float32_Medium(b *testing.B) {
	src := make([]float32, 1024)
	dst := make([]float32, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copyFlat(dst, src)
	}
}

func BenchmarkCopyFlat_Float32_Large(b *testing.B) {
	src := make([]float32, 65536)
	dst := make([]float32, 65536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copyFlat(dst, src)
	}
}

func BenchmarkCopyFlatReflect_Float32_Small(b *testing.B) {
	src := make([]float32, 64)
	dst := make([]float32, 64)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copyFlatReflect(dst, src)
	}
}

func BenchmarkCopyFlatReflect_Float32_Medium(b *testing.B) {
	src := make([]float32, 1024)
	dst := make([]float32, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copyFlatReflect(dst, src)
	}
}

func BenchmarkCopyFlatReflect_Float32_Large(b *testing.B) {
	src := make([]float32, 65536)
	dst := make([]float32, 65536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copyFlatReflect(dst, src)
	}
}

func BenchmarkMakeSlice_Float32_Small(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = makeSliceForDType(dtypes.Float32, 64)
	}
}

func BenchmarkMakeSlice_Float32_Medium(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = makeSliceForDType(dtypes.Float32, 1024)
	}
}

func BenchmarkMakeSlice_Float32_Large(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = makeSliceForDType(dtypes.Float32, 65536)
	}
}

func BenchmarkMakeSliceReflect_Float32_Small(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = makeSliceReflect(dtypes.Float32, 64)
	}
}

func BenchmarkMakeSliceReflect_Float32_Medium(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = makeSliceReflect(dtypes.Float32, 1024)
	}
}

func BenchmarkMakeSliceReflect_Float32_Large(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = makeSliceReflect(dtypes.Float32, 65536)
	}
}

func BenchmarkMakeSlice_Int64_Small(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = makeSliceForDType(dtypes.Int64, 64)
	}
}

func BenchmarkMakeSliceReflect_Int64_Small(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = makeSliceReflect(dtypes.Int64, 64)
	}
}
