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

// Iterator pooling benchmarks

func BenchmarkBroadcastIterator_Pooled(b *testing.B) {
	for i := 0; i < b.N; i++ {
		it := getBroadcastIterator(4)
		putBroadcastIterator(it)
	}
}

func BenchmarkBroadcastIterator_Alloc(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = &broadcastIterator{
			perAxesIdx:  make([]int, 4),
			targetDims:  make([]int, 4),
			isBroadcast: make([]bool, 4),
			strides:     make([]int, 4),
		}
	}
}

func BenchmarkTransposeIterator_Pooled(b *testing.B) {
	for i := 0; i < b.N; i++ {
		it := getTransposeIterator(4)
		putTransposeIterator(it)
	}
}

func BenchmarkTransposeIterator_Alloc(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = &transposeIterator{
			perAxisIdx:     make([]int, 4),
			perAxisStrides: make([]int, 4),
			dimensions:     make([]int, 4),
		}
	}
}

func BenchmarkReduceIterator_Pooled(b *testing.B) {
	for i := 0; i < b.N; i++ {
		it := getReduceIterator(4)
		putReduceIterator(it)
	}
}

func BenchmarkReduceIterator_Alloc(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = &reduceOutputIterator{
			perAxisIdx:    make([]int, 4),
			dimensions:    make([]int, 4),
			perAxisStride: make([]int, 4),
		}
	}
}

// sink prevents compiler from optimizing away allocations
var sink any

// These benchmarks force heap escape to simulate real usage

func BenchmarkBroadcastIterator_Pooled_Escape(b *testing.B) {
	for i := 0; i < b.N; i++ {
		it := getBroadcastIterator(4)
		sink = it // force escape
		putBroadcastIterator(it)
	}
}

func BenchmarkBroadcastIterator_Alloc_Escape(b *testing.B) {
	for i := 0; i < b.N; i++ {
		it := &broadcastIterator{
			perAxesIdx:  make([]int, 4),
			targetDims:  make([]int, 4),
			isBroadcast: make([]bool, 4),
			strides:     make([]int, 4),
		}
		sink = it // force escape
	}
}

func BenchmarkTransposeIterator_Pooled_Escape(b *testing.B) {
	for i := 0; i < b.N; i++ {
		it := getTransposeIterator(4)
		sink = it // force escape
		putTransposeIterator(it)
	}
}

func BenchmarkTransposeIterator_Alloc_Escape(b *testing.B) {
	for i := 0; i < b.N; i++ {
		it := &transposeIterator{
			perAxisIdx:     make([]int, 4),
			perAxisStrides: make([]int, 4),
			dimensions:     make([]int, 4),
		}
		sink = it // force escape
	}
}

func BenchmarkReduceIterator_Pooled_Escape(b *testing.B) {
	for i := 0; i < b.N; i++ {
		it := getReduceIterator(4)
		sink = it // force escape
		putReduceIterator(it)
	}
}

func BenchmarkReduceIterator_Alloc_Escape(b *testing.B) {
	for i := 0; i < b.N; i++ {
		it := &reduceOutputIterator{
			perAxisIdx:    make([]int, 4),
			dimensions:    make([]int, 4),
			perAxisStride: make([]int, 4),
		}
		sink = it // force escape
	}
}

// While state workspace benchmarks

func BenchmarkWhileStateWorkspace_Pooled(b *testing.B) {
	for i := 0; i < b.N; i++ {
		ws := getWhileStateWorkspace(4)
		putWhileStateWorkspace(ws)
	}
}

func BenchmarkWhileStateWorkspace_Alloc(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = &whileStateWorkspace{
			state:       make([]*Buffer, 4),
			donateState: make([]bool, 4),
		}
	}
}

func BenchmarkWhileStateWorkspace_Pooled_Escape(b *testing.B) {
	for i := 0; i < b.N; i++ {
		ws := getWhileStateWorkspace(4)
		sink = ws // force escape
		putWhileStateWorkspace(ws)
	}
}

func BenchmarkWhileStateWorkspace_Alloc_Escape(b *testing.B) {
	for i := 0; i < b.N; i++ {
		ws := &whileStateWorkspace{
			state:       make([]*Buffer, 4),
			donateState: make([]bool, 4),
		}
		sink = ws // force escape
	}
}

// Sort workspace benchmarks

func BenchmarkSortWorkspace_Pooled(b *testing.B) {
	for i := 0; i < b.N; i++ {
		ws := getSortWorkspace(2, 100)
		putSortWorkspace(ws)
	}
}

func BenchmarkSortWorkspace_Alloc(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = &sortWorkspace{
			outputs:    make([]*Buffer, 2),
			indices:    make([]int, 100),
			compInputs: make([]*Buffer, 4),
		}
	}
}

func BenchmarkSortWorkspace_Pooled_Escape(b *testing.B) {
	for i := 0; i < b.N; i++ {
		ws := getSortWorkspace(2, 100)
		sink = ws // force escape
		putSortWorkspace(ws)
	}
}

func BenchmarkSortWorkspace_Alloc_Escape(b *testing.B) {
	for i := 0; i < b.N; i++ {
		ws := &sortWorkspace{
			outputs:    make([]*Buffer, 2),
			indices:    make([]int, 100),
			compInputs: make([]*Buffer, 4),
		}
		sink = ws // force escape
	}
}

// Closure inputs workspace benchmarks

func BenchmarkClosureInputsWorkspace_Pooled(b *testing.B) {
	captureCounts := []int{3, 5}
	for i := 0; i < b.N; i++ {
		ws := getClosureInputsWorkspace(captureCounts)
		putClosureInputsWorkspace(ws)
	}
}

func BenchmarkClosureInputsWorkspace_Alloc(b *testing.B) {
	for i := 0; i < b.N; i++ {
		closureInputs := make([]ClosureInputs, 2)
		closureInputs[0] = ClosureInputs{
			Buffers: make([]*Buffer, 3),
			Owned:   make([]bool, 3),
		}
		closureInputs[1] = ClosureInputs{
			Buffers: make([]*Buffer, 5),
			Owned:   make([]bool, 5),
		}
		sink = closureInputs
	}
}

func BenchmarkClosureInputsWorkspace_Pooled_Escape(b *testing.B) {
	captureCounts := []int{3, 5}
	for i := 0; i < b.N; i++ {
		ws := getClosureInputsWorkspace(captureCounts)
		sink = ws // force escape
		putClosureInputsWorkspace(ws)
	}
}

func BenchmarkClosureInputsWorkspace_Alloc_Escape(b *testing.B) {
	for i := 0; i < b.N; i++ {
		closureInputs := make([]ClosureInputs, 2)
		closureInputs[0] = ClosureInputs{
			Buffers: make([]*Buffer, 3),
			Owned:   make([]bool, 3),
		}
		closureInputs[1] = ClosureInputs{
			Buffers: make([]*Buffer, 5),
			Owned:   make([]bool, 5),
		}
		sink = closureInputs // force escape
	}
}
