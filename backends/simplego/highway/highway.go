// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package highway provides SIMD-accelerated matrix multiplication using go-highway.
// This package requires Go 1.26+ due to its dependency on go-highway.
//
// To enable highway support, import this package for its side effects:
//
//	import _ "github.com/gomlx/gomlx/backends/simplego/highway"
package highway

import (
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul"
	"github.com/ajroetker/go-highway/hwy/contrib/workerpool"
	"github.com/gomlx/gomlx/backends/simplego"
	"github.com/gomlx/gomlx/backends/simplego/packgemm"
	"github.com/gomlx/gomlx/internal/workerspool"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/gomlx/gomlx/pkg/support/xsync"
	"github.com/pkg/errors"
	"github.com/x448/float16"
)

// hwyPool is the shared go-highway worker pool for intra-matrix parallelism.
// Created lazily on first use.
var hwyPool *workerpool.Pool

func init() {
	simplego.RegisterHighway(impl{})
	// Create a shared highway worker pool for intra-matrix parallelism
	// Pass 0 to use GOMAXPROCS workers
	hwyPool = workerpool.New(0)
}

// impl implements simplego.HighwayMatMul interface.
type impl struct{}

func (impl) HasDTypeSupport(input, output dtypes.DType) bool {
	return HasDTypeSupport(input, output)
}

func (impl) MatMulDynamic(inputDType, outputDType dtypes.DType,
	lhsFlat, rhsFlat any, batchSize,
	lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat any,
	bufAllocAnyFn packgemm.BufAllocAnyFn, bufReleaseFn packgemm.BufReleaseFn,
	pool *workerspool.Pool) error {
	return MatMulDynamic(inputDType, outputDType, lhsFlat, rhsFlat, batchSize,
		lhsCrossSize, rhsCrossSize, contractingSize, outputFlat,
		bufAllocAnyFn, bufReleaseFn, pool)
}

func (impl) MatMulKLast(inputDType, outputDType dtypes.DType,
	lhsFlat, rhsFlat any, batchSize,
	lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat any,
	pool *workerspool.Pool) error {
	return MatMulKLast(inputDType, outputDType, lhsFlat, rhsFlat, batchSize,
		lhsCrossSize, rhsCrossSize, contractingSize, outputFlat, pool)
}

func (impl) Transpose2D(dtype dtypes.DType, src any, m, k int, dst any) bool {
	return Transpose2D(dtype, src, m, k, dst)
}

// Transpose2D transposes an M×K row-major matrix to K×M using SIMD.
// Returns false if the dtype is not supported.
func Transpose2D(dtype dtypes.DType, src any, m, k int, dst any) bool {
	switch dtype {
	case dtypes.Float32:
		matmul.Transpose2DFloat32(src.([]float32), m, k, dst.([]float32))
		return true
	case dtypes.Float64:
		matmul.Transpose2DFloat64(src.([]float64), m, k, dst.([]float64))
		return true
	case dtypes.Float16:
		srcSlice := src.([]float16.Float16)
		dstSlice := dst.([]float16.Float16)
		srcHwy := unsafe.Slice((*hwy.Float16)(unsafe.Pointer(unsafe.SliceData(srcSlice))), len(srcSlice))
		dstHwy := unsafe.Slice((*hwy.Float16)(unsafe.Pointer(unsafe.SliceData(dstSlice))), len(dstSlice))
		matmul.Transpose2DFloat16(srcHwy, m, k, dstHwy)
		return true
	case dtypes.BFloat16:
		srcSlice := src.([]bfloat16.BFloat16)
		dstSlice := dst.([]bfloat16.BFloat16)
		srcHwy := unsafe.Slice((*hwy.BFloat16)(unsafe.Pointer(unsafe.SliceData(srcSlice))), len(srcSlice))
		dstHwy := unsafe.Slice((*hwy.BFloat16)(unsafe.Pointer(unsafe.SliceData(dstSlice))), len(dstSlice))
		matmul.Transpose2DBFloat16(srcHwy, m, k, dstHwy)
		return true
	default:
		return false
	}
}

// HasDTypeSupport returns true if a MatMulDynamic function is registered for the given dtypes.
func HasDTypeSupport(input, output dtypes.DType) bool {
	// Support float types where input and output dtypes match
	switch input {
	case dtypes.Float32:
		return output == dtypes.Float32
	case dtypes.Float64:
		return output == dtypes.Float64
	case dtypes.Float16:
		return output == dtypes.Float16
	case dtypes.BFloat16:
		return output == dtypes.BFloat16
	}
	return false
}

// MatMulDynamic dispatches the MatMul function for the given dtypes.
// It computes C = A * B for matrices where:
//   - A is [batchSize, lhsCrossSize, contractingSize] (M x K per batch)
//   - B is [batchSize, contractingSize, rhsCrossSize] (K x N per batch)
//   - C is [batchSize, lhsCrossSize, rhsCrossSize] (M x N per batch)
func MatMulDynamic(inputDType, outputDType dtypes.DType,
	lhsFlat, rhsFlat any, batchSize,
	lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat any,
	bufAllocAnyFn packgemm.BufAllocAnyFn, bufReleaseFn packgemm.BufReleaseFn, pool *workerspool.Pool) error {

	switch inputDType {
	case dtypes.Float32:
		if outputDType != dtypes.Float32 {
			return errors.Errorf("highway: input dtype Float32 requires output dtype Float32, got %s", outputDType)
		}
		return matMulFloat32(
			lhsFlat.([]float32), rhsFlat.([]float32), outputFlat.([]float32),
			batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
			pool)

	case dtypes.Float64:
		if outputDType != dtypes.Float64 {
			return errors.Errorf("highway: input dtype Float64 requires output dtype Float64, got %s", outputDType)
		}
		return matMulFloat64(
			lhsFlat.([]float64), rhsFlat.([]float64), outputFlat.([]float64),
			batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
			pool)

	case dtypes.Float16:
		if outputDType != dtypes.Float16 {
			return errors.Errorf("highway: input dtype Float16 requires output dtype Float16, got %s", outputDType)
		}
		return matMulFloat16(
			lhsFlat.([]float16.Float16), rhsFlat.([]float16.Float16), outputFlat.([]float16.Float16),
			batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
			pool)

	case dtypes.BFloat16:
		if outputDType != dtypes.BFloat16 {
			return errors.Errorf("highway: input dtype BFloat16 requires output dtype BFloat16, got %s", outputDType)
		}
		return matMulBFloat16(
			lhsFlat.([]bfloat16.BFloat16), rhsFlat.([]bfloat16.BFloat16), outputFlat.([]bfloat16.BFloat16),
			batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
			pool)

	default:
		return errors.Errorf("highway: unsupported input dtype %s", inputDType)
	}
}

// matMulFloat32 performs batched matrix multiplication for float32.
func matMulFloat32(lhs, rhs, output []float32, batchSize, m, n, k int, pool *workerspool.Pool) error {
	lhsBatchStride := m * k
	rhsBatchStride := k * n
	outBatchStride := m * n

	// For single batch, use highway pool for intra-matrix parallelism
	if batchSize == 1 {
		matmul.MatMulAutoWithPool(hwyPool, lhs, rhs, output, m, n, k)
		return nil
	}

	// Use WaitGroup to synchronize parallel batch processing
	wg := xsync.NewDynamicWaitGroup()

	for batchIdx := range batchSize {
		wg.Add(1)
		task := func() {
			lhsStart := batchIdx * lhsBatchStride
			rhsStart := batchIdx * rhsBatchStride
			outStart := batchIdx * outBatchStride
			matmul.MatMulAuto(
				lhs[lhsStart:lhsStart+lhsBatchStride],
				rhs[rhsStart:rhsStart+rhsBatchStride],
				output[outStart:outStart+outBatchStride],
				m, n, k)
			wg.Done()
		}

		// Try to offload to worker pool, otherwise run inline
		if pool == nil || !pool.StartIfAvailable(task) {
			task()
		}
	}

	wg.Wait()
	return nil
}

// matMulFloat64 performs batched matrix multiplication for float64.
func matMulFloat64(lhs, rhs, output []float64, batchSize, m, n, k int, pool *workerspool.Pool) error {
	lhsBatchStride := m * k
	rhsBatchStride := k * n
	outBatchStride := m * n

	// For single batch, use highway pool for intra-matrix parallelism
	if batchSize == 1 {
		matmul.MatMulAutoWithPool(hwyPool, lhs, rhs, output, m, n, k)
		return nil
	}

	// Use WaitGroup to synchronize parallel batch processing
	wg := xsync.NewDynamicWaitGroup()

	for batchIdx := range batchSize {
		wg.Add(1)
		task := func() {
			lhsStart := batchIdx * lhsBatchStride
			rhsStart := batchIdx * rhsBatchStride
			outStart := batchIdx * outBatchStride
			matmul.MatMulAuto(
				lhs[lhsStart:lhsStart+lhsBatchStride],
				rhs[rhsStart:rhsStart+rhsBatchStride],
				output[outStart:outStart+outBatchStride],
				m, n, k)
			wg.Done()
		}

		// Try to offload to worker pool, otherwise run inline
		if pool == nil || !pool.StartIfAvailable(task) {
			task()
		}
	}

	wg.Wait()
	return nil
}

// matMulFloat16 performs batched matrix multiplication for float16.
// It converts between x448/float16.Float16 and hwy.Float16 using unsafe pointer casting
// since both are uint16 under the hood.
func matMulFloat16(lhs, rhs, output []float16.Float16, batchSize, m, n, k int, pool *workerspool.Pool) error {
	// Convert slices using unsafe - both types are uint16 underneath
	lhsHwy := unsafe.Slice((*hwy.Float16)(unsafe.Pointer(unsafe.SliceData(lhs))), len(lhs))
	rhsHwy := unsafe.Slice((*hwy.Float16)(unsafe.Pointer(unsafe.SliceData(rhs))), len(rhs))
	outputHwy := unsafe.Slice((*hwy.Float16)(unsafe.Pointer(unsafe.SliceData(output))), len(output))

	lhsBatchStride := m * k
	rhsBatchStride := k * n
	outBatchStride := m * n

	// For single batch, use highway pool for intra-matrix parallelism
	if batchSize == 1 {
		matmul.MatMulAutoWithPool(hwyPool, lhsHwy, rhsHwy, outputHwy, m, n, k)
		return nil
	}

	// Use WaitGroup to synchronize parallel batch processing
	wg := xsync.NewDynamicWaitGroup()

	for batchIdx := range batchSize {
		wg.Add(1)
		task := func() {
			lhsStart := batchIdx * lhsBatchStride
			rhsStart := batchIdx * rhsBatchStride
			outStart := batchIdx * outBatchStride
			matmul.MatMulAuto(
				lhsHwy[lhsStart:lhsStart+lhsBatchStride],
				rhsHwy[rhsStart:rhsStart+rhsBatchStride],
				outputHwy[outStart:outStart+outBatchStride],
				m, n, k)
			wg.Done()
		}

		// Try to offload to worker pool, otherwise run inline
		if pool == nil || !pool.StartIfAvailable(task) {
			task()
		}
	}

	wg.Wait()
	return nil
}

// matMulBFloat16 performs batched matrix multiplication for bfloat16.
// It converts between bfloat16.BFloat16 and hwy.BFloat16 using unsafe pointer casting
// since both are uint16 under the hood.
func matMulBFloat16(lhs, rhs, output []bfloat16.BFloat16, batchSize, m, n, k int, pool *workerspool.Pool) error {
	// Convert slices using unsafe - both types are uint16 underneath
	lhsHwy := unsafe.Slice((*hwy.BFloat16)(unsafe.Pointer(unsafe.SliceData(lhs))), len(lhs))
	rhsHwy := unsafe.Slice((*hwy.BFloat16)(unsafe.Pointer(unsafe.SliceData(rhs))), len(rhs))
	outputHwy := unsafe.Slice((*hwy.BFloat16)(unsafe.Pointer(unsafe.SliceData(output))), len(output))

	lhsBatchStride := m * k
	rhsBatchStride := k * n
	outBatchStride := m * n

	// For single batch, use highway pool for intra-matrix parallelism
	if batchSize == 1 {
		matmul.MatMulAutoWithPool(hwyPool, lhsHwy, rhsHwy, outputHwy, m, n, k)
		return nil
	}

	// Use WaitGroup to synchronize parallel batch processing
	wg := xsync.NewDynamicWaitGroup()

	for batchIdx := range batchSize {
		wg.Add(1)
		task := func() {
			lhsStart := batchIdx * lhsBatchStride
			rhsStart := batchIdx * rhsBatchStride
			outStart := batchIdx * outBatchStride
			matmul.MatMulAuto(
				lhsHwy[lhsStart:lhsStart+lhsBatchStride],
				rhsHwy[rhsStart:rhsStart+rhsBatchStride],
				outputHwy[outStart:outStart+outBatchStride],
				m, n, k)
			wg.Done()
		}

		// Try to offload to worker pool, otherwise run inline
		if pool == nil || !pool.StartIfAvailable(task) {
			task()
		}
	}

	wg.Wait()
	return nil
}

// MatMulKLast dispatches the MatMulKLast function for the given dtypes.
// It computes C = A * B^T for matrices where both have K as the last dimension:
//   - A is [batchSize, lhsCrossSize, contractingSize] (M x K per batch)
//   - B is [batchSize, rhsCrossSize, contractingSize] (N x K per batch) - note K last!
//   - C is [batchSize, lhsCrossSize, rhsCrossSize] (M x N per batch)
//
// This is the natural format for PyTorch weights and enables efficient SIMD
// since both A rows and B rows have sequential memory access.
func MatMulKLast(inputDType, outputDType dtypes.DType,
	lhsFlat, rhsFlat any, batchSize,
	lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat any,
	pool *workerspool.Pool) error {

	switch inputDType {
	case dtypes.Float32:
		if outputDType != dtypes.Float32 {
			return errors.Errorf("highway: input dtype Float32 requires output dtype Float32, got %s", outputDType)
		}
		return matMulKLastFloat32(
			lhsFlat.([]float32), rhsFlat.([]float32), outputFlat.([]float32),
			batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
			pool)

	case dtypes.Float64:
		if outputDType != dtypes.Float64 {
			return errors.Errorf("highway: input dtype Float64 requires output dtype Float64, got %s", outputDType)
		}
		return matMulKLastFloat64(
			lhsFlat.([]float64), rhsFlat.([]float64), outputFlat.([]float64),
			batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
			pool)

	case dtypes.Float16:
		if outputDType != dtypes.Float16 {
			return errors.Errorf("highway: input dtype Float16 requires output dtype Float16, got %s", outputDType)
		}
		return matMulKLastFloat16(
			lhsFlat.([]float16.Float16), rhsFlat.([]float16.Float16), outputFlat.([]float16.Float16),
			batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
			pool)

	case dtypes.BFloat16:
		if outputDType != dtypes.BFloat16 {
			return errors.Errorf("highway: input dtype BFloat16 requires output dtype BFloat16, got %s", outputDType)
		}
		return matMulKLastBFloat16(
			lhsFlat.([]bfloat16.BFloat16), rhsFlat.([]bfloat16.BFloat16), outputFlat.([]bfloat16.BFloat16),
			batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
			pool)

	default:
		return errors.Errorf("highway: unsupported input dtype %s", inputDType)
	}
}

// matMulKLastFloat32 performs batched K-last matrix multiplication for float32.
func matMulKLastFloat32(lhs, rhs, output []float32, batchSize, m, n, k int, pool *workerspool.Pool) error {
	lhsBatchStride := m * k
	rhsBatchStride := n * k // Note: n*k not k*n since RHS is [N, K]
	outBatchStride := m * n

	// For single batch, use highway pool for intra-matrix parallelism
	if batchSize == 1 {
		matmul.MatMulKLastAutoWithPool(hwyPool, lhs, rhs, output, m, n, k)
		return nil
	}

	// Use WaitGroup to synchronize parallel batch processing
	wg := xsync.NewDynamicWaitGroup()

	for batchIdx := range batchSize {
		wg.Add(1)
		task := func() {
			lhsStart := batchIdx * lhsBatchStride
			rhsStart := batchIdx * rhsBatchStride
			outStart := batchIdx * outBatchStride
			matmul.MatMulKLastAuto(
				lhs[lhsStart:lhsStart+lhsBatchStride],
				rhs[rhsStart:rhsStart+rhsBatchStride],
				output[outStart:outStart+outBatchStride],
				m, n, k)
			wg.Done()
		}

		// Try to offload to worker pool, otherwise run inline
		if pool == nil || !pool.StartIfAvailable(task) {
			task()
		}
	}

	wg.Wait()
	return nil
}

// matMulKLastFloat64 performs batched K-last matrix multiplication for float64.
func matMulKLastFloat64(lhs, rhs, output []float64, batchSize, m, n, k int, pool *workerspool.Pool) error {
	lhsBatchStride := m * k
	rhsBatchStride := n * k // Note: n*k not k*n since RHS is [N, K]
	outBatchStride := m * n

	// For single batch, use highway pool for intra-matrix parallelism
	if batchSize == 1 {
		matmul.MatMulKLastAutoWithPool(hwyPool, lhs, rhs, output, m, n, k)
		return nil
	}

	// Use WaitGroup to synchronize parallel batch processing
	wg := xsync.NewDynamicWaitGroup()

	for batchIdx := range batchSize {
		wg.Add(1)
		task := func() {
			lhsStart := batchIdx * lhsBatchStride
			rhsStart := batchIdx * rhsBatchStride
			outStart := batchIdx * outBatchStride
			matmul.MatMulKLastAuto(
				lhs[lhsStart:lhsStart+lhsBatchStride],
				rhs[rhsStart:rhsStart+rhsBatchStride],
				output[outStart:outStart+outBatchStride],
				m, n, k)
			wg.Done()
		}

		// Try to offload to worker pool, otherwise run inline
		if pool == nil || !pool.StartIfAvailable(task) {
			task()
		}
	}

	wg.Wait()
	return nil
}

// matMulKLastFloat16 performs batched K-last matrix multiplication for float16.
func matMulKLastFloat16(lhs, rhs, output []float16.Float16, batchSize, m, n, k int, pool *workerspool.Pool) error {
	// Convert slices using unsafe - both types are uint16 underneath
	lhsHwy := unsafe.Slice((*hwy.Float16)(unsafe.Pointer(unsafe.SliceData(lhs))), len(lhs))
	rhsHwy := unsafe.Slice((*hwy.Float16)(unsafe.Pointer(unsafe.SliceData(rhs))), len(rhs))
	outputHwy := unsafe.Slice((*hwy.Float16)(unsafe.Pointer(unsafe.SliceData(output))), len(output))

	lhsBatchStride := m * k
	rhsBatchStride := n * k // Note: n*k not k*n since RHS is [N, K]
	outBatchStride := m * n

	// For single batch, use highway pool for intra-matrix parallelism
	if batchSize == 1 {
		matmul.MatMulKLastAutoWithPool(hwyPool, lhsHwy, rhsHwy, outputHwy, m, n, k)
		return nil
	}

	// Use WaitGroup to synchronize parallel batch processing
	wg := xsync.NewDynamicWaitGroup()

	for batchIdx := range batchSize {
		wg.Add(1)
		task := func() {
			lhsStart := batchIdx * lhsBatchStride
			rhsStart := batchIdx * rhsBatchStride
			outStart := batchIdx * outBatchStride
			matmul.MatMulKLastAuto(
				lhsHwy[lhsStart:lhsStart+lhsBatchStride],
				rhsHwy[rhsStart:rhsStart+rhsBatchStride],
				outputHwy[outStart:outStart+outBatchStride],
				m, n, k)
			wg.Done()
		}

		// Try to offload to worker pool, otherwise run inline
		if pool == nil || !pool.StartIfAvailable(task) {
			task()
		}
	}

	wg.Wait()
	return nil
}

// matMulKLastBFloat16 performs batched K-last matrix multiplication for bfloat16.
func matMulKLastBFloat16(lhs, rhs, output []bfloat16.BFloat16, batchSize, m, n, k int, pool *workerspool.Pool) error {
	// Convert slices using unsafe - both types are uint16 underneath
	lhsHwy := unsafe.Slice((*hwy.BFloat16)(unsafe.Pointer(unsafe.SliceData(lhs))), len(lhs))
	rhsHwy := unsafe.Slice((*hwy.BFloat16)(unsafe.Pointer(unsafe.SliceData(rhs))), len(rhs))
	outputHwy := unsafe.Slice((*hwy.BFloat16)(unsafe.Pointer(unsafe.SliceData(output))), len(output))

	lhsBatchStride := m * k
	rhsBatchStride := n * k // Note: n*k not k*n since RHS is [N, K]
	outBatchStride := m * n

	// For single batch, use highway pool for intra-matrix parallelism
	if batchSize == 1 {
		matmul.MatMulKLastAutoWithPool(hwyPool, lhsHwy, rhsHwy, outputHwy, m, n, k)
		return nil
	}

	// Use WaitGroup to synchronize parallel batch processing
	wg := xsync.NewDynamicWaitGroup()

	for batchIdx := range batchSize {
		wg.Add(1)
		task := func() {
			lhsStart := batchIdx * lhsBatchStride
			rhsStart := batchIdx * rhsBatchStride
			outStart := batchIdx * outBatchStride
			matmul.MatMulKLastAuto(
				lhsHwy[lhsStart:lhsStart+lhsBatchStride],
				rhsHwy[rhsStart:rhsStart+rhsBatchStride],
				outputHwy[outStart:outStart+outBatchStride],
				m, n, k)
			wg.Done()
		}

		// Try to offload to worker pool, otherwise run inline
		if pool == nil || !pool.StartIfAvailable(task) {
			task()
		}
	}

	wg.Wait()
	return nil
}
