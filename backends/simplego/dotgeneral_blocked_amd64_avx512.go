// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

// AVX512 implementation of DotGeneralBlocked for float32.
//
// This uses the "broadcast A, stream B" algorithm which avoids expensive
// horizontal SIMD reductions. Instead of computing dot products that require
// reducing across lanes, we broadcast each LHS element and stream through
// output columns, accumulating with FMA instructions.
//
// The RHS block is transposed in-kernel to enable sequential memory access
// when streaming through output columns. The transpose is O(n²) while the
// matmul is O(n³), so this overhead is negligible for reasonable block sizes.

package simplego

import (
	"simd/archsimd"
	"sync"
	"unsafe"

	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
)

func init() {
	if archsimd.X86.AVX512() {
		dotGeneralKernelDTypeMap.Register(dtypes.Float32, priorityArch, buildDotGeneralBlockKernel_avx512_float32)
		// Adjust block-size: we can be more aggressive with AVX512 support:
		DotGeneralTargetBlockSize = 32 * 1024
		DotGeneralBlockedPathThreshold = 8
	}
}

func castToArray16(ptr *float32) *[16]float32 {
	return (*[16]float32)(unsafe.Pointer(ptr))
}

// rhsTransposeBufferPool provides per-goroutine buffers for transposing RHS blocks.
// Using sync.Pool avoids allocation overhead in the hot path.
var rhsTransposeBufferPool = sync.Pool{
	New: func() any {
		// Allocate buffer for max expected block size (128x128 = 16384 elements)
		buf := make([]float32, 128*128)
		return &buf
	},
}

// buildDotGeneralBlockKernel_avx512_float32 returns a kernel function that does a DotGeneral (matrix multiplication)
// of the lhs/rhs block to the corresponding output buffer block.
//
// It uses the "broadcast A, stream B" algorithm with AVX512 instructions:
// - For each LHS row i, for each element k in the contracting dimension:
//   - Broadcast lhs[i][k] to all 16 SIMD lanes
//   - Stream through RHS row k (after transpose), loading 16 consecutive output columns
//   - FMA: output[i][j:j+16] += broadcast(lhs[i][k]) * rhs_transposed[k][j:j+16]
//
// This avoids expensive horizontal reductions entirely.
func buildDotGeneralBlockKernel_avx512_float32(
	lhs, rhs, output *Buffer, blockDim int) kernelFuncType {
	lhsFlat := lhs.flat.([]float32)
	rhsFlat := rhs.flat.([]float32)
	outputFlat := output.flat.([]float32)

	blockSize := blockDim * blockDim

	return func(lhsBlockIdx, rhsBlockIdx, outputBlockIdx int) {
		if blockDim%16 != 0 {
			exceptions.Panicf("blockDim must be a multiple of 16, got %d", blockDim)
		}

		baseLhsIdx := lhsBlockIdx * blockSize
		baseRhsIdx := rhsBlockIdx * blockSize
		baseOutputIdx := outputBlockIdx * blockSize

		// Get a buffer from the pool for transposing RHS
		bufPtr := rhsTransposeBufferPool.Get().(*[]float32)
		rhsT := (*bufPtr)[:blockSize]
		defer rhsTransposeBufferPool.Put(bufPtr)

		// Transpose RHS block: rhsT[k][j] = rhs[j][k]
		// Original layout: rhs[j*blockDim + k] (row j, col k)
		// Transposed layout: rhsT[k*blockDim + j] (row k, col j)
		// This enables sequential access when streaming through j for fixed k.
		for j := 0; j < blockDim; j++ {
			srcRowStart := baseRhsIdx + j*blockDim
			for k := 0; k < blockDim; k++ {
				rhsT[k*blockDim+j] = rhsFlat[srcRowStart+k]
			}
		}

		// Process each output row (from LHS row)
		for i := 0; i < blockDim; i++ {
			lhsRowStart := baseLhsIdx + i*blockDim
			outputRowStart := baseOutputIdx + i*blockDim

			// Process output row in chunks of 16 columns
			for j := 0; j < blockDim; j += 16 {
				// Load current output accumulator
				accum := archsimd.LoadFloat32x16(castToArray16(&outputFlat[outputRowStart+j]))

				// Accumulate contributions from all elements in contracting dimension
				for k := 0; k < blockDim; k++ {
					// Broadcast lhs[i][k] to all 16 lanes
					aik := lhsFlat[lhsRowStart+k]
					vA := archsimd.BroadcastFloat32x16(aik)

					// Load 16 consecutive elements from transposed RHS row k
					vB := archsimd.LoadFloat32x16(castToArray16(&rhsT[k*blockDim+j]))

					// FMA: accum += vA * vB
					accum = vA.MulAdd(vB, accum)
				}

				// Store result
				accum.Store(castToArray16(&outputFlat[outputRowStart+j]))
			}
		}
	}
}
