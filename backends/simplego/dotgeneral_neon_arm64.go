// Copyright 2025 The GoMLX Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !noasm && arm64

package simplego

import (
	"runtime"
	"unsafe"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
)

// dotProduct_neon_asm is implemented in dotgeneral_neon_arm64.s
// It computes a single dot product of n float32 values using ARM NEON.
//
//go:noescape
func dotProduct_neon_asm(a, b unsafe.Pointer, n int64) float32

// dotProductGroup4_neon_asm is implemented in dotgeneral_neon_arm64.s
// It computes 4 dot products simultaneously sharing the same LHS vector.
// b_stride is the stride in elements (float32) between the start of each RHS vector.
//
//go:noescape
func dotProductGroup4_neon_asm(a, b unsafe.Pointer, b_stride, n int64) (r0, r1, r2, r3 float32)

// dotProduct_neon computes dot product using NEON and keeps the source slices alive.
// This prevents the compiler from optimizing away or relocating the slice backing arrays.
func dotProduct_neon(aSlice, bSlice []float32, aIdx, bIdx int, n int64) float32 {
	result := dotProduct_neon_asm(
		unsafe.Pointer(&aSlice[aIdx]),
		unsafe.Pointer(&bSlice[bIdx]),
		n)
	// Keep slices alive until after assembly completes
	runtime.KeepAlive(aSlice)
	runtime.KeepAlive(bSlice)
	return result
}

// dotProductInnerLoopNEON is a wrapper that uses NEON to accelerate the inner dot product loop.
// It processes the entire inner loop with NEON for maximum performance.
//
// This function matches the signature needed by buildDotGeneralKernel to replace the inner loop.
//
// Performance: ~2-3x faster than scalar for large vectors on ARM64.
func dotProductInnerLoopNEON(lhsFlat, rhsFlat, outputFlat []float32,
	lhsIdx, rhsIdx, outputIdx, blockDim int) (sum0, sum1, sum2, sum3 float32) {

	// Initialize sums from current output values
	sum0 = outputFlat[outputIdx]
	sum1 = outputFlat[outputIdx+1]
	sum2 = outputFlat[outputIdx+2]
	sum3 = outputFlat[outputIdx+3]

	// Compute 4 independent dot products using NEON Group4 optimization.
	// This loads the LHS vector once and streams 4 RHS vectors against it.
	// blockDim acts as the stride for RHS vectors because they are columns in the block.
	r0, r1, r2, r3 := dotProductGroup4_neon_asm(
		unsafe.Pointer(&lhsFlat[lhsIdx]),
		unsafe.Pointer(&rhsFlat[rhsIdx]),
		int64(blockDim), // stride in elements
		int64(blockDim)) // length n

	runtime.KeepAlive(lhsFlat)
	runtime.KeepAlive(rhsFlat)

	sum0 += r0
	sum1 += r1
	sum2 += r2
	sum3 += r3

	return
}

// buildDotGeneralKernelFloat32NEON returns a NEON-optimized kernel function for float32 blocked
// matrix multiplication. This is registered with priorityArch to override the generic kernel
// when NEON is available.
//
// By having a separate kernel builder, we eliminate the type check and hasNEON branch from the
// hot inner loop, improving performance through better branch prediction and inlining.
func buildDotGeneralKernelFloat32NEON(lhs, rhs, output *Buffer, blockDim int) kernelFuncType {
	lhsFlat := lhs.flat.([]float32)
	rhsFlat := rhs.flat.([]float32)
	outputFlat := output.flat.([]float32)
	blockSize := blockDim * blockDim

	return func(lhsBlockIdx, rhsBlockIdx, outputBlockIdx int) {
		baseLhsIdx := lhsBlockIdx * blockSize
		baseRhsIdx := rhsBlockIdx * blockSize
		outputIdx := outputBlockIdx * blockSize

		for range blockDim { // Loop over lhs rows
			rhsIdx := baseRhsIdx

			// Loop 4 rows at a time using NEON Group4
			for rhsRow := 0; rhsRow < blockDim; rhsRow += 4 {
				lhsIdx := baseLhsIdx

				// NEON Group4 path - computes 4 dot products simultaneously
				sum0, sum1, sum2, sum3 := dotProductInnerLoopNEON(
					lhsFlat, rhsFlat, outputFlat,
					lhsIdx, rhsIdx, outputIdx, blockDim)

				outputFlat[outputIdx] = sum0
				outputFlat[outputIdx+1] = sum1
				outputFlat[outputIdx+2] = sum2
				outputFlat[outputIdx+3] = sum3
				outputIdx += 4

				// Skip to next group of 4 RHS rows
				rhsIdx += 4 * blockDim
			}

			baseLhsIdx += blockDim
		}
	}
}

func init() {
	// Register NEON-optimized float32 kernel for large matrix path.
	// priorityArch overrides the generic buildDotGeneralKernel[float32] in gen_register_dtypes.go.
	// This eliminates the type check and hasNEON branch from the hot inner loop.
	if hasNEON {
		dotGeneralKernelDTypeMap.Register(dtypes.Float32, priorityArch, buildDotGeneralKernelFloat32NEON)
	}
}
