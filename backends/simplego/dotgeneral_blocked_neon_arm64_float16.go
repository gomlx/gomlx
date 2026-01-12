//go:build !noasm && arm64

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"runtime"
	"unsafe"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/x448/float16"
)

// buildDotGeneralKernelFloat16NEON returns a kernel function for FP16×FP16→FP32 blocked matrix multiplication.
// Uses NEON FMLAL instructions when blockDim >= 8, otherwise falls back to scalar.
func buildDotGeneralKernelFloat16NEON(lhs, rhs, output *Buffer, blockDim int) kernelFuncType {
	lhsFlat := lhs.flat.([]float16.Float16)
	rhsFlat := rhs.flat.([]float16.Float16)
	outputFlat := output.flat.([]float32)
	blockSize := blockDim * blockDim

	// Check blockDim at kernel build time to avoid branch in hot path
	if blockDim < 8 {
		// Scalar fallback for small block dimensions
		return buildDotGeneralKernelFloat16Scalar(lhsFlat, rhsFlat, outputFlat, blockDim, blockSize)
	}

	// NEON-accelerated kernel for blockDim >= 8
	return func(lhsBlockIdx, rhsBlockIdx, outputBlockIdx int) {
		baseLhsIdx := lhsBlockIdx * blockSize
		baseRhsIdx := rhsBlockIdx * blockSize
		outputIdx := outputBlockIdx * blockSize

		for range blockDim {
			rhsIdx := baseRhsIdx

			for rhsRow := 0; rhsRow < blockDim; rhsRow += 4 {
				lhsIdx := baseLhsIdx

				// Use NEON Group4 for 4 parallel dot products
				lhsPtr := unsafe.Pointer(&lhsFlat[lhsIdx])
				sum0, sum1, sum2, sum3 := dotProductFP16Group4_neon_asm(
					lhsPtr,
					unsafe.Pointer(&rhsFlat[rhsIdx]),
					int64(blockDim),
					int64(blockDim),
				)
				runtime.KeepAlive(lhsFlat)
				runtime.KeepAlive(rhsFlat)

				outputFlat[outputIdx] = outputFlat[outputIdx] + sum0
				outputFlat[outputIdx+1] = outputFlat[outputIdx+1] + sum1
				outputFlat[outputIdx+2] = outputFlat[outputIdx+2] + sum2
				outputFlat[outputIdx+3] = outputFlat[outputIdx+3] + sum3
				outputIdx += 4

				rhsIdx += 4 * blockDim
			}

			baseLhsIdx += blockDim
		}
	}
}

// buildDotGeneralKernelFloat16Scalar returns a scalar kernel for small block dimensions.
func buildDotGeneralKernelFloat16Scalar(lhsFlat []float16.Float16, rhsFlat []float16.Float16, outputFlat []float32, blockDim, blockSize int) kernelFuncType {
	return func(lhsBlockIdx, rhsBlockIdx, outputBlockIdx int) {
		baseLhsIdx := lhsBlockIdx * blockSize
		baseRhsIdx := rhsBlockIdx * blockSize
		outputIdx := outputBlockIdx * blockSize

		for range blockDim {
			rhsIdx := baseRhsIdx

			for range blockDim {
				lhsIdx := baseLhsIdx
				sum := outputFlat[outputIdx]

				for range blockDim {
					sum += lhsFlat[lhsIdx].Float32() * rhsFlat[rhsIdx].Float32()
					lhsIdx++
					rhsIdx++
				}

				outputFlat[outputIdx] = sum
				outputIdx++
			}

			baseLhsIdx += blockDim
		}
	}
}

func init() {
	// Register FP16 NEON-optimized kernel for blocked path when NEON FP16 is available.
	// priorityArch overrides priorityTyped from the generated alternates.
	if hasFP16NEON {
		dotGeneralKernelDTypeMap.Register(dtypes.Float16, priorityArch, buildDotGeneralKernelFloat16NEON)
	}
}
