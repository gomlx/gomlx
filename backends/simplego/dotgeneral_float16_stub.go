//go:build noasm || !arm64

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

// Scalar fallback implementations for Float16/BFloat16 dot general operations.
// These are used on non-ARM64 platforms or when noasm build tag is set.

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/x448/float16"
)

// execNormalizedDotGeneralFloat16ToFloat32 is the scalar fallback for FP16×FP16→FP32.
func execNormalizedDotGeneralFloat16ToFloat32(lhs, rhs, output *Buffer, params *dotGeneralNodeData, batchStartIdx, batchEndIdx int) {
	lhsFlat := lhs.flat.([]float16.Float16)
	rhsFlat := rhs.flat.([]float16.Float16)
	outputFlat := output.flat.([]float32)

	contractingSize := params.contractingSize
	lhsCrossSize := params.lhsCrossSize
	rhsCrossSize := params.rhsCrossSize

	lhsBatchStride := lhsCrossSize * contractingSize
	rhsBatchStride := rhsCrossSize * contractingSize
	outputBatchStride := lhsCrossSize * rhsCrossSize

	const blockSize = 64

	for batchIdx := batchStartIdx; batchIdx < batchEndIdx; batchIdx++ {
		lhsBaseIdx := batchIdx * lhsBatchStride
		rhsBaseIdx := batchIdx * rhsBatchStride
		outputBaseIdx := batchIdx * outputBatchStride

		for outerIdxLhsCross := 0; outerIdxLhsCross < lhsCrossSize; outerIdxLhsCross += blockSize {
			lhsCrossBlockEnd := min(outerIdxLhsCross+blockSize, lhsCrossSize)

			for outerIdxRhsCross := 0; outerIdxRhsCross < rhsCrossSize; outerIdxRhsCross += blockSize {
				rhsCrossBlockEnd := min(outerIdxRhsCross+blockSize, rhsCrossSize)

				for outerIdxContracting := 0; outerIdxContracting < contractingSize; outerIdxContracting += blockSize {
					contractingBlockEnd := min(outerIdxContracting+blockSize, contractingSize)

					for idxLhsCross := outerIdxLhsCross; idxLhsCross < lhsCrossBlockEnd; idxLhsCross++ {
						lhsRowStartIdx := lhsBaseIdx + idxLhsCross*contractingSize
						outputRowStartIdx := outputBaseIdx + idxLhsCross*rhsCrossSize

						for idxRhsCross := outerIdxRhsCross; idxRhsCross < rhsCrossBlockEnd; idxRhsCross++ {
							rhsColStartIdx := rhsBaseIdx + idxRhsCross*contractingSize
							sum := outputFlat[outputRowStartIdx+idxRhsCross]

							for idxContracting := outerIdxContracting; idxContracting < contractingBlockEnd; idxContracting++ {
								lhsVal := lhsFlat[lhsRowStartIdx+idxContracting].Float32()
								rhsVal := rhsFlat[rhsColStartIdx+idxContracting].Float32()
								sum += lhsVal * rhsVal
							}

							outputFlat[outputRowStartIdx+idxRhsCross] = sum
						}
					}
				}
			}
		}
	}
}

// buildDotGeneralKernelFloat16ToFloat32 returns a scalar kernel for FP16×FP16→FP32.
func buildDotGeneralKernelFloat16ToFloat32(lhs, rhs, output *Buffer, blockDim int) kernelFuncType {
	lhsFlat := lhs.flat.([]float16.Float16)
	rhsFlat := rhs.flat.([]float16.Float16)
	outputFlat := output.flat.([]float32)
	blockSize := blockDim * blockDim

	return func(lhsBlockIdx, rhsBlockIdx, outputBlockIdx int) {
		baseLhsIdx := lhsBlockIdx * blockSize
		baseRhsIdx := rhsBlockIdx * blockSize
		outputIdx := outputBlockIdx * blockSize

		for range blockDim { // output's columns loop
			rhsIdx := baseRhsIdx

			for range blockDim { // output's rows loop
				lhsIdx := baseLhsIdx
				sum := outputFlat[outputIdx]

				for range blockDim { // contracting indices loop (only on the rhs and lhs)
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
	// Register FP16 fallback kernels for non-NEON platforms.
	// Uses priorityTyped so NEON implementations (priorityArch) can override.
	dotGeneralNormalizedDTypeMap.Register(dtypes.Float16, priorityTyped, execNormalizedDotGeneralFloat16ToFloat32)
	dotGeneralKernelDTypeMap.Register(dtypes.Float16, priorityTyped, buildDotGeneralKernelFloat16ToFloat32)
}
