//go:build noasm || !arm64

package simplego

// Scalar fallback implementations for Float16/BFloat16 dot general operations.
// These are used on non-ARM64 platforms or when noasm build tag is set.

import (
	"unsafe"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/x448/float16"
)

// ASM stub functions - panic on non-ARM64 platforms

func dotProductFP16_neon_asm(a, b unsafe.Pointer, n int64) float32 {
	panic("dotProductFP16_neon_asm not available on this platform")
}

func dotProductFP16Group4_neon_asm(a, b unsafe.Pointer, b_stride, n int64) (r0, r1, r2, r3 float32) {
	panic("dotProductFP16Group4_neon_asm not available on this platform")
}

func dotProductBF16_neon_asm(a, b unsafe.Pointer, n int64) float32 {
	panic("dotProductBF16_neon_asm not available on this platform")
}

var hasFP16NEON = false
var hasBF16NEON = false

// Debug functions - stubs for non-ARM64 builds
func dotProductBF16_debug_asm(a, b unsafe.Pointer, n int64) (lane0, lane1, lane2, lane3 float32) {
	panic("dotProductBF16_debug_asm not available on this platform")
}

func dotProductBFMLALB_only_asm(a, b unsafe.Pointer, n int64) (lane0, lane1, lane2, lane3 float32) {
	panic("dotProductBFMLALB_only_asm not available on this platform")
}

func dotProductBFMLALT_only_asm(a, b unsafe.Pointer, n int64) (lane0, lane1, lane2, lane3 float32) {
	panic("dotProductBFMLALT_only_asm not available on this platform")
}

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

// execNormalizedDotGeneralBFloat16ToFloat32 is the scalar fallback for BF16×BF16→FP32.
func execNormalizedDotGeneralBFloat16ToFloat32(lhs, rhs, output *Buffer, params *dotGeneralNodeData, batchStartIdx, batchEndIdx int) {
	lhsFlat := lhs.flat.([]bfloat16.BFloat16)
	rhsFlat := rhs.flat.([]bfloat16.BFloat16)
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

		for range blockDim {
			rhsIdx := baseRhsIdx

			for rhsRow := 0; rhsRow < blockDim; rhsRow++ {
				lhsIdx := baseLhsIdx
				sum := outputFlat[outputIdx]

				for contractingIdx := 0; contractingIdx < blockDim; contractingIdx++ {
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

	// Register BF16 fallback kernels for non-NEON platforms.
	dotGeneralNormalizedDTypeMap.Register(dtypes.BFloat16, priorityTyped, execNormalizedDotGeneralBFloat16ToFloat32)
}

// dotProductBF16InnerLoop is the scalar fallback for BF16 dot product.
// This is used by the BFloat16 kernel in dotgeneral_large.go on non-NEON platforms.
func dotProductBF16InnerLoop(lhsFlat, rhsFlat []bfloat16.BFloat16, lhsIdx, rhsIdx, size int) float32 {
	var sum float32
	for i := 0; i < size; i++ {
		sum += lhsFlat[lhsIdx+i].Float32() * rhsFlat[rhsIdx+i].Float32()
	}
	return sum
}
