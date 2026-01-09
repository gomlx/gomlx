//go:build !noasm && arm64

package simplego

import (
	"runtime"
	"unsafe"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/x448/float16"
)

// Assembly functions for FP16 dot products (defined in dotgeneral_fp16_neon_arm64.s)
//
//go:noescape
func dotProductFP16_neon_asm(a, b unsafe.Pointer, n int64) float32

//go:noescape
func dotProductFP16Group4_neon_asm(a, b unsafe.Pointer, b_stride, n int64) (r0, r1, r2, r3 float32)

//go:noescape
func dotProductBF16_neon_asm(a, b unsafe.Pointer, n int64) float32

// hasFP16NEON indicates whether FP16 NEON instructions are available.
// FMLAL/FMLAL2 require ARMv8.2-A with FP16 extension (FEAT_FHM).
// Most modern ARM64 chips (Apple M1+, recent Cortex-A) support this.
var hasFP16NEON = detectFP16NEON()

// hasBF16NEON indicates whether BF16 NEON instructions are available.
// BFMLALB/BFMLALT require ARMv8.6-A (FEAT_BF16).
var hasBF16NEON = detectBF16NEON()

// execNormalizedDotGeneralFloat16ToFloat32 is a specialized implementation for FP16×FP16→FP32
// using native FMLAL instructions when available.
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

							dotSize := contractingBlockEnd - outerIdxContracting
							if hasFP16NEON && dotSize >= 8 {
								// Use NEON FMLAL for native FP16→FP32 accumulation
								lhsPtr := unsafe.Pointer(&lhsFlat[lhsRowStartIdx+outerIdxContracting])
								rhsPtr := unsafe.Pointer(&rhsFlat[rhsColStartIdx+outerIdxContracting])
								dotResult := dotProductFP16_neon_asm(lhsPtr, rhsPtr, int64(dotSize))
								runtime.KeepAlive(lhsFlat)
								runtime.KeepAlive(rhsFlat)
								sum += dotResult
							} else {
								// Scalar fallback with explicit conversion
								for idxContracting := outerIdxContracting; idxContracting < contractingBlockEnd; idxContracting++ {
									lhsVal := lhsFlat[lhsRowStartIdx+idxContracting].Float32()
									rhsVal := rhsFlat[rhsColStartIdx+idxContracting].Float32()
									sum += lhsVal * rhsVal
								}
							}

							outputFlat[outputRowStartIdx+idxRhsCross] = sum
						}
					}
				}
			}
		}
	}
}

// execNormalizedDotGeneralBFloat16ToFloat32 is a specialized implementation for BF16×BF16→FP32
// using native BFMLAL instructions when available (ARMv8.6+).
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

							dotSize := contractingBlockEnd - outerIdxContracting
							if hasBF16NEON && dotSize >= 8 {
								// Use NEON BFMLAL for native BF16→FP32 accumulation
								lhsPtr := unsafe.Pointer(&lhsFlat[lhsRowStartIdx+outerIdxContracting])
								rhsPtr := unsafe.Pointer(&rhsFlat[rhsColStartIdx+outerIdxContracting])
								dotResult := dotProductBF16_neon_asm(lhsPtr, rhsPtr, int64(dotSize))
								runtime.KeepAlive(lhsFlat)
								runtime.KeepAlive(rhsFlat)
								sum += dotResult
							} else {
								// Scalar fallback
								for idxContracting := outerIdxContracting; idxContracting < contractingBlockEnd; idxContracting++ {
									lhsVal := lhsFlat[lhsRowStartIdx+idxContracting].Float32()
									rhsVal := rhsFlat[rhsColStartIdx+idxContracting].Float32()
									sum += lhsVal * rhsVal
								}
							}

							outputFlat[outputRowStartIdx+idxRhsCross] = sum
						}
					}
				}
			}
		}
	}
}

// buildDotGeneralKernelFloat16ToFloat32 returns a kernel function for FP16×FP16→FP32 blocked matrix multiplication.
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

			for rhsRow := 0; rhsRow < blockDim; rhsRow += 4 {
				lhsIdx := baseLhsIdx
				var sum0, sum1, sum2, sum3 float32

				if hasFP16NEON && blockDim >= 8 {
					// Use NEON Group4 for 4 parallel dot products
					lhsPtr := unsafe.Pointer(&lhsFlat[lhsIdx])
					sum0, sum1, sum2, sum3 = dotProductFP16Group4_neon_asm(
						lhsPtr,
						unsafe.Pointer(&rhsFlat[rhsIdx]),
						int64(blockDim),
						int64(blockDim),
					)
					runtime.KeepAlive(lhsFlat)
					runtime.KeepAlive(rhsFlat)
					sum0 += outputFlat[outputIdx]
					sum1 += outputFlat[outputIdx+1]
					sum2 += outputFlat[outputIdx+2]
					sum3 += outputFlat[outputIdx+3]
					rhsIdx += blockDim
					goto fp16done
				}

				// Scalar fallback
				{
					sum0 = outputFlat[outputIdx]
					sum1 = outputFlat[outputIdx+1]
					sum2 = outputFlat[outputIdx+2]
					sum3 = outputFlat[outputIdx+3]

					for contractingIdx := 0; contractingIdx < blockDim; contractingIdx++ {
						rhsIdx1 := rhsIdx + blockDim
						rhsIdx2 := rhsIdx + 2*blockDim
						rhsIdx3 := rhsIdx + 3*blockDim
						lhsVal := lhsFlat[lhsIdx].Float32()
						sum0 += lhsVal * rhsFlat[rhsIdx].Float32()
						sum1 += lhsVal * rhsFlat[rhsIdx1].Float32()
						sum2 += lhsVal * rhsFlat[rhsIdx2].Float32()
						sum3 += lhsVal * rhsFlat[rhsIdx3].Float32()
						lhsIdx++
						rhsIdx++
					}
				}

			fp16done:
				outputFlat[outputIdx] = sum0
				outputFlat[outputIdx+1] = sum1
				outputFlat[outputIdx+2] = sum2
				outputFlat[outputIdx+3] = sum3
				outputIdx += 4

				rhsIdx += 3 * blockDim
			}

			baseLhsIdx += blockDim
		}
	}
}

func init() {
	// Register FP16 NEON-optimized kernels only when NEON FP16 is available.
	// priorityArch overrides priorityTyped fallback in dotgeneral_fp16_stub.go
	if hasFP16NEON {
		dotGeneralNormalizedDTypeMap.Register(dtypes.Float16, priorityArch, execNormalizedDotGeneralFloat16ToFloat32)
		dotGeneralKernelDTypeMap.Register(dtypes.Float16, priorityArch, buildDotGeneralKernelFloat16ToFloat32)
	}

	// Register BF16 NEON-optimized kernels (uses NEON BFMLALB/BFMLALT).
	// This overrides the scalar version when BF16 NEON is available.
	if hasBF16NEON {
		dotGeneralNormalizedDTypeMap.Register(dtypes.BFloat16, priorityArch, execNormalizedDotGeneralBFloat16ToFloat32)
		// Note: BF16 kernel builder for large matrix path would go here if needed
	}
}

// dotProductBF16InnerLoop computes the dot product of lhs[lhsIdx:lhsIdx+size] and rhs[rhsIdx:rhsIdx+size]
// using NEON BFMLAL instructions when available.
// This is used by the BFloat16 kernel in dotgeneral_large.go.
func dotProductBF16InnerLoop(lhsFlat, rhsFlat []bfloat16.BFloat16, lhsIdx, rhsIdx, size int) float32 {
	lhsPtr := unsafe.Pointer(&lhsFlat[lhsIdx])
	rhsPtr := unsafe.Pointer(&rhsFlat[rhsIdx])
	result := dotProductBF16_neon_asm(lhsPtr, rhsPtr, int64(size))
	runtime.KeepAlive(lhsFlat)
	runtime.KeepAlive(rhsFlat)
	return result
}
