//go:build !noasm && arm64

package simplego

import (
	"unsafe"

	"github.com/gomlx/go-xla/pkg/types/dtypes"
)

// Assembly functions for int8/uint8 dot products (defined in dotgeneral_int8_neon_arm64.s)
//go:noescape
func dotProductInt8_neon_asm(a, b unsafe.Pointer, n int64) int32

//go:noescape
func dotProductUint8_neon_asm(a, b unsafe.Pointer, n int64) int32

// execNormalizedDotGeneralInt8ToInt32 is a specialized implementation for int8×int8→int32
// matrix multiplication. It accumulates in int32 to avoid overflow.
func execNormalizedDotGeneralInt8ToInt32(lhs, rhs, output *Buffer, params *dotGeneralNodeData, batchStartIdx, batchEndIdx int) {
	lhsFlat := lhs.flat.([]int8)
	rhsFlat := rhs.flat.([]int8)
	outputFlat := output.flat.([]int32)

	contractingSize := params.contractingSize
	lhsCrossSize := params.lhsCrossSize
	rhsCrossSize := params.rhsCrossSize

	lhsBatchStride := lhsCrossSize * contractingSize
	rhsBatchStride := rhsCrossSize * contractingSize
	outputBatchStride := lhsCrossSize * rhsCrossSize

	// Block size of 64 for cache-efficient tiled matrix multiplication.
	// This creates 64x64 tiles that fit well in L1 cache (~32KB on most ARM64).
	// Each tile is 64*64*1 = 4KB for int8, leaving room for LHS, RHS, and output tiles.
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

							// Call NEON assembly for int8 dot product if available.
							// Threshold of 16 elements matches SDOT's native 16-byte (128-bit) vector width.
							// Below 16: scalar path is faster due to NEON call overhead.
							// At 16+: SDOT processes all elements in one vector operation.
							dotSize := contractingBlockEnd - outerIdxContracting
							if hasNEON && dotSize >= 16 {
								lhsPtr := unsafe.Pointer(&lhsFlat[lhsRowStartIdx+outerIdxContracting])
								rhsPtr := unsafe.Pointer(&rhsFlat[rhsColStartIdx+outerIdxContracting])
								dotResult := dotProductInt8_neon_asm(lhsPtr, rhsPtr, int64(dotSize))
								sum += dotResult
							} else {
								// Scalar path for small vectors
								for idxContracting := outerIdxContracting; idxContracting < contractingBlockEnd; idxContracting++ {
									lhsVal := int32(lhsFlat[lhsRowStartIdx+idxContracting])
									rhsVal := int32(rhsFlat[rhsColStartIdx+idxContracting])
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

// execNormalizedDotGeneralUint8ToInt32 is a specialized implementation for uint8×uint8→int32
// matrix multiplication. It accumulates in int32 to avoid overflow.
//
// Note: This function is also called for mixed int8/uint8 operations. In that case,
// the int8 values are reinterpreted as uint8 (same bit pattern), which means:
//   - Positive int8 values (0-127) work correctly
//   - Negative int8 values are treated as large unsigned values (128-255)
//
// This behavior matches common quantization schemes where activations and weights
// use the same signedness. If you need proper signed × unsigned multiplication,
// you should convert both operands to a wider signed type first.
func execNormalizedDotGeneralUint8ToInt32(lhs, rhs, output *Buffer, params *dotGeneralNodeData, batchStartIdx, batchEndIdx int) {
	// Handle both uint8 and int8 inputs by converting to uint8 view
	var lhsFlat, rhsFlat []uint8

	// Convert lhs to uint8 view
	switch lhs.shape.DType {
	case dtypes.Uint8:
		lhsFlat = lhs.flat.([]uint8)
	case dtypes.Int8:
		// Reinterpret int8 as uint8 (same bit pattern, different interpretation).
		// This is safe for the UDOT instruction which treats inputs as unsigned.
		int8Flat := lhs.flat.([]int8)
		lhsFlat = unsafe.Slice((*uint8)(unsafe.Pointer(&int8Flat[0])), len(int8Flat))
	}

	// Convert rhs to uint8 view
	switch rhs.shape.DType {
	case dtypes.Uint8:
		rhsFlat = rhs.flat.([]uint8)
	case dtypes.Int8:
		int8Flat := rhs.flat.([]int8)
		rhsFlat = unsafe.Slice((*uint8)(unsafe.Pointer(&int8Flat[0])), len(int8Flat))
	}
	outputFlat := output.flat.([]int32)

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

							// Call NEON assembly for uint8 dot product if available
							dotSize := contractingBlockEnd - outerIdxContracting
							if hasNEON && dotSize >= 16 {
								lhsPtr := unsafe.Pointer(&lhsFlat[lhsRowStartIdx+outerIdxContracting])
								rhsPtr := unsafe.Pointer(&rhsFlat[rhsColStartIdx+outerIdxContracting])
								dotResult := dotProductUint8_neon_asm(lhsPtr, rhsPtr, int64(dotSize))
								sum += dotResult
							} else {
								// Scalar path
								for idxContracting := outerIdxContracting; idxContracting < contractingBlockEnd; idxContracting++ {
									lhsVal := int32(lhsFlat[lhsRowStartIdx+idxContracting])
									rhsVal := int32(rhsFlat[rhsColStartIdx+idxContracting])
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

// buildDotGeneralKernelInt8ToInt32 returns a kernel function for int8×int8→int32 blocked matrix multiplication.
// The inputs are int8, but the blocked output is int32 to avoid overflow when accumulating results.
// This follows the same pattern as buildDotGeneralKernelBFloat16.
func buildDotGeneralKernelInt8ToInt32(lhs, rhs, output *Buffer, blockDim int) kernelFuncType {
	lhsFlat := lhs.flat.([]int8)
	rhsFlat := rhs.flat.([]int8)
	outputFlat := output.flat.([]int32)
	blockSize := blockDim * blockDim

	return func(lhsBlockIdx, rhsBlockIdx, outputBlockIdx int) {
		baseLhsIdx := lhsBlockIdx * blockSize
		baseRhsIdx := rhsBlockIdx * blockSize
		outputIdx := outputBlockIdx * blockSize

		for range blockDim { // Loop over lhs rows
			rhsIdx := baseRhsIdx

			// Loop 4 rows at a time for better performance
			for rhsRow := 0; rhsRow < blockDim; rhsRow += 4 {
				lhsIdx := baseLhsIdx
				var sum0, sum1, sum2, sum3 int32

				if hasNEON && blockDim >= 16 {
					// We can use the NEON assembly for dot products
					lhsPtr := unsafe.Pointer(&lhsFlat[lhsIdx])

					rhsPtr0 := unsafe.Pointer(&rhsFlat[rhsIdx])
					rhsPtr1 := unsafe.Pointer(&rhsFlat[rhsIdx+blockDim])
					rhsPtr2 := unsafe.Pointer(&rhsFlat[rhsIdx+2*blockDim])
					rhsPtr3 := unsafe.Pointer(&rhsFlat[rhsIdx+3*blockDim])

					sum0 = outputFlat[outputIdx] + dotProductInt8_neon_asm(lhsPtr, rhsPtr0, int64(blockDim))
					sum1 = outputFlat[outputIdx+1] + dotProductInt8_neon_asm(lhsPtr, rhsPtr1, int64(blockDim))
					sum2 = outputFlat[outputIdx+2] + dotProductInt8_neon_asm(lhsPtr, rhsPtr2, int64(blockDim))
					sum3 = outputFlat[outputIdx+3] + dotProductInt8_neon_asm(lhsPtr, rhsPtr3, int64(blockDim))

					// Advance rhsIdx by blockDim to compensate for skipping the scalar loop
					// The scalar loop would have incremented rhsIdx once per contracting element
					rhsIdx += blockDim
					goto done
				}

				// Scalar fallback
				{
					sum0 = outputFlat[outputIdx]
					sum1 = outputFlat[outputIdx+1]
					sum2 = outputFlat[outputIdx+2]
					sum3 = outputFlat[outputIdx+3]

					// Loop unrolled 4 at a time
					contractingIdx := 0
					for ; contractingIdx+3 < blockDim; contractingIdx += 4 {
						rhsIdx1 := rhsIdx + blockDim
						rhsIdx2 := rhsIdx + 2*blockDim
						rhsIdx3 := rhsIdx + 3*blockDim

						for i := 0; i < 4; i++ {
							lhsVal := int32(lhsFlat[lhsIdx+i])
							sum0 += lhsVal * int32(rhsFlat[rhsIdx+i])
							sum1 += lhsVal * int32(rhsFlat[rhsIdx1+i])
							sum2 += lhsVal * int32(rhsFlat[rhsIdx2+i])
							sum3 += lhsVal * int32(rhsFlat[rhsIdx3+i])
						}
						lhsIdx += 4
						rhsIdx += 4
					}

					// Tail loop
					for ; contractingIdx < blockDim; contractingIdx++ {
						rhsIdx1 := rhsIdx + blockDim
						rhsIdx2 := rhsIdx + 2*blockDim
						rhsIdx3 := rhsIdx + 3*blockDim
						lhsVal := int32(lhsFlat[lhsIdx])
						sum0 += lhsVal * int32(rhsFlat[rhsIdx])
						sum1 += lhsVal * int32(rhsFlat[rhsIdx1])
						sum2 += lhsVal * int32(rhsFlat[rhsIdx2])
						sum3 += lhsVal * int32(rhsFlat[rhsIdx3])
						lhsIdx++
						rhsIdx++
					}
				}

			done:
				outputFlat[outputIdx] = sum0
				outputFlat[outputIdx+1] = sum1
				outputFlat[outputIdx+2] = sum2
				outputFlat[outputIdx+3] = sum3
				outputIdx += 4

				// We unrolled 4 rows of RHS, skip the remaining 3
				rhsIdx += 3 * blockDim
			}

			baseLhsIdx += blockDim
		}
	}
}

// buildDotGeneralKernelUint8ToInt32 returns a kernel function for uint8×uint8→int32 blocked matrix multiplication.
func buildDotGeneralKernelUint8ToInt32(lhs, rhs, output *Buffer, blockDim int) kernelFuncType {
	// Handle both uint8 and int8 inputs by converting to uint8 view
	var lhsFlat, rhsFlat []uint8

	switch lhs.shape.DType {
	case dtypes.Uint8:
		lhsFlat = lhs.flat.([]uint8)
	case dtypes.Int8:
		int8Flat := lhs.flat.([]int8)
		lhsFlat = unsafe.Slice((*uint8)(unsafe.Pointer(&int8Flat[0])), len(int8Flat))
	}

	switch rhs.shape.DType {
	case dtypes.Uint8:
		rhsFlat = rhs.flat.([]uint8)
	case dtypes.Int8:
		int8Flat := rhs.flat.([]int8)
		rhsFlat = unsafe.Slice((*uint8)(unsafe.Pointer(&int8Flat[0])), len(int8Flat))
	}

	outputFlat := output.flat.([]int32)
	blockSize := blockDim * blockDim

	return func(lhsBlockIdx, rhsBlockIdx, outputBlockIdx int) {
		baseLhsIdx := lhsBlockIdx * blockSize
		baseRhsIdx := rhsBlockIdx * blockSize
		outputIdx := outputBlockIdx * blockSize

		for range blockDim {
			rhsIdx := baseRhsIdx

			for rhsRow := 0; rhsRow < blockDim; rhsRow += 4 {
				lhsIdx := baseLhsIdx
				var sum0, sum1, sum2, sum3 int32

				if hasNEON && blockDim >= 16 {
					lhsPtr := unsafe.Pointer(&lhsFlat[lhsIdx])

					rhsPtr0 := unsafe.Pointer(&rhsFlat[rhsIdx])
					rhsPtr1 := unsafe.Pointer(&rhsFlat[rhsIdx+blockDim])
					rhsPtr2 := unsafe.Pointer(&rhsFlat[rhsIdx+2*blockDim])
					rhsPtr3 := unsafe.Pointer(&rhsFlat[rhsIdx+3*blockDim])

					sum0 = outputFlat[outputIdx] + dotProductUint8_neon_asm(lhsPtr, rhsPtr0, int64(blockDim))
					sum1 = outputFlat[outputIdx+1] + dotProductUint8_neon_asm(lhsPtr, rhsPtr1, int64(blockDim))
					sum2 = outputFlat[outputIdx+2] + dotProductUint8_neon_asm(lhsPtr, rhsPtr2, int64(blockDim))
					sum3 = outputFlat[outputIdx+3] + dotProductUint8_neon_asm(lhsPtr, rhsPtr3, int64(blockDim))

					// Advance rhsIdx by blockDim to compensate for skipping the scalar loop
					rhsIdx += blockDim
					goto udone
				}

				// Scalar fallback
				{
					sum0 = outputFlat[outputIdx]
					sum1 = outputFlat[outputIdx+1]
					sum2 = outputFlat[outputIdx+2]
					sum3 = outputFlat[outputIdx+3]

					contractingIdx := 0
					for ; contractingIdx+3 < blockDim; contractingIdx += 4 {
						rhsIdx1 := rhsIdx + blockDim
						rhsIdx2 := rhsIdx + 2*blockDim
						rhsIdx3 := rhsIdx + 3*blockDim

						for i := 0; i < 4; i++ {
							lhsVal := int32(lhsFlat[lhsIdx+i])
							sum0 += lhsVal * int32(rhsFlat[rhsIdx+i])
							sum1 += lhsVal * int32(rhsFlat[rhsIdx1+i])
							sum2 += lhsVal * int32(rhsFlat[rhsIdx2+i])
							sum3 += lhsVal * int32(rhsFlat[rhsIdx3+i])
						}
						lhsIdx += 4
						rhsIdx += 4
					}

					for ; contractingIdx < blockDim; contractingIdx++ {
						rhsIdx1 := rhsIdx + blockDim
						rhsIdx2 := rhsIdx + 2*blockDim
						rhsIdx3 := rhsIdx + 3*blockDim
						lhsVal := int32(lhsFlat[lhsIdx])
						sum0 += lhsVal * int32(rhsFlat[rhsIdx])
						sum1 += lhsVal * int32(rhsFlat[rhsIdx1])
						sum2 += lhsVal * int32(rhsFlat[rhsIdx2])
						sum3 += lhsVal * int32(rhsFlat[rhsIdx3])
						lhsIdx++
						rhsIdx++
					}
				}

			udone:
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
	// Register specialized int8×int8→int32 and uint8×uint8→int32 kernels
	// These will be used when output dtype is Int32 and inputs are Int8/Uint8
	// We need to register these in the dtype map that handles mixed-type operations

	// Register for the normalized dotgeneral operations (small path)
	dotGeneralNormalizedDTypeMap.RegisterIfNotSet(dtypes.Int8, execNormalizedDotGeneralInt8ToInt32)
	dotGeneralNormalizedDTypeMap.RegisterIfNotSet(dtypes.Uint8, execNormalizedDotGeneralUint8ToInt32)

	// Register for the kernel builders (large path)
	// Use Register() to override the generic int8/uint8 kernels with our NEON-optimized versions
	dotGeneralKernelDTypeMap.Register(dtypes.Int8, buildDotGeneralKernelInt8ToInt32)
	dotGeneralKernelDTypeMap.Register(dtypes.Uint8, buildDotGeneralKernelUint8ToInt32)
}
