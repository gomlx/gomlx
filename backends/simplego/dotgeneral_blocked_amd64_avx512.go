// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

// EXPERIMENTAL: AVX512 implementation of DotGeneralBlocked for float32.
// It gets a ~2.5x speedup on an AMD9550X3D processor.
//
// This should change to a generic implementation once we get a go-highway version working,
// and it is expected to change in the future.

package simplego

import (
	"simd/archsimd"
	"unsafe"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/support/exceptions"
)

func init() {
	if archsimd.X86.AVX512() {
		dotGeneralKernelDTypeMap.Register(dtypes.Float32, priorityArch, buildDotGeneralBlockKernel_avx512_float32)
		// Adjust block-size: we can be more aggressive with AVX512 support:
		setDotGeneralTargetBlockSize(16 * 1024)
		DotGeneralBlockedPathThreshold = 8
	}
}

func castToArray16[T float32](ptr *T) *[16]T {
	return (*[16]T)(unsafe.Pointer(ptr))
}

// reduceSumFloat32x16 reduces a Float32x16 to a float32.
func reduceSumFloat32x16(x16 archsimd.Float32x16) float32 {
	x8 := x16.GetHi().Add(x16.GetLo())
	x4 := x8.GetHi().Add(x8.GetLo())
	x4sum := x4.AddPairs(x4)
	return x4sum.GetElem(0) + x4sum.GetElem(1)
}

// buildDotGeneralBlockKernel_avx512_float32 returns a kernel function that does a DotGeneral (matrix multiplication)
// of the lhs/rhs block to the corresponding output buffer block.
//
// It uses AVX512 instructions to perform the multiplication.
func buildDotGeneralBlockKernel_avx512_float32(
	lhs, rhs, output *Buffer, blockDim int) kernelFuncType {
	lhsFlat := lhs.flat.([]float32)
	rhsFlat := rhs.flat.([]float32)
	outputFlat := output.flat.([]float32)

	blockSize := blockDim * blockDim

	return func(lhsBlockIdx, rhsBlockIdx, outputBlockIdx int) {
		baseLhsIdx := lhsBlockIdx * blockSize
		baseRhsIdx := rhsBlockIdx * blockSize
		outputIdx := outputBlockIdx * blockSize
		if blockDim%16 != 0 {
			exceptions.Panicf("blockDim must be a multiple of 16, got %d", blockDim)
		}
		for range blockDim { // Loop over lhs rows:
			rhsIdx := baseRhsIdx
			// Loop 8 rows at a time.
			for rhsRow := 0; rhsRow < blockDim; rhsRow += 8 { // loop over rhs rows:
				lhsIdx := baseLhsIdx
				contractingIdx := 0

				// Loop unrolled 16 at a time.
				var sumRow0x16, sumRow1x16, sumRow2x16, sumRow3x16 archsimd.Float32x16
				var sumRow4x16, sumRow5x16, sumRow6x16, sumRow7x16 archsimd.Float32x16
				for ; contractingIdx+15 < blockDim; contractingIdx += 16 {
					lhsRow0 := archsimd.LoadFloat32x16(castToArray16(&lhsFlat[lhsIdx]))

					rhsRow0 := archsimd.LoadFloat32x16(castToArray16(&rhsFlat[rhsIdx]))
					sumRow0x16 = lhsRow0.MulAdd(rhsRow0, sumRow0x16)
					rhsRow1 := archsimd.LoadFloat32x16(castToArray16(&rhsFlat[rhsIdx+blockDim]))
					sumRow1x16 = lhsRow0.MulAdd(rhsRow1, sumRow1x16)
					rhsRow2 := archsimd.LoadFloat32x16(castToArray16(&rhsFlat[rhsIdx+2*blockDim]))
					sumRow2x16 = lhsRow0.MulAdd(rhsRow2, sumRow2x16)
					rhsRow3 := archsimd.LoadFloat32x16(castToArray16(&rhsFlat[rhsIdx+3*blockDim]))
					sumRow3x16 = lhsRow0.MulAdd(rhsRow3, sumRow3x16)
					rhsRow4 := archsimd.LoadFloat32x16(castToArray16(&rhsFlat[rhsIdx+4*blockDim]))
					sumRow4x16 = lhsRow0.MulAdd(rhsRow4, sumRow4x16)
					rhsRow5 := archsimd.LoadFloat32x16(castToArray16(&rhsFlat[rhsIdx+5*blockDim]))
					sumRow5x16 = lhsRow0.MulAdd(rhsRow5, sumRow5x16)
					rhsRow6 := archsimd.LoadFloat32x16(castToArray16(&rhsFlat[rhsIdx+6*blockDim]))
					sumRow6x16 = lhsRow0.MulAdd(rhsRow6, sumRow6x16)
					rhsRow7 := archsimd.LoadFloat32x16(castToArray16(&rhsFlat[rhsIdx+7*blockDim]))
					sumRow7x16 = lhsRow0.MulAdd(rhsRow7, sumRow7x16)

					lhsIdx += 16
					rhsIdx += 16
				}

				sum0 := reduceSumFloat32x16(sumRow0x16)
				sum1 := reduceSumFloat32x16(sumRow1x16)
				sum2 := reduceSumFloat32x16(sumRow2x16)
				sum3 := reduceSumFloat32x16(sumRow3x16)
				sum4 := reduceSumFloat32x16(sumRow4x16)
				sum5 := reduceSumFloat32x16(sumRow5x16)
				sum6 := reduceSumFloat32x16(sumRow6x16)
				sum7 := reduceSumFloat32x16(sumRow7x16)
				outputFlat[outputIdx] += sum0
				outputFlat[outputIdx+1] += sum1
				outputFlat[outputIdx+2] += sum2
				outputFlat[outputIdx+3] += sum3
				outputFlat[outputIdx+4] += sum4
				outputFlat[outputIdx+5] += sum5
				outputFlat[outputIdx+6] += sum6
				outputFlat[outputIdx+7] += sum7
				outputIdx += 8

				// We unrolled 8 rows of RHS, so we need to skip the remaining 7 rows:
				rhsIdx += 7 * blockDim
			} // loop over rhs rows

			// Start next lhs row.
			baseLhsIdx += blockDim
		}
	}
}
