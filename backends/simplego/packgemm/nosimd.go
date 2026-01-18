// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package packgemm

import (
	"runtime"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/support/xsync"
)

var (
	// NoSIMD32Params are generic assumptions for L1/L2/L3 cache sizes for 32 bits dtypes (float32, int32, uint32)
	//
	// These values are somewhat arbitrary, assuming "standard" modern cache sizes.
	// They are parameterized so they can be tuned or determined dynamically later.
	NoSIMD32Params = CacheParams{
		LHSL1KernelRows:      8,    // Mr: Rows of LHS in registers.
		RHSL1KernelCols:      8,    // Nr: Cols of RHS in registers.
		PanelContractingSize: 2048, // 2048, // Kc: L1 Block Depth.
		LHSL2PanelCrossSize:  32,   //32,   // Mc: L2 Block Height.
		RHSL3PanelCrossSize:  64,   // 64,   // Nc: L3 Block Width.
	}

	// Threshold in byte size for switching to the small matrix multiplication kernel.
	// If the total number of operations is below this threshold, the small
	// matrix multiplication kernel is used instead of the tiled implementation.
	// This is a heuristic and may need to be tuned for different architectures.
	// Expressed in number of bytes.
	nosimdSmallMatMulSizeThreshold = 4 * 1024 * 1024

	// Minimum number of flops per worker: above this number, if possible we should
	// parallelize computation on separate goroutines.
	nosimdMinMatMulFlopsPerWorker = 1024
)

func init() {
	RegisterGEMM("Basic(non-SIMD)", basicSymmetricGeneric[float32], &NoSIMD32Params, PriorityBase)
	RegisterGEMM("Basic(non-SIMD)", basicSymmetricGeneric[int32], &NoSIMD32Params, PriorityBase)
	RegisterGEMM("Basic(non-SIMD)", basicSymmetricGeneric[uint32], &NoSIMD32Params, PriorityBase)
}

// basicSymmetricGeneric implements basic symmetric (input and output dtypes are the same) non-SIMD
// GEMM for various types of inputs and outputs.
//
// It is used when no SIMD-optimized implementation is available.
func basicSymmetricGeneric[T dtypes.Number](alpha, beta T, lhsFlat, rhsFlat []T,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	outputFlat []T,
	bufAllocFn BufAllocFn[T], bufReleaseFn BufReleaseFn, starter GoroutineStarter) error {

	// 1. Resolve Strides
	lhsBatchStride := lhsCrossSize * contractingSize
	rhsBatchStride := contractingSize * rhsCrossSize
	outputBatchStride := lhsCrossSize * rhsCrossSize
	dtype := dtypes.FromGenericsType[T]()
	gemmSize := (lhsBatchStride + rhsBatchStride + outputBatchStride) * dtype.Size()
	// gemmFlops := lhsCrossSize * rhsCrossSize * contractingSize

	// 2. Check if small matrix multiplication kernel can be used.
	if (forceVariant == VariantNone && gemmSize < nosimdSmallMatMulSizeThreshold) || forceVariant == VariantSmall {
		basicSymmetricGenericSmallGEMMParallel(
			alpha, beta,
			lhsFlat, rhsFlat, outputFlat,
			batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
			lhsBatchStride, rhsBatchStride, outputBatchStride,
			starter)
		return nil
	}

	// 3. Recursive work loop
	wg := xsync.NewDynamicWaitGroup() // Control workers started.
	var process func(batchStart, batchCount, rhsColStart, rhsColCount, depth int)
	process = func(batchStart, batchCount, rhsColStart, rhsColCount, depth int) {
		switch splitStrategy(depth, batchCount, rhsColCount, lhsCrossSize, contractingSize, &NoSIMD32Params) {
		case noSplit:
			colEnd := rhsColStart + rhsColCount
			if colEnd > rhsCrossSize {
				colEnd = rhsCrossSize
			}
			for b := range batchCount {
				batchIdx := batchStart + b
				batchLhs := lhsFlat[batchIdx*lhsBatchStride : (batchIdx+1)*lhsBatchStride]
				batchRhs := rhsFlat[batchIdx*rhsBatchStride : (batchIdx+1)*rhsBatchStride]
				batchOutput := outputFlat[batchIdx*outputBatchStride : (batchIdx+1)*outputBatchStride]
				basicSymmetricGemmChunk(
					alpha, beta,
					batchLhs, batchRhs, batchOutput,
					lhsCrossSize, rhsCrossSize, contractingSize,
					NoSIMD32Params, rhsColStart, colEnd,
					bufAllocFn, bufReleaseFn,
				)
			}
			return
		case splitBatch:
			split := batchCount / 2
			firstHalf := func() {
				process(batchStart, split, rhsColStart, rhsColCount, depth+1)
				wg.Done()
			}
			if wg.Add(1); !starter(firstHalf) {
				firstHalf() // Execute first-half sequentially otherwise.
			}
			// Execute second-half on current goroutine.
			process(batchStart+split, batchCount-split, rhsColStart, rhsColCount, depth+1)
		case splitRHSCol:
			split := rhsColCount / 2
			firstHalf := func() {
				process(batchStart, batchCount, rhsColStart, split, depth+1)
				wg.Done()
			}
			if wg.Add(1); !starter(firstHalf) {
				firstHalf() // Execute first-half sequentially otherwise.
			}
			// Execute second-half on current goroutine.
			process(batchStart, batchCount, rhsColStart+split, rhsColCount-split, depth+1)
		}
	}

	// Start recursion.
	process(0, batchSize, 0, rhsCrossSize, 0)
	return nil
}

// basicSymmetricGenericSmallGEMMParallel implements basic symmetric (input and output dtypes are the same) non-SIMD
// GEMM for various types of inputs and outputs for **small matrices** (not counting the batch size).
//
// This function will attempt to parallelize the computation on the batch dimension, if it evaluate it as
// worth parallelizing.
//
// It is used when no SIMD-optimized implementation is available.
func basicSymmetricGenericSmallGEMMParallel[T dtypes.Number](
	alpha, beta T,
	lhsFlat, rhsFlat []T, outputFlat []T,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	lhsBatchStride, rhsBatchStride, outputBatchStride int,
	starter GoroutineStarter) error {

	gemmFlops := lhsCrossSize * rhsCrossSize * contractingSize
	if starter == nil || batchSize == 1 || batchSize*gemmFlops < nosimdMinMatMulFlopsPerWorker {
		// Not worth parallelizing: just run the small matmul kernel sequentially.
		basicSymmetricGenericSmallGEMM(
			alpha, beta,
			lhsFlat, rhsFlat, outputFlat,
			batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
		)
		return nil
	}

	// Parallelize on the batch dimension:
	wg := xsync.NewDynamicWaitGroup() // Control workers started.
	maxWorkers := runtime.GOMAXPROCS(0)
	minChunk := 1 // nosimdSmallMatMulThreshold / matmulSize
	batchCountPerTask := max(minChunk, batchSize/maxWorkers)
	for b := 0; b < batchSize; b += batchCountPerTask {
		batchCount := min(batchCountPerTask, batchSize-b)
		batchLhs := lhsFlat[b*lhsBatchStride : (b+batchCount)*lhsBatchStride]
		batchRhs := rhsFlat[b*rhsBatchStride : (b+batchCount)*rhsBatchStride]
		batchOutput := outputFlat[b*outputBatchStride : (b+batchCount)*outputBatchStride]
		if b+batchCount == batchSize || !starter(func() {
			// Started on a separate worker.
			wg.Add(1)
			defer wg.Done()
			basicSymmetricGenericSmallGEMM(
				alpha, beta,
				batchLhs, batchRhs, batchOutput,
				batchCount, lhsCrossSize, rhsCrossSize, contractingSize,
			)
		}) {
			// Last chunk or if no more workers available, run in the current goroutine.
			basicSymmetricGenericSmallGEMM(
				alpha, beta,
				batchLhs, batchRhs, batchOutput,
				batchCount, lhsCrossSize, rhsCrossSize, contractingSize,
			)
		}
	}
	wg.Wait()
	return nil
}

func basicSymmetricGenericSmallGEMM[T dtypes.Number](
	alpha, beta T,
	lhs, rhs, output []T,
	batchCount, lhsCrossSize, rhsCrossSize, contractingSize int,
) {
	lhsStride := contractingSize * lhsCrossSize
	rhsStride := rhsCrossSize * contractingSize
	outputStride := rhsCrossSize * lhsCrossSize
	_ = lhs[lhsStride*batchCount-1]
	_ = rhs[rhsStride*batchCount-1]
	_ = output[outputStride*batchCount-1]
	for b := range batchCount {
		lhsBase := b * lhsStride
		rhsBase := b * rhsStride
		outputBase := b * outputStride
		for row := range lhsCrossSize {
			lhsRowStart := lhsBase + row*contractingSize
			for col := range rhsCrossSize {
				acc := T(0)
				var rhsColStart = rhsBase + col
				var rhsColContracting0 = rhsColStart
				var rhsColContracting1 = rhsColStart + rhsCrossSize
				var rhsColContracting2 = rhsColStart + 2*rhsCrossSize
				var rhsColContracting3 = rhsColStart + 3*rhsCrossSize
				var rhsStep = 4 * rhsCrossSize
				var contractingIdx int
				for ; contractingIdx+3 < contractingSize; contractingIdx += 4 {
					v0 := lhs[lhsRowStart+contractingIdx] * rhs[rhsColContracting0]
					v1 := lhs[lhsRowStart+contractingIdx+1] * rhs[rhsColContracting1]
					v2 := lhs[lhsRowStart+contractingIdx+2] * rhs[rhsColContracting2]
					v3 := lhs[lhsRowStart+contractingIdx+3] * rhs[rhsColContracting3]
					acc += v0 + v1 + v2 + v3
					rhsColContracting0 += rhsStep
					rhsColContracting1 += rhsStep
					rhsColContracting2 += rhsStep
					rhsColContracting3 += rhsStep
				}
				for ; contractingIdx < contractingSize; contractingIdx++ {
					v := lhs[lhsBase+row*contractingSize+contractingIdx] * rhs[rhsBase+contractingIdx*rhsCrossSize+col]
					acc += v
				}
				if beta != 0 {
					output[outputBase+row*rhsCrossSize+col] = beta*output[outputBase+row*rhsCrossSize+col] + alpha*acc
				} else {
					output[outputBase+row*rhsCrossSize+col] = alpha * acc
				}
			}
		}
	}
}

// basicSymmetricGemmChunk performs the matrix multiplication on a chunk of columns.
func basicSymmetricGemmChunk[T dtypes.Number](
	alpha, beta T,
	lhs, rhs, output []T,
	lhsCrossSize, rhsCrossSize, contractingSize int,
	params CacheParams, colStart, colEnd int,
	bufAllocFn BufAllocFn[T], bufReleaseFn BufReleaseFn,
) {
	packedLhsRef, packedLHS := bufAllocFn(params.LHSL2PanelCrossSize * params.PanelContractingSize)
	packedRhsRef, packedRHS := bufAllocFn(params.PanelContractingSize * params.RHSL3PanelCrossSize)
	var accum [64]T

	defer func() {
		bufReleaseFn(packedLhsRef)
		bufReleaseFn(packedRhsRef)
	}()

	// Loop 5 (jc): Tiling N (Output Columns)
	for rhsPanelColIdx := colStart; rhsPanelColIdx < colEnd; rhsPanelColIdx += params.RHSL3PanelCrossSize {
		rhsPanelWidth := min(params.RHSL3PanelCrossSize, colEnd-rhsPanelColIdx)

		// Loop 4 (p): Tiling K (Depth)
		for contractingPanelIdx := 0; contractingPanelIdx < contractingSize; contractingPanelIdx += params.PanelContractingSize {
			effectiveBeta := beta
			if contractingPanelIdx > 0 {
				effectiveBeta = 1
			}
			contractingPanelWidth := min(params.PanelContractingSize, contractingSize-contractingPanelIdx)

			// PACK RHS
			packRHS(rhs, packedRHS, contractingPanelIdx, rhsPanelColIdx, rhsCrossSize, contractingPanelWidth, rhsPanelWidth, params.RHSL1KernelCols)

			// Loop 3 (ic): Tiling M (Output Rows)
			for lhsPanelRowIdx := 0; lhsPanelRowIdx < lhsCrossSize; lhsPanelRowIdx += params.LHSL2PanelCrossSize {
				lhsPanelHeight := min(params.LHSL2PanelCrossSize, lhsCrossSize-lhsPanelRowIdx)

				// PACK LHS
				packLHS(lhs, packedLHS, lhsPanelRowIdx, contractingPanelIdx, contractingSize, lhsPanelHeight, contractingPanelWidth, params.LHSL1KernelRows)

				// Loop 2 (jr): Micro-Kernel Columns
				for microColIdx := 0; microColIdx < rhsPanelWidth; microColIdx += params.RHSL1KernelCols {
					microKernelActiveWidth := min(params.RHSL1KernelCols, rhsPanelWidth-microColIdx)

					// Loop 1 (ir): Micro-Kernel Rows
					for microRowIdx := 0; microRowIdx < lhsPanelHeight; microRowIdx += params.LHSL1KernelRows {
						microKernelActiveHeight := min(params.LHSL1KernelRows, lhsPanelHeight-microRowIdx)

						offsetRhs := (microColIdx / params.RHSL1KernelCols) * (contractingPanelWidth * params.RHSL1KernelCols)
						offsetLhs := (microRowIdx / params.LHSL1KernelRows) * (contractingPanelWidth * params.LHSL1KernelRows)

						outputRow := lhsPanelRowIdx + microRowIdx
						outputCol := rhsPanelColIdx + microColIdx

						basicSymmetricMicroKernel(
							alpha, effectiveBeta,
							packedLHS[offsetLhs:],
							packedRHS[offsetRhs:],
							&accum,
							output,
							outputRow, outputCol,
							rhsCrossSize,
							params.LHSL1KernelRows,
							params.RHSL1KernelCols,
							contractingPanelWidth,
							microKernelActiveHeight, microKernelActiveWidth,
						)
					}
				}
			}
		}
	}
}

// basicSymmetricMicroKernel updates one [lhsKernelRows, rhsKernelCols] tile
// from the panels in lhsPack and rhsPack, along a contractingLen elements
// of the contracting dimensions.
//
// - lhsPackSlice: the slice of the packed panel for this kernel, shaped [contractingCols, lhsL1KernelRows]
// - rhsPackSlice: the slice of the package panel for this kernel, shaped [contractingRows, rhsL1KernelCols]
// - accum: array of accumulators in stack, with enough space to fit [lhsL1KernelRows, rhsL1KernelCols]
// - output: output buffer, organized as [lhsCrossSize, rhsCrossSize]
func basicSymmetricMicroKernel[T dtypes.Number](
	alpha, beta T,
	lhsPackSlice, rhsPackSlice []T,
	accum *[64]T,
	output []T,
	outputRowStart, outputColStart int,
	outputRowStride int,
	lhsL1KernelRows, rhsL1KernelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
) {
	// 1. Initialize Accumulators
	for i := range *accum {
		(*accum)[i] = 0
	}

	// 2. The K-Loop (Dot Product)
	// ptrLhs is stored as [k][r] (where r is inner dimension of the packed strip)
	// ptrRhs is stored as [k][c] (where c is inner dimension of the packed strip)
	idxLhs := 0
	idxRhs := 0

	for range contractingLen {
		// Force early bound-check to eliminate bounds checks in the inner loops.
		lhsWindow := lhsPackSlice[idxLhs : idxLhs+lhsL1KernelRows]
		_ = lhsWindow[lhsL1KernelRows-1]
		rhsWindow := rhsPackSlice[idxRhs : idxRhs+rhsL1KernelCols]
		_ = rhsWindow[rhsL1KernelCols-1]
		for lhsRow := 0; lhsRow < lhsL1KernelRows; lhsRow += 4 {
			lhsV0 := lhsWindow[lhsRow]
			lhsV1 := lhsWindow[lhsRow+1]
			lhsV2 := lhsWindow[lhsRow+2]
			lhsV3 := lhsWindow[lhsRow+3]

			for rhsCol := 0; rhsCol+1 < rhsL1KernelCols; rhsCol += 2 {
				rhsV0 := rhsWindow[rhsCol]
				rhsV1 := rhsWindow[rhsCol+1]

				// BCE (bound check elimination)
				(*accum)[lhsRow*rhsL1KernelCols+rhsCol] += lhsV0 * rhsV0
				(*accum)[(lhsRow+1)*rhsL1KernelCols+rhsCol] += lhsV1 * rhsV0
				(*accum)[(lhsRow+2)*rhsL1KernelCols+rhsCol] += lhsV2 * rhsV0
				(*accum)[(lhsRow+3)*rhsL1KernelCols+rhsCol] += lhsV3 * rhsV0

				(*accum)[lhsRow*rhsL1KernelCols+rhsCol+1] += lhsV0 * rhsV1
				(*accum)[(lhsRow+1)*rhsL1KernelCols+rhsCol+1] += lhsV1 * rhsV1
				(*accum)[(lhsRow+2)*rhsL1KernelCols+rhsCol+1] += lhsV2 * rhsV1
				(*accum)[(lhsRow+3)*rhsL1KernelCols+rhsCol+1] += lhsV3 * rhsV1
			}
		}
		idxLhs += lhsL1KernelRows
		idxRhs += rhsL1KernelCols
	}

	// 3. Write Back to Output
	if alpha == 1 && beta == 0 {
		_ = (*accum)[(lhsActiveRows-1)*rhsL1KernelCols+rhsActiveCols-1]
		for r := range lhsActiveRows {
			res := (*accum)[r*rhsL1KernelCols : r*rhsL1KernelCols+rhsActiveCols]
			outIdx := (outputRowStart+r)*outputRowStride + outputColStart
			copy(output[outIdx:outIdx+rhsActiveCols], res)
		}
	} else {
		for r := range lhsActiveRows {
			for c := range rhsActiveCols {
				res := (*accum)[r*rhsL1KernelCols+c]
				outIdx := (outputRowStart+r)*outputRowStride + (outputColStart + c)
				if beta == 0 {
					output[outIdx] = alpha * res
				} else {
					output[outIdx] = alpha*res + beta*output[outIdx]
				}
			}
		}
	}
}
