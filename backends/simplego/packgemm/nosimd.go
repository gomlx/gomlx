// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package packgemm

import (
	"github.com/gomlx/gomlx/internal/workerspool"
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
	bufAllocFn BufAllocFn[T], bufReleaseFn BufReleaseFn, pool *workerspool.Pool) error {

	// 1. Resolve Strides
	lhsBatchStride := lhsCrossSize * contractingSize
	rhsBatchStride := contractingSize * rhsCrossSize
	outputBatchStride := lhsCrossSize * rhsCrossSize
	dtype := dtypes.FromGenericsType[T]()
	gemmSize := (lhsBatchStride + rhsBatchStride + outputBatchStride) * dtype.Size()
	// gemmFlops := lhsCrossSize * rhsCrossSize * contractingSize

	// 2. Check if small matrix multiplication kernel can be used.
	if (forceVariant == VariantNone && gemmSize < nosimdSmallMatMulSizeThreshold) || forceVariant == VariantSmall {
		return basicSymmetricGenericSmallGEMMParallel(
			alpha, beta,
			lhsFlat, rhsFlat, outputFlat,
			batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
			lhsBatchStride, rhsBatchStride, outputBatchStride,
			pool)
	}

	return basicSymmetricGenericLargeGEMMParallel(
		alpha, beta,
		lhsFlat, rhsFlat, outputFlat,
		batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
		lhsBatchStride, rhsBatchStride, outputBatchStride,
		bufAllocFn, bufReleaseFn,
		pool)
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
	pool *workerspool.Pool) error {

	gemmFlops := lhsCrossSize * rhsCrossSize * contractingSize
	var maxWorkers int
	if pool != nil {
		maxWorkers = pool.MaxParallelism()
	}
	if maxWorkers == 0 || maxWorkers == 1 || batchSize == 1 || batchSize*gemmFlops < nosimdMinMatMulFlopsPerWorker {
		// Not worth parallelizing: just run the small matmul kernel sequentially.
		basicSymmetricGenericSmallGEMM(
			alpha, beta,
			lhsFlat, rhsFlat, outputFlat,
			batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
		)
		return nil
	}

	// Parallelize on the batch dimension:
	batchCountPerTask := nosimdMinMatMulFlopsPerWorker / gemmFlops
	if maxWorkers > 0 {
		// Make parallelization more fine-grained if there are enough workers
		batchCountPerTask = min(batchCountPerTask, batchSize/maxWorkers)
	}
	batchCountPerTask = max(batchCountPerTask, 1)

	// Crate work that needs doing in a buffered channel.
	type chunkData struct {
		batchIdx, batchCount int
	}
	numChunks := (batchSize + batchCountPerTask - 1) / batchCountPerTask
	work := make(chan chunkData, numChunks)
	for batchIdx := 0; batchIdx < batchSize; batchIdx += batchCountPerTask {
		batchCount := min(batchCountPerTask, batchSize-batchIdx)
		work <- chunkData{batchIdx, batchCount}
	}
	close(work)

	// Execute the work in as many workers as available.
	pool.Saturate(func() {
		for w := range work {
			batchLhs := lhsFlat[w.batchIdx*lhsBatchStride : (w.batchIdx+w.batchCount)*lhsBatchStride]
			batchRhs := rhsFlat[w.batchIdx*rhsBatchStride : (w.batchIdx+w.batchCount)*rhsBatchStride]
			batchOutput := outputFlat[w.batchIdx*outputBatchStride : (w.batchIdx+w.batchCount)*outputBatchStride]
			basicSymmetricGenericSmallGEMM(
				alpha, beta,
				batchLhs, batchRhs, batchOutput,
				w.batchCount, lhsCrossSize, rhsCrossSize, contractingSize,
			)
		}
	})
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

	// Bounds check hint for the compiler
	if len(lhs) < lhsStride*batchCount || len(rhs) < rhsStride*batchCount || len(output) < outputStride*batchCount {
		return
	}

	for batchIdx := 0; batchIdx < batchCount; batchIdx++ {
		lhsBase := batchIdx * lhsStride
		rhsBase := batchIdx * rhsStride
		outputBase := batchIdx * outputStride

		row := 0
		// Main Loop: Process 3 rows at a time
		for ; row+2 < lhsCrossSize; row += 3 {
			// Pre-calculate base indices for the 3 LHS rows
			lRow0Base := lhsBase + row*contractingSize
			lRow1Base := lRow0Base + contractingSize
			lRow2Base := lRow1Base + contractingSize

			col := 0
			// Main Tile: Process 4 columns at a time
			for ; col+3 < rhsCrossSize; col += 4 {
				var c00, c01, c02, c03 T
				var c10, c11, c12, c13 T
				var c20, c21, c22, c23 T

				// rIdx tracks the current row in the RHS for these 4 columns
				rIdx := rhsBase + col

				for k := 0; k < contractingSize; k++ {
					// Load RHS row segment
					r0, r1, r2, r3 := rhs[rIdx], rhs[rIdx+1], rhs[rIdx+2], rhs[rIdx+3]

					// Row 0
					l0 := lhs[lRow0Base+k]
					c00 += l0 * r0
					c01 += l0 * r1
					c02 += l0 * r2
					c03 += l0 * r3
					// Row 1
					l1 := lhs[lRow1Base+k]
					c10 += l1 * r0
					c11 += l1 * r1
					c12 += l1 * r2
					c13 += l1 * r3
					// Row 2
					l2 := lhs[lRow2Base+k]
					c20 += l2 * r0
					c21 += l2 * r1
					c22 += l2 * r2
					c23 += l2 * r3

					rIdx += rhsCrossSize
				}

				// Write 3x4 tile results
				basicWriteCol4(output, outputBase+row*rhsCrossSize+col, alpha, beta, c00, c01, c02, c03)
				basicWriteCol4(output, outputBase+(row+1)*rhsCrossSize+col, alpha, beta, c10, c11, c12, c13)
				basicWriteCol4(output, outputBase+(row+2)*rhsCrossSize+col, alpha, beta, c20, c21, c22, c23)
			}

			// Columns-fringe: handle remaining columns for the current 3 rows
			for ; col < rhsCrossSize; col++ {
				var c0, c1, c2 T
				rIdx := rhsBase + col
				for k := 0; k < contractingSize; k++ {
					rk := rhs[rIdx]
					c0 += lhs[lRow0Base+k] * rk
					c1 += lhs[lRow1Base+k] * rk
					c2 += lhs[lRow2Base+k] * rk
					rIdx += rhsCrossSize
				}
				outputIdx := outputBase + row*rhsCrossSize + col
				basicWriteScalar(output, outputIdx, alpha, beta, c0)
				basicWriteScalar(output, outputIdx+rhsCrossSize, alpha, beta, c1)
				basicWriteScalar(output, outputIdx+2*rhsCrossSize, alpha, beta, c2)
			}
		}

		// Row-Fringe: Handle remaining rows (fewer than 3)
		outputIdx := outputBase + row*rhsCrossSize
		for ; row < lhsCrossSize; row++ {
			for col := range rhsCrossSize {
				var acc T
				lhsIdx := lhsBase + row*contractingSize
				rhsIdx0 := rhsBase + col
				rhsIdx1 := rhsBase + col + rhsCrossSize
				rhsIdx2 := rhsBase + col + 2*rhsCrossSize
				rhsIdx3 := rhsBase + col + 3*rhsCrossSize
				rhsStride := rhsCrossSize * 4
				var contractingIdx int
				for ; contractingIdx+3 < contractingSize; contractingIdx += 4 {
					v0 := lhs[lhsIdx] *
						rhs[rhsIdx0]
					v1 := lhs[lhsIdx+1] * rhs[rhsIdx1]
					v2 := lhs[lhsIdx+2] * rhs[rhsIdx2]
					v3 := lhs[lhsIdx+3] * rhs[rhsIdx3]
					acc += v0 + v1 + v2 + v3
					lhsIdx += 4
					rhsIdx0 += rhsStride
					rhsIdx1 += rhsStride
					rhsIdx2 += rhsStride
					rhsIdx3 += rhsStride
				}
				for ; contractingIdx < contractingSize; contractingIdx++ {
					acc += lhs[lhsIdx] * rhs[rhsIdx0]
					lhsIdx++
					rhsIdx0 += rhsCrossSize
				}
				basicWriteScalar(output, outputIdx, alpha, beta, acc)
				outputIdx++
			}
		}
	}
}

// basicWriteCol4 handles a single row of 4 columns to maximize store-throughput
func basicWriteCol4[T dtypes.Number](out []T, offset int, alpha, beta T, v0, v1, v2, v3 T) {
	if beta != 0 {
		out[offset+0] = beta*out[offset+0] + alpha*v0
		out[offset+1] = beta*out[offset+1] + alpha*v1
		out[offset+2] = beta*out[offset+2] + alpha*v2
		out[offset+3] = beta*out[offset+3] + alpha*v3
	} else {
		out[offset+0] = alpha * v0
		out[offset+1] = alpha * v1
		out[offset+2] = alpha * v2
		out[offset+3] = alpha * v3
	}
}

// basicWriteScalar handles a single scalar write to maximize store-throughput
func basicWriteScalar[T dtypes.Number](out []T, idx int, alpha, beta T, value T) {
	if beta != 0 {
		out[idx] = beta*out[idx] + alpha*value
	} else {
		out[idx] = alpha * value
	}
}

func basicSymmetricGenericLargeGEMMParallel[T dtypes.Number](
	alpha, beta T,
	lhsFlat, rhsFlat []T, outputFlat []T,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	lhsBatchStride, rhsBatchStride, outputBatchStride int,
	bufAllocFn BufAllocFn[T], bufReleaseFn BufReleaseFn,
	pool *workerspool.Pool) error {

	// Split work in reasonable number of "chunks".
	var maxWorkers int
	if pool != nil {
		maxWorkers = pool.MaxParallelism()
	}
	if maxWorkers == 0 || maxWorkers == 1 {
		// Do everything sequentially.
		basicSymmetricGemmChunk(
			alpha, beta,
			lhsFlat, rhsFlat, outputFlat,
			lhsCrossSize, rhsCrossSize, contractingSize,
			NoSIMD32Params, 0, rhsCrossSize,
			bufAllocFn, bufReleaseFn,
		)
		return nil
	}

	// 1. Split work in chunks.

	// 2. Recursive work loop
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
			if wg.Add(1); pool == nil || !pool.StartIfAvailable(firstHalf) {
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
			if wg.Add(1); !pool.StartIfAvailable(firstHalf) {
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
