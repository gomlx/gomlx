// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package packgemm

import "runtime"

var (
	// GenericFloat32Params are generic assumptions for L1/L2/L3 cache sizes.
	//
	// These values are somewhat arbitrary, assuming "standard" modern cache sizes.
	// They are parameterized so they can be tuned or determined dynamically later.
	GenericFloat32Params = CacheParams{
		LHSL1KernelRows:      8,    // Mr: Rows of LHS in registers.
		RHSL1KernelCols:      8,    // Nr: Cols of RHS in registers.
		PanelContractingSize: 2048, // Kc: L1 Block Depth.
		LHSL2PanelCrossSize:  32,   // Mc: L2 Block Height.
		RHSL3PanelCrossSize:  64,   // Nc: L3 Block Width.
	}
)

func init() {
	RegisterGEMM("GenericScalar", genericFloat32, &GenericFloat32Params, PriorityBase)
}

// genericFloat32 implements generic matrix multiplication for float32 inputs and outputs.
// It is used when no SIMD-optimized implementation is available.
func genericFloat32(alpha, beta float32, lhsFlat, rhsFlat []float32, batchSize, lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat []float32,
	bufAllocFn BufAllocFn[float32], bufReleaseFn BufReleaseFn, starter GoroutineStarter) error {

	// 1. Resolve Strides
	lhsBatchStride := lhsCrossSize * contractingSize
	rhsBatchStride := contractingSize * rhsCrossSize
	outputBatchStride := lhsCrossSize * rhsCrossSize

	// 2. Determine "Quantum" Size (Splitting Strategy)
	targetParallelism := runtime.GOMAXPROCS(0)
	rhsColSplitSize := rhsCrossSize

	if batchSize < targetParallelism {
		// Minimum strip size.
		minSplit := 16
		if rhsColSplitSize > minSplit {
			neededSplits := (targetParallelism + batchSize - 1) / batchSize
			calculatedSplit := rhsCrossSize / neededSplits

			if calculatedSplit < minSplit {
				calculatedSplit = minSplit
			}

			// Align to Nr (8)
			rhsColSplitSize = (calculatedSplit + 7) &^ 7
		}
	}

	// 3. The Work Loop
	for batchIdx := range batchSize {
		batchLhs := lhsFlat[batchIdx*lhsBatchStride : (batchIdx+1)*lhsBatchStride]
		batchRhs := rhsFlat[batchIdx*rhsBatchStride : (batchIdx+1)*rhsBatchStride]
		batchOutput := outputFlat[batchIdx*outputBatchStride : (batchIdx+1)*outputBatchStride]

		for colStart := 0; colStart < rhsCrossSize; colStart += rhsColSplitSize {
			colEnd := colStart + rhsColSplitSize
			if colEnd > rhsCrossSize {
				colEnd = rhsCrossSize
			}

			task := func() {
				genericFloat32GemmChunk(
					alpha, beta,
					batchLhs, batchRhs, batchOutput,
					lhsCrossSize, rhsCrossSize, contractingSize,
					GenericFloat32Params, colStart, colEnd,
					bufAllocFn, bufReleaseFn,
				)
			}

			if !starter(task) {
				task()
			}
		}
	}
	return nil
}

// genericFloat32GemmChunk performs the matrix multiplication on a chunk of columns.
func genericFloat32GemmChunk(
	alpha, beta float32,
	lhs, rhs, output []float32,
	lhsCrossSize, rhsCrossSize, contractingSize int,
	params CacheParams, colStart, colEnd int,
	bufAllocFn BufAllocFn[float32], bufReleaseFn BufReleaseFn,
) {
	packedLhsRef, packedLHS := bufAllocFn(params.LHSL2PanelCrossSize * params.PanelContractingSize)
	packedRhsRef, packedRHS := bufAllocFn(params.PanelContractingSize * params.RHSL3PanelCrossSize)
	var accum [64]float32

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

						genericFloat32MicroKernel(
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

// genericFloat32MicroKernel updates one [lhsKernelRows, rhsKernelCols] tile
// from the panels in lhsPack and rhsPack, along a contractingLen elements
// of the contracting dimensions.
//
// - lhsPackSlice: the slice of the packed panel for this kernel, shaped [contractingCols, lhsL1KernelRows]
// - rhsPackSlice: the slice of the package panel for this kernel, shaped [contractingRows, rhsL1KernelCols]
// - accum: array of accumulators in stack, with enough space to fit [lhsL1KernelRows, rhsL1KernelCols]
// - output: output buffer, organized as [lhsCrossSize, rhsCrossSize]
func genericFloat32MicroKernel(
	alpha, beta float32,
	lhsPackSlice, rhsPackSlice []float32,
	accum *[64]float32,
	output []float32,
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
