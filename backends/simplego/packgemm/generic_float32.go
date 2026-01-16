// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package packgemm

import "runtime"

func init() {
	if Float32 == nil {
		Float32 = genericFloat32
		Float32Params = GenericFloat32Params
	}
}

var (
	// GenericFloat32Params are generic assumptions for L1/L2/L3 cache sizes.
	//
	// These values are somewhat arbitrary, assuming "standard" modern cache sizes.
	// They are parameterized so they can be tuned or determined dynamically later.
	GenericFloat32Params = CacheParams{
		LHSL1KernelRows:      8,    // Mr: Rows of LHS in registers.
		RHSL1KernelCols:      8,    // Nr: Cols of RHS in registers.
		ContractingPanelSize: 256,  // Kc: L1 Block Depth.
		LHSL2PanelCrossSize:  256,  // Mc: L2 Block Height.
		RHSL3PanelCrossSize:  2048, // Nc: L3 Block Width.
	}
)

// genericFloat32 implements generic matrix multiplication for float32 inputs and outputs.
// It is used when no SIMD-optimized implementation is available.
func genericFloat32(alpha, beta float32, lhsFlat, rhsFlat []float32, batchSize, lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat []float32,
	bufAllocFn BufAllocFn[float32], bufReleaseFn BufReleaseFn, starter GoroutineStarter) {

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
}

// genericFloat32GemmChunk performs the matrix multiplication on a chunk of columns.
func genericFloat32GemmChunk(
	alpha, beta float32,
	lhs, rhs, output []float32,
	lhsCrossSize, rhsCrossSize, contractingSize int,
	params CacheParams, colStart, colEnd int,
	bufAllocFn BufAllocFn[float32], bufReleaseFn BufReleaseFn,
) {
	packedLhsRef, packedLhs := bufAllocFn(params.LHSL2PanelCrossSize * params.ContractingPanelSize)
	packedRhsRef, packedRhs := bufAllocFn(params.ContractingPanelSize * params.RHSL3PanelCrossSize)

	// Accumulator buffer for the micro-kernel.
	accum := make([]float32, params.LHSL1KernelRows*params.RHSL1KernelCols)

	defer func() {
		bufReleaseFn(packedLhsRef)
		bufReleaseFn(packedRhsRef)
	}()

	// Loop 5 (jc): Tiling N (Output Columns)
	for rhsPanelColIdx := colStart; rhsPanelColIdx < colEnd; rhsPanelColIdx += params.RHSL3PanelCrossSize {
		rhsPanelWidth := min(params.RHSL3PanelCrossSize, colEnd-rhsPanelColIdx)

		// Loop 4 (p): Tiling K (Depth)
		for contractingPanelIdx := 0; contractingPanelIdx < contractingSize; contractingPanelIdx += params.ContractingPanelSize {
			effectiveBeta := beta
			if contractingPanelIdx > 0 {
				effectiveBeta = 1
			}
			contractingPanelWidth := min(params.ContractingPanelSize, contractingSize-contractingPanelIdx)

			// PACK RHS
			packRhs(rhs, packedRhs, contractingPanelIdx, rhsPanelColIdx, rhsCrossSize, contractingPanelWidth, rhsPanelWidth, params.RHSL1KernelCols)

			// Loop 3 (ic): Tiling M (Output Rows)
			for lhsPanelRowIdx := 0; lhsPanelRowIdx < lhsCrossSize; lhsPanelRowIdx += params.LHSL2PanelCrossSize {
				lhsPanelHeight := min(params.LHSL2PanelCrossSize, lhsCrossSize-lhsPanelRowIdx)

				// PACK LHS
				packLhs(lhs, packedLhs, lhsPanelRowIdx, contractingPanelIdx, contractingSize, lhsPanelHeight, contractingPanelWidth, params.LHSL1KernelRows)

				// Loop 2 (jr): Micro-Kernel Columns
				for microColIdx := 0; microColIdx < rhsPanelWidth; microColIdx += params.RHSL1KernelCols {
					microKernelActualWidth := min(params.RHSL1KernelCols, rhsPanelWidth-microColIdx)

					// Loop 1 (ir): Micro-Kernel Rows
					for microRowIdx := 0; microRowIdx < lhsPanelHeight; microRowIdx += params.LHSL1KernelRows {
						microKernelActualHeight := min(params.LHSL1KernelRows, lhsPanelHeight-microRowIdx)

						offsetRhs := (microColIdx / params.RHSL1KernelCols) * (contractingPanelWidth * params.RHSL1KernelCols)
						offsetLhs := (microRowIdx / params.LHSL1KernelRows) * (contractingPanelWidth * params.LHSL1KernelRows)

						outputRow := lhsPanelRowIdx + microRowIdx
						outputCol := rhsPanelColIdx + microColIdx

						genericFloat32MicroKernel(
							contractingPanelWidth,
							alpha, effectiveBeta,
							packedLhs[offsetLhs:],
							packedRhs[offsetRhs:],
							output,
							accum,
							outputRow, outputCol,
							rhsCrossSize,
							params.LHSL1KernelRows,
							params.RHSL1KernelCols,
							microKernelActualHeight, microKernelActualWidth,
						)
					}
				}
			}
		}
	}
}

// genericFloat32MicroKernel computes a [lhsKernelRows, rhsKernelCols] tile.
func genericFloat32MicroKernel(
	contractingLen int,
	alpha, beta float32,
	ptrLhs, ptrRhs []float32,
	output, accum []float32,
	outputRowStart, outputColStart int,
	outputStride int,
	lhsKernelRows, rhsKernelCols int,
	lhsActiveRows, rhsActiveCols int,
) {
	// 1. Initialize Accumulators
	for i := range accum {
		accum[i] = 0
	}

	// 2. The K-Loop (Dot Product)
	// ptrLhs is stored as [k][r] (where r is inner dimension of the packed strip)
	// ptrRhs is stored as [k][c] (where c is inner dimension of the packed strip)
	idxLhs := 0
	idxRhs := 0

	for range contractingLen {
		for r := 0; r < lhsKernelRows; r++ {
			valA := ptrLhs[idxLhs+r]
			for c := 0; c < rhsKernelCols; c++ {
				valB := ptrRhs[idxRhs+c]
				accum[r*rhsKernelCols+c] += valA * valB
			}
		}
		idxLhs += lhsKernelRows
		idxRhs += rhsKernelCols
	}

	// 3. Write Back to Output
	for r := 0; r < lhsActiveRows; r++ {
		for c := 0; c < rhsActiveCols; c++ {
			res := accum[r*rhsKernelCols+c]
			outIdx := (outputRowStart+r)*outputStride + (outputColStart + c)
			if beta == 0 {
				output[outIdx] = alpha * res
			} else {
				output[outIdx] = alpha * res + beta * output[outIdx]
			}
		}
	}
}
