// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package packgemm

import (
	"runtime"
	"simd/archsimd"
	"unsafe"
)

func init() {
	if archsimd.X86.AVX512() {
		Float32 = avx512Float32
		Float32Params = avx512Float32Params
	}
}

var avx512Float32Params = CacheParams{
	LHSL1KernelRows:      6,    // Mr: Uses 6 ZMM registers for accumulation rows
	RHSL1KernelCols:      32,   // Nr: Uses 2 ZMM registers (2x16) for accumulation cols
	ContractingPanelSize: 256,  // Kc: A strip fits in L1 cache
	LHSL2PanelCrossSize:  528,  // Mc: Fits in L2 cache (multiple of 6)
	RHSL3PanelCrossSize:  4096, // Nc: Fits in L3 cache (multiple of 32)
}

// avx512Float32 implements generic matrix multiplication for float32 inputs and outputs.
// output = alpha * (lhs x rhs) + beta * output
func avx512Float32(alpha, beta float32, lhsFlat, rhsFlat []float32, batchSize, lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat []float32,
	bufAllocFn BufAllocFn[float32], bufReleaseFn BufReleaseFn, starter GoroutineStarter) {

	// 1. Resolve Strides
	lhsBatchStride := lhsCrossSize * contractingSize
	rhsBatchStride := contractingSize * rhsCrossSize
	outputBatchStride := lhsCrossSize * rhsCrossSize

	// 2. Determine "Quantum" Size (Splitting Strategy)
	// We want enough tasks to fill the machine, but not so many that we trash the cache/packing.
	// Target parallelism: Number of physical cores (or default GOMAXPROCS).
	targetParallelism := runtime.GOMAXPROCS(0)

	// Default: Treat the whole N (rhsCrossSize) as one quantum.
	// This minimizes redundant packing of Matrix A.
	rhsColSplitSize := rhsCrossSize

	// Heuristic: If we don't have enough batches to saturate the cores,
	// we MUST split the columns to get parallelism.
	if batchSize < targetParallelism {
		// Example: 16 cores, Batch=2. We want ~8 tasks per batch.
		// tasksPerBatch := targetParallelism / batchSize
		// This is a rough estimate.

		// Ensure we don't split too small (avoid overhead for tiny strips).
		// Minimum strip size e.g., 256 columns or the L1 Block Size.
		minSplit := 256
		if rhsColSplitSize > minSplit {
			neededSplits := (targetParallelism + batchSize - 1) / batchSize
			calculatedSplit := rhsCrossSize / neededSplits

			// Clamp to valid range
			if calculatedSplit < minSplit {
				calculatedSplit = minSplit
			}

			// Align to 32 (Nr) for SIMD efficiency
			rhsColSplitSize = (calculatedSplit + 31) &^ 31
		}
	}

	// 3. The Work Loop
	// We iterate sequentially. If the pool is full, we do the work ourselves.
	for batchIdx := range batchSize {
		// Capture batch offsets once per batch
		batchLhs := lhsFlat[batchIdx*lhsBatchStride : (batchIdx+1)*lhsBatchStride]
		batchRhs := rhsFlat[batchIdx*rhsBatchStride : (batchIdx+1)*rhsBatchStride]
		batchOutput := outputFlat[batchIdx*outputBatchStride : (batchIdx+1)*outputBatchStride]

		// Iterate over Column Strips -- a "chunk" of work that can be parallelized.
		for colStart := 0; colStart < rhsCrossSize; colStart += rhsColSplitSize {
			colEnd := colStart + rhsColSplitSize
			if colEnd > rhsCrossSize {
				colEnd = rhsCrossSize
			}

			// Define the task closure
			task := func() {
				gemmChunk(
					alpha, beta,
					batchLhs, batchRhs, batchOutput,
					lhsCrossSize, rhsCrossSize, contractingSize,
					avx512Float32Params, colStart, colEnd,
					bufAllocFn, bufReleaseFn,
				)
			}

			// 4. Try to Offload
			if !starter(task) {
				// Pool is busy/full.
				// Execute immediately on this thread to prevent starvation.
				task()
			}
		}
	}
}

// gemmChunk performs the 5-loop GotoBLAS algorithm on a slice of a single batch matrix.
func gemmChunk(
	alpha, beta float32,
	lhs, rhs, output []float32,
	lhsCrossSize, rhsCrossSize, contractingSize int,
	params CacheParams, colStart, colEnd int,
	bufAllocFn BufAllocFn[float32], bufReleaseFn BufReleaseFn,
) {
	// fmt.Printf("gemmChunk(colStart=%d, colEnd=%d)\n", colStart, colEnd)
	packedLhsRef, packedLhs := bufAllocFn(params.LHSL2PanelCrossSize * params.ContractingPanelSize)
	packedRhsRef, packedRhs := bufAllocFn(params.ContractingPanelSize * params.RHSL3PanelCrossSize)
	defer func() {
		bufReleaseFn(packedLhsRef)
		bufReleaseFn(packedRhsRef)
	}()

	// Loop 5 (jc): Tiling N (Output Columns) - Fits in L3
	// Iterates over the assigned strip [colStart, colEnd) in chunks of rhsL3PanelCrossSize.
	for rhsPanelColIdx := colStart; rhsPanelColIdx < colEnd; rhsPanelColIdx += params.RHSL3PanelCrossSize {

		// The width of the current panel is limited by the L3 block size (Nc)
		// AND the end of our assigned chunk (colEnd).
		rhsPanelWidth := min(params.RHSL3PanelCrossSize, colEnd-rhsPanelColIdx)

		// Loop 4 (p): Tiling K (Depth) - Fits in L1
		// Iterates over the contracting dimension in chunks of contractingPanelSize
		for contractingPanelIdx := 0; contractingPanelIdx < contractingSize; contractingPanelIdx += params.ContractingPanelSize {
			// fmt.Printf("- contractingPanelIdx=%d\n", contractingPanelIdx)
			effectiveBeta := beta
			if contractingPanelIdx > 0 {
				// We only apply (multiply) the current output by beta the first time we touch the output buffer
				// at this panel, after that the output is already accumulating the results of the matmul.
				effectiveBeta = 1
			}
			contractingPanelWidth := min(params.ContractingPanelSize, contractingSize-contractingPanelIdx)

			// ---------------------------------------------------------
			// PACK RHS (Bit) -> ~B
			// We pack a [contractingPanelWidth, rhsPanelWidth] block of RHS into contiguous memory.
			// Format: Vertical strips of width rhsL1KernelCols (Nr).
			// ---------------------------------------------------------
			packRhs(rhs, packedRhs, contractingPanelIdx, rhsPanelColIdx, rhsCrossSize, contractingPanelWidth, rhsPanelWidth, params.RHSL1KernelCols)

			// Loop 3 (ic): Tiling M (Output Rows) - Fits in L2
			// Iterates over the LHS height in chunks of lhsL2PanelCrossSize
			for lhsPanelRowIdx := 0; lhsPanelRowIdx < lhsCrossSize; lhsPanelRowIdx += params.LHSL2PanelCrossSize {
				lhsPanelHeight := min(params.LHSL2PanelCrossSize, lhsCrossSize-lhsPanelRowIdx)

				// -----------------------------------------------------
				// PACK LHS (Ait) -> ~A
				// We pack a [lhsPanelHeight, contractingPanelWidth] block of LHS into contiguous memory.
				// Format: Horizontal strips of height lhsL1KernelRows (Mr).
				// -----------------------------------------------------
				packLhs(lhs, packedLhs, lhsPanelRowIdx, contractingPanelIdx, lhsCrossSize, contractingSize, lhsPanelHeight, contractingPanelWidth, params.LHSL1KernelRows)

				// Loop 2 (jr): Micro-Kernel Columns (Nr == rhsL1BlockCols)
				for microColIdx := 0; microColIdx < rhsPanelWidth; microColIdx += params.RHSL1KernelCols {
					// Actual width to process (might be < Nr at matrix edge)
					microKernelActualWidth := min(params.RHSL1KernelCols, rhsPanelWidth-microColIdx)

					// Loop 1 (ir): Micro-Kernel Rows (Mr == lhsL1BlockRows)
					for microRowIdx := 0; microRowIdx < lhsPanelHeight; microRowIdx += params.LHSL1KernelRows {
						microKernelActualHeight := min(params.LHSL1KernelRows, lhsPanelHeight-microRowIdx)

						// ---------------------------------------------
						// MICRO KERNEL
						// Computes a [Mr, Nr] tile of Output
						// ---------------------------------------------

						// Calculate pointers into packed buffers.
						// PackRhs is organized in strips of Nr. We need the current strip (microColIdx / Nr).
						// Each strip has size (contractingPanelWidth * Nr).
						offsetRhs := (microColIdx / params.RHSL1KernelCols) * (contractingPanelWidth * params.RHSL1KernelCols)

						// PackLhs is organized in strips of Mr. We need the current strip (microRowIdx / Mr).
						// Each strip has size (contractingPanelWidth * Mr).
						offsetLhs := (microRowIdx / params.LHSL1KernelRows) * (contractingPanelWidth * params.LHSL1KernelRows)

						// Output index calculation (Absolute coordinates)
						outputRow := lhsPanelRowIdx + microRowIdx
						outputCol := rhsPanelColIdx + microColIdx

						microKernelFloat32(
							contractingPanelWidth,
							alpha, effectiveBeta,
							packedLhs[offsetLhs:],
							packedRhs[offsetRhs:],
							output,
							outputRow, outputCol,
							rhsCrossSize, // Stride of Output
							microKernelActualHeight, microKernelActualWidth,
						)
					}
				}
			}
		}
	}
}

// microKernelFloat32 computes a 6x32 tile.
// It uses simulated SIMD logic based on the user's pseudo-code.
// This assumes lhsCrossSize=6 and rhsCrossSize=32 (2x Float32x16).
//
// lhsActiveRows and rhsActiveCols are the number of rows/cols that should be
// written back to the output. Notice the input is padded.
func microKernelFloat32(
	contractingLen int,
	alpha, beta float32,
	ptrLhs, ptrRhs []float32, // Packed Buffers
	output []float32, // Output Matrix
	outputRowStart, outputColStart int, // Coordinates
	outputStride int,
	lhsActiveRows, rhsActiveCols int, // Active rows/cols (for edge handling)
) {
	// // fmt.Printf("\t- microKernelFloat32(beta=%g, contractingLen=%d, lhsActiveRows=%d, rhsActiveCols=%d)\n",
	// 	beta, contractingLen, lhsActiveRows, rhsActiveCols)

	// ---------------------------------------------------------
	// 1. Initialize Accumulators (Registers) to 0.0
	// ---------------------------------------------------------
	// We have 6 rows (Mr), each needs 2 vectors (Nr=32, vec=16)
	// Expanded to individual variables to ensure register allocation.
	accum_lhs0_rhs0 := archsimd.BroadcastFloat32x16(0.0)
	accum_lhs0_rhs1 := archsimd.BroadcastFloat32x16(0.0)
	accum_lhs1_rhs0 := archsimd.BroadcastFloat32x16(0.0)
	accum_lhs1_rhs1 := archsimd.BroadcastFloat32x16(0.0)
	accum_lhs2_rhs0 := archsimd.BroadcastFloat32x16(0.0)
	accum_lhs2_rhs1 := archsimd.BroadcastFloat32x16(0.0)
	accum_lhs3_rhs0 := archsimd.BroadcastFloat32x16(0.0)
	accum_lhs3_rhs1 := archsimd.BroadcastFloat32x16(0.0)
	accum_lhs4_rhs0 := archsimd.BroadcastFloat32x16(0.0)
	accum_lhs4_rhs1 := archsimd.BroadcastFloat32x16(0.0)
	accum_lhs5_rhs0 := archsimd.BroadcastFloat32x16(0.0)
	accum_lhs5_rhs1 := archsimd.BroadcastFloat32x16(0.0)

	// ---------------------------------------------------------
	// 2. The K-Loop (Dot Product)
	// ---------------------------------------------------------
	idxLhs := 0
	idxRhs := 0
	for range contractingLen {
		// Load RHS (Broadcasting/Streaming)
		// We know packedRhs is arranged as [Nr] linear.
		// Since Nr=32, we load two 16-wide vectors.
		// Because we zero-padded in packRhs, this is always safe!
		rhsVec0 := archsimd.LoadFloat32x16Slice(ptrRhs[idxRhs : idxRhs+16])
		rhsVec1 := archsimd.LoadFloat32x16Slice(ptrRhs[idxRhs+16 : idxRhs+32])
		idxRhs += 32 // This automatically skips to the next contracting strip in rhs kernel.

		// Unrolled loop over 6 rows of LHS

		// Row 0
		lhsVal0 := ptrLhs[idxLhs+0]
		lhsVec0 := archsimd.BroadcastFloat32x16(lhsVal0)
		accum_lhs0_rhs0 = lhsVec0.MulAdd(rhsVec0, accum_lhs0_rhs0)
		accum_lhs0_rhs1 = lhsVec0.MulAdd(rhsVec1, accum_lhs0_rhs1)

		// Row 1
		lhsVal1 := ptrLhs[idxLhs+1]
		lhsVec1 := archsimd.BroadcastFloat32x16(lhsVal1)
		accum_lhs1_rhs0 = lhsVec1.MulAdd(rhsVec0, accum_lhs1_rhs0)
		accum_lhs1_rhs1 = lhsVec1.MulAdd(rhsVec1, accum_lhs1_rhs1)

		// Row 2
		lhsVal2 := ptrLhs[idxLhs+2]
		lhsVec2 := archsimd.BroadcastFloat32x16(lhsVal2)
		accum_lhs2_rhs0 = lhsVec2.MulAdd(rhsVec0, accum_lhs2_rhs0)
		accum_lhs2_rhs1 = lhsVec2.MulAdd(rhsVec1, accum_lhs2_rhs1)

		// Row 3
		lhsVal3 := ptrLhs[idxLhs+3]
		lhsVec3 := archsimd.BroadcastFloat32x16(lhsVal3)
		accum_lhs3_rhs0 = lhsVec3.MulAdd(rhsVec0, accum_lhs3_rhs0)
		accum_lhs3_rhs1 = lhsVec3.MulAdd(rhsVec1, accum_lhs3_rhs1)

		// Row 4
		lhsVal4 := ptrLhs[idxLhs+4]
		lhsVec4 := archsimd.BroadcastFloat32x16(lhsVal4)
		accum_lhs4_rhs0 = lhsVec4.MulAdd(rhsVec0, accum_lhs4_rhs0)
		accum_lhs4_rhs1 = lhsVec4.MulAdd(rhsVec1, accum_lhs4_rhs1)

		// Row 5
		lhsVal5 := ptrLhs[idxLhs+5]
		lhsVec5 := archsimd.BroadcastFloat32x16(lhsVal5)
		accum_lhs5_rhs0 = lhsVec5.MulAdd(rhsVec0, accum_lhs5_rhs0)
		accum_lhs5_rhs1 = lhsVec5.MulAdd(rhsVec1, accum_lhs5_rhs1)

		idxLhs += 6 // Skips to the next contracting strip in lhs kernel.
	}

	// Apply alpha factor.
	if alpha != 1 {
		alphaBroadcast := archsimd.BroadcastFloat32x16(alpha)
		accum_lhs0_rhs0 = accum_lhs0_rhs0.Mul(alphaBroadcast)
		accum_lhs0_rhs1 = accum_lhs0_rhs1.Mul(alphaBroadcast)
		accum_lhs1_rhs0 = accum_lhs1_rhs0.Mul(alphaBroadcast)
		accum_lhs1_rhs1 = accum_lhs1_rhs1.Mul(alphaBroadcast)
		accum_lhs2_rhs0 = accum_lhs2_rhs0.Mul(alphaBroadcast)
		accum_lhs2_rhs1 = accum_lhs2_rhs1.Mul(alphaBroadcast)
		accum_lhs3_rhs0 = accum_lhs3_rhs0.Mul(alphaBroadcast)
		accum_lhs3_rhs1 = accum_lhs3_rhs1.Mul(alphaBroadcast)
		accum_lhs4_rhs0 = accum_lhs4_rhs0.Mul(alphaBroadcast)
		accum_lhs4_rhs1 = accum_lhs4_rhs1.Mul(alphaBroadcast)
		accum_lhs5_rhs0 = accum_lhs5_rhs0.Mul(alphaBroadcast)
		accum_lhs5_rhs1 = accum_lhs5_rhs1.Mul(alphaBroadcast)
	}

	// ---------------------------------------------------------
	// 2. Write Back to Output (Scaling with Beta/Alpha)
	// ---------------------------------------------------------

	betaBroadcast := archsimd.BroadcastFloat32x16(beta)
	cols0Bits := min(16, rhsActiveCols)
	cols1Bits := min(16, max(0, rhsActiveCols-16))
	maskForCols0 := archsimd.Mask32x16FromBits(uint16(uint64(1<<cols0Bits) - 1))
	maskForCols1 := archsimd.Mask32x16FromBits(uint16(uint64(1<<cols1Bits) - 1))

	// fmt.Printf("\t- accum_lhs0_rhs0=%v\n", accum_lhs0_rhs0)
	// fmt.Printf("\t- maskForCols0=%v\n", maskForCols0)

	writeRow := func(row int, acc0, acc1 archsimd.Float32x16) {
		outputIdx := (outputRowStart+row)*outputStride + outputColStart
		if rhsActiveCols >= 16 {
			// Store first strip of columns.
			outSlice := output[outputIdx : outputIdx+16]
			curValue := archsimd.LoadFloat32x16Slice(outSlice)
			// curValue = curValue.Mul(betaBroadcast)
			curValue = curValue.Add(acc0)
			curValue.StoreSlice(outSlice)

			// Store second strip of columns.
			if rhsActiveCols > 16 {
				if rhsActiveCols == 32 {
					// Full second strip of columns.
					outSlice = output[outputIdx+16 : outputIdx+32]
					curValue = archsimd.LoadFloat32x16Slice(outSlice)
					// curValue = curValue.Mul(betaBroadcast)
					curValue = curValue.Add(acc1)
					curValue.StoreSlice(outSlice)
				} else {
					// Partial second strip of columns.
					outSlice := output[outputIdx+16 : outputIdx+rhsActiveCols]
					curValue = archsimd.LoadFloat32x16SlicePart(outSlice)
					// curValue = curValue.Mul(betaBroadcast)
					curValue = curValue.Add(acc1)
					curValue.StoreMasked(castToArray16(&outSlice[0]), maskForCols1)
				}
			}

		} else {
			// Store partial first strip of columns.
			outSlice := output[outputIdx : outputIdx+rhsActiveCols]
			curValue := archsimd.LoadFloat32x16SlicePart(outSlice)
			curValue = curValue.Mul(betaBroadcast)
			curValue = curValue.Add(acc0)
			curValue.StoreMasked(castToArray16(&outSlice[0]), maskForCols0)
		}
	}

	switch {
	case lhsActiveRows > 5:
		writeRow(5, accum_lhs5_rhs0, accum_lhs5_rhs1)
		fallthrough
	case lhsActiveRows > 4:
		writeRow(4, accum_lhs4_rhs0, accum_lhs4_rhs1)
		fallthrough
	case lhsActiveRows > 3:
		writeRow(3, accum_lhs3_rhs0, accum_lhs3_rhs1)
		fallthrough
	case lhsActiveRows > 2:
		writeRow(2, accum_lhs2_rhs0, accum_lhs2_rhs1)
		fallthrough
	case lhsActiveRows > 1:
		writeRow(1, accum_lhs1_rhs0, accum_lhs1_rhs1)
		fallthrough
	case lhsActiveRows > 0:
		writeRow(0, accum_lhs0_rhs0, accum_lhs0_rhs1)
	}
}

// packRhs packs a [depth, width] block from RHS into packedRhs.
// It rearranges data into vertical strips of width Nr (rhsL1BlockCols).
// If the block is smaller than Nr, it ZERO-PADS.
func packRhs(src, dst []float32, rowStart, colStart, strideCol, depth, width, nr int) {
	dstIdx := 0
	// Iterate over strips of width nr
	for stripColIdx := 0; stripColIdx < width; stripColIdx += nr {
		// How many columns valid in this strip?
		validCols := min(nr, width-stripColIdx)

		// Iterate over rows (k)
		for row := range depth {
			srcRow := rowStart + row
			srcColBase := colStart + stripColIdx
			// Copy valid columns
			srcIdx := (srcRow * strideCol) + srcColBase
			copy(dst[dstIdx:], src[srcIdx:srcIdx+validCols])
			dstIdx += validCols
			// Zero-pad if strip is incomplete (edge of matrix)
			for c := validCols; c < nr; c++ {
				dst[dstIdx] = 0.0
				dstIdx++
			}
		}
	}
}

// packLhs packs a [height, depth] block from LHS into packedLhs.
// It rearranges data into horizontal strips of height Mr (lhsL1BlockRows).
func packLhs(src, dst []float32, rowStart, colStart, strideRow, strideCol, height, depth, lhsL1KernelRows int) {
	dstIdx := 0
	// Iterate over strips of height mr
	for stripRowIdx := 0; stripRowIdx < height; stripRowIdx += lhsL1KernelRows {
		validRows := min(lhsL1KernelRows, height-stripRowIdx)

		// Iterate over columns (k) (We want LHS to be traversed K-first in the kernel)
		for col := range depth {
			srcCol := colStart + col
			srcRowBase := rowStart + stripRowIdx

			// Copy valid rows
			for row := range validRows {
				srcIdx := ((srcRowBase + row) * strideCol) + srcCol
				dst[dstIdx] = src[srcIdx]
				dstIdx++
			}
			// Zero-pad
			for r := validRows; r < lhsL1KernelRows; r++ {
				dst[dstIdx] = 0.0
				dstIdx++
			}
		}
	}
}

func castToArray16[T float32](ptr *T) *[16]T {
	return (*[16]T)(unsafe.Pointer(ptr))
}
