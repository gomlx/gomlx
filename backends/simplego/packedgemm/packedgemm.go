// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package packedgemm

import (
	"simd/archsimd"
	"unsafe"

	"k8s.io/klog/v2"
)

// "internal/archsimd" // Pseudo-import based on your example

// Block/packs parameters for current architecture.
type CacheParams struct {
	lhsL1KernelRows int // or Mr: number of lhs kernel rows going to registers.
	rhsL1KernelCols int // or Nr: Register Block Width

	contractingPanelSize int // Kc: L1 Block Depth
	lhsL2PanelCrossSize  int // Mc: L2 Block Height
	rhsL3PanelCrossSize  int // Nc: L3 Block Width
}

var DefaultCacheParams = CacheParams{
	lhsL1KernelRows:      6,    // Mr: Uses 6 ZMM registers for accumulation rows
	rhsL1KernelCols:      32,   // Nr: Uses 2 ZMM registers (2x16) for accumulation cols
	contractingPanelSize: 256,  // Kc: A strip fits in L1 cache
	lhsL2PanelCrossSize:  528,  // Mc: Fits in L2 cache (multiple of 6)
	rhsL3PanelCrossSize:  4096, // Nc: Fits in L3 cache (multiple of 32)
}

// BufAllocFn is a function that allocates a buffer of type T, of the given size.
type BufAllocFn[T any] func(size int) (ref any, data []T)

// BufReleaseFn is a function that releases a buffer allocated with BufAllocFn.
type BufReleaseFn[T any] func(ref any)

// Float32 implements generic matrix multiplication for float32 inputs and outputs.
// output = alpha * (lhs x rhs) + beta * output
func Float32(alpha, beta float32, lhsFlat, rhsFlat []float32, batchSize, lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat []float32,
	bufAllocFn BufAllocFn[float32], bufReleaseFn BufReleaseFn[float32]) {

	// 1. Resolve Strides
	lhsBatchStride := lhsCrossSize * contractingSize
	rhsBatchStride := contractingSize * rhsCrossSize
	outputBatchStride := lhsCrossSize * rhsCrossSize

	params := DefaultCacheParams

	// 2. Allocate packing buffers for panels.
	packedLhsRef, packedLhs := bufAllocFn(params.lhsL2PanelCrossSize * params.contractingPanelSize)
	packedRhsRef, packedRhs := bufAllocFn(params.contractingPanelSize * params.rhsL3PanelCrossSize)
	defer func() {
		bufReleaseFn(packedLhsRef)
		bufReleaseFn(packedRhsRef)
	}()

	// 3. Iterate Batch
	for batchIdx := range batchSize {
		// Calculate slice offsets for this batch
		lhsStart := batchIdx * lhsBatchStride
		rhsStart := batchIdx * rhsBatchStride
		outputStart := batchIdx * outputBatchStride

		gemmSingleBatch(
			alpha, beta,
			lhsFlat[lhsStart:lhsStart+lhsBatchStride],
			rhsFlat[rhsStart:rhsStart+rhsBatchStride],
			outputFlat[outputStart:outputStart+outputBatchStride],
			lhsCrossSize, rhsCrossSize, contractingSize,
			params,
			packedLhs, packedRhs,
		)
	}
}

// gemmSingleBatch performs the 5-loop GotoBLAS algorithm on a single batch matrix.
func gemmSingleBatch(
	alpha, beta float32,
	lhs, rhs, output []float32,
	lhsCrossSize, rhsCrossSize, contractingSize int,
	params CacheParams,
	packedLhs, packedRhs []float32,
) {
	defer func() {
		if err := recover(); err != nil {
			klog.Exitf("Fatal error in gemmSingleBatch: %+v", err)
		}
	}()

	// Loop 5 (jc): Tiling N (Output Columns) - Fits in L3
	// Iterates over the RHS width in chunks of rhsL3PanelCrossSize
	for rhsPanelColIdx := 0; rhsPanelColIdx < rhsCrossSize; rhsPanelColIdx += params.rhsL3PanelCrossSize {
		rhsPanelWidth := min(params.rhsL3PanelCrossSize, rhsCrossSize-rhsPanelColIdx)

		// Loop 4 (p): Tiling K (Depth) - Fits in L1
		// Iterates over the contracting dimension in chunks of contractingPanelSize
		for contractingPanelIdx := 0; contractingPanelIdx < contractingSize; contractingPanelIdx += params.contractingPanelSize {
			contractingPanelWidth := min(params.contractingPanelSize, contractingSize-contractingPanelIdx)

			// ---------------------------------------------------------
			// PACK RHS (Bit) -> ~B
			// We pack a [contractingPanelWidth, rhsPanelWidth] block of RHS into contiguous memory.
			// Format: Vertical strips of width rhsL1KernelCols (Nr).
			// ---------------------------------------------------------
			packRhs(rhs, packedRhs, contractingPanelIdx, rhsPanelColIdx, contractingSize, rhsCrossSize, contractingPanelWidth, rhsPanelWidth, params.rhsL1KernelCols)

			// Loop 3 (ic): Tiling M (Output Rows) - Fits in L2
			// Iterates over the LHS height in chunks of lhsL2PanelCrossSize
			for lhsPanelRowIdx := 0; lhsPanelRowIdx < lhsCrossSize; lhsPanelRowIdx += params.lhsL2PanelCrossSize {
				lhsPanelHeight := min(params.lhsL2PanelCrossSize, lhsCrossSize-lhsPanelRowIdx)

				// -----------------------------------------------------
				// PACK LHS (Ait) -> ~A
				// We pack a [lhsPanelHeight, contractingPanelWidth] block of LHS into contiguous memory.
				// Format: Horizontal strips of height lhsL1KernelRows (Mr).
				// -----------------------------------------------------
				packLhs(lhs, packedLhs, lhsPanelRowIdx, contractingPanelIdx, lhsCrossSize, contractingSize, lhsPanelHeight, contractingPanelWidth, params.lhsL1KernelRows)

				// Loop 2 (jr): Micro-Kernel Columns (Nr == rhsL1BlockCols)
				for microColIdx := 0; microColIdx < rhsPanelWidth; microColIdx += params.rhsL1KernelCols {
					// Actual width to process (might be < Nr at matrix edge)
					microKernelActualWidth := min(params.rhsL1KernelCols, rhsPanelWidth-microColIdx)

					// Loop 1 (ir): Micro-Kernel Rows (Mr == lhsL1BlockRows)
					for microRowIdx := 0; microRowIdx < lhsPanelHeight; microRowIdx += params.lhsL1KernelRows {
						microKernelActualHeight := min(params.lhsL1KernelRows, lhsPanelHeight-microRowIdx)

						// ---------------------------------------------
						// MICRO KERNEL
						// Computes a [Mr, Nr] tile of Output
						// ---------------------------------------------

						// Calculate pointers into packed buffers.
						// PackRhs is organized in strips of Nr. We need the current strip (microColIdx / Nr).
						// Each strip has size (contractingPanelWidth * Nr).
						offsetRhs := (microColIdx / params.rhsL1KernelCols) * (contractingPanelWidth * params.rhsL1KernelCols)

						// PackLhs is organized in strips of Mr. We need the current strip (microRowIdx / Mr).
						// Each strip has size (contractingPanelWidth * Mr).
						offsetLhs := (microRowIdx / params.lhsL1KernelRows) * (contractingPanelWidth * params.lhsL1KernelRows)

						// Output index calculation (Absolute coordinates)
						outputRow := lhsPanelRowIdx + microRowIdx
						outputCol := rhsPanelColIdx + microColIdx

						microKernelFloat32(
							contractingPanelWidth,
							alpha, beta,
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
		accum_lhs0_rhs0 = accum_lhs0_rhs0.MulAdd(lhsVec0, rhsVec0)
		accum_lhs0_rhs1 = accum_lhs0_rhs1.MulAdd(lhsVec0, rhsVec1)

		// Row 1
		lhsVal1 := ptrLhs[idxLhs+1]
		lhsVec1 := archsimd.BroadcastFloat32x16(lhsVal1)
		accum_lhs1_rhs0 = accum_lhs1_rhs0.MulAdd(lhsVec1, rhsVec0)
		accum_lhs1_rhs1 = accum_lhs1_rhs1.MulAdd(lhsVec1, rhsVec1)

		// Row 2
		lhsVal2 := ptrLhs[idxLhs+2]
		lhsVec2 := archsimd.BroadcastFloat32x16(lhsVal2)
		accum_lhs2_rhs0 = accum_lhs2_rhs0.MulAdd(lhsVec2, rhsVec0)
		accum_lhs2_rhs1 = accum_lhs2_rhs1.MulAdd(lhsVec2, rhsVec1)

		// Row 3
		lhsVal3 := ptrLhs[idxLhs+3]
		lhsVec3 := archsimd.BroadcastFloat32x16(lhsVal3)
		accum_lhs3_rhs0 = accum_lhs3_rhs0.MulAdd(lhsVec3, rhsVec0)
		accum_lhs3_rhs1 = accum_lhs3_rhs1.MulAdd(lhsVec3, rhsVec1)

		// Row 4
		lhsVal4 := ptrLhs[idxLhs+4]
		lhsVec4 := archsimd.BroadcastFloat32x16(lhsVal4)
		accum_lhs4_rhs0 = accum_lhs4_rhs0.MulAdd(lhsVec4, rhsVec0)
		accum_lhs4_rhs1 = accum_lhs4_rhs1.MulAdd(lhsVec4, rhsVec1)

		// Row 5
		lhsVal5 := ptrLhs[idxLhs+5]
		lhsVec5 := archsimd.BroadcastFloat32x16(lhsVal5)
		accum_lhs5_rhs0 = accum_lhs5_rhs0.MulAdd(lhsVec5, rhsVec0)
		accum_lhs5_rhs1 = accum_lhs5_rhs1.MulAdd(lhsVec5, rhsVec1)

		idxLhs += 6 // Skips to the next contracting strip in lhs kernel.
	}

	// ---------------------------------------------------------
	// 2. Write Back to Output (Scaling with Beta/Alpha)
	// ---------------------------------------------------------
	// Helper closure to write a row
	alphaBroadcast := archsimd.BroadcastFloat32x16(alpha)
	betaBroadcast := archsimd.BroadcastFloat32x16(beta)
	cols0Bits := min(16, rhsActiveCols)
	cols1Bits := min(16, max(0, rhsActiveCols-16))
	maskForCols0 := archsimd.Mask32x16FromBits(uint16(uint64(1<<cols0Bits) - 1))
	maskForCols1 := archsimd.Mask32x16FromBits(uint16(uint64(1<<cols1Bits) - 1))

	writeRow := func(row int, acc0, acc1 archsimd.Float32x16) {
		outputIdx := (outputRowStart+row)*outputStride + outputColStart
		if row > 0 {
			if rhsActiveCols >= 16 {
				// Store first row.
				outSlice := output[outputIdx : outputIdx+16]
				curValue := archsimd.LoadFloat32x16Slice(outSlice)
				curValue = curValue.Mul(betaBroadcast)
				curValue = curValue.Add(acc0)
				curValue.StoreSlice(outSlice)

				// Store second row.
				if rhsActiveCols == 32 {
					// Full second row.
					outSlice = output[outputIdx+16 : outputIdx+32]
					curValue = archsimd.LoadFloat32x16Slice(outSlice)
					curValue = curValue.Mul(betaBroadcast)
					curValue = curValue.Add(acc1)
					curValue.StoreSlice(outSlice)
				} else {
					// Partial second row.
					outSlice := output[outputIdx+16 : outputIdx+rhsActiveCols]
					curValue = archsimd.LoadFloat32x16SlicePart(outSlice)
					curValue = curValue.Mul(betaBroadcast)
					curValue = curValue.Add(acc1)
					curValue.StoreMasked(castToArray16(&outSlice[0]), maskForCols1)
				}
			} else {
				// Store partial first row.
				outSlice := output[outputIdx : outputIdx+rhsActiveCols]
				curValue := archsimd.LoadFloat32x16SlicePart(outSlice)
				curValue = curValue.Mul(betaBroadcast)
				curValue = curValue.Add(acc0)
				curValue.StoreMasked(castToArray16(&outSlice[0]), maskForCols0)
			}
		}
	}

	switch {
	case lhsActiveRows > 5:
		writeRow(5,
			accum_lhs5_rhs0.Mul(alphaBroadcast),
			accum_lhs5_rhs1.Mul(alphaBroadcast))
		fallthrough
	case lhsActiveRows > 4:
		writeRow(4,
			accum_lhs4_rhs0.Mul(alphaBroadcast),
			accum_lhs4_rhs1.Mul(alphaBroadcast))
		fallthrough
	case lhsActiveRows > 3:
		writeRow(3,
			accum_lhs3_rhs0.Mul(alphaBroadcast),
			accum_lhs3_rhs1.Mul(alphaBroadcast))
		fallthrough
	case lhsActiveRows > 2:
		writeRow(2,
			accum_lhs2_rhs0.Mul(alphaBroadcast),
			accum_lhs2_rhs1.Mul(alphaBroadcast))
		fallthrough
	case lhsActiveRows > 1:
		writeRow(1,
			accum_lhs1_rhs0.Mul(alphaBroadcast),
			accum_lhs1_rhs1.Mul(alphaBroadcast))
		fallthrough
	case lhsActiveRows > 0:
		writeRow(0,
			accum_lhs0_rhs0.Mul(alphaBroadcast),
			accum_lhs0_rhs1.Mul(alphaBroadcast))
	}
}

// packRhs packs a [depth, width] block from RHS into packedRhs.
// It rearranges data into vertical strips of width Nr (rhsL1BlockCols).
// If the block is smaller than Nr, it ZERO-PADS.
func packRhs(src, dst []float32, rowStart, colStart, strideRow, strideCol, depth, width, nr int) {
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
