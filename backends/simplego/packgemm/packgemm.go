// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package packgemm

// BufAllocFn is a function that allocates a buffer of type T, of the given size.
type BufAllocFn[T any] func(size int) (ref any, data []T)

// BufReleaseFn is a function that releases a buffer allocated with BufAllocFn.
type BufReleaseFn func(ref any)

// GoroutineStarter is a function that starts a goroutine, if available from the global pool.
// It returns false if no goroutine was started.
type GoroutineStarter func(work func()) bool

// Block/packs parameters for current architecture.
type CacheParams struct {
	LHSL1KernelRows int // or Mr: number of lhs kernel rows going to registers.
	RHSL1KernelCols int // or Nr: Register Block Width

	ContractingPanelSize int // Kc: L1 Block Depth
	LHSL2PanelCrossSize  int // Mc: L2 Block Height
	RHSL3PanelCrossSize  int // Nc: L3 Block Width
}

var (
	// Float32Params is set by the architecture files.
	Float32Params CacheParams

	// Float32 implements generic matrix multiplication for float32 inputs and outputs.
	// output = alpha * (lhs x rhs) + beta * output
	//
	// Check it is not nil in your platform before using.
	Float32 func(alpha, beta float32, lhsFlat, rhsFlat []float32, batchSize, lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat []float32,
		bufAllocFn BufAllocFn[float32], bufReleaseFn BufReleaseFn, starter GoroutineStarter)
)

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

// packLhs packs a [lhsPanelHeight/lhsL1KernelRows, contractingPanelWidth, lhsL1KernelRows] "panel" (a block of size Mr x Kc) from LHS.
// It rearranges data into horizontal strips of height Mr (lhsL1BlockRows).
// packLhs(lhs, packedLhs, lhsPanelRowIdx, contractingPanelIdx, contractingSize, lhsPanelHeight, contractingPanelWidth, params.LHSL1KernelRows)
func packLhs(src, dst []float32, rowStart, colStart, rowStride, lhsPanelHeight, contractingPanelWidth, lhsL1KernelRows int) {
	dstIdx := 0
	// Iterate over strips of height mr
	for stripRowIdx := 0; stripRowIdx < lhsPanelHeight; stripRowIdx += lhsL1KernelRows {
		validRows := min(lhsL1KernelRows, lhsPanelHeight-stripRowIdx)

		// Iterate over columns (contracting size k), we want LHS to be traversed K-first in the kernel
		for col := range contractingPanelWidth {
			srcCol := colStart + col
			srcRowBase := rowStart + stripRowIdx

			// Copy valid "rows" (they are the last axis in the returned panel)
			for row := range validRows {
				srcIdx := ((srcRowBase + row) * rowStride) + srcCol
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
