package packgemm

import (
	"github.com/ajroetker/go-highway/hwy"
)

//go:generate go tool github.com/ajroetker/go-highway/cmd/hwygen -input packing_base.go -output_prefix=gen_packing_impl -dispatch gen_packing_dispatch -targets avx2,avx512,fallback

// packRHS packs a slice of size [contractingRows, rhsCols] block from RHS into
// the panel reshaped+transposed to [ceil(rhsCols/RHSL1KernelCols), contractingRows, RHSL1KernelCols],
// padding the cols of the last strip with zeros if necessary.
//
//   - src: [contractingSize, rhsCrossSize]
//   - dst: a slice with enough size to hold the panel
//   - srcRowStart: start row in src
//   - srcColStart: start col in src
//   - srcStrideCol: stride of src
//   - contractingRows: number of rows to be copied in the panel (must fit total panel allocated size)
//   - rhsCols: number of columns to be copied in the panel (excluding padding), will be padded to a RHSL1KernelCols
//     multiple with zeros.
//   - RHSL1KernelCols: number of columns in each "L1 kernel"
func BasePackRHS[T hwy.Floats](src, dst []T, srcRowStart, srcColStart, srcStrideCol,
	contractingRows, rhsCols, RHSL1KernelCols int) {
	dstIdx := 0
	// Iterate over strips of width nr
	for stripColIdx := 0; stripColIdx < rhsCols; stripColIdx += RHSL1KernelCols {
		// How many columns valid in this strip?
		validCols := min(RHSL1KernelCols, rhsCols-stripColIdx)

		// Iterate over rows (k)
		for row := range contractingRows {
			srcRow := srcRowStart + row
			srcColBase := srcColStart + stripColIdx
			srcIdx := (srcRow * srcStrideCol) + srcColBase
			// Copy valid columns
			copy(dst[dstIdx:], src[srcIdx:srcIdx+validCols])
			dstIdx += validCols
			// Zero-pad if strip is incomplete (edge of matrix)
			for c := validCols; c < RHSL1KernelCols; c++ {
				dst[dstIdx] = T(0)
				dstIdx++
			}
		}
	}
}
