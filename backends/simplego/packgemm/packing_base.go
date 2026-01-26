package packgemm

import (
	"github.com/ajroetker/go-highway/hwy"
)

//go:generate go tool github.com/ajroetker/go-highway/cmd/hwygen -input packing_base.go -output_prefix=gen_packing_impl -dispatch gen_packing_dispatch -targets avx2,avx512,fallback

// packRHS packs a slice of size [contractingRows, numCols] block from RHS into
// the panel reshaped+transposed to [ceil(numCols/kernelCols), contractingRows, kernelCols],
// padding the cols of the last strip with zeros if necessary.
//
//   - src: [contractingSize, numCols]
//   - dst: a slice with enough size to hold the panel
//   - srcRowStart: start row in src
//   - srcColStart: start col in src
//   - srcRowStride: row-stride of src (number of columns per row in src)
//   - contractingRows: number of rows to be copied in the panel (must fit total panel allocated size)
//   - numCols: number of columns to be copied in the panel (excluding padding), will be padded to a kernelCols
//     multiple with zeros.
//   - kernelCols: number of columns in each "L1 kernel" (nr)
func BasePackRHS[T hwy.Floats](src, dst []T, srcRowStart, srcColStart, srcRowStride,
	contractingRows, numCols, kernelCols int) {
	dstIdx := 0
	numFullStrips := numCols / kernelCols
	fullStripsCol := numFullStrips * kernelCols
	srcStartRowIdx := srcRowStart * srcRowStride
	// Iterate over strips of width kernelCols (nr)
	for stripColIdx := 0; stripColIdx < fullStripsCol; stripColIdx += kernelCols {
		// Iterate over rows (k)
		srcIdx := srcStartRowIdx + srcColStart + stripColIdx
		for range contractingRows {
			copy(dst[dstIdx:], src[srcIdx:srcIdx+kernelCols])
			dstIdx += kernelCols
			srcIdx += srcRowStride
		}
	}

	// Last strip, with incomplete number of columns.
	validCols := numCols - fullStripsCol
	if validCols == 0 {
		// We are done.
		return
	}
	srcIdx := srcStartRowIdx + srcColStart + fullStripsCol
	for range contractingRows {
		// Copy valid columns
		copy(dst[dstIdx:], src[srcIdx:srcIdx+validCols])
		dstIdx += validCols
		// Zero-pad if strip is incomplete (edge of matrix)
		for c := validCols; c < kernelCols; c++ {
			dst[dstIdx] = T(0)
			dstIdx++
		}
	}
}
