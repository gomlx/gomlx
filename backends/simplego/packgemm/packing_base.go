package packgemm

import (
	"github.com/ajroetker/go-highway/hwy"
)

//go:generate go tool github.com/ajroetker/go-highway/cmd/hwygen -input packing_base.go -output_prefix=gen_packing_impl -dispatch gen_packing_dispatch -targets avx2,avx512,neon,fallback

// BasePackRHS packs a slice of size [contractingRows, numCols] block from RHS into
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

	useScalar := true
	if hwy.CurrentLevel() != hwy.DispatchScalar {
		// Using SIMD
		numLanes := hwy.NumLanes[T]()
		switch {
		case kernelCols == numLanes:
			// Use highway intrinsics, with one register.
			useScalar = false // No need for the scalar version.
			var v0 hwy.Vec[T]
			for stripColIdx := 0; stripColIdx < fullStripsCol; stripColIdx += kernelCols {
				// Iterate over rows (k)
				srcIdx := srcStartRowIdx + srcColStart + stripColIdx
				for range contractingRows {
					v0 = hwy.Load(src[srcIdx:])
					hwy.Store(v0, dst[dstIdx:])
					dstIdx += kernelCols
					srcIdx += srcRowStride
				}
			}
		case kernelCols == numLanes*2:
			useScalar = false // No need for the scalar version.
			for stripColIdx := 0; stripColIdx < fullStripsCol; stripColIdx += kernelCols {
				// Iterate over rows (k)
				srcIdx := srcStartRowIdx + srcColStart + stripColIdx
				for range contractingRows {
					v0 := hwy.Load(src[srcIdx:])
					v1 := hwy.Load(src[srcIdx+numLanes:])
					hwy.Store(v0, dst[dstIdx:])
					hwy.Store(v1, dst[dstIdx+numLanes:])
					dstIdx += kernelCols
					srcIdx += srcRowStride
				}
			}
		case kernelCols == numLanes*4:
			useScalar = false // No need for the scalar version.
			var v0, v1, v2, v3 hwy.Vec[T]
			for stripColIdx := 0; stripColIdx < fullStripsCol; stripColIdx += kernelCols {
				// Iterate over rows (k)
				srcIdx := srcStartRowIdx + srcColStart + stripColIdx
				for range contractingRows {
					v0 = hwy.Load(src[srcIdx:])
					v1 = hwy.Load(src[srcIdx+numLanes:])
					v2 = hwy.Load(src[srcIdx+numLanes*2:])
					v3 = hwy.Load(src[srcIdx+numLanes*3:])
					hwy.Store(v0, dst[dstIdx:])
					hwy.Store(v1, dst[dstIdx+numLanes:])
					hwy.Store(v2, dst[dstIdx+numLanes*2:])
					hwy.Store(v3, dst[dstIdx+numLanes*3:])
					dstIdx += kernelCols
					srcIdx += srcRowStride
				}
			}
		}
	}

	if useScalar {
		// Use copy() instead:
		for stripColIdx := 0; stripColIdx < fullStripsCol; stripColIdx += kernelCols {
			// Iterate over rows (k)
			srcIdx := srcStartRowIdx + srcColStart + stripColIdx
			for range contractingRows {
				copy(dst[dstIdx:], src[srcIdx:srcIdx+kernelCols])
				dstIdx += kernelCols
				srcIdx += srcRowStride
			}
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
		for range kernelCols - validCols {
			dst[dstIdx] = 0
			dstIdx++
		}
	}
}

// BaseApplyPackedOutput apply a packageOutput "panel" back into the output matrix.
func BaseApplyPackedOutput[T hwy.Floats](
	packedOutput, output []T,
	alpha, beta T,
	packedOutputRowStride int,
	rowOffset, colOffset int, // Global output offsets
	outputRowStride int,
	height, width int, // actual amount of data to copy
) {
	// Vectorized constants
	alphaVec := hwy.Set[T](alpha)
	betaVec := hwy.Set[T](beta)

	for r := range height {
		packedIdx := r * packedOutputRowStride
		outputIdx := (rowOffset+r)*outputRowStride + colOffset

		c := 0
		// Vectorized loop
		if hwy.CurrentLevel() != hwy.DispatchScalar {
			numLanes := hwy.NumLanes[T]()
			for ; c+numLanes <= width; c += numLanes {
				packedVal := hwy.Load(packedOutput[packedIdx:])
				outputVal := hwy.Load(output[outputIdx:])

				// output = alpha * packed + beta * output
				newVal := hwy.MulAdd(alphaVec, packedVal, hwy.Mul(betaVec, outputVal))
				hwy.Store(newVal, output[outputIdx:])

				packedIdx += numLanes
				outputIdx += numLanes
			}
		}

		// Scalar tail
		for ; c < width; c++ {
			val := packedOutput[packedIdx]
			output[outputIdx] = beta*output[outputIdx] + alpha*val
			packedIdx++
			outputIdx++
		}
	}
}
