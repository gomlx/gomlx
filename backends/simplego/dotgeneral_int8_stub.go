// Copyright 2025 The GoMLX Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build noasm || !arm64

package simplego

import (
	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"unsafe"
)

// execNormalizedDotGeneralInt8ToInt32 is a fallback implementation for int8×int8→int32
// matrix multiplication on non-ARM64 platforms. Uses scalar operations.
func execNormalizedDotGeneralInt8ToInt32(lhs, rhs, output *Buffer, params *dotGeneralNodeData, batchStartIdx, batchEndIdx int) {
	lhsFlat := lhs.flat.([]int8)
	rhsFlat := rhs.flat.([]int8)
	outputFlat := output.flat.([]int32)

	contractingSize := params.contractingSize
	lhsCrossSize := params.lhsCrossSize
	rhsCrossSize := params.rhsCrossSize

	lhsBatchStride := lhsCrossSize * contractingSize
	rhsBatchStride := rhsCrossSize * contractingSize
	outputBatchStride := lhsCrossSize * rhsCrossSize

	for batchIdx := batchStartIdx; batchIdx < batchEndIdx; batchIdx++ {
		lhsBaseIdx := batchIdx * lhsBatchStride
		rhsBaseIdx := batchIdx * rhsBatchStride
		outputBaseIdx := batchIdx * outputBatchStride

		for idxLhsCross := 0; idxLhsCross < lhsCrossSize; idxLhsCross++ {
			lhsRowStartIdx := lhsBaseIdx + idxLhsCross*contractingSize
			outputRowStartIdx := outputBaseIdx + idxLhsCross*rhsCrossSize

			for idxRhsCross := 0; idxRhsCross < rhsCrossSize; idxRhsCross++ {
				rhsColStartIdx := rhsBaseIdx + idxRhsCross*contractingSize
				var sum int32

				// Scalar implementation
				for idxContracting := 0; idxContracting < contractingSize; idxContracting++ {
					lhsVal := int32(lhsFlat[lhsRowStartIdx+idxContracting])
					rhsVal := int32(rhsFlat[rhsColStartIdx+idxContracting])
					sum += lhsVal * rhsVal
				}

				outputFlat[outputRowStartIdx+idxRhsCross] += sum
			}
		}
	}
}

// execNormalizedDotGeneralUint8ToInt32 is a fallback implementation for uint8×uint8→int32
// Also handles mixed int8/uint8 cases by treating everything as unsigned
func execNormalizedDotGeneralUint8ToInt32(lhs, rhs, output *Buffer, params *dotGeneralNodeData, batchStartIdx, batchEndIdx int) {
	// Handle both uint8 and int8 inputs by converting to uint8 view
	var lhsFlat, rhsFlat []uint8

	// Convert lhs to uint8 view
	switch lhs.shape.DType {
	case dtypes.Uint8:
		lhsFlat = lhs.flat.([]uint8)
	case dtypes.Int8:
		// Reinterpret int8 as uint8 (same bit pattern, different interpretation)
		int8Flat := lhs.flat.([]int8)
		lhsFlat = unsafe.Slice((*uint8)(unsafe.Pointer(&int8Flat[0])), len(int8Flat))
	}

	// Convert rhs to uint8 view
	switch rhs.shape.DType {
	case dtypes.Uint8:
		rhsFlat = rhs.flat.([]uint8)
	case dtypes.Int8:
		int8Flat := rhs.flat.([]int8)
		rhsFlat = unsafe.Slice((*uint8)(unsafe.Pointer(&int8Flat[0])), len(int8Flat))
	}

	outputFlat := output.flat.([]int32)

	contractingSize := params.contractingSize
	lhsCrossSize := params.lhsCrossSize
	rhsCrossSize := params.rhsCrossSize

	lhsBatchStride := lhsCrossSize * contractingSize
	rhsBatchStride := rhsCrossSize * contractingSize
	outputBatchStride := lhsCrossSize * rhsCrossSize

	for batchIdx := batchStartIdx; batchIdx < batchEndIdx; batchIdx++ {
		lhsBaseIdx := batchIdx * lhsBatchStride
		rhsBaseIdx := batchIdx * rhsBatchStride
		outputBaseIdx := batchIdx * outputBatchStride

		for idxLhsCross := 0; idxLhsCross < lhsCrossSize; idxLhsCross++ {
			lhsRowStartIdx := lhsBaseIdx + idxLhsCross*contractingSize
			outputRowStartIdx := outputBaseIdx + idxLhsCross*rhsCrossSize

			for idxRhsCross := 0; idxRhsCross < rhsCrossSize; idxRhsCross++ {
				rhsColStartIdx := rhsBaseIdx + idxRhsCross*contractingSize
				var sum int32

				// Scalar implementation
				for idxContracting := 0; idxContracting < contractingSize; idxContracting++ {
					lhsVal := int32(lhsFlat[lhsRowStartIdx+idxContracting])
					rhsVal := int32(rhsFlat[rhsColStartIdx+idxContracting])
					sum += lhsVal * rhsVal
				}

				outputFlat[outputRowStartIdx+idxRhsCross] += sum
			}
		}
	}
}
