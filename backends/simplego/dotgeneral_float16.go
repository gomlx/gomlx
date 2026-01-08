// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

// Float16 DotGeneral operations

import (
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/x448/float16"
)

// Float16 specific copy-to-block function for DotGeneral
func dgCopyFlatToBlockShapeFloat16(
	source, blkOutput *Buffer, contractingAxes, batchAxes []int, batchSize, crossSize, contractingSize, blkLog2Dim int) {
	rank := source.shape.Rank()

	// Map source axes to their types (0: cross, 1: contracting, 2: batch)
	axesTypes := make([]int, rank)
	for _, axis := range contractingAxes {
		axesTypes[axis] = 1
	}
	for _, axis := range batchAxes {
		axesTypes[axis] = 2
	}
	sourceDims := source.shape.Dimensions
	sourceStrides := make([]int, rank)
	sourceRewindAmount := make([]int, rank)
	batchStride, crossStride, contractStride := 1, 1, 1
	for axis := rank - 1; axis >= 0; axis-- {
		if axesTypes[axis] != 0 {
			continue
		}
		sourceStrides[axis] = crossStride
		sourceRewindAmount[axis] = crossStride * (sourceDims[axis] - 1)
		crossStride *= sourceDims[axis]
	}
	lenContracting := len(contractingAxes)
	for ii := lenContracting - 1; ii >= 0; ii-- {
		axis := contractingAxes[ii]
		sourceStrides[axis] = contractStride
		sourceRewindAmount[axis] = contractStride * (sourceDims[axis] - 1)
		contractStride *= sourceDims[axis]
	}
	lenBatch := len(batchAxes)
	for ii := lenBatch - 1; ii >= 0; ii-- {
		axis := batchAxes[ii]
		sourceStrides[axis] = batchStride
		sourceRewindAmount[axis] = batchStride * (sourceDims[axis] - 1)
		batchStride *= sourceDims[axis]
	}

	blkDim := 1 << blkLog2Dim
	blkMask := blkDim - 1
	crossBlocks := (crossSize + blkDim - 1) / blkDim
	contractBlocks := (contractingSize + blkDim - 1) / blkDim

	outputDims := [5]int{batchSize, crossBlocks, contractBlocks, blkDim, blkDim}
	outputStrides := [5]int{1, 1, 1, 1, 1}
	for ii := 3; ii >= 0; ii-- {
		outputStrides[ii] = outputStrides[ii+1] * outputDims[ii+1]
	}
	var outputIdx [5]int
	var outputCrossIdx, outputContractIdx int

	sourceData := source.flat.([]float16.Float16)
	outputData := blkOutput.flat.([]float16.Float16)
	sourceIdx := make([]int, rank)

	for sourceFlatIdx := range len(sourceData) {
		outputIdx[4] = outputContractIdx & blkMask
		outputIdx[2] = outputContractIdx >> blkLog2Dim
		outputIdx[3] = outputCrossIdx & blkMask
		outputIdx[1] = outputCrossIdx >> blkLog2Dim
		outputFlatIdx := outputIdx[4] +
			outputIdx[3]*outputStrides[3] +
			outputIdx[2]*outputStrides[2] +
			outputIdx[1]*outputStrides[1] +
			outputIdx[0]*outputStrides[0]
		outputData[outputFlatIdx] = sourceData[sourceFlatIdx]

		for axis := rank - 1; axis >= 0; axis-- {
			if sourceDims[axis] == 1 {
				continue
			}

			sourceIdx[axis]++
			if sourceIdx[axis] < sourceDims[axis] {
				switch axesTypes[axis] {
				case 0:
					outputCrossIdx += sourceStrides[axis]
				case 1:
					outputContractIdx += sourceStrides[axis]
				case 2:
					outputIdx[0] += sourceStrides[axis]
				}
				break
			}

			sourceIdx[axis] = 0
			switch axesTypes[axis] {
			case 0:
				outputCrossIdx -= sourceRewindAmount[axis]
			case 1:
				outputContractIdx -= sourceRewindAmount[axis]
			case 2:
				outputIdx[0] -= sourceRewindAmount[axis]
			}
		}
	}
}

// dgNormalizeShapeFloat16 is the Float16 version of dgNormalizeShape
func dgNormalizeShapeFloat16(backend *Backend, source *Buffer, contractingAxes, batchAxes []int, batchSize, crossSize, contractingSize int) (output *Buffer) {
	rank := source.shape.Rank()

	axesTypes := make([]int, rank)
	for _, axis := range contractingAxes {
		axesTypes[axis] = 1
	}
	for _, axis := range batchAxes {
		axesTypes[axis] = 2
	}

	// Check if reshaping is needed
	needsReshaping := false
	if rank != 3 {
		needsReshaping = true
	} else {
		// Check if axes need reordering
		if len(batchAxes) > 0 && batchAxes[len(batchAxes)-1] != 0 {
			needsReshaping = true
		}
		contractExpected := 1
		if len(batchAxes) > 0 {
			contractExpected = 1
		}
		for _, axis := range contractingAxes {
			if axis != contractExpected {
				needsReshaping = true
				break
			}
			contractExpected++
		}
	}

	if !needsReshaping {
		return source
	}

	// Calculate the permutation and strides
	perm := make([]int, rank)
	permIdx := 0
	for _, axis := range batchAxes {
		perm[permIdx] = axis
		permIdx++
	}
	for axis := 0; axis < rank; axis++ {
		if axesTypes[axis] == 0 {
			perm[permIdx] = axis
			permIdx++
		}
	}
	for _, axis := range contractingAxes {
		perm[permIdx] = axis
		permIdx++
	}

	// Calculate source strides
	sourceDims := source.shape.Dimensions
	sourceStrides := calculateStrides(sourceDims)

	// Allocate output
	outputShape := shapes.Make(dtypes.Float16, batchSize, crossSize, contractingSize)
	output = backend.getBuffer(dtypes.Float16, outputShape.Size())
	output.shape = outputShape

	sourceData := source.flat.([]float16.Float16)
	outputData := output.flat.([]float16.Float16)

	// Iterate
	sourceIdx := make([]int, rank)
	for outputFlatIdx := range outputData {
		// Calculate source flat index
		sourceFlatIdx := 0
		for axis := 0; axis < rank; axis++ {
			sourceFlatIdx += sourceIdx[perm[axis]] * sourceStrides[perm[axis]]
		}
		outputData[outputFlatIdx] = sourceData[sourceFlatIdx]

		// Increment source indices
		for axis := rank - 1; axis >= 0; axis-- {
			sourceIdx[axis]++
			if sourceIdx[axis] < sourceDims[axis] {
				break
			}
			sourceIdx[axis] = 0
		}
	}

	return output
}

func init() {
	dotGeneralFlatToBlockDTypeMap.Register(dtypes.Float16, priorityTyped, dgCopyFlatToBlockShapeFloat16)
	dotGeneralNormalizeShapeDTypeMap.Register(dtypes.Float16, priorityTyped, dgNormalizeShapeFloat16)
}
