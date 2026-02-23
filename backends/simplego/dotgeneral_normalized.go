// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"sync"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/x448/float16"
)

var dotGeneralNormalizeShapeDTypeMap = NewDTypeMap("DotGeneralNormalizeShape")

// dgNormalizationInfo holds pre-calculated information for dgNormalizeShape.
// This is calculated at graph construction time.
type dgNormalizationInfo struct {
	needsTranspose     bool
	axisToOutputAxis   []int // For each source axis, which output axis (0=batch, 1=cross, 2=contracting) it maps to.
	sourceStrides      []int
	sourceRewindAmount []int
}

// dgNormalizePrepare pre-calculates the information needed for dgNormalizeShape.
func dgNormalizePrepare(shape shapes.Shape, contractingAxes, batchAxes []int) *dgNormalizationInfo {
	rank := shape.Rank()
	info := &dgNormalizationInfo{
		axisToOutputAxis:   make([]int, rank),
		sourceStrides:      make([]int, rank),
		sourceRewindAmount: make([]int, rank),
	}

	// Map source axes to their types (0: cross, 1: contracting, 2: batch)
	axesTypes := make([]int, rank)
	currentAxis := -1
	for _, axis := range contractingAxes {
		axesTypes[axis] = 1
		if axis < currentAxis {
			info.needsTranspose = true
		}
		currentAxis = axis
	}
	currentAxis = -1
	for _, axis := range batchAxes {
		axesTypes[axis] = 2
		if axis < currentAxis {
			info.needsTranspose = true
		}
		currentAxis = axis
	}
	sourceDims := shape.Dimensions

	// Check whether the axes types are in the right order:
	currentType := 2 // 2: batch, 1: contracting, 0: cross
	for _, axisType := range axesTypes {
		if axisType == currentType {
			continue
		}
		if (axisType == 2) || (currentType == 1) {
			// Invalid transition.
			info.needsTranspose = true
			break
		}
		currentType = axisType
	}

	// Pre-fill axisToOutputAxis
	for axis, axisType := range axesTypes {
		switch axisType {
		case 0: // Cross
			info.axisToOutputAxis[axis] = 1
		case 1: // Contracting
			info.axisToOutputAxis[axis] = 2
		case 2: // Batch
			info.axisToOutputAxis[axis] = 0
		}
	}

	if !info.needsTranspose {
		return info
	}

	// sourceStrides stores strides per axis-type: crossStride, contractStride or batchStride.
	// sourceRewindAmount stores the amount needed to rewind when the axis index goes back to zero (see the loop that updates the index below)
	batchStride, crossStride, contractStride := 1, 1, 1
	// - crossStride:
	for axis := rank - 1; axis >= 0; axis-- {
		if axesTypes[axis] != 0 {
			continue
		}
		info.sourceStrides[axis] = crossStride
		info.sourceRewindAmount[axis] = crossStride * (sourceDims[axis] - 1)
		crossStride *= sourceDims[axis]
	}
	// batchStride and contractStride must be computed in order of the axes given: they may be transposed.
	// - contractStride: strides go from the last axis to the first.
	lenContracting := len(contractingAxes)
	for ii := lenContracting - 1; ii >= 0; ii-- {
		axis := contractingAxes[ii]
		info.sourceStrides[axis] = contractStride
		info.sourceRewindAmount[axis] = contractStride * (sourceDims[axis] - 1)
		contractStride *= sourceDims[axis]
	}
	// - batchStride: strides go from the last axis to the first.
	lenBatch := len(batchAxes)
	for ii := lenBatch - 1; ii >= 0; ii-- {
		axis := batchAxes[ii]
		info.sourceStrides[axis] = batchStride
		info.sourceRewindAmount[axis] = batchStride * (sourceDims[axis] - 1)
		batchStride *= sourceDims[axis]
	}
	return info
}

// dgNormalizeShape reshapes the source to a rank-3 shape [batchSize, crossSize, contractingSize].
//
// It returns a buffer with the transposed/reshaped source.
//
// In the chance that the source needs no transposing, output is returned nil.
// TODO: handle the error.
func dgNormalizeShape[T interface {
	PODNumericConstraints | bfloat16.BFloat16 | float16.Float16
}](backend *Backend, source *Buffer, info *dgNormalizationInfo, batchSize,
	crossSize, contractingSize int) (output *Buffer) {
	if !info.needsTranspose {
		return nil
	}

	// Create the output buffer.
	outputShape := shapes.Make(source.shape.DType, batchSize, crossSize, contractingSize)
	output, _ = backend.getBufferForShape(outputShape)
	outputStrides := [3]int{crossSize * contractingSize, contractingSize, 1}
	var outputIdx [3]int

	sourceDims := source.shape.Dimensions
	rank := source.shape.Rank()

	// Indices we are going to iterate.
	sourceData := source.flat.([]T)
	outputData := output.flat.([]T)
	sourceIdx := make([]int, rank)
	for sourceFlatIdx := range len(sourceData) {
		// Copy value at current index:
		outputFlatIdx := outputStrides[0]*outputIdx[0] + outputStrides[1]*outputIdx[1] + outputStrides[2]*outputIdx[2]
		outputData[outputFlatIdx] = sourceData[sourceFlatIdx]

		// Increment indices in source and output.
		for axis := rank - 1; axis >= 0; axis-- {
			if sourceDims[axis] == 1 {
				continue
			}
			sourceIdx[axis]++

			// The source axis corresponds to one of the 3 output axes depending on the axis type.
			outputAxis := info.axisToOutputAxis[axis]

			if sourceIdx[axis] < sourceDims[axis] {
				// Not reached the end of this axis, continue to next copy position.
				outputIdx[outputAxis] += info.sourceStrides[axis]
				break
			}

			// Reached the end of this axis, rewind the index to 0: both in sourceIdx and the corresponding output index.
			sourceIdx[axis] = 0
			outputIdx[outputAxis] -= info.sourceRewindAmount[axis]
		}
	}
	return
}

// execDotGeneralNormalized executes the dot general operation for normalized shapes:
// both rhs and lhs are shaped [batchSize, crossSize, contractingSize]
func execDotGeneralNormalized(backend *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, output *Buffer) error {
	dtype := lhs.shape.DType
	normalizeFn := dotGeneralNormalizeShapeDTypeMap.Get(dtype).(func(backend *Backend, source *Buffer,
		info *dgNormalizationInfo, batchSize, crossSize, contractingSize int) *Buffer)

	batchSize := params.batchSize
	contractingSize := params.contractingSize
	lhsCrossSize := params.lhsCrossSize
	rhsCrossSize := params.rhsCrossSize

	// Normalize lhs and rhs if needed.
	lhsNormalized := lhs
	rhsNormalized := rhs
	if params.lhsNormalization.needsTranspose {
		lhsNormalized = normalizeFn(backend, lhs, params.lhsNormalization,
			batchSize, lhsCrossSize, contractingSize)
	}
	if params.rhsNormalization.needsTranspose {
		rhsNormalized = normalizeFn(backend, rhs, params.rhsNormalization,
			batchSize, rhsCrossSize, contractingSize)
	}

	tmpOutput := output
	castToFloat32 := dtype == dtypes.BFloat16 || dtype == dtypes.Float16
	if castToFloat32 {
		outputShape := shapes.Make(dtypes.Float32, params.batchSize, params.lhsCrossSize, params.rhsCrossSize)
		var err error
		tmpOutput, err = backend.getBufferForShape(outputShape)
		if err != nil {
			return err
		}
		tmpOutput.Zeros()
	}

	normalizeDotGeneral := dotGeneralNormalizedDTypeMap.Get(dtype).(func(lhs, rhs, output *Buffer,
		params *dotGeneralNodeData, batchStartIdx, batchEndIdx int))

	// Decide on using parallelism across the batch -- each example is started on a separate worker.
	useBatchParallelism := backend.workers.IsEnabled()
	maxParallelism := backend.workers.MaxParallelism()
	batchSplitSize := 1
	if useBatchParallelism && !backend.workers.IsUnlimited() {
		batchSplitSize = (params.batchSize + maxParallelism - 1) / maxParallelism
	}

	if !useBatchParallelism {
		// Process the whole batch in one call inline in the current worker.
		normalizeDotGeneral(lhsNormalized, rhsNormalized, tmpOutput, params, 0, batchSize)
	} else {
		// Split in batchSplitSize
		wg := sync.WaitGroup{}
		for batchStartIdx := 0; batchStartIdx < batchSize; batchStartIdx += batchSplitSize {
			batchEndIdx := min(batchStartIdx+batchSplitSize, batchSize)
			wg.Add(1)
			backend.workers.WaitToStart(func() {
				normalizeDotGeneral(lhsNormalized, rhsNormalized, tmpOutput, params, batchStartIdx, batchEndIdx)
				wg.Done()
			})
		}
		wg.Wait()
	}

	// If we created a temporary float32 output, convert it back to the original dtype.
	if castToFloat32 {
		convertFn := convertDTypePairMap.Get(dtypes.Float32, output.shape.DType).(convertFnType)
		convertFn(tmpOutput, output)
		backend.putBuffer(tmpOutput) // Return the temporary buffer to the pool.
	}
	return nil
}

var dotGeneralNormalizedDTypeMap = NewDTypeMap("DotGeneralNormalized")

// Auto-generate alternate specialized versions of execNormalizedDotGeneral
// (that can't easily be refactored into smaller functions due to latency penalities)
//go:generate go run ../../internal/cmd/alternates_generator -base=dotgeneral_normalized_alt_base.go -tags=bf16,f16

func init() {
	dotGeneralNormalizedDTypeMap.Register(dtypes.BFloat16, priorityTyped, execNormalizedDotGeneralBFloat16)
	dotGeneralNormalizedDTypeMap.Register(dtypes.Float16, priorityTyped, execNormalizedDotGeneralFloat16)
}
