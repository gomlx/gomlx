package simplego

import (
	"sync"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/pkg/errors"

	"github.com/gomlx/gomlx/pkg/core/shapes"
)

var dotGeneralNormalizeShapeDTypeMap = NewDTypeMap("DotGeneralNormalizeShape")

// dgNormalizeShape reshapes the source to a rank-3 shape [batchSize, crossSize, contractingSize].
//
// It returns a buffer with the transposed/reshaped source.
//
// In the chance that the source needs no transposing, output is returned nil.
func dgNormalizeShape[T interface {
	PODNumericConstraints | bfloat16.BFloat16
}](backend *Backend, source *Buffer, contractingAxes, batchAxes []int, batchSize, crossSize, contractingSize int) (output *Buffer) {
	rank := source.shape.Rank()

	// Map source axes to their types (0: cross, 1: contracting, 2: batch)
	var needsTranspose bool
	axesTypes := make([]int, rank)
	currentAxis := -1
	for _, axis := range contractingAxes {
		axesTypes[axis] = 1
		if axis < currentAxis {
			needsTranspose = true
		}
		currentAxis = axis
	}
	currentAxis = -1
	for _, axis := range batchAxes {
		axesTypes[axis] = 2
		if axis < currentAxis {
			needsTranspose = true
		}
		currentAxis = axis
	}
	sourceDims := source.shape.Dimensions

	// Check whether the axes types are in the right order:
	currentType := 2 // 2: batch, 1: contracting, 0: cross
	for _, axisType := range axesTypes {
		if axisType == currentType {
			continue
		}
		if (axisType == 2) || (currentType == 1) {
			// Invalid transition.
			needsTranspose = true
			break
		}
		currentType = axisType
	}
	if !needsTranspose {
		// Axes are given in the correct order, no need to transpose (only maybe a reshape).
		return nil
	}

	// sourceStrides stores strides per axis-type: crossStride, contractStride or batchStride.
	// sourceRewindAmount stores the amount needed to rewind when the axis index goes back to zero (see the loop that updates the index below)
	sourceStrides := make([]int, rank)      // Stride is per type of axis.
	sourceRewindAmount := make([]int, rank) // dim-1 * stride.
	batchStride, crossStride, contractStride := 1, 1, 1
	// - crossStride:
	for axis := rank - 1; axis >= 0; axis-- {
		if axesTypes[axis] != 0 {
			continue
		}
		sourceStrides[axis] = crossStride
		sourceRewindAmount[axis] = crossStride * (sourceDims[axis] - 1)
		crossStride *= sourceDims[axis]
	}
	// batchStride and contractStride must be computed in order of the axes given: they may be transposed.
	// - contractStride: strides go from the last axis to the first.
	lenContracting := len(contractingAxes)
	for ii := lenContracting - 1; ii >= 0; ii-- {
		axis := contractingAxes[ii]
		sourceStrides[axis] = contractStride
		sourceRewindAmount[axis] = contractStride * (sourceDims[axis] - 1)
		contractStride *= sourceDims[axis]
	}
	// - batchStride: strides go from the last axis to the first.
	lenBatch := len(batchAxes)
	for ii := lenBatch - 1; ii >= 0; ii-- {
		axis := batchAxes[ii]
		sourceStrides[axis] = batchStride
		sourceRewindAmount[axis] = batchStride * (sourceDims[axis] - 1)
		batchStride *= sourceDims[axis]
	}

	// Create the output buffer.
	outputShape := shapes.Make(source.shape.DType, batchSize, crossSize, contractingSize)
	output = backend.getBufferForShape(outputShape)
	outputStrides := [3]int{crossSize * contractingSize, contractingSize, 1}
	var outputIdx [3]int

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
			var outputAxis int
			switch axesTypes[axis] {
			case 0: // Cross
				outputAxis = 1
			case 1: // Contracting
				outputAxis = 2
			case 2: // Batch
				outputAxis = 0
			}

			if sourceIdx[axis] < sourceDims[axis] {
				// Not reached the end of this axis, continue to next copy position.
				outputIdx[outputAxis] += sourceStrides[axis]
				break
			}

			// Reached the end of this axis, rewind the index to 0: both in sourceIdx and the corresponding output index.
			sourceIdx[axis] = 0
			outputIdx[outputAxis] -= sourceRewindAmount[axis]
		}
	}
	return
}

// execDotGeneralNormalized executes DotGeneral by first transposing operands to normalized form.
//
// The normalized form is [Batch, Cross, Contract] where:
//   - Batch dimensions come first
//   - Cross dimensions (output dimensions) come next
//   - Contracting dimensions come LAST
//
// This ordering ensures that the contracting dimension is contiguous in memory for BOTH
// operands, enabling efficient sequential access and vectorization.
//
// Memory access pattern after normalization:
//
//   LHS normalized to [B, M, K]: row (m) has K elements contiguous ✓
//   RHS normalized to [B, N, K]: row (n) has K elements contiguous ✓
//
// Compare to the direct path (no transpose) where RHS column access is strided.
// The transpose overhead is worthwhile for large matrices where cache locality matters.
//
// This function handles:
//   - Arbitrary axis orderings (transposes as needed)
//   - int8/uint8 quantized operations with int32 accumulation
//   - float16/bfloat16 with float32 accumulation for precision
//   - Batch parallelism across workers
func execDotGeneralNormalized(backend *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, output *Buffer) error {
	dtype := lhs.shape.DType
	normalizeFn := dotGeneralNormalizeShapeDTypeMap.Get(dtype).(func(backend *Backend, source *Buffer, contractingAxes, batchAxes []int, batchSize, crossSize, contractingSize int) *Buffer)

	batchSize := params.batchSize
	contractingSize := params.contractingSize
	lhsCrossSize := params.lhsCrossSize
	rhsCrossSize := params.rhsCrossSize

	lhsNormalized := normalizeFn(backend, lhs, params.lhsContractingAxes, params.lhsBatchAxes,
		batchSize, lhsCrossSize, contractingSize)
	if lhsNormalized == nil {
		lhsNormalized = lhs // The shape is wrong, but the flat values are correct.
	}

	rhsNormalized := normalizeFn(backend, rhs, params.rhsContractingAxes, params.rhsBatchAxes,
		batchSize, rhsCrossSize, contractingSize)
	if rhsNormalized == nil {
		rhsNormalized = rhs // The shape is wrong, but the flat values are correct.
	}

	tmpOutput := output
	outputDType := output.shape.DType

	// Check if we need a temporary output buffer with a different accumulator dtype.
	// This is needed for:
	// - int8/uint8: input is int8 but accumulator is int32
	// - float16/bfloat16: input is fp16 but accumulator is float32 for precision
	needsTempOutput := false
	var accumulatorDType dtypes.DType
	if dtype == dtypes.Int8 || dtype == dtypes.Uint8 {
		accumulatorDType = dtypes.Int32
		needsTempOutput = dtype != outputDType // int8→int32 case
	} else if dtype == dtypes.BFloat16 || dtype == dtypes.Float16 {
		accumulatorDType = dtypes.Float32
		needsTempOutput = true // Always need float32 temp for float16/bfloat16
	}

	if needsTempOutput {
		// Create temporary output with intermediate dtype (float32 for fp16, int32 for int8)
		tmpShape := shapes.Make(accumulatorDType, output.shape.Dimensions...)
		tmpOutput = backend.getBufferForShape(tmpShape)
		tmpOutput.Zeros()
	}

	// Select kernel based on input dtype.
	// Specialized kernels for int8/uint8 are registered in the dtype map and handle
	// accumulation in int32 internally with saturation to output dtype.
	normalizeDotGeneral := dotGeneralNormalizedDTypeMap.Get(dtype).(func(lhs, rhs, output *Buffer, params *dotGeneralNodeData, batchStartIdx, batchEndIdx int))

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

	// If we created a temporary output with different dtype, handle conversion.
	// This is only needed for float16/bfloat16 where we accumulated in float32.
	// For int8/uint8, the specialized kernels handle int32 accumulation and saturation internally.
	if needsTempOutput {
		// For float16→float32, convert back to float16
		convertFn := convertDTypePairMap.Get(tmpOutput.shape.DType, output.shape.DType).(convertFnType)
		convertFn(tmpOutput, output)
		backend.putBuffer(tmpOutput)
	}
	return nil
}

var dotGeneralNormalizedDTypeMap = NewDTypeMap("DotGeneralNormalized")

func execNormalizedDotGeneralGeneric[T PODNumericConstraints](lhs, rhs, output *Buffer, params *dotGeneralNodeData, batchStartIdx, batchEndIdx int) {
	lhsFlat := lhs.flat.([]T)
	rhsFlat := rhs.flat.([]T)
	outputFlat := output.flat.([]T)

	// Notice we cannot trust lhs.shape and rhs.shape, in case they haven't been transposed or reshaped.
	contractingSize := params.contractingSize
	lhsCrossSize := params.lhsCrossSize
	rhsCrossSize := params.rhsCrossSize

	// Pre-compute strides to avoid repeated calculations
	lhsBatchStride := lhsCrossSize * contractingSize
	rhsBatchStride := rhsCrossSize * contractingSize
	outputBatchStride := lhsCrossSize * rhsCrossSize

	// Cache block sizes - adjust based on typical matrix sizes and CPU cache
	const blockSize = 64 // Tune this based on your typical workload and L1 cache size
	for batchIdx := batchStartIdx; batchIdx < batchEndIdx; batchIdx++ {
		lhsBaseIdx := batchIdx * lhsBatchStride
		rhsBaseIdx := batchIdx * rhsBatchStride
		outputBaseIdx := batchIdx * outputBatchStride

		// Use blocking to improve cache locality
		for outerIdxLhsCross := 0; outerIdxLhsCross < lhsCrossSize; outerIdxLhsCross += blockSize {
			lhsCrossBlockEnd := min(outerIdxLhsCross+blockSize, lhsCrossSize)

			for outerIdxRhsCross := 0; outerIdxRhsCross < rhsCrossSize; outerIdxRhsCross += blockSize {
				rhsCrossBlockEnd := min(outerIdxRhsCross+blockSize, rhsCrossSize)

				for outerIdxContracting := 0; outerIdxContracting < contractingSize; outerIdxContracting += blockSize {
					contractingBlockEnd := min(outerIdxContracting+blockSize, contractingSize)

					// Process the current block
					for idxLhsCross := outerIdxLhsCross; idxLhsCross < lhsCrossBlockEnd; idxLhsCross++ {
						lhsRowStartIdx := lhsBaseIdx + idxLhsCross*contractingSize
						outputRowStartIdx := outputBaseIdx + idxLhsCross*rhsCrossSize

						for idxRhsCross := outerIdxRhsCross; idxRhsCross < rhsCrossBlockEnd; idxRhsCross++ {
							rhsColStartIdx := rhsBaseIdx + idxRhsCross*contractingSize
							sum := outputFlat[outputRowStartIdx+idxRhsCross]

							// Unroll the innermost loop for better vectorization
							idxContracting := outerIdxContracting
							for ; idxContracting+7 < contractingBlockEnd; idxContracting += 8 {
								if lhsRowStartIdx+idxContracting+7 >= len(lhsFlat) {
									panic(errors.Errorf("Out-of-bounds for lhs: batchIdx=%d, idxLhsCross=%d, idxRhsCross=%d, idxContracting=%d, len(lhsFlat)=%d, lhsFlatIdx=%d",
										batchIdx, idxLhsCross, idxRhsCross, idxContracting, len(lhsFlat), lhsRowStartIdx+idxContracting+7))
								}
								if rhsColStartIdx+idxContracting+7 >= len(rhsFlat) {
									panic(errors.Errorf("Out-of-bounds for rhs: batchIdx=%d, idxLhsCross=%d, idxRhsCross=%d, idxContracting=%d, len(rhsFlat)=%d, rhsFlatIdx=%d",
										batchIdx, idxLhsCross, idxRhsCross, idxContracting, len(rhsFlat), rhsColStartIdx+idxContracting+7))
								}
								sum += lhsFlat[lhsRowStartIdx+idxContracting]*rhsFlat[rhsColStartIdx+idxContracting] +
									lhsFlat[lhsRowStartIdx+idxContracting+1]*rhsFlat[rhsColStartIdx+idxContracting+1] +
									lhsFlat[lhsRowStartIdx+idxContracting+2]*rhsFlat[rhsColStartIdx+idxContracting+2] +
									lhsFlat[lhsRowStartIdx+idxContracting+3]*rhsFlat[rhsColStartIdx+idxContracting+3] +
									lhsFlat[lhsRowStartIdx+idxContracting+4]*rhsFlat[rhsColStartIdx+idxContracting+4] +
									lhsFlat[lhsRowStartIdx+idxContracting+5]*rhsFlat[rhsColStartIdx+idxContracting+5] +
									lhsFlat[lhsRowStartIdx+idxContracting+6]*rhsFlat[rhsColStartIdx+idxContracting+6] +
									lhsFlat[lhsRowStartIdx+idxContracting+7]*rhsFlat[rhsColStartIdx+idxContracting+7]
							}

							// Handle remaining elements
							for ; idxContracting < contractingBlockEnd; idxContracting++ {
								sum += lhsFlat[lhsRowStartIdx+idxContracting] * rhsFlat[rhsColStartIdx+idxContracting]
							}

							outputFlat[outputRowStartIdx+idxRhsCross] = sum
						}
					}
				}
			}
		}
	}
}

func init() {
	// Use priorityTyped so NEON-optimized version from dotgeneral_fp16_neon_arm64.go can take precedence
	dotGeneralNormalizedDTypeMap.Register(dtypes.BFloat16, priorityTyped, execNormalizedDotGeneralBfloat16)
}
func execNormalizedDotGeneralBfloat16(lhs, rhs, output *Buffer, params *dotGeneralNodeData, batchStartIdx, batchEndIdx int) {
	lhsFlat := lhs.flat.([]bfloat16.BFloat16)
	rhsFlat := rhs.flat.([]bfloat16.BFloat16)
	outputFlat := output.flat.([]float32) // Notice we use float32 for the output of BF16 DotGeneral.

	// Notice we cannot trust lhs.shape and rhs.shape, in case they haven't been transposed or reshaped.
	contractingSize := params.contractingSize
	lhsCrossSize := params.lhsCrossSize
	rhsCrossSize := params.rhsCrossSize

	// Pre-compute strides to avoid repeated calculations
	lhsBatchStride := lhsCrossSize * contractingSize
	rhsBatchStride := rhsCrossSize * contractingSize
	outputBatchStride := lhsCrossSize * rhsCrossSize

	// Cache block sizes - adjust based on typical matrix sizes and CPU cache
	const blockSize = 64 // Tune this based on your typical workload and L1 cache size
	for batchIdx := batchStartIdx; batchIdx < batchEndIdx; batchIdx++ {
		lhsBaseIdx := batchIdx * lhsBatchStride
		rhsBaseIdx := batchIdx * rhsBatchStride
		outputBaseIdx := batchIdx * outputBatchStride

		// Use blocking to improve cache locality
		for outerIdxLhsCross := 0; outerIdxLhsCross < lhsCrossSize; outerIdxLhsCross += blockSize {
			lhsCrossBlockEnd := min(outerIdxLhsCross+blockSize, lhsCrossSize)

			for outerIdxRhsCross := 0; outerIdxRhsCross < rhsCrossSize; outerIdxRhsCross += blockSize {
				rhsCrossBlockEnd := min(outerIdxRhsCross+blockSize, rhsCrossSize)

				for outerIdxContracting := 0; outerIdxContracting < contractingSize; outerIdxContracting += blockSize {
					contractingBlockEnd := min(outerIdxContracting+blockSize, contractingSize)

					// Process the current block
					for idxLhsCross := outerIdxLhsCross; idxLhsCross < lhsCrossBlockEnd; idxLhsCross++ {
						lhsRowStartIdx := lhsBaseIdx + idxLhsCross*contractingSize
						outputRowStartIdx := outputBaseIdx + idxLhsCross*rhsCrossSize

						for idxRhsCross := outerIdxRhsCross; idxRhsCross < rhsCrossBlockEnd; idxRhsCross++ {
							rhsColStartIdx := rhsBaseIdx + idxRhsCross*contractingSize
							sum := outputFlat[outputRowStartIdx+idxRhsCross]

							// Unroll the innermost loop for better vectorization
							idxContracting := outerIdxContracting
							for ; idxContracting+7 < contractingBlockEnd; idxContracting += 8 {
								if lhsRowStartIdx+idxContracting+7 >= len(lhsFlat) {
									panic(errors.Errorf("Out-of-bounds for lhs: batchIdx=%d, idxLhsCross=%d, idxRhsCross=%d, idxContracting=%d, len(lhsFlat)=%d, lhsFlatIdx=%d",
										batchIdx, idxLhsCross, idxRhsCross, idxContracting, len(lhsFlat), lhsRowStartIdx+idxContracting+7))
								}
								if rhsColStartIdx+idxContracting+7 >= len(rhsFlat) {
									panic(errors.Errorf("Out-of-bounds for rhs: batchIdx=%d, idxLhsCross=%d, idxRhsCross=%d, idxContracting=%d, len(rhsFlat)=%d, rhsFlatIdx=%d",
										batchIdx, idxLhsCross, idxRhsCross, idxContracting, len(rhsFlat), rhsColStartIdx+idxContracting+7))
								}
								sum += lhsFlat[lhsRowStartIdx+idxContracting].Float32()*rhsFlat[rhsColStartIdx+idxContracting].Float32() +
									lhsFlat[lhsRowStartIdx+idxContracting+1].Float32()*rhsFlat[rhsColStartIdx+idxContracting+1].Float32() +
									lhsFlat[lhsRowStartIdx+idxContracting+2].Float32()*rhsFlat[rhsColStartIdx+idxContracting+2].Float32() +
									lhsFlat[lhsRowStartIdx+idxContracting+3].Float32()*rhsFlat[rhsColStartIdx+idxContracting+3].Float32() +
									lhsFlat[lhsRowStartIdx+idxContracting+4].Float32()*rhsFlat[rhsColStartIdx+idxContracting+4].Float32() +
									lhsFlat[lhsRowStartIdx+idxContracting+5].Float32()*rhsFlat[rhsColStartIdx+idxContracting+5].Float32() +
									lhsFlat[lhsRowStartIdx+idxContracting+6].Float32()*rhsFlat[rhsColStartIdx+idxContracting+6].Float32() +
									lhsFlat[lhsRowStartIdx+idxContracting+7].Float32()*rhsFlat[rhsColStartIdx+idxContracting+7].Float32()
							}

							// Handle remaining elements
							for ; idxContracting < contractingBlockEnd; idxContracting++ {
								sum += lhsFlat[lhsRowStartIdx+idxContracting].Float32() * rhsFlat[rhsColStartIdx+idxContracting].Float32()
							}

							outputFlat[outputRowStartIdx+idxRhsCross] = sum
						}
					}
				}
			}
		}
	}
}
