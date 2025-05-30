package simplego

import (
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/xsync"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
)

// This file contains the implementation for dot-general operations on large tensors -- except the batch
// dimension, which is handled the same way for large or small tensors.
//
// The underlying algorithm is based on the wikipedia description here:
// https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#Non-square_matrices
//
// We also parallelize the algorithm where possible and worth the parallelization costs.

var (
	// DotGeneralTargetBlockSize is hardware-specific, it should be aligned with the L1 cache size
	// and maybe page-size.
	// It should be the number per thread, not necessarily the number per core.
	// It was empirically optimized in an AMD 9950x3d.
	// TODO: find out how to initialize this number in runtime.
	DotGeneralTargetBlockSize = 4 * 1024

	// DotGeneralTargetBlockLog2Dim is set per dtype, such that it is square and fits DotGeneralTargetBlockSize.
	// The block dim is 2^(DotGeneralTargetBlockLog2Dim[dtype]).
	DotGeneralTargetBlockLog2Dim [MaxDTypes]int
)

func init() {
	for _, dtype := range []dtypes.DType{dtypes.F32, dtypes.F64, dtypes.BFloat16} {
		sizePerElem := dtype.Size()
		if dtype == dtypes.BFloat16 || dtype == dtypes.Float16 {
			// Because for BFloat16/Float16 we store the results in float32 and only later convert to
			// BFloat16/Float16. This avoids numeric issues with accumulating sums in small precision
			// types.
			sizePerElem = 4
		}
		dim := 2
		log2Dim := 1
		for dim*dim*sizePerElem < DotGeneralTargetBlockSize {
			dim *= 2
			log2Dim++
		}
		dim /= 2
		log2Dim--
		DotGeneralTargetBlockLog2Dim[dtype] = log2Dim
	}
}

// dgCreateBlockedShape returns a shape that is able to split the original shape into blocks, with extra
// padding (zero initialized) to make it fit.
//
// Input shape: [batchSize, crossSize, contractingSize]
// Output shape: [batchSize, crossBlocks * blkDim, contractBlocks * blkDim]
func dgCreateBlockedShape(dtype dtypes.DType, batchSize, crossSize, contractingSize, blkLog2Dim int) shapes.Shape {
	blkDim := 1 << blkLog2Dim
	newCrossDim := (crossSize + blkDim - 1) / blkDim
	newContractDim := (contractingSize + blkDim - 1) / blkDim
	return shapes.Make(dtype, batchSize, newCrossDim, newContractDim, blkDim, blkDim)
}

var dotGeneralFlatToBlockDTypeMap = NewDTypeMap("DotGeneralFlatToBlock")

// dgCopyDataToBlockedShape copies the data from the original (with a non-normalized shape, with the contracting axes
// and batch axes given) to blocked, whose shape is normalized to [batchSize, crossSize, contractingSize] and
// is organized in blocks (packages) of shape [1, blkDim, blkDim].
//
// blkOutput is assumed to have been created with a size that is multiple of blkDim for the cross and contracting axes.
//
// source shape: any combination of batch, cross or contracting dimensions.
// blkOutput shape: [batchSize, crossBlocks * blkDim, contractBlocks * blkDim]
func dgCopyFlatToBlockShape[T interface {
	PODNumericConstraints | bfloat16.BFloat16
}](
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

	// Calculate sizes
	blkDim := 1 << blkLog2Dim
	blkMask := blkDim - 1
	crossBlocks := (crossSize + blkDim - 1) / blkDim
	contractBlocks := (contractingSize + blkDim - 1) / blkDim

	// Calculate the virtual output shape as: [batchSize, outerCross, outerContracting, innerCross, innerContracting],
	// where innerCross = innerContracting = blkDim.
	outputDims := [5]int{batchSize, crossBlocks, contractBlocks, blkDim, blkDim}
	outputStrides := [5]int{1, 1, 1, 1, 1}
	for ii := 3; ii >= 0; ii-- {
		outputStrides[ii] = outputStrides[ii+1] * outputDims[ii+1]
	}
	var outputIdx [5]int
	var outputCrossIdx, outputContractIdx int

	// Pre-compute axis counters and limits
	sourceData := source.flat.([]T)
	outputData := blkOutput.flat.([]T)
	sourceIdx := make([]int, rank)

	// Sequential iteration over source data
	for sourceFlatIdx := range len(sourceData) {
		// Copy over value.
		outputIdx[4] = outputContractIdx & blkMask     // Take only the innerContracting bits.
		outputIdx[2] = outputContractIdx >> blkLog2Dim // Shift innerContracting bits away.
		outputIdx[3] = outputCrossIdx & blkMask
		outputIdx[1] = outputCrossIdx >> blkLog2Dim
		outputFlatIdx := outputIdx[4] +
			outputIdx[3]*outputStrides[3] +
			outputIdx[2]*outputStrides[2] +
			outputIdx[1]*outputStrides[1] +
			outputIdx[0]*outputStrides[0]
		outputData[outputFlatIdx] = sourceData[sourceFlatIdx]
		//fmt.Printf("\toutput%v (flat %d) = source%v (flat %d)\n", outputIdx, outputFlatIdx, sourceIdx, sourceFlatIdx)

		// Increment position.
		for axis := rank - 1; axis >= 0; axis-- {
			if sourceDims[axis] == 1 {
				continue
			}

			sourceIdx[axis]++
			if sourceIdx[axis] < sourceDims[axis] {
				// Not reached the end of this axis.
				switch axesTypes[axis] {
				case 0: // Cross
					outputCrossIdx += sourceStrides[axis]
				case 1: // Contracting
					outputContractIdx += sourceStrides[axis]
				case 2: // Batch
					outputIdx[0] += sourceStrides[axis]
				}
				break
			}

			// Reached the end of this axis, rewind the index to 0: both in sourceIdx and the corresponding output index.
			sourceIdx[axis] = 0
			switch axesTypes[axis] {
			case 0: // Cross
				outputCrossIdx -= sourceRewindAmount[axis]
			case 1: // Contracting
				outputContractIdx -= sourceRewindAmount[axis]
			case 2: // Batch
				outputIdx[0] -= sourceRewindAmount[axis]
			}
		}
	}
}

var dotGeneralOutputBlockToFlatDTypeMap = NewDTypeMap("DotGeneralNormalizedBlockToFlat")

// dgCopyOutputBlockToFlat copies the blocked output to a flat output, removing the padding.
//
// blockedSource shape: [batchSize, lhsCrossBlocks, rhsCrossBlocks, blockDim, blockDim]
// output shape: [batchSize, lhsCrossSize, rhsCrossSize]
func dgCopyOutputBlockToFlat[T interface {
	PODNumericConstraints | bfloat16.BFloat16
}](blockSource, output *Buffer) {
	sourceDims := blockSource.shape.Dimensions
	outputDims := output.shape.Dimensions

	batchSize := sourceDims[0]
	lhsBlockCross := sourceDims[1]
	rhsBlockCross := sourceDims[2]
	blockDim := sourceDims[3] // Same as sourceDims[4]
	lhsCrossSize := outputDims[1]
	rhsCrossSize := outputDims[2]

	// Pre-calculate strides
	outputRhsStride := 1
	outputLhsStride := rhsCrossSize
	outputBatchStride := lhsCrossSize * rhsCrossSize

	sourceBlockSize := blockDim * blockDim
	sourceRhsBlockStride := sourceBlockSize
	sourceLhsBlockStride := rhsBlockCross * sourceBlockSize
	sourceBatchStride := lhsBlockCross * rhsBlockCross * sourceBlockSize

	sourceData := blockSource.flat.([]T)
	outputData := output.flat.([]T)

	for batch := 0; batch < batchSize; batch++ {
		sourceBatchOffset := batch * sourceBatchStride
		outputBatchOffset := batch * outputBatchStride

		for lhsBlock := 0; lhsBlock < lhsBlockCross && lhsBlock*blockDim < lhsCrossSize; lhsBlock++ {
			lhsStart := lhsBlock * blockDim
			lhsEnd := min(lhsStart+blockDim, lhsCrossSize)
			sourceLhsOffset := sourceBatchOffset + lhsBlock*sourceLhsBlockStride
			outputLhsOffset := outputBatchOffset + lhsStart*outputLhsStride

			for rhsBlock := 0; rhsBlock < rhsBlockCross && rhsBlock*blockDim < rhsCrossSize; rhsBlock++ {
				rhsStart := rhsBlock * blockDim
				rhsEnd := min(rhsStart+blockDim, rhsCrossSize)
				sourceBlockOffset := sourceLhsOffset + rhsBlock*sourceRhsBlockStride
				outputBlockOffset := outputLhsOffset + rhsStart*outputRhsStride

				// Copy valid elements from the block
				for i := 0; i < lhsEnd-lhsStart; i++ {
					sourceRowOffset := sourceBlockOffset + i*blockDim
					outputRowOffset := outputBlockOffset + i*outputLhsStride
					copy(outputData[outputRowOffset:outputRowOffset+rhsEnd-rhsStart],
						sourceData[sourceRowOffset:sourceRowOffset+rhsEnd-rhsStart])
				}
			}
		}
	}
}

func init() {
	dotGeneralOutputBlockToFlatDTypeMap.Register(dtypes.BFloat16, dgCopyOutputBlockToFlatBFloat16)
}

// dgCopyOutputBlockToFlatBFloat16 copies the blocked output to a flat output, removing the padding.
// The blockSource is assumed to be float32 -- matrix multiplication uses float32 to s
//
// blockedSource shape: float32[batchSize, lhsCrossBlocks, rhsCrossBlocks, blockDim, blockDim]
// output shape: bfloat16[batchSize, lhsCrossSize, rhsCrossSize]
func dgCopyOutputBlockToFlatBFloat16(blockSource, output *Buffer) {
	sourceDims := blockSource.shape.Dimensions
	outputDims := output.shape.Dimensions

	batchSize := sourceDims[0]
	lhsBlockCross := sourceDims[1]
	rhsBlockCross := sourceDims[2]
	blockDim := sourceDims[3] // Same as sourceDims[4]
	lhsCrossSize := outputDims[1]
	rhsCrossSize := outputDims[2]

	// Pre-calculate strides
	outputRhsStride := 1
	outputLhsStride := rhsCrossSize
	outputBatchStride := lhsCrossSize * rhsCrossSize

	sourceBlockSize := blockDim * blockDim
	sourceRhsBlockStride := sourceBlockSize
	sourceLhsBlockStride := rhsBlockCross * sourceBlockSize
	sourceBatchStride := lhsBlockCross * rhsBlockCross * sourceBlockSize

	sourceData := blockSource.flat.([]float32)
	outputData := output.flat.([]bfloat16.BFloat16)

	for batch := 0; batch < batchSize; batch++ {
		sourceBatchOffset := batch * sourceBatchStride
		outputBatchOffset := batch * outputBatchStride

		for lhsBlock := 0; lhsBlock < lhsBlockCross && lhsBlock*blockDim < lhsCrossSize; lhsBlock++ {
			lhsStart := lhsBlock * blockDim
			lhsEnd := min(lhsStart+blockDim, lhsCrossSize)
			sourceLhsOffset := sourceBatchOffset + lhsBlock*sourceLhsBlockStride
			outputLhsOffset := outputBatchOffset + lhsStart*outputLhsStride

			for rhsBlock := 0; rhsBlock < rhsBlockCross && rhsBlock*blockDim < rhsCrossSize; rhsBlock++ {
				rhsStart := rhsBlock * blockDim
				rhsEnd := min(rhsStart+blockDim, rhsCrossSize)
				sourceBlockOffset := sourceLhsOffset + rhsBlock*sourceRhsBlockStride
				outputBlockOffset := outputLhsOffset + rhsStart*outputRhsStride

				// Copy valid elements from the block
				for blockRow := 0; blockRow < lhsEnd-lhsStart; blockRow++ {
					sourceRowOffset := sourceBlockOffset + blockRow*blockDim
					outputRowOffset := outputBlockOffset + blockRow*outputLhsStride
					for blockCol := range rhsEnd - rhsStart {
						outputData[outputRowOffset+blockCol] = bfloat16.FromFloat32(sourceData[sourceRowOffset+blockCol])
					}
				}
			}
		}
	}
}

func execDotGeneralLarge(backend *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, output *Buffer) error {
	dtype := lhs.shape.DType

	// Get block buffers.
	blkLog2Dim := DotGeneralTargetBlockLog2Dim[dtype]
	blockDim := 1 << blkLog2Dim
	lhsBlocks := backend.getBuffer(dtype, params.lhsBlockedShape.Size())
	lhsBlocks.shape = params.lhsBlockedShape
	lhsBlocks.Zeros()
	copyFlatToBlock := dotGeneralFlatToBlockDTypeMap.Get(dtype).(func(source, blkOutput *Buffer, contractingAxes, batchAxes []int, batchSize, crossSize, contractingSize, blkLog2Dim int))
	copyFlatToBlock(lhs, lhsBlocks, params.lhsContractingAxes, params.lhsBatchAxes,
		params.batchSize, params.lhsCrossSize, params.contractingSize, blkLog2Dim)

	rhsBlocks := backend.getBuffer(dtype, params.rhsBlockedShape.Size())
	rhsBlocks.shape = params.rhsBlockedShape
	rhsBlocks.Zeros()
	copyFlatToBlock(rhs, rhsBlocks, params.rhsContractingAxes, params.rhsBatchAxes,
		params.batchSize, params.rhsCrossSize, params.contractingSize, blkLog2Dim)

	outputBlocks := backend.getBuffer(params.outputBlockedShape.DType, params.outputBlockedShape.Size())
	outputBlocks.shape = params.outputBlockedShape
	outputBlocks.Zeros()

	var recursive dotGeneralRecursiveData
	recursive.backend = backend

	// Get the matrix multiplication kernel for a block.
	kernelBuilder := dotGeneralKernelDTypeMap.Get(dtype).(func(lhs, rhs, output *Buffer, blockDim int) kernelFuncType)
	recursive.kernelFn = kernelBuilder(lhsBlocks, rhsBlocks, outputBlocks, blockDim)

	// Straight block multiplying (as opposed to recursive)
	recursive.lhsCrossBlocks = lhsBlocks.shape.Dimensions[1]
	recursive.rhsCrossBlocks = rhsBlocks.shape.Dimensions[1]
	recursive.contractBlocks = lhsBlocks.shape.Dimensions[2]

	// Decide on intra-example parallelism: up to which depth we should use a new worker.
	maxParallelism := backend.workers.MaxParallelism()
	recursive.maxDepthParallelization = -1 // Disable sub-batch parallelization.
	if backend.workers.IsEnabled() {
		if backend.workers.IsUnlimited() {
			recursive.maxDepthParallelization = 8 // At most 2^8 = 256 goroutines are spawned.
		} else {
			recursive.maxDepthParallelization = log2int(maxParallelism)
			recursive.maxDepthParallelization += 1 // We want to allow slightly more fine-grained parallelization.
		}
	}

	// Decide on using parallelism across the batch -- each example is started on a separate worker.
	useBatchParallelism := backend.workers.IsEnabled()
	batchSplitSize := 1
	if useBatchParallelism && !backend.workers.IsUnlimited() {
		batchSplitSize = (params.batchSize + maxParallelism - 1) / maxParallelism
	}

	// Loop over examples in the batch:
	wg := xsync.NewDynamicWaitGroup() // Control workers started.
	for outerBatchIdx := 0; outerBatchIdx < params.batchSize; outerBatchIdx += batchSplitSize {
		wg.Add(1)
		batchSplitFn := func() {
			for innerBatchIdx := outerBatchIdx; innerBatchIdx < min(outerBatchIdx+batchSplitSize, params.batchSize); innerBatchIdx++ {
				var batchRecursive dotGeneralRecursiveData
				batchRecursive = recursive
				batchRecursive.lhsBatchOffset = innerBatchIdx * recursive.lhsCrossBlocks * recursive.contractBlocks
				batchRecursive.rhsBatchOffset = innerBatchIdx * recursive.rhsCrossBlocks * recursive.contractBlocks
				batchRecursive.outputBatchOffset = innerBatchIdx * recursive.lhsCrossBlocks * recursive.rhsCrossBlocks
				wg.Add(1)
				batchRecursive.apply(0, recursive.lhsCrossBlocks, 0, recursive.rhsCrossBlocks, 0, recursive.contractBlocks, 0, wg)
			}
			wg.Done()
		}
		if useBatchParallelism {
			backend.workers.WaitToStart(batchSplitFn)
		} else {
			batchSplitFn()
		}
	}
	wg.Wait()

	// Free the block buffers.
	backend.putBuffer(lhsBlocks)
	backend.putBuffer(rhsBlocks)

	// Copy over outputBlocks to the normal output.
	copyOutputFn := dotGeneralOutputBlockToFlatDTypeMap.Get(dtype).(func(blockedSource, output *Buffer))
	copyOutputFn(outputBlocks, output)
	backend.putBuffer(outputBlocks)
	return nil
}

// Information passed along the recursive splitting of the dot-general.
type dotGeneralRecursiveData struct {
	backend                                           *Backend
	kernelFn                                          kernelFuncType
	lhsCrossBlocks, rhsCrossBlocks, contractBlocks    int
	lhsBatchOffset, rhsBatchOffset, outputBatchOffset int
	maxDepthParallelization                           int
}

// apply recursively splits the dot-general into smaller blocks and applies the kernel to each block.
//
// At the lowest splitting levels, the kernel is applied to blocks of the form.
//
// The function may return before the work is completed -- if it's being processed by a worker on a separate goroutine,
// but wg.Done() will be called when the work is completed.
//
// If the work is further parallelized, wg.Add() is called for each new worker used, and wg.Done() is called when each
// is completed.
func (r *dotGeneralRecursiveData) apply(
	lhsCrossStart, lhsCrossEnd,
	rhsCrossStart, rhsCrossEnd,
	contractStart, contractEnd int,
	depth int,
	wg *xsync.DynamicWaitGroup) {
	lhsCrossLen := lhsCrossEnd - lhsCrossStart
	rhsCrossLen := rhsCrossEnd - rhsCrossStart
	contractingLen := contractEnd - contractStart
	maxLen := max(max(lhsCrossLen, rhsCrossLen), contractingLen)

	// Base case: no splitting, simple go over all the crosses and calculate the matrix multiplication for this
	// slice.
	if maxLen <= 2 {
		for lhsCross := lhsCrossStart; lhsCross < lhsCrossEnd; lhsCross++ {
			for rhsCross := rhsCrossStart; rhsCross < rhsCrossEnd; rhsCross++ {
				outputBlockIdx := r.outputBatchOffset + lhsCross*r.rhsCrossBlocks + rhsCross
				rhsBlockIdx := r.rhsBatchOffset + rhsCross*r.contractBlocks + contractStart
				lhsBlockIdx := r.lhsBatchOffset + lhsCross*r.contractBlocks + contractStart
				for contract := contractStart; contract < contractEnd; contract++ {
					r.kernelFn(lhsBlockIdx, rhsBlockIdx, outputBlockIdx)
					rhsBlockIdx++
					lhsBlockIdx++
				}
			}
		}
		wg.Done()
		return
	}

	// Recursively split on the largest axis:
	// - The opportunity to parallelize the split, if possible.
	parallelize := depth < r.maxDepthParallelization
	if maxLen == lhsCrossLen {
		// Split on lhs cross dimension.
		wg.Add(1) // The current plus 1.
		split := lhsCrossStart + lhsCrossLen/2
		if !parallelize || !r.backend.workers.StartIfAvailable(func() {
			// If running in a worker:
			r.apply(lhsCrossStart, split, rhsCrossStart, rhsCrossEnd, contractStart, contractEnd, depth+1, wg)
		}) {
			// If not parallelizing, just run the work synchronously.
			r.apply(lhsCrossStart, split, rhsCrossStart, rhsCrossEnd, contractStart, contractEnd, depth+1, wg)
		}
		r.apply(split, lhsCrossEnd, rhsCrossStart, rhsCrossEnd, contractStart, contractEnd, depth+1, wg)

	} else if maxLen == rhsCrossLen {
		// Split on rhs cross dimension.
		wg.Add(1) // The current plus 1.
		split := rhsCrossStart + rhsCrossLen/2
		if !parallelize || !r.backend.workers.StartIfAvailable(func() {
			r.apply(lhsCrossStart, lhsCrossEnd, rhsCrossStart, split, contractStart, contractEnd, depth+1, wg)
		}) {
			// If not parallelizing, just run the work synchronously.
			r.apply(lhsCrossStart, lhsCrossEnd, rhsCrossStart, split, contractStart, contractEnd, depth+1, wg)
		}
		r.apply(lhsCrossStart, lhsCrossEnd, split, rhsCrossEnd, contractStart, contractEnd, depth+1, wg)

	} else {
		// No parallelization when splitting on the contracting axis because both splits will be writing
		// to the same output blocks, so there will be memory contention.
		// This also means we don't increase the depth of the recursion.
		split := contractStart + contractingLen/2
		// Create a new working group to force serialization of work here:
		r.backend.workers.WorkerIsAsleep() // Add temporary extra worker, because we are going to wait.
		newWg := xsync.NewDynamicWaitGroup()
		newWg.Add(1)
		r.apply(lhsCrossStart, lhsCrossEnd, rhsCrossStart, rhsCrossEnd, contractStart, split, depth, newWg)
		newWg.Wait()
		r.backend.workers.WorkerRestarted()
		r.apply(lhsCrossStart, lhsCrossEnd, rhsCrossStart, rhsCrossEnd, split, contractEnd, depth, wg)
		return
	}
}

var dotGeneralKernelDTypeMap = NewDTypeMap("DotGeneralKernel")

//go:generate go run ../../internal/cmd/simplego_dispatcher -dispatcher=dispatchDotGeneral -generic=execNormalizedDotGeneralGeneric -int -uint -float

// kernelFuncType is a function that does a matrix mult of the lhs/rhs and adds it to the output buffer, given the indices of the square blocks.
// So output[outputIdx] += lhs[lhsIdx] * rhs[rhsIdx], a block at a time.
// The contracting axis is 1 for both, lhs and rhs.
type kernelFuncType func(lhsBlockIdx, rhsBlockIdx, outputBlockIdx int)

// buildDotGeneralKernel returns a kernel function that does a DotGeneral (matrix multiplication) of the lhs/rhs block
// to the corresponding output buffer block, given the indices of the square blocks.
func buildDotGeneralKernel[T PODNumericConstraints](lhs, rhs, output *Buffer, blockDim int) kernelFuncType {
	lhsFlat := lhs.flat.([]T)
	rhsFlat := rhs.flat.([]T)
	outputFlat := output.flat.([]T)
	blockSize := blockDim * blockDim

	return func(lhsBlockIdx, rhsBlockIdx, outputBlockIdx int) {
		baseLhsIdx := lhsBlockIdx * blockSize
		baseRhsIdx := rhsBlockIdx * blockSize
		outputIdx := outputBlockIdx * blockSize
		for range blockDim { // Loop over lhs rows:
			rhsIdx := baseRhsIdx
			// Loop 4 rows at a time.
			for rhsRow := 0; rhsRow < blockDim; rhsRow += 4 { // range blockDim { // loop over rhs rows:
				lhsIdx := baseLhsIdx
				contractingIdx := 0
				sum0 := outputFlat[outputIdx]
				sum1 := outputFlat[outputIdx+1]
				sum2 := outputFlat[outputIdx+2]
				sum3 := outputFlat[outputIdx+3]
				// Loop unrolled 8 at a time.
				for ; contractingIdx+7 < blockDim; contractingIdx += 8 {
					rhsIdx1 := rhsIdx + blockDim
					rhsIdx2 := rhsIdx + 2*blockDim
					rhsIdx3 := rhsIdx + 3*blockDim
					sum0 += lhsFlat[lhsIdx]*rhsFlat[rhsIdx] +
						lhsFlat[lhsIdx+1]*rhsFlat[rhsIdx+1] +
						lhsFlat[lhsIdx+2]*rhsFlat[rhsIdx+2] +
						lhsFlat[lhsIdx+3]*rhsFlat[rhsIdx+3] +
						lhsFlat[lhsIdx+4]*rhsFlat[rhsIdx+4] +
						lhsFlat[lhsIdx+5]*rhsFlat[rhsIdx+5] +
						lhsFlat[lhsIdx+6]*rhsFlat[rhsIdx+6] +
						lhsFlat[lhsIdx+7]*rhsFlat[rhsIdx+7]
					sum1 += lhsFlat[lhsIdx]*rhsFlat[rhsIdx1] +
						lhsFlat[lhsIdx+1]*rhsFlat[rhsIdx1+1] +
						lhsFlat[lhsIdx+2]*rhsFlat[rhsIdx1+2] +
						lhsFlat[lhsIdx+3]*rhsFlat[rhsIdx1+3] +
						lhsFlat[lhsIdx+4]*rhsFlat[rhsIdx1+4] +
						lhsFlat[lhsIdx+5]*rhsFlat[rhsIdx1+5] +
						lhsFlat[lhsIdx+6]*rhsFlat[rhsIdx1+6] +
						lhsFlat[lhsIdx+7]*rhsFlat[rhsIdx1+7]
					sum2 += lhsFlat[lhsIdx]*rhsFlat[rhsIdx2] +
						lhsFlat[lhsIdx+1]*rhsFlat[rhsIdx2+1] +
						lhsFlat[lhsIdx+2]*rhsFlat[rhsIdx2+2] +
						lhsFlat[lhsIdx+3]*rhsFlat[rhsIdx2+3] +
						lhsFlat[lhsIdx+4]*rhsFlat[rhsIdx2+4] +
						lhsFlat[lhsIdx+5]*rhsFlat[rhsIdx2+5] +
						lhsFlat[lhsIdx+6]*rhsFlat[rhsIdx2+6] +
						lhsFlat[lhsIdx+7]*rhsFlat[rhsIdx2+7]
					sum3 += lhsFlat[lhsIdx]*rhsFlat[rhsIdx3] +
						lhsFlat[lhsIdx+1]*rhsFlat[rhsIdx3+1] +
						lhsFlat[lhsIdx+2]*rhsFlat[rhsIdx3+2] +
						lhsFlat[lhsIdx+3]*rhsFlat[rhsIdx3+3] +
						lhsFlat[lhsIdx+4]*rhsFlat[rhsIdx3+4] +
						lhsFlat[lhsIdx+5]*rhsFlat[rhsIdx3+5] +
						lhsFlat[lhsIdx+6]*rhsFlat[rhsIdx3+6] +
						lhsFlat[lhsIdx+7]*rhsFlat[rhsIdx3+7]
					lhsIdx += 8
					rhsIdx += 8
				}
				// Tail loop.
				for ; contractingIdx < blockDim; contractingIdx++ {
					rhsIdx1 := rhsIdx + blockDim
					rhsIdx2 := rhsIdx + 2*blockDim
					rhsIdx3 := rhsIdx + 3*blockDim
					sum0 += lhsFlat[lhsIdx] * rhsFlat[rhsIdx]
					sum1 += lhsFlat[lhsIdx] * rhsFlat[rhsIdx1]
					sum2 += lhsFlat[lhsIdx] * rhsFlat[rhsIdx2]
					sum3 += lhsFlat[lhsIdx] * rhsFlat[rhsIdx3]
					lhsIdx++
					rhsIdx++
				}
				outputFlat[outputIdx] = sum0
				outputFlat[outputIdx+1] = sum1
				outputFlat[outputIdx+2] = sum2
				outputFlat[outputIdx+3] = sum3
				outputIdx += 4

				// We unrolled 4 rows of RHS, so we need to skip the remaining 3 rows:
				rhsIdx += 3 * blockDim
			} // loop over rhs rows

			// Start next lhs row.
			baseLhsIdx += blockDim
		}
	}
}

func init() {
	dotGeneralKernelDTypeMap.Register(dtypes.BFloat16, buildDotGeneralKernelBFloat16)
}

// buildDotGeneralKernelBFloat16 returns a kernel function that does a DotGeneral (matrix multiplication) of the lhs/rhs block
// to the corresponding output buffer block, given the indices of the square blocks.
//
// This is the version for BFloat16: the inputs are in BFloat16, and the blocked output is in float32 -- to avoid
// numeric errors when accumulating results in the output -- later it gets converted back to BFloat16.
//
// The contracting axis is 1 for both, lhs and rhs.
func buildDotGeneralKernelBFloat16(lhs, rhs, output *Buffer, blockDim int) kernelFuncType {
	lhsFlat := lhs.flat.([]bfloat16.BFloat16)
	rhsFlat := rhs.flat.([]bfloat16.BFloat16)
	outputFlat := output.flat.([]float32)
	blockSize := blockDim * blockDim

	return func(lhsBlockIdx, rhsBlockIdx, outputBlockIdx int) {
		baseLhsIdx := lhsBlockIdx * blockSize
		baseRhsIdx := rhsBlockIdx * blockSize
		outputIdx := outputBlockIdx * blockSize
		for range blockDim { // Loop over lhs rows:
			rhsIdx := baseRhsIdx
			// Loop 4 rows at a time.
			for rhsRow := 0; rhsRow < blockDim; rhsRow += 4 { // range blockDim { // loop over rhs rows:
				lhsIdx := baseLhsIdx
				contractingIdx := 0
				sum0 := outputFlat[outputIdx]
				sum1 := outputFlat[outputIdx+1]
				sum2 := outputFlat[outputIdx+2]
				sum3 := outputFlat[outputIdx+3]
				// Loop unrolled 8 at a time.
				for ; contractingIdx+7 < blockDim; contractingIdx += 8 {
					rhsIdx1 := rhsIdx + blockDim
					rhsIdx2 := rhsIdx + 2*blockDim
					rhsIdx3 := rhsIdx + 3*blockDim
					sum0 += lhsFlat[lhsIdx].Float32()*rhsFlat[rhsIdx].Float32() +
						lhsFlat[lhsIdx+1].Float32()*rhsFlat[rhsIdx+1].Float32() +
						lhsFlat[lhsIdx+2].Float32()*rhsFlat[rhsIdx+2].Float32() +
						lhsFlat[lhsIdx+3].Float32()*rhsFlat[rhsIdx+3].Float32() +
						lhsFlat[lhsIdx+4].Float32()*rhsFlat[rhsIdx+4].Float32() +
						lhsFlat[lhsIdx+5].Float32()*rhsFlat[rhsIdx+5].Float32() +
						lhsFlat[lhsIdx+6].Float32()*rhsFlat[rhsIdx+6].Float32() +
						lhsFlat[lhsIdx+7].Float32()*rhsFlat[rhsIdx+7].Float32()
					sum1 += lhsFlat[lhsIdx].Float32()*rhsFlat[rhsIdx1].Float32() +
						lhsFlat[lhsIdx+1].Float32()*rhsFlat[rhsIdx1+1].Float32() +
						lhsFlat[lhsIdx+2].Float32()*rhsFlat[rhsIdx1+2].Float32() +
						lhsFlat[lhsIdx+3].Float32()*rhsFlat[rhsIdx1+3].Float32() +
						lhsFlat[lhsIdx+4].Float32()*rhsFlat[rhsIdx1+4].Float32() +
						lhsFlat[lhsIdx+5].Float32()*rhsFlat[rhsIdx1+5].Float32() +
						lhsFlat[lhsIdx+6].Float32()*rhsFlat[rhsIdx1+6].Float32() +
						lhsFlat[lhsIdx+7].Float32()*rhsFlat[rhsIdx1+7].Float32()
					sum2 += lhsFlat[lhsIdx].Float32()*rhsFlat[rhsIdx2].Float32() +
						lhsFlat[lhsIdx+1].Float32()*rhsFlat[rhsIdx2+1].Float32() +
						lhsFlat[lhsIdx+2].Float32()*rhsFlat[rhsIdx2+2].Float32() +
						lhsFlat[lhsIdx+3].Float32()*rhsFlat[rhsIdx2+3].Float32() +
						lhsFlat[lhsIdx+4].Float32()*rhsFlat[rhsIdx2+4].Float32() +
						lhsFlat[lhsIdx+5].Float32()*rhsFlat[rhsIdx2+5].Float32() +
						lhsFlat[lhsIdx+6].Float32()*rhsFlat[rhsIdx2+6].Float32() +
						lhsFlat[lhsIdx+7].Float32()*rhsFlat[rhsIdx2+7].Float32()
					sum3 += lhsFlat[lhsIdx].Float32()*rhsFlat[rhsIdx3].Float32() +
						lhsFlat[lhsIdx+1].Float32()*rhsFlat[rhsIdx3+1].Float32() +
						lhsFlat[lhsIdx+2].Float32()*rhsFlat[rhsIdx3+2].Float32() +
						lhsFlat[lhsIdx+3].Float32()*rhsFlat[rhsIdx3+3].Float32() +
						lhsFlat[lhsIdx+4].Float32()*rhsFlat[rhsIdx3+4].Float32() +
						lhsFlat[lhsIdx+5].Float32()*rhsFlat[rhsIdx3+5].Float32() +
						lhsFlat[lhsIdx+6].Float32()*rhsFlat[rhsIdx3+6].Float32() +
						lhsFlat[lhsIdx+7].Float32()*rhsFlat[rhsIdx3+7].Float32()
					lhsIdx += 8
					rhsIdx += 8
				}
				// Tail loop.
				for ; contractingIdx < blockDim; contractingIdx++ {
					rhsIdx1 := rhsIdx + blockDim
					rhsIdx2 := rhsIdx + 2*blockDim
					rhsIdx3 := rhsIdx + 3*blockDim
					sum0 += lhsFlat[lhsIdx].Float32() * rhsFlat[rhsIdx].Float32()
					sum1 += lhsFlat[lhsIdx].Float32() * rhsFlat[rhsIdx1].Float32()
					sum2 += lhsFlat[lhsIdx].Float32() * rhsFlat[rhsIdx2].Float32()
					sum3 += lhsFlat[lhsIdx].Float32() * rhsFlat[rhsIdx3].Float32()
					lhsIdx++
					rhsIdx++
				}
				outputFlat[outputIdx] = sum0
				outputFlat[outputIdx+1] = sum1
				outputFlat[outputIdx+2] = sum2
				outputFlat[outputIdx+3] = sum3
				outputIdx += 4

				// We unrolled 4 rows of RHS, so we need to skip the remaining 3 rows:
				rhsIdx += 3 * blockDim
			} // loop over rhs rows

			// Start next lhs row.
			baseLhsIdx += blockDim
		}
	}
}
