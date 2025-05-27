package simplego

import (
	"runtime"
	"sync"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/pkg/errors"
)

func init() {
	nodeExecutors[backends.OpTypeDotGeneral] = execDotGeneral
}

type dotGeneralNodeData struct {
	lhsContractingAxes, lhsBatchAxes                       []int
	rhsContractingAxes, rhsBatchAxes                       []int
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int
	lhsBlockedShape, rhsBlockedShape, outputBlockedShape   shapes.Shape
}

// adjustAxisToRank returns a positive axis, adjusting negative numbers to the correct rank.
func adjustAxisToRank(rank, axis int) (int, error) {
	if axis < 0 {
		axis += rank
	}
	if axis < 0 || axis >= rank {
		return -1, errors.Errorf("axis %d is out of range [0, %d)", axis, rank)
	}
	return axis, nil
}

// DotGeneral takes as input lhs (left-hand-side) and rhs (right-hand-side) specifications
// for a general vector product -- a generalized "Einsum". Each axis can be:
//   - Just aligned (batch axes), so the output has the same axes as the inputs. The dimensions
//     must match in lhs and rhs.
//   - Crossed (default), in which case the output is the combination (concatenation) of the
//     dimensions.
//   - Contracted (contracting axes), where the output does multiply the values and reduce sum
//     those dimensions.
//
// It follows that the resulting dimension number starts with the batch dimension, then the 'lhs'
// non-contracting/non-batch dimension and finally the 'rhs' non-contracting/non-batch dimension.
// It provides the basic means of implementing Einsum.
//
// This function implements backends.Builder interface.
//
// This is the graph building part of DotGeneral. It first transposes the operands to a normalized
// shape with rank=3 ([batchSize, crossSize, contractingSize]), and then it issues the DotGeneral
// node with normalized inputs. Finally, it reshapes back to the final result.
//
// See execDotGeneral for the implementation.
func (b *Builder) DotGeneral(lhsOp backends.Op, lhsContractingAxes, lhsBatchAxes []int, rhsOp backends.Op, rhsContractingAxes, rhsBatchAxes []int) (backends.Op, error) {
	inputs, err := b.checkOps(backends.OpTypeDotGeneral.String(), lhsOp, rhsOp)
	if err != nil {
		return nil, err
	}
	lhs, rhs := inputs[0], inputs[1]
	dtype := lhs.shape.DType
	if dtype != rhs.shape.DType {
		return nil, errors.Errorf("DotGeneral lhs (left-hand-side) and rhs operands don't match data types: %s and %s", dtype, rhs.shape.DType)
	}
	if len(lhsContractingAxes) != len(rhsContractingAxes) {
		return nil, errors.Errorf("DotGeneral number of contracting axes for lhs (%d) doesn't match rhs (%d)",
			len(lhsContractingAxes), len(rhsContractingAxes))
	}
	if len(lhsBatchAxes) != len(rhsBatchAxes) {
		return nil, errors.Errorf("DotGeneral number of contracting axes for lhs (%d) doesn't match rhs (%d)",
			len(lhsContractingAxes), len(rhsContractingAxes))
	}

	lhsRank := lhs.shape.Rank()
	rhsRank := rhs.shape.Rank()
	params := dotGeneralNodeData{
		lhsContractingAxes: lhsContractingAxes,
		lhsBatchAxes:       lhsBatchAxes,
		rhsContractingAxes: rhsContractingAxes,
		rhsBatchAxes:       rhsBatchAxes,
	}

	// Validate and adjust axes.
	for ii, axis := range lhsContractingAxes {
		params.lhsContractingAxes[ii], err = adjustAxisToRank(lhsRank, axis)
		if err != nil {
			return nil, errors.WithMessagef(err, "while adjusting contractingAxes for DotGeneral(lhs=%s, lhsContractingAxes=%v)", lhs.shape, lhsContractingAxes)
		}
	}
	for ii, axis := range lhsBatchAxes {
		params.lhsBatchAxes[ii], err = adjustAxisToRank(lhsRank, axis)
		if err != nil {
			return nil, errors.WithMessagef(err, "while adjusting batchAxes for DotGeneral(lhs=%s, lhsBatchAxes=%v)", lhs.shape, lhsBatchAxes)
		}
	}
	for ii, axis := range rhsContractingAxes {
		params.rhsContractingAxes[ii], err = adjustAxisToRank(rhsRank, axis)
		if err != nil {
			return nil, errors.WithMessagef(err, "while adjusting contractingAxes for DotGeneral(rhs=%s, rhsContractingAxes=%v)", rhs.shape, rhsContractingAxes)
		}
	}
	for ii, axis := range rhsBatchAxes {
		params.rhsBatchAxes[ii], err = adjustAxisToRank(rhsRank, axis)
		if err != nil {
			return nil, errors.WithMessagef(err, "while adjusting batchAxes for DotGeneral(rhs=%s, rhsBatchAxes=%v)", rhs.shape, rhsBatchAxes)
		}
	}

	// Check that batch and contracting dimensions from lhs and rhs match.
	batchDims := make([]int, len(lhsBatchAxes))
	contractingDims := make([]int, len(lhsContractingAxes))
	for ii, lhsAxis := range params.lhsContractingAxes {
		rhsAxis := params.rhsContractingAxes[ii]
		if lhs.shape.Dimensions[lhsAxis] != rhs.shape.Dimensions[rhsAxis] {
			return nil, errors.Errorf("DotGeneral contracting dimensions don't match: lhs[%d]=%d != rhs[%d]=%d",
				lhsAxis, lhs.shape.Dimensions[lhsAxis], rhsAxis, rhs.shape.Dimensions[rhsAxis])
		}
		contractingDims[ii] = lhs.shape.Dimensions[lhsAxis]
	}
	for ii, lhsAxis := range params.lhsBatchAxes {
		rhsAxis := params.rhsBatchAxes[ii]
		if lhs.shape.Dimensions[lhsAxis] != rhs.shape.Dimensions[rhsAxis] {
			return nil, errors.Errorf("DotGeneral batch dimensions don't match: lhs[%d]=%d != rhs[%d]=%d",
				lhsAxis, lhs.shape.Dimensions[lhsAxis], rhsAxis, rhs.shape.Dimensions[rhsAxis])
		}
		batchDims[ii] = lhs.shape.Dimensions[lhsAxis]
	}

	// Find sizes of the normalized operands ([batchSize, crossSize, contractSize]).
	var lhsCrossDims, rhsCrossDims []int
	params.batchSize, params.lhsCrossSize, params.contractingSize, lhsCrossDims = dgFindSizes(lhs.shape, lhsContractingAxes, lhsBatchAxes)
	_, params.rhsCrossSize, _, rhsCrossDims = dgFindSizes(rhs.shape, rhsContractingAxes, rhsBatchAxes)

	// Check that all sizes are positive
	if params.batchSize <= 0 || params.lhsCrossSize <= 0 || params.contractingSize <= 0 || params.rhsCrossSize <= 0 {
		return nil, errors.Errorf("DotGeneral sizes must be positive: lhs(batch=%d, cross=%d, contracting=%d), rhs(cross=%d)",
			params.batchSize, params.lhsCrossSize, params.contractingSize,
			params.rhsCrossSize)
	}

	blockLog2Dim := DotGeneralTargetBlockLog2Dim[dtype]
	params.lhsBlockedShape = dgCreateBlockedShape(dtype, params.batchSize, params.lhsCrossSize, params.contractingSize, blockLog2Dim)
	params.rhsBlockedShape = dgCreateBlockedShape(dtype, params.batchSize, params.rhsCrossSize, params.contractingSize, blockLog2Dim)
	outputDType := dtype
	if dtype == dtypes.BFloat16 || dtype == dtypes.Float16 {
		// For 16 bits, store the intermediary results as float32 to minimize numerical errors during accumulation.
		// Notice the blockLog2Dim must be the same, because the block dimensions much match the inputs.
		outputDType = dtypes.Float32
	}
	params.outputBlockedShape = dgCreateBlockedShape(outputDType, params.batchSize, params.lhsCrossSize, params.rhsCrossSize, blockLog2Dim)

	// Create dot-general node: it will generate a normalized output [batchSize, lhsCrossSize, rhsCrossSize].
	dotGeneral := b.newNode(backends.OpTypeDotGeneral, shapes.Make(dtype, params.batchSize, params.lhsCrossSize, params.rhsCrossSize), lhs, rhs)
	dotGeneral.data = &params

	// Reshape result to recover batch and cross dimensions.
	resultingDims := make([]int, 0, len(batchDims)+len(lhsCrossDims)+len(rhsCrossDims))
	resultingDims = append(resultingDims, batchDims...)
	resultingDims = append(resultingDims, lhsCrossDims...)
	resultingDims = append(resultingDims, rhsCrossDims...)
	result, err := b.Reshape(dotGeneral, resultingDims...)

	//fmt.Printf("DotGeneral(*lhs*: %s, c:%v, b:%v; *rhs*:  %s, c:%v, b:%v) -> %s\n",
	//	lhs.shape, lhsContractingAxes, lhsBatchAxes, rhs.shape, rhsContractingAxes, rhsBatchAxes,
	//	result.(*Node).shape)

	if err != nil {
		return nil, err
	}
	return result, nil
}

func dgFindSizes(shape shapes.Shape, contractingAxes, batchAxes []int) (batchSize, crossSize, contractingSize int, crossDims []int) {
	rank := shape.Rank()
	axesTypes := make([]int, rank)

	// Mark axes types: 1 for contracting, 2 for batch
	for _, axis := range contractingAxes {
		axesTypes[axis] = 1
	}
	for _, axis := range batchAxes {
		axesTypes[axis] = 2
	}

	// Calculate sizes by multiplying dimensions according to axis type
	batchSize, crossSize, contractingSize = 1, 1, 1
	crossDims = make([]int, 0, rank-len(contractingAxes)-len(batchAxes))
	for axis, axisType := range axesTypes {
		dim := shape.Dimensions[axis]
		switch axisType {
		case 0: // Cross axes (unmarked)
			crossSize *= dim
			crossDims = append(crossDims, dim)
		case 1: // Contracting axes
			contractingSize *= dim
		case 2: // Batch axes
			batchSize *= dim
		}
	}
	return
}

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
			// Because for BFloat16/Float16 we store the results in float32, and only later convert to
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

// dgCopyDataToBlockedShape copies the data from original (with a non-normalized shape, with the contracting axes
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
	// sourceRewindAmount stores the amount needed to rewind when the axis index goes back to zero (see loop that updates the index below)
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

// minBatchParallelize is the batchSize threshold above which we parallelize on the batch-level,
// if backend.maxParallelism unlimited (<0)
var minBatchParallelize = runtime.NumCPU()

// execDotGeneral executes the DotGeneral by first normalizing and repackaging the tensors into blocks.
func execDotGeneral(backend *Backend, node *Node, inputs []*Buffer, _ []bool) (*Buffer, error) {
	lhs, rhs := inputs[0], inputs[1]
	params := node.data.(*dotGeneralNodeData)
	outputShape := node.shape
	dtype := lhs.shape.DType
	output := backend.getBuffer(dtype, outputShape.Size())
	output.shape = outputShape

	// Problem size (per example of the batch):
	problemSize := params.rhsCrossSize * params.lhsCrossSize * params.contractingSize
	_ = problemSize
	err := execDotGeneralLarge(backend, lhs, rhs, params, output)
	if err != nil {
		backend.putBuffer(output)
		return nil, err
	}
	return output, nil
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

	var wg sync.WaitGroup
	if (backend.maxParallelism > 0 && params.batchSize > backend.maxParallelism) ||
		(backend.maxParallelism < 0 && params.batchSize > minBatchParallelize) {
		// Parallelize at the batch level.
		recursive.parallizeIfPossible = false // Disable sub-batch parallelization.
		for batch := 0; batch < params.batchSize; batch++ {
			batchRecursive := recursive
			batchRecursive.lhsBatchOffset = batch * recursive.lhsCrossBlocks * recursive.contractBlocks
			batchRecursive.rhsBatchOffset = batch * recursive.rhsCrossBlocks * recursive.contractBlocks
			batchRecursive.outputBatchOffset = batch * recursive.lhsCrossBlocks * recursive.rhsCrossBlocks
			wg.Add(1)
			batchRecursive.apply(0, recursive.lhsCrossBlocks, 0, recursive.rhsCrossBlocks, 0, recursive.contractBlocks, &wg)
		}
	} else {
		// Parallelize within the batch examples if possible:
		recursive.parallizeIfPossible = true
		for batch := 0; batch < params.batchSize; batch++ {
			recursive.lhsBatchOffset = batch * recursive.lhsCrossBlocks * recursive.contractBlocks
			recursive.rhsBatchOffset = batch * recursive.rhsCrossBlocks * recursive.contractBlocks
			recursive.outputBatchOffset = batch * recursive.lhsCrossBlocks * recursive.rhsCrossBlocks
			wg.Add(1)
			recursive.apply(0, recursive.lhsCrossBlocks, 0, recursive.rhsCrossBlocks, 0, recursive.contractBlocks, &wg)
		}
	}
	wg.Wait()

	// Copy over outputBlocks to the normal output.
	copyOutputFn := dotGeneralOutputBlockToFlatDTypeMap.Get(dtype).(func(blockedSource, output *Buffer))
	copyOutputFn(outputBlocks, output)
	return nil
}

// Information passed along the recursive splitting of the dot-general.
type dotGeneralRecursiveData struct {
	backend                                           *Backend
	kernelFn                                          kernelFuncType
	parallizeIfPossible                               bool
	lhsCrossBlocks, rhsCrossBlocks, contractBlocks    int
	lhsBatchOffset, rhsBatchOffset, outputBatchOffset int
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
	wg *sync.WaitGroup) {
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
	parallelize := r.parallizeIfPossible && maxLen >= 2
	if maxLen == lhsCrossLen {
		// Split on lhs cross dimension.
		wg.Add(1) // The current plus 1.
		split := lhsCrossStart + lhsCrossLen/2
		if !parallelize || !r.backend.StartWorker(func() {
			// If running in a worker:
			r.apply(lhsCrossStart, split, rhsCrossStart, rhsCrossEnd, contractStart, contractEnd, wg)
		}) {
			// If not parallelizing, just run the work synchronously.
			r.apply(lhsCrossStart, split, rhsCrossStart, rhsCrossEnd, contractStart, contractEnd, wg)
		}
		r.apply(split, lhsCrossEnd, rhsCrossStart, rhsCrossEnd, contractStart, contractEnd, wg)

	} else if maxLen == rhsCrossLen {
		// Split on rhs cross dimension.
		wg.Add(1) // The current plus 1.
		split := rhsCrossStart + rhsCrossLen/2
		if !parallelize || !r.backend.StartWorker(func() {
			r.apply(lhsCrossStart, lhsCrossEnd, rhsCrossStart, split, contractStart, contractEnd, wg)
		}) {
			// If not parallelizing, just run the work synchronously.
			r.apply(lhsCrossStart, lhsCrossEnd, rhsCrossStart, split, contractStart, contractEnd, wg)
		}
		r.apply(lhsCrossStart, lhsCrossEnd, split, rhsCrossEnd, contractStart, contractEnd, wg)

	} else {
		// No parallelization when splitting on the contracting axis because both splits will be writing
		// to the same output blocks, so there will be memory contention.
		split := contractStart + contractingLen/2
		// Create a new working group to force serialization of work here:
		var newWg sync.WaitGroup
		newWg.Add(1)
		r.apply(lhsCrossStart, lhsCrossEnd, rhsCrossStart, rhsCrossEnd, contractStart, split, &newWg)
		newWg.Wait()
		r.apply(lhsCrossStart, lhsCrossEnd, rhsCrossStart, rhsCrossEnd, split, contractEnd, wg)
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

// Dot ------------------------------------------------------------------------------------------------------
// Dot implements backends.Builder interface.
//
// It is implemented using DotGeneral and Reshape.
//
// Dot returns the "dot product" operation.
// The exact semantics of this operation depend on the ranks of the operands:
// | Input | Output | Semantics |
// | vector [n] dot vector [n] | scalar | vector dot product |
// | matrix [m x k] dot vector [k] | vector [m]	matrix-vector multiplication |
// | matrix [m x k] dot matrix [k x n] | matrix [m x n] | matrix-matrix multiplication |
// The operation performs sum of products over the second dimension of x0 (or the first if it has rank 1) and
// the first dimension of x1.
// These are the "contracted" dimensions.
// The contracted dimensions of x0 and x1 must be of the same size.
// In practice, it can be used to perform dot products between vectors, vector/matrix multiplications or
// matrix/matrix multiplications.
// The op is created on the same XlaBuilder as used for x0 and x1.
func (b *Builder) Dot(lhsOp, rhsOp backends.Op) (backends.Op, error) {
	inputs, err := b.checkOps(backends.OpTypeDot.String(), lhsOp, rhsOp)
	if err != nil {
		return nil, err
	}
	lhs, rhs := inputs[0], inputs[1]
	var output backends.Op
	if lhs.shape.Rank() == 1 && rhs.shape.Rank() == 1 {
		// Contracting both vectors.
		output, err = b.DotGeneral(lhs, []int{0}, []int{}, rhs, []int{0}, []int{})
	} else if lhs.shape.Rank() == 2 && rhs.shape.Rank() == 1 {
		// Contract rhs vector.
		output, err = b.DotGeneral(lhs, []int{1}, []int{}, rhs, []int{0}, []int{})
	} else if lhs.shape.Rank() == 2 && rhs.shape.Rank() == 2 {
		// Traditional matrix multiplication:
		output, err = b.DotGeneral(lhs, []int{1}, []int{}, rhs, []int{0}, []int{})
	} else {
		return nil, errors.Errorf("Dot operands have invalid ranks: lhs=%v, rhs=%v", lhs.shape, rhs.shape)
	}
	if err != nil {
		return nil, errors.WithMessagef(err, "while building op Dot()")
	}
	return output, nil
}
