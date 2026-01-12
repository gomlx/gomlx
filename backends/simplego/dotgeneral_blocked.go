// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"slices"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/xsync"
	"github.com/x448/float16"
)

// This file contains the implementation for the blocked (cache-tiled) DotGeneral algorithm.
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
	DotGeneralTargetBlockSize = 16 * 1024

	// DotGeneralTargetBlockLog2Dim is set per dtype, such that it is square and fits DotGeneralTargetBlockSize.
	// The block dim is 2^(DotGeneralTargetBlockLog2Dim[dtype]).
	DotGeneralTargetBlockLog2Dim [MaxDTypes]int

	// DotGeneralBlockedPathThreshold is the multiplier for determining when to use the blocked execution path.
	// When crossesSize (lhsCrossSize * rhsCrossSize) exceeds this multiplier times blockSize,
	// the blocked path is chosen over the normalized path.
	//
	// Empirically determined: below this threshold, the overhead of cache-tiled blocking
	// outweighs its benefits. Above this threshold, the blocked path's cache efficiency wins.
	DotGeneralBlockedPathThreshold = 16
)

func init() {
	// Initialize block dimensions for all numeric types that support DotGeneral.
	// This includes float types and integer types (used by quantized models).
	setDotGeneralTargetBlockSize(DotGeneralTargetBlockSize)
}

// setDotGeneralTargetBlockSize sets the target block size for DotGeneral.
func setDotGeneralTargetBlockSize(blockSize int) {
	DotGeneralTargetBlockSize = blockSize
	for _, dtype := range numericDTypes {
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
		log2Dim--
		// Ensure minimum block dimension of 8 (log2Dim >= 3) for the kernel's loop unrolling.
		if log2Dim < 3 {
			log2Dim = 3
		}
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

// ============================================================================
// Pre-blocking for DotGeneral
// ============================================================================

// blockForDotGeneralData holds parameters for the BlockForDotGeneral operation.
// This operation pre-blocks a tensor (LHS or RHS) for efficient DotGeneral execution.
// Works with any shape after normalization to [batchSize, crossSize, contractingSize].
type blockForDotGeneralData struct {
	// blockLog2Dim is log2 of the block dimension
	blockLog2Dim int

	// blockedShape is the output shape after blocking
	// Format: [batchSize, crossBlocks, contractBlocks, blockDim, blockDim]
	blockedShape shapes.Shape

	// Original tensor characteristics (after axis adjustment, before blocking)
	batchSize       int
	crossSize       int
	contractingSize int

	// Axes from the original tensor shape (needed for copying flat -> blocked)
	contractingAxes []int
	batchAxes       []int
}

// EqualNodeData implements nodeDataComparable for de-duplication.
func (d *blockForDotGeneralData) EqualNodeData(other nodeDataComparable) bool {
	o, ok := other.(*blockForDotGeneralData)
	if !ok {
		return false
	}
	return d.blockLog2Dim == o.blockLog2Dim &&
		d.blockedShape.Equal(o.blockedShape) &&
		d.batchSize == o.batchSize &&
		d.crossSize == o.crossSize &&
		d.contractingSize == o.contractingSize &&
		slices.Equal(d.contractingAxes, o.contractingAxes) &&
		slices.Equal(d.batchAxes, o.batchAxes)
}

// Compile-time check that blockForDotGeneralData implements nodeDataComparable.
var _ nodeDataComparable = (*blockForDotGeneralData)(nil)

func init() {
	setNodeExecutor(backends.OpTypeBlockForDotGeneral, priorityGeneric, execBlockForDotGeneral)
}

// blockForDotGeneral returns a BlockForDotGeneral node for the given input tensor.
// Uses de-duplication via getOrCreateNode to return an existing node if available.
//
// This is the generalized version that works for both LHS and RHS operands with any shape.
//
// Parameters:
//   - input: the node to block
//   - contractingAxes, batchAxes: axes from the original tensor shape
//   - batchSize, crossSize, contractingSize: normalized sizes
func (b *Builder) blockForDotGeneral(input *Node,
	contractingAxes, batchAxes []int,
	batchSize, crossSize, contractingSize int) *Node {

	dtype := input.shape.DType
	blockLog2Dim := DotGeneralTargetBlockLog2Dim[dtype]
	blockedShape := dgCreateBlockedShape(dtype, batchSize, crossSize, contractingSize, blockLog2Dim)

	data := &blockForDotGeneralData{
		blockLog2Dim:    blockLog2Dim,
		blockedShape:    blockedShape,
		batchSize:       batchSize,
		crossSize:       crossSize,
		contractingSize: contractingSize,
		contractingAxes: slices.Clone(contractingAxes),
		batchAxes:       slices.Clone(batchAxes),
	}

	blocked, _ := b.getOrCreateNode(backends.OpTypeBlockForDotGeneral, blockedShape, []*Node{input}, data)
	return blocked
}

// execBlockForDotGeneral executes the pre-blocking operation.
// It takes a tensor (any shape) and converts it to blocked format
// for efficient DotGeneral execution.
func execBlockForDotGeneral(backend *Backend, node *Node, inputs []*Buffer, _ []bool) (*Buffer, error) {
	input := inputs[0]
	data := node.data.(*blockForDotGeneralData)

	dtype := input.shape.DType

	// Allocate output buffer for blocked data
	output := backend.getBuffer(dtype, data.blockedShape.Size())
	output.shape = data.blockedShape
	output.Zeros()

	// Copy data from flat to blocked format using the generic copy function
	copyFlatToBlock := dotGeneralFlatToBlockDTypeMap.Get(dtype).(func(source, blkOutput *Buffer, contractingAxes, batchAxes []int, batchSize, crossSize, contractingSize, blkLog2Dim int))
	copyFlatToBlock(input, output, data.contractingAxes, data.batchAxes, data.batchSize, data.crossSize, data.contractingSize, data.blockLog2Dim)

	return output, nil
}

// Auto-generate alternate specialized versions of dgCopyOutputBlockToFlat
// (that can't easily be refactored into smaller functions due to latency penalities)
//go:generate go run ../../internal/cmd/alternates_generator -base=dotgeneral_blocked_alt_base.go -tags=bf16,f16

// ============================================================================
// Blocked DotGeneral Execution
// ============================================================================

// execDotGeneralBlocked executes DotGeneral using the blocked (cache-tiled) algorithm.
// Both inputs MUST be pre-blocked (coming from BlockForDotGeneral nodes).
// This is the main blocked execution path used when blockedPath is selected at build time.
//
// Parameters:
//   - lhs, rhs: input buffers in blocked format (from BlockForDotGeneral)
//   - lhsBlockData, rhsBlockData: pre-blocking metadata from the input nodes
//   - params: DotGeneral parameters
//   - output: output buffer in flat format
func execDotGeneralBlocked(backend *Backend, lhsBlocks, rhsBlocks *Buffer, hasBatch bool, params *dotGeneralNodeData, output *Buffer) error {
	dtype := lhsBlocks.shape.DType
	blockDim := 1 << DotGeneralTargetBlockLog2Dim[dtype]

	// Allocate output buffer in blocked format.
	// Use params.outputBlockedShape.DType which is the accumulator type (Float32 for FP16/BF16).
	accumulatorDType := params.outputBlockedShape.DType
	outputBlocks := backend.getBuffer(accumulatorDType, params.outputBlockedShape.Size())
	outputBlocks.shape = params.outputBlockedShape
	outputBlocks.Zeros()

	// Set up recursive data for kernel execution
	var recursive dotGeneralRecursiveData
	recursive.backend = backend

	// Get the matrix multiplication kernel for a block
	kernelBuilder := dotGeneralKernelDTypeMap.Get(dtype).(func(lhs, rhs, output *Buffer, blockDim int) kernelFuncType)
	recursive.kernelFn = kernelBuilder(lhsBlocks, rhsBlocks, outputBlocks, blockDim)

	// Set block counts from blocked buffer dimensions
	recursive.lhsCrossBlocks = lhsBlocks.shape.Dimensions[1]
	recursive.rhsCrossBlocks = rhsBlocks.shape.Dimensions[1]
	recursive.contractBlocks = lhsBlocks.shape.Dimensions[2]

	// Execute the batch loop with parallelism
	runDotGeneralBatchLoop(backend, &recursive, params.batchSize, hasBatch)

	// Copy output from blocked to flat format
	finalOutputDType := output.shape.DType
	copyOutputBlockToFlat := dotGeneralOutputBlockToFlatDTypeMap.Get(finalOutputDType).(func(blockedSource, output *Buffer))
	copyOutputBlockToFlat(outputBlocks, output)
	backend.putBuffer(outputBlocks)
	return nil
}

// ============================================================================
// Data Copy Functions (Flat <-> Blocked)
// ============================================================================

var dotGeneralFlatToBlockDTypeMap = NewDTypeMap("DotGeneralFlatToBlock")

// dgCopyFlatToBlockShape copies the data from the original (with a non-normalized shape, with the contracting axes
// and batch axes given) to blocked, whose shape is normalized to [batchSize, crossSize, contractingSize] and
// is organized in blocks (packages) of shape [1, blkDim, blkDim].
//
// blkOutput is assumed to have been created with a size that is multiple of blkDim for the cross and contracting axes.
//
// source shape: any combination of batch, cross or contracting dimensions.
// blkOutput shape: [batchSize, crossBlocks * blkDim, contractBlocks * blkDim]
func dgCopyFlatToBlockShape[T interface {
	PODNumericConstraints | bfloat16.BFloat16 | float16.Float16
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
		// fmt.Printf("\toutput%v (flat %d) = source%v (flat %d)\n", outputIdx, outputFlatIdx, sourceIdx, sourceFlatIdx)

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

func init() {
	dotGeneralOutputBlockToFlatDTypeMap.Register(dtypes.BFloat16, priorityTyped, dgCopyOutputBlockToFlatF32ToBF16)
	dotGeneralOutputBlockToFlatDTypeMap.Register(dtypes.Float16, priorityTyped, dgCopyOutputBlockToFlatF32ToF16)
}

// ============================================================================
// Batch Loop and Recursive Splitting
// ============================================================================

// runDotGeneralBatchLoop runs the batch loop for blocked DotGeneral execution.
// It handles parallelism across batch examples and within each example.
func runDotGeneralBatchLoop(backend *Backend, recursive *dotGeneralRecursiveData, batchSize int, rhsHasBatch bool) {
	// Decide on intra-example parallelism: up to which depth we should use a new worker.
	maxParallelism := backend.workers.MaxParallelism()
	recursive.maxDepthParallelization = -1 // Disable sub-batch parallelization.
	if backend.workers.IsEnabled() {
		if backend.workers.IsUnlimited() {
			recursive.maxDepthParallelization = 8 // At most 2^8 = 256 goroutines are spawned.
		} else {
			// Use log2 of parallelism to reduce goroutine overhead.
			recursive.maxDepthParallelization = log2int(maxParallelism)
		}
	}

	// Decide on using parallelism across the batch -- each example is started on a separate worker.
	useBatchParallelism := backend.workers.IsEnabled()
	batchSplitSize := 1
	if useBatchParallelism && !backend.workers.IsUnlimited() {
		batchSplitSize = (batchSize + maxParallelism - 1) / maxParallelism
	}

	// Loop over examples in the batch:
	wg := xsync.NewDynamicWaitGroup() // Control workers started.
	for outerBatchIdx := 0; outerBatchIdx < batchSize; outerBatchIdx += batchSplitSize {
		wg.Add(1)
		batchSplitFn := func() {
			for innerBatchIdx := outerBatchIdx; innerBatchIdx < min(outerBatchIdx+batchSplitSize, batchSize); innerBatchIdx++ {
				var batchRecursive dotGeneralRecursiveData
				batchRecursive = *recursive
				batchRecursive.lhsBatchOffset = innerBatchIdx * recursive.lhsCrossBlocks * recursive.contractBlocks
				if rhsHasBatch {
					batchRecursive.rhsBatchOffset = innerBatchIdx * recursive.rhsCrossBlocks * recursive.contractBlocks
				} else {
					batchRecursive.rhsBatchOffset = 0 // RHS is shared across all batches
				}
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
	switch maxLen {
	case lhsCrossLen:
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
	case rhsCrossLen:
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
	default:
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
	}
}

// ============================================================================
// Matrix Multiplication Kernels
// ============================================================================

var dotGeneralKernelDTypeMap = NewDTypeMap("DotGeneralKernel")

// kernelFuncType is a function that does a matrix mult of the lhs/rhs and adds it to the output buffer, given the indices of the square blocks.
// So output[outputIdx] += lhs[lhsIdx] * rhs[rhsIdx], a block at a time.
// The contracting axis is 1 for both, lhs and rhs.
type kernelFuncType func(lhsBlockIdx, rhsBlockIdx, outputBlockIdx int)

func init() {
	dotGeneralKernelDTypeMap.Register(dtypes.BFloat16, priorityTyped, buildDotGeneralKernelBFloat16)
	dotGeneralKernelDTypeMap.Register(dtypes.Float16, priorityTyped, buildDotGeneralKernelFloat16)
}
