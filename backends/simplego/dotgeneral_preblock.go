package simplego

import (
	"slices"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/xsync"
)

// This file contains the pre-blocking infrastructure for DotGeneral.
// Pre-blocking converts tensors (like weight matrices) to blocked format once at graph build time,
// avoiding the runtime blocking cost on every execution.

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

// shouldPreBlock determines if a tensor should be pre-blocked for DotGeneral.
//
// Pre-blocking is beneficial when:
// - The tensor is a constant or parameter (not computed dynamically)
// - The tensor is large enough to benefit from blocking
// - The dtype is supported for blocking
//
// Parameters:
//   - node: the input node to potentially block
//   - crossSize, contractingSize: normalized sizes for blocking calculation
func shouldPreBlock(node *Node, crossSize, contractingSize int) bool {
	dtype := node.shape.DType

	// Check dtype is supported for blocking
	switch dtype {
	case dtypes.Float32, dtypes.Float64,
		dtypes.Float16, dtypes.BFloat16,
		dtypes.Int8, dtypes.Uint8,
		dtypes.Int16, dtypes.Int32, dtypes.Int64,
		dtypes.Uint16, dtypes.Uint32, dtypes.Uint64:
		// dtype supported, continue to size check
	default:
		return false
	}

	// Must be large enough to benefit from blocking
	blockDim := 1 << DotGeneralTargetBlockLog2Dim[dtype]
	if crossSize < blockDim || contractingSize < blockDim {
		return false
	}

	// Only pre-block constants or parameters (not computed values)
	// This ensures we only pay the blocking cost once
	switch node.opType {
	case backends.OpTypeParameter, backends.OpTypeConstant:
		return true
	default:
		return false
	}
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

// execDotGeneralWithPreBlocked executes DotGeneral when one or both inputs are pre-blocked.
// This function handles the case where inputs may be in blocked format already.
//
// Parameters:
//   - lhs, rhs: input buffers (may be in blocked or flat format)
//   - lhsBlockData, rhsBlockData: pre-blocking metadata (nil if that operand is not pre-blocked)
//   - params: DotGeneral parameters
//   - output: output buffer in flat format
func execDotGeneralWithPreBlocked(backend *Backend, lhs, rhs *Buffer, lhsBlockData, rhsBlockData *blockForDotGeneralData, params *dotGeneralNodeData, output *Buffer) error {
	dtype := lhs.shape.DType

	// Determine block dimension from pre-blocked data or compute from dtype
	var blkLog2Dim int
	if lhsBlockData != nil {
		blkLog2Dim = lhsBlockData.blockLog2Dim
	} else if rhsBlockData != nil {
		blkLog2Dim = rhsBlockData.blockLog2Dim
	} else {
		blkLog2Dim = DotGeneralTargetBlockLog2Dim[dtype]
	}
	blockDim := 1 << blkLog2Dim

	// Get or create blocked LHS
	var lhsBlocks *Buffer
	var freeLHS bool
	if lhsBlockData != nil {
		// Already pre-blocked
		lhsBlocks = lhs
	} else {
		// Block at runtime
		lhsBlocks = backend.getBuffer(dtype, params.lhsBlockedShape.Size())
		lhsBlocks.shape = params.lhsBlockedShape
		lhsBlocks.Zeros()
		copyFlatToBlock := dotGeneralFlatToBlockDTypeMap.Get(dtype).(func(source, blkOutput *Buffer, contractingAxes, batchAxes []int, batchSize, crossSize, contractingSize, blkLog2Dim int))
		copyFlatToBlock(lhs, lhsBlocks, params.lhsContractingAxes, params.lhsBatchAxes, params.batchSize, params.lhsCrossSize, params.contractingSize, blkLog2Dim)
		freeLHS = true
	}

	// Get or create blocked RHS
	var rhsBlocks *Buffer
	var freeRHS bool
	if rhsBlockData != nil {
		// Already pre-blocked
		rhsBlocks = rhs
	} else {
		// Block at runtime
		rhsBlocks = backend.getBuffer(dtype, params.rhsBlockedShape.Size())
		rhsBlocks.shape = params.rhsBlockedShape
		rhsBlocks.Zeros()
		copyFlatToBlock := dotGeneralFlatToBlockDTypeMap.Get(dtype).(func(source, blkOutput *Buffer, contractingAxes, batchAxes []int, batchSize, crossSize, contractingSize, blkLog2Dim int))
		copyFlatToBlock(rhs, rhsBlocks, params.rhsContractingAxes, params.rhsBatchAxes, params.batchSize, params.rhsCrossSize, params.contractingSize, blkLog2Dim)
		freeRHS = true
	}

	// Allocate output buffer in blocked format
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

	// Set block counts
	recursive.lhsCrossBlocks = lhsBlocks.shape.Dimensions[1]
	recursive.rhsCrossBlocks = rhsBlocks.shape.Dimensions[1]
	recursive.contractBlocks = lhsBlocks.shape.Dimensions[2]

	// Determine if RHS has batch dimension
	var rhsHasBatch bool
	if rhsBlockData != nil {
		rhsHasBatch = rhsBlockData.batchSize > 1
	} else {
		rhsHasBatch = len(params.rhsBatchAxes) > 0 && params.batchSize > 1
	}

	// Decide on intra-example parallelism
	maxParallelism := backend.workers.MaxParallelism()
	recursive.maxDepthParallelization = -1
	if backend.workers.IsEnabled() {
		if backend.workers.IsUnlimited() {
			recursive.maxDepthParallelization = 8
		} else {
			recursive.maxDepthParallelization = log2int(maxParallelism) + 1
		}
	}

	// Decide on using parallelism across the batch
	useBatchParallelism := backend.workers.IsEnabled()
	batchSplitSize := 1
	if useBatchParallelism && !backend.workers.IsUnlimited() {
		batchSplitSize = (params.batchSize + maxParallelism - 1) / maxParallelism
	}

	// Loop over examples in the batch
	wg := xsync.NewDynamicWaitGroup()
	for outerBatchIdx := 0; outerBatchIdx < params.batchSize; outerBatchIdx += batchSplitSize {
		wg.Add(1)
		batchSplitFn := func() {
			for innerBatchIdx := outerBatchIdx; innerBatchIdx < min(outerBatchIdx+batchSplitSize, params.batchSize); innerBatchIdx++ {
				var batchRecursive dotGeneralRecursiveData
				batchRecursive = recursive
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

	// Free only the buffers we allocated at runtime (not pre-blocked graph buffers)
	if freeLHS {
		backend.putBuffer(lhsBlocks)
	}
	if freeRHS {
		backend.putBuffer(rhsBlocks)
	}

	// Copy output from blocked to flat format
	finalOutputDType := output.shape.DType
	copyOutputBlockToFlat := dotGeneralOutputBlockToFlatDTypeMap.Get(finalOutputDType).(func(blockedSource, output *Buffer))
	copyOutputBlockToFlat(outputBlocks, output)
	backend.putBuffer(outputBlocks)

	return nil
}
