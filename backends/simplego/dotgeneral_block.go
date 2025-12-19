package simplego

import (
	"slices"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

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

	// Axes from the original tensor shape (needed for copying flat → blocked)
	contractingAxes []int
	batchAxes       []int
}

// Equal implements NodeDataComparable for de-duplication.
func (d *blockForDotGeneralData) Equal(other NodeDataComparable) bool {
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

// Compile-time check that blockForDotGeneralData implements NodeDataComparable.
var _ NodeDataComparable = (*blockForDotGeneralData)(nil)

func init() {
	setNodeExecutor(backends.OpTypeBlockForDotGeneral, priorityGeneric, execBlockForDotGeneral)
}

// shouldPreBlock determines if a tensor should be pre-blocked for DotGeneral.
// Unlike shouldPreBlockRHS, this works for both LHS and RHS with any normalized shape.
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

	// Don't pre-block small matrices where DirectPath or SmallNormalized would be faster.
	// Pre-blocking forces us through the blocked path which has more overhead.
	// For Float32 with small contracting dimensions, DirectPath is faster.
	if dtype == dtypes.Float32 && contractingSize <= DirectPathMaxContractingSize {
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

// shouldPreBlockRHS determines if the RHS (weights) should be pre-blocked for DotGeneral.
// This is a specialized version for backward compatibility with 2D weight matrices.
//
// Pre-blocking is beneficial when:
// - RHS is a 2D tensor [K, N] (standard weight layout)
// - RHS has no batch dimensions (weights are shared across batch)
// - Single contracting axis on axis 0 for RHS
// - The dtype is supported for blocking
// - The matrix is large enough to benefit from blocking (at least one full block in each dimension)
func shouldPreBlockRHS(rhs *Node, lhsContractingAxes, rhsContractingAxes, rhsBatchAxes []int) bool {
	// RHS (weights) must be 2D: [K, N]
	if rhs.shape.Rank() != 2 {
		return false
	}

	// RHS must have no batch dimensions (weights are shared across batch)
	if len(rhsBatchAxes) != 0 {
		return false
	}

	// Single contracting axis for RHS
	if len(rhsContractingAxes) != 1 {
		return false
	}

	// RHS contracts on axis 0 (standard [K, N] weight layout)
	if rhsContractingAxes[0] != 0 {
		return false
	}

	// LHS must also have single contracting axis
	if len(lhsContractingAxes) != 1 {
		return false
	}

	// Use the general shouldPreBlock with derived cross/contracting sizes
	K := rhs.shape.Dimensions[0] // contracting size
	N := rhs.shape.Dimensions[1] // cross size
	return shouldPreBlock(rhs, N, K)
}

// getOrCreateBlockedInputGeneral returns a BlockForDotGeneral node for the given input tensor.
// If a blocked version already exists (de-duplication), it returns the existing node.
// Otherwise, it creates a new BlockForDotGeneral node.
//
// This is the generalized version that works for both LHS and RHS operands with any shape.
//
// Parameters:
//   - input: the node to block
//   - contractingAxes, batchAxes: axes from the original tensor shape
//   - batchSize, crossSize, contractingSize: normalized sizes
func (b *Builder) getOrCreateBlockedInputGeneral(input *Node,
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

	// Try to find existing equivalent node via de-duplication
	if existing := b.findDuplicateNode(backends.OpTypeBlockForDotGeneral, []*Node{input}, data); existing != nil {
		return existing
	}

	// Create new BlockForDotGeneral node
	blocked := b.newNode(backends.OpTypeBlockForDotGeneral, blockedShape, input)
	blocked.data = data

	// Register for de-duplication
	b.registerForDeduplication(blocked)

	return blocked
}

// getOrCreateBlockedInput returns a BlockForDotGeneral node for the given 2D RHS input.
// This is kept for backward compatibility with the existing 2D weight blocking pattern.
// If a blocked version already exists (de-duplication), it returns the existing node.
// Otherwise, it creates a new BlockForDotGeneral node.
func (b *Builder) getOrCreateBlockedInput(rhs *Node) *Node {
	// For 2D RHS [K, N], the contracting axis is 0 and there are no batch axes
	K := rhs.shape.Dimensions[0] // contracting size
	N := rhs.shape.Dimensions[1] // cross size

	// Use the generalized version with the appropriate parameters
	blocked := b.getOrCreateBlockedInputGeneral(rhs, []int{0}, []int{}, 1, N, K)

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

// execDotGeneralWithGraphBlockedRHS executes DotGeneral when RHS was pre-blocked at graph build time.
// The rhsBlocked buffer is already in blocked format [1, crossBlocks, contractBlocks, blockDim, blockDim].
// This function only blocks LHS (activations) and performs the blocked matrix multiplication.
func execDotGeneralWithGraphBlockedRHS(backend *Backend, lhs, rhsBlocked *Buffer, blockData *blockForDotGeneralData, params *dotGeneralNodeData, output *Buffer) error {
	dtype := lhs.shape.DType
	blkLog2Dim := blockData.blockLog2Dim
	blockDim := 1 << blkLog2Dim

	// Block the LHS (activations)
	lhsBlocks := backend.getBuffer(dtype, params.lhsBlockedShape.Size())
	lhsBlocks.shape = params.lhsBlockedShape
	lhsBlocks.Zeros()

	// Copy LHS to blocked format
	copyFlatToBlock := dotGeneralFlatToBlockDTypeMap.Get(dtype).(func(source, blkOutput *Buffer, contractingAxes, batchAxes []int, batchSize, crossSize, contractingSize, blkLog2Dim int))
	copyFlatToBlock(lhs, lhsBlocks, params.lhsContractingAxes, params.lhsBatchAxes, params.batchSize, params.lhsCrossSize, params.contractingSize, blkLog2Dim)

	// Get output blocked buffer
	// Use params.outputBlockedShape.DType which is the accumulator type (Float32 for FP16/BF16, Int32 for Int8/Uint8)
	accumulatorDType := params.outputBlockedShape.DType
	outputBlocks := backend.getBuffer(accumulatorDType, params.outputBlockedShape.Size())
	outputBlocks.shape = params.outputBlockedShape
	outputBlocks.Zeros()

	// Set up base recursive data for kernel execution
	var recursive dotGeneralRecursiveData
	recursive.backend = backend

	// Get the matrix multiplication kernel for a block
	kernelBuilder := dotGeneralKernelDTypeMap.Get(dtype).(func(lhs, rhs, output *Buffer, blockDim int) kernelFuncType)
	recursive.kernelFn = kernelBuilder(lhsBlocks, rhsBlocked, outputBlocks, blockDim)

	// Set block counts
	recursive.lhsCrossBlocks = lhsBlocks.shape.Dimensions[1]
	recursive.rhsCrossBlocks = rhsBlocked.shape.Dimensions[1]
	recursive.contractBlocks = lhsBlocks.shape.Dimensions[2]

	// Execute the batch loop with parallelism (RHS is shared, no batch dimension)
	runDotGeneralBatchLoop(backend, &recursive, params.batchSize, false)

	// Free the LHS block buffer (RHS is already in graph buffer, don't free it)
	backend.putBuffer(lhsBlocks)

	// Copy output from blocked to flat format
	// Use final output dtype (e.g., Float16) to get correct conversion function
	finalOutputDType := output.shape.DType
	copyOutputBlockToFlat := dotGeneralOutputBlockToFlatDTypeMap.Get(finalOutputDType).(func(blockedSource, output *Buffer))
	copyOutputBlockToFlat(outputBlocks, output)
	backend.putBuffer(outputBlocks)

	return nil
}

// execDotGeneralWithGraphBlockedLHS executes DotGeneral when LHS was pre-blocked at graph build time.
// The lhsBlocked buffer is already in blocked format [batchSize, crossBlocks, contractBlocks, blockDim, blockDim].
// This function only blocks RHS and performs the blocked matrix multiplication.
func execDotGeneralWithGraphBlockedLHS(backend *Backend, lhsBlocked, rhs *Buffer, lhsBlockData *blockForDotGeneralData, params *dotGeneralNodeData, output *Buffer) error {
	dtype := rhs.shape.DType
	blkLog2Dim := lhsBlockData.blockLog2Dim
	blockDim := 1 << blkLog2Dim

	// Block the RHS
	rhsBlocks := backend.getBuffer(dtype, params.rhsBlockedShape.Size())
	rhsBlocks.shape = params.rhsBlockedShape
	rhsBlocks.Zeros()

	// Copy RHS to blocked format
	copyFlatToBlock := dotGeneralFlatToBlockDTypeMap.Get(dtype).(func(source, blkOutput *Buffer, contractingAxes, batchAxes []int, batchSize, crossSize, contractingSize, blkLog2Dim int))
	copyFlatToBlock(rhs, rhsBlocks, params.rhsContractingAxes, params.rhsBatchAxes, params.batchSize, params.rhsCrossSize, params.contractingSize, blkLog2Dim)

	// Get output blocked buffer
	// Use params.outputBlockedShape.DType which is the accumulator type (Float32 for FP16/BF16, Int32 for Int8/Uint8)
	accumulatorDType := params.outputBlockedShape.DType
	outputBlocks := backend.getBuffer(accumulatorDType, params.outputBlockedShape.Size())
	outputBlocks.shape = params.outputBlockedShape
	outputBlocks.Zeros()

	// Set up base recursive data for kernel execution
	var recursive dotGeneralRecursiveData
	recursive.backend = backend

	// Get the matrix multiplication kernel for a block
	kernelBuilder := dotGeneralKernelDTypeMap.Get(dtype).(func(lhs, rhs, output *Buffer, blockDim int) kernelFuncType)
	recursive.kernelFn = kernelBuilder(lhsBlocked, rhsBlocks, outputBlocks, blockDim)

	// Set block counts
	recursive.lhsCrossBlocks = lhsBlocked.shape.Dimensions[1]
	recursive.rhsCrossBlocks = rhsBlocks.shape.Dimensions[1]
	recursive.contractBlocks = lhsBlocked.shape.Dimensions[2]

	// Determine if RHS has batch dimension
	rhsHasBatch := len(params.rhsBatchAxes) > 0

	// Execute the batch loop with parallelism
	runDotGeneralBatchLoop(backend, &recursive, params.batchSize, rhsHasBatch)

	// Free the RHS block buffer (LHS is already in graph buffer, don't free it)
	backend.putBuffer(rhsBlocks)

	// Copy output from blocked to flat format
	// Use final output dtype (e.g., Float16) to get correct conversion function
	finalOutputDType := output.shape.DType
	copyOutputBlockToFlat := dotGeneralOutputBlockToFlatDTypeMap.Get(finalOutputDType).(func(blockedSource, output *Buffer))
	copyOutputBlockToFlat(outputBlocks, output)
	backend.putBuffer(outputBlocks)

	return nil
}

// execDotGeneralWithBothBlocked executes DotGeneral when both LHS and RHS were pre-blocked at graph build time.
// Both buffers are already in blocked format [batchSize, crossBlocks, contractBlocks, blockDim, blockDim].
// This is the most efficient path as no runtime blocking is needed.
func execDotGeneralWithBothBlocked(backend *Backend, lhsBlocked, rhsBlocked *Buffer, lhsBlockData, rhsBlockData *blockForDotGeneralData, params *dotGeneralNodeData, output *Buffer) error {
	dtype := lhsBlocked.shape.DType
	blkLog2Dim := lhsBlockData.blockLog2Dim
	blockDim := 1 << blkLog2Dim

	// Get output blocked buffer
	// Use params.outputBlockedShape.DType which is the accumulator type (Float32 for FP16/BF16, Int32 for Int8/Uint8)
	accumulatorDType := params.outputBlockedShape.DType
	outputBlocks := backend.getBuffer(accumulatorDType, params.outputBlockedShape.Size())
	outputBlocks.shape = params.outputBlockedShape
	outputBlocks.Zeros()

	// Set up base recursive data for kernel execution
	var recursive dotGeneralRecursiveData
	recursive.backend = backend

	// Get the matrix multiplication kernel for a block
	kernelBuilder := dotGeneralKernelDTypeMap.Get(dtype).(func(lhs, rhs, output *Buffer, blockDim int) kernelFuncType)
	recursive.kernelFn = kernelBuilder(lhsBlocked, rhsBlocked, outputBlocks, blockDim)

	// Set block counts
	recursive.lhsCrossBlocks = lhsBlocked.shape.Dimensions[1]
	recursive.rhsCrossBlocks = rhsBlocked.shape.Dimensions[1]
	recursive.contractBlocks = lhsBlocked.shape.Dimensions[2]

	// Determine if RHS has batch dimension (both can have batch, or RHS is shared)
	rhsHasBatch := rhsBlockData.batchSize > 1

	// Execute the batch loop with parallelism
	runDotGeneralBatchLoop(backend, &recursive, params.batchSize, rhsHasBatch)

	// No block buffers to free - both are in graph buffers

	// Copy output from blocked to flat format
	// Use final output dtype (e.g., Float16) to get correct conversion function
	finalOutputDType := output.shape.DType
	copyOutputBlockToFlat := dotGeneralOutputBlockToFlatDTypeMap.Get(finalOutputDType).(func(blockedSource, output *Buffer))
	copyOutputBlockToFlat(outputBlocks, output)
	backend.putBuffer(outputBlocks)

	return nil
}
