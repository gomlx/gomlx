package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// blockForDotGeneralData holds parameters for the BlockForDotGeneral operation.
// This operation pre-blocks a 2D weight tensor for efficient DotGeneral execution.
type blockForDotGeneralData struct {
	// blockLog2Dim is log2 of the block dimension
	blockLog2Dim int

	// blockedShape is the output shape after blocking
	// Format: [1, crossBlocks, contractBlocks, blockDim, blockDim]
	blockedShape shapes.Shape

	// originalK and originalN are the dimensions of the original [K, N] weight matrix
	originalK, originalN int
}

func init() {
	setNodeExecutor(backends.OpTypeBlockForDotGeneral, priorityGeneric, execBlockForDotGeneral)
}

// shouldPreBlockRHS determines if the RHS (weights) should be pre-blocked for DotGeneral.
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

	// Check dtype is supported for blocking
	dtype := rhs.shape.DType
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

	// Check minimum size threshold: matrix should be large enough to benefit from blocking.
	// Require at least one full block in each dimension to justify the blocking overhead.
	K := rhs.shape.Dimensions[0]
	N := rhs.shape.Dimensions[1]
	blockDim := 1 << DotGeneralTargetBlockLog2Dim[dtype]
	if K < blockDim || N < blockDim {
		return false
	}

	return true
}

// getOrCreateBlockedInput returns a BlockForDotGeneral node for the given RHS input.
// If a blocked version already exists (de-duplication), it returns the existing node.
// Otherwise, it creates a new BlockForDotGeneral node.
func (b *Builder) getOrCreateBlockedInput(rhs *Node) *Node {
	// Check if we already have a blocked version of this input
	if blocked, ok := b.blockedForDotGeneral[rhs]; ok {
		return blocked
	}

	// Create the blocking parameters
	dtype := rhs.shape.DType
	K := rhs.shape.Dimensions[0]
	N := rhs.shape.Dimensions[1]

	blockLog2Dim := DotGeneralTargetBlockLog2Dim[dtype]
	blockedShape := dgCreateBlockedShape(dtype, 1, N, K, blockLog2Dim)

	// Create the BlockForDotGeneral node
	blocked := b.newNode(backends.OpTypeBlockForDotGeneral, blockedShape, rhs)
	blocked.data = &blockForDotGeneralData{
		blockLog2Dim: blockLog2Dim,
		blockedShape: blockedShape,
		originalK:    K,
		originalN:    N,
	}

	// Cache for de-duplication
	b.blockedForDotGeneral[rhs] = blocked
	return blocked
}

// execBlockForDotGeneral executes the pre-blocking operation.
// It takes a 2D weight tensor [K, N] and converts it to blocked format
// for efficient DotGeneral execution.
func execBlockForDotGeneral(backend *Backend, node *Node, inputs []*Buffer, _ []bool) (*Buffer, error) {
	rhs := inputs[0]
	data := node.data.(*blockForDotGeneralData)

	dtype := rhs.shape.DType
	blockDim := 1 << data.blockLog2Dim

	// Allocate output buffer for blocked data
	output := backend.getBuffer(dtype, data.blockedShape.Size())
	output.shape = data.blockedShape
	output.Zeros()

	// Copy data from flat to blocked format using the existing function
	copyWeightToBlocked(rhs, output.flat, data.originalK, data.originalN, blockDim, data.blockLog2Dim)

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
