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

// execDotGeneralBlockedUnified executes DotGeneral using the blocked (cache-tiled) algorithm.
// This unified function handles all cases: both pre-blocked, one pre-blocked, or neither pre-blocked.
//
// Parameters:
//   - lhs, rhs: input buffers (may be in flat or blocked format depending on pre-blocking)
//   - lhsBlockData, rhsBlockData: pre-blocking metadata (nil if that operand is not pre-blocked)
//   - params: DotGeneral parameters
//   - output: output buffer in flat format
//
// The function will block any operand that isn't already pre-blocked, run the kernel, and free
// only the buffers that were allocated at runtime (not the graph-owned pre-blocked buffers).
func execDotGeneralBlockedUnified(backend *Backend, lhs, rhs *Buffer, lhsBlockData, rhsBlockData *blockForDotGeneralData, params *dotGeneralNodeData, output *Buffer) error {
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
	// Use params.outputBlockedShape.DType which is the accumulator type (Float32 for FP16/BF16, Int32 for Int8/Uint8)
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
	// - If RHS is pre-blocked, check the stored batchSize
	// - If RHS is not pre-blocked, check the params
	var rhsHasBatch bool
	if rhsBlockData != nil {
		rhsHasBatch = rhsBlockData.batchSize > 1
	} else {
		rhsHasBatch = len(params.rhsBatchAxes) > 0 && params.batchSize > 1
	}

	// Execute the batch loop with parallelism
	runDotGeneralBatchLoop(backend, &recursive, params.batchSize, rhsHasBatch)

	// Free only the buffers we allocated at runtime (not pre-blocked graph buffers)
	if freeLHS {
		backend.putBuffer(lhsBlocks)
	}
	if freeRHS {
		backend.putBuffer(rhsBlocks)
	}

	// Copy output from blocked to flat format
	// Use final output dtype (e.g., Float16) to get correct conversion function
	finalOutputDType := output.shape.DType
	copyOutputBlockToFlat := dotGeneralOutputBlockToFlatDTypeMap.Get(finalOutputDType).(func(blockedSource, output *Buffer))
	copyOutputBlockToFlat(outputBlocks, output)
	backend.putBuffer(outputBlocks)

	return nil
}

