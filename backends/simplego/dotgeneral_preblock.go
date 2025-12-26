package simplego

import (
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/x448/float16"
)

// PreBlockedTensor holds a tensor that has been pre-converted to blocked format
// for efficient DotGeneral execution. This avoids the runtime cost of dgCopyFlatToBlockShape.
type PreBlockedTensor struct {
	// OriginalShape is the original shape before blocking (e.g., [K, N] for a 2D tensor)
	OriginalShape shapes.Shape

	// BlockedShape is the blocked shape (e.g., [1, crossBlocks, contractBlocks, blockDim, blockDim])
	BlockedShape shapes.Shape

	// BlockedData holds the pre-blocked data
	BlockedData any

	// BlockLog2Dim is the log2 of the block dimension used
	BlockLog2Dim int

	// IsTransposed indicates if this tensor should be used as the RHS (transposed) in matmul
	// For tensor [K, N], if IsTransposed=true, it's blocked as [1, N/blk, K/blk, blk, blk]
	// to be compatible with the standard matmul pattern [M, K] × [K, N]
	IsTransposed bool
}

// PreBlockTensorForDotGeneral pre-blocks a 2D tensor for efficient DotGeneral execution.
// The tensor is assumed to be used as the RHS in matmul: [M, K] × [K, N] → [M, N]
// where the tensor has shape [K, N].
//
// This pre-computes the blocked format so that during DotGeneral execution, we can skip
// the dgCopyFlatToBlockShape transformation entirely.
func PreBlockTensorForDotGeneral(buf *Buffer) *PreBlockedTensor {
	if buf.shape.Rank() != 2 {
		return nil // Only support 2D tensors
	}

	dtype := buf.shape.DType
	K := buf.shape.Dimensions[0]
	N := buf.shape.Dimensions[1]

	blockLog2Dim := DotGeneralTargetBlockLog2Dim[dtype]
	blockDim := 1 << blockLog2Dim

	// For RHS in matmul: crossSize = N, contractingSize = K
	// The blocked shape is [batchSize=1, crossBlocks, contractBlocks, blockDim, blockDim]
	blockedShape := dgCreateBlockedShape(dtype, 1, N, K, blockLog2Dim)

	// Allocate blocked data
	blockedData := allocateBlockedData(dtype, blockedShape.Size())

	// Copy data from flat to blocked format
	// For RHS tensor [K, N], we need to reorder so that:
	// - crossSize (N) becomes the outer iteration
	// - contractingSize (K) becomes the inner iteration
	copyTensorToBlocked(buf, blockedData, K, N, blockDim, blockLog2Dim)

	return &PreBlockedTensor{
		OriginalShape: buf.shape,
		BlockedShape:  blockedShape,
		BlockedData:   blockedData,
		BlockLog2Dim:  blockLog2Dim,
		IsTransposed:  false,
	}
}

// allocateBlockedData allocates the appropriate slice type for the blocked data
func allocateBlockedData(dtype dtypes.DType, size int) any {
	switch dtype {
	case dtypes.Float32:
		return make([]float32, size)
	case dtypes.Float64:
		return make([]float64, size)
	case dtypes.Float16:
		return make([]float16.Float16, size)
	case dtypes.BFloat16:
		return make([]bfloat16.BFloat16, size)
	case dtypes.Int8:
		return make([]int8, size)
	case dtypes.Int16:
		return make([]int16, size)
	case dtypes.Int32:
		return make([]int32, size)
	case dtypes.Int64:
		return make([]int64, size)
	case dtypes.Uint8:
		return make([]uint8, size)
	case dtypes.Uint16:
		return make([]uint16, size)
	case dtypes.Uint32:
		return make([]uint32, size)
	case dtypes.Uint64:
		return make([]uint64, size)
	default:
		return nil
	}
}

// copyTensorToBlocked copies a 2D tensor [K, N] to blocked format for RHS usage.
// The blocked format is [1, crossBlocks, contractBlocks, blockDim, blockDim]
// where crossSize = N and contractingSize = K.
func copyTensorToBlocked(buf *Buffer, blockedData any, K, N, blockDim, blockLog2Dim int) {
	dtype := buf.shape.DType

	crossBlocks := (N + blockDim - 1) / blockDim
	contractBlocks := (K + blockDim - 1) / blockDim

	switch dtype {
	case dtypes.Float32:
		copyTensorToBlockedTyped(buf.flat.([]float32), blockedData.([]float32), K, N, blockDim, crossBlocks, contractBlocks)
	case dtypes.Float64:
		copyTensorToBlockedTyped(buf.flat.([]float64), blockedData.([]float64), K, N, blockDim, crossBlocks, contractBlocks)
	case dtypes.Float16:
		copyTensorToBlockedTyped(buf.flat.([]float16.Float16), blockedData.([]float16.Float16), K, N, blockDim, crossBlocks, contractBlocks)
	case dtypes.BFloat16:
		copyTensorToBlockedTyped(buf.flat.([]bfloat16.BFloat16), blockedData.([]bfloat16.BFloat16), K, N, blockDim, crossBlocks, contractBlocks)
	case dtypes.Int8:
		copyTensorToBlockedTyped(buf.flat.([]int8), blockedData.([]int8), K, N, blockDim, crossBlocks, contractBlocks)
	case dtypes.Int16:
		copyTensorToBlockedTyped(buf.flat.([]int16), blockedData.([]int16), K, N, blockDim, crossBlocks, contractBlocks)
	case dtypes.Int32:
		copyTensorToBlockedTyped(buf.flat.([]int32), blockedData.([]int32), K, N, blockDim, crossBlocks, contractBlocks)
	case dtypes.Int64:
		copyTensorToBlockedTyped(buf.flat.([]int64), blockedData.([]int64), K, N, blockDim, crossBlocks, contractBlocks)
	case dtypes.Uint8:
		copyTensorToBlockedTyped(buf.flat.([]uint8), blockedData.([]uint8), K, N, blockDim, crossBlocks, contractBlocks)
	case dtypes.Uint16:
		copyTensorToBlockedTyped(buf.flat.([]uint16), blockedData.([]uint16), K, N, blockDim, crossBlocks, contractBlocks)
	case dtypes.Uint32:
		copyTensorToBlockedTyped(buf.flat.([]uint32), blockedData.([]uint32), K, N, blockDim, crossBlocks, contractBlocks)
	case dtypes.Uint64:
		copyTensorToBlockedTyped(buf.flat.([]uint64), blockedData.([]uint64), K, N, blockDim, crossBlocks, contractBlocks)
	}
}

// copyTensorToBlockedTyped is the generic implementation for copying tensor data to blocked format.
// Input: flat [K, N] - row-major, so element [k, n] is at index k*N + n
// Output: blocked [1, crossBlocks, contractBlocks, blockDim, blockDim]
//
// The kernel expects blocks where:
// - First dimension (rows) = cross dimension (N)
// - Second dimension (cols) = contracting dimension (K)
// So the block layout is [crossLocal, contractLocal] = [n_local, k_local]
//
// This means we need to transpose from flat [K, N] to block [N_local, K_local]:
// - flat[k, n] → block[n_local, k_local]
func copyTensorToBlockedTyped[T any](flat []T, blocked []T, K, N, blockDim, crossBlocks, contractBlocks int) {
	blockSize := blockDim * blockDim

	for crossBlockIdx := 0; crossBlockIdx < crossBlocks; crossBlockIdx++ {
		nStart := crossBlockIdx * blockDim
		nEnd := min(nStart+blockDim, N)

		for contractBlockIdx := 0; contractBlockIdx < contractBlocks; contractBlockIdx++ {
			kStart := contractBlockIdx * blockDim
			kEnd := min(kStart+blockDim, K)

			// Calculate the start of this block in the blocked output
			// Shape: [1, crossBlocks, contractBlocks, blockDim, blockDim]
			blockStartIdx := crossBlockIdx*(contractBlocks*blockSize) + contractBlockIdx*blockSize

			// Copy data into the block with transpose
			// Block layout: [crossLocal, contractLocal] = [n_local, k_local]
			// We iterate over the flat input and place values in transposed positions
			for localN := 0; localN < blockDim; localN++ {
				globalN := nStart + localN
				for localK := 0; localK < blockDim; localK++ {
					globalK := kStart + localK

					// Index in blocked output: row=localN (cross), col=localK (contract)
					blockedIdx := blockStartIdx + localN*blockDim + localK

					// Get value from flat input (or zero if out of bounds)
					if globalK < kEnd && globalN < nEnd {
						flatIdx := globalK*N + globalN
						blocked[blockedIdx] = flat[flatIdx]
					}
					// Note: blocked array should be zero-initialized, so padding is already zero
				}
			}
		}
	}
}

// GetPreBlockedBuffer returns a Buffer containing the pre-blocked data.
// This can be used directly in DotGeneral execution, skipping the blocking step.
func (pbt *PreBlockedTensor) GetPreBlockedBuffer() *Buffer {
	return &Buffer{
		shape: pbt.BlockedShape,
		flat:  pbt.BlockedData,
	}
}

// CanUsePreBlockedTensor checks if a pre-blocked tensor can be used for this matmul operation.
// Returns true if the tensor shape and blocking parameters match.
// Supports both unbatched [M, K] × [K, N] and batched [B, M, K] × [K, N] operations
// where RHS tensors are shared across the batch.
func CanUsePreBlockedTensor(pbt *PreBlockedTensor, rhsShape shapes.Shape, params *dotGeneralNodeData) bool {
	if pbt == nil {
		return false
	}

	// RHS must have no batch dimensions (tensor shared across batch)
	if len(params.rhsBatchAxes) != 0 {
		return false
	}

	// Single contracting axis for both LHS and RHS
	if len(params.lhsContractingAxes) != 1 || len(params.rhsContractingAxes) != 1 {
		return false
	}

	// RHS contracting must be axis 0 for standard [K, N] tensor layout
	if params.rhsContractingAxes[0] != 0 {
		return false
	}

	// Check shape matches
	if !pbt.OriginalShape.Equal(rhsShape) {
		return false
	}

	// Check block dimension matches
	if pbt.BlockLog2Dim != DotGeneralTargetBlockLog2Dim[rhsShape.DType] {
		return false
	}

	return true
}
