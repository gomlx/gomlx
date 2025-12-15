package simplego

import (
	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// isStandardMatmul checks if the DotGeneral operation is a standard matrix multiplication
// that doesn't require any transposition or complex axis manipulation.
//
// Standard patterns that can skip normalization:
// 1. Matrix × Matrix: [M, K] × [K, N] → [M, N] (contracting on last axis of lhs, first of rhs)
// 2. Matrix × Vector: [M, K] × [K] → [M] (contracting on last axis of lhs, only axis of rhs)
// 3. Batched MatMul: [B, M, K] × [B, K, N] → [B, M, N] (batch on first axis)
//
// Returns true if we can use the fast path (no transpose needed).
func isStandardMatmul(lhsShape, rhsShape shapes.Shape, lhsContractingAxes, rhsContractingAxes, lhsBatchAxes, rhsBatchAxes []int) bool {
	lhsRank := lhsShape.Rank()
	rhsRank := rhsShape.Rank()

	// Check for standard matrix-matrix multiplication: [M, K] × [K, N]
	if lhsRank == 2 && rhsRank == 2 &&
		len(lhsContractingAxes) == 1 && len(rhsContractingAxes) == 1 &&
		len(lhsBatchAxes) == 0 && len(rhsBatchAxes) == 0 {
		// Contracting: lhs last axis (1) with rhs first axis (0)
		if lhsContractingAxes[0] == 1 && rhsContractingAxes[0] == 0 {
			return true
		}
	}

	// Check for matrix-vector multiplication: [M, K] × [K]
	if lhsRank == 2 && rhsRank == 1 &&
		len(lhsContractingAxes) == 1 && len(rhsContractingAxes) == 1 &&
		len(lhsBatchAxes) == 0 && len(rhsBatchAxes) == 0 {
		// Contracting: lhs last axis (1) with rhs only axis (0)
		if lhsContractingAxes[0] == 1 && rhsContractingAxes[0] == 0 {
			return true
		}
	}

	// Check for batched matrix multiplication: [B, M, K] × [B, K, N]
	if lhsRank == 3 && rhsRank == 3 &&
		len(lhsContractingAxes) == 1 && len(rhsContractingAxes) == 1 &&
		len(lhsBatchAxes) == 1 && len(rhsBatchAxes) == 1 {
		// Batch on first axis (0)
		if lhsBatchAxes[0] == 0 && rhsBatchAxes[0] == 0 {
			// Contracting: lhs axis 2 with rhs axis 1
			if lhsContractingAxes[0] == 2 && rhsContractingAxes[0] == 1 {
				return true
			}
		}
	}

	// Check for multi-batch matmul: [B1, B2, M, K] × [B1, B2, K, N]
	if lhsRank == 4 && rhsRank == 4 &&
		len(lhsContractingAxes) == 1 && len(rhsContractingAxes) == 1 &&
		len(lhsBatchAxes) == 2 && len(rhsBatchAxes) == 2 {
		// Batch on first two axes
		if lhsBatchAxes[0] == 0 && lhsBatchAxes[1] == 1 &&
			rhsBatchAxes[0] == 0 && rhsBatchAxes[1] == 1 {
			// Contracting: lhs axis 3 with rhs axis 2
			if lhsContractingAxes[0] == 3 && rhsContractingAxes[0] == 2 {
				return true
			}
		}
	}

	return false
}

// isMemoryContiguous checks if the tensor layout is already contiguous in memory
// for the given contracting pattern.
func isMemoryContiguous(shape shapes.Shape, contractingAxes, batchAxes []int) bool {
	rank := shape.Rank()
	if rank == 0 {
		return true
	}

	// Build expected order: batch axes first, then cross, then contracting
	expectedOrder := make([]int, 0, rank)
	expectedOrder = append(expectedOrder, batchAxes...)

	// Add cross axes (non-batch, non-contracting)
	isContracting := make(map[int]bool)
	isBatch := make(map[int]bool)
	for _, a := range contractingAxes {
		isContracting[a] = true
	}
	for _, a := range batchAxes {
		isBatch[a] = true
	}

	for i := 0; i < rank; i++ {
		if !isContracting[i] && !isBatch[i] {
			expectedOrder = append(expectedOrder, i)
		}
	}
	expectedOrder = append(expectedOrder, contractingAxes...)

	// Check if expected order is 0, 1, 2, ... (natural order)
	for i, axis := range expectedOrder {
		if axis != i {
			return false
		}
	}
	return true
}

// canUseFastPath determines if we can use the optimized fast path for this DotGeneral operation.
func canUseFastPath(lhs, rhs *Buffer, params *dotGeneralNodeData) bool {
	// Only support float32 fast path for now (most common)
	if lhs.shape.DType != dtypes.Float32 {
		return false
	}

	// Check if it's a standard matmul pattern
	if !isStandardMatmul(lhs.shape, rhs.shape,
		params.lhsContractingAxes, params.rhsContractingAxes,
		params.lhsBatchAxes, params.rhsBatchAxes) {
		return false
	}

	return true
}

// execDotGeneralFastPath executes a standard matrix multiplication without normalization.
// This is a significant optimization for the common case of A × B matrix multiplication.
// Returns true if fast path was used, false if caller should use standard path.
func execDotGeneralFastPath(backend *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, output *Buffer) bool {
	if !canUseFastPath(lhs, rhs, params) {
		return false
	}

	// Execute the optimized float32 path
	execDotGeneralFastPathFloat32(backend, lhs, rhs, params, output)
	return true
}

// execDotGeneralFastPathFloat32 is the fast path for float32 matrix multiplication.
// It directly operates on the input data without transposing to normalized form.
//
// Memory layout for row-major tensors:
// - LHS [M, K]: element [m, k] is at index m*K + k (rows are contiguous)
// - RHS [K, N]: element [k, n] is at index k*N + n (rows are contiguous)
// - Output [M, N]: element [m, n] is at index m*N + n
//
// For the dot product of row m with column n:
//   sum over k: LHS[m,k] * RHS[k,n] = sum over k: lhs[m*K+k] * rhs[k*N+n]
//
// Note: Column n in RHS has stride N between elements (not contiguous),
// so we cannot use the Group4 NEON path which requires contiguous columns.
// We use the standard scalar loop with explicit strided access.
func execDotGeneralFastPathFloat32(backend *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, output *Buffer) {
	lhsFlat := lhs.flat.([]float32)
	rhsFlat := rhs.flat.([]float32)
	outputFlat := output.flat.([]float32)

	batchSize := params.batchSize
	lhsCrossSize := params.lhsCrossSize      // M
	rhsCrossSize := params.rhsCrossSize      // N
	contractingSize := params.contractingSize // K

	lhsBatchStride := lhsCrossSize * contractingSize  // M * K
	rhsBatchStride := rhsCrossSize * contractingSize  // N * K (but actually K * N for [K,N])
	outputBatchStride := lhsCrossSize * rhsCrossSize  // M * N

	// For row-major RHS [K, N], the stride between elements in the same column is N
	rhsColStride := rhsCrossSize // N

	for batchIdx := 0; batchIdx < batchSize; batchIdx++ {
		lhsBaseIdx := batchIdx * lhsBatchStride
		rhsBaseIdx := batchIdx * rhsBatchStride
		outputBaseIdx := batchIdx * outputBatchStride

		for m := 0; m < lhsCrossSize; m++ {
			lhsRowStart := lhsBaseIdx + m*contractingSize
			outputRowStart := outputBaseIdx + m*rhsCrossSize

			for n := 0; n < rhsCrossSize; n++ {
				// For column n in row-major [K,N], element [k,n] is at k*N + n
				rhsColStart := rhsBaseIdx + n
				var sum float32

				// Scalar loop with strided RHS access
				// We cannot use NEON here because RHS column elements are not contiguous
				k := 0
				for ; k+3 < contractingSize; k += 4 {
					sum += lhsFlat[lhsRowStart+k]*rhsFlat[rhsColStart+k*rhsColStride] +
						lhsFlat[lhsRowStart+k+1]*rhsFlat[rhsColStart+(k+1)*rhsColStride] +
						lhsFlat[lhsRowStart+k+2]*rhsFlat[rhsColStart+(k+2)*rhsColStride] +
						lhsFlat[lhsRowStart+k+3]*rhsFlat[rhsColStart+(k+3)*rhsColStride]
				}
				for ; k < contractingSize; k++ {
					sum += lhsFlat[lhsRowStart+k] * rhsFlat[rhsColStart+k*rhsColStride]
				}

				outputFlat[outputRowStart+n] = sum
			}
		}
	}
}
