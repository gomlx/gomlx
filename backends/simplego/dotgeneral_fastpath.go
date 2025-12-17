package simplego

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// isContractLastOrder checks if the DotGeneral operands have axes ordered such that
// the contracting dimension is last for LHS and first for RHS - the standard matmul layout.
//
// This is the "BatchCrossContract" ordering for LHS: [Batch..., Cross, Contract]
// And "ContractCross" ordering for RHS: [Batch..., Contract, Cross]
//
// Memory access pattern analysis for row-major storage:
//
//   For [M, K] × [K, N] → [M, N]:
//   - LHS row m: elements at [m*K, m*K+1, ..., m*K+K-1] → SEQUENTIAL (good cache locality)
//   - RHS col n: elements at [n, N+n, 2N+n, ...] → STRIDED with stride N (poor cache locality)
//
// This function returns true when inputs are already in this standard order, meaning we can
// skip the transpose/normalization step. However, note that for LARGE matrices, the strided
// RHS access causes cache thrashing, so the normalized path (which transposes RHS to make
// both operands have sequential access) may be faster despite the transpose overhead.
//
// Standard patterns detected:
//   1. Matrix × Matrix: [M, K] × [K, N] → [M, N] (contract on lhs axis 1, rhs axis 0)
//   2. Matrix × Vector: [M, K] × [K] → [M] (contract on lhs axis 1, rhs axis 0)
//   3. Batched MatMul: [B, M, K] × [B, K, N] → [B, M, N] (contract on lhs axis 2, rhs axis 1)
//   4. Multi-batch: [B1, B2, M, K] × [B1, B2, K, N] → [B1, B2, M, N]
//
// See also: execDotGeneralNormalized which transposes to [Batch, Cross, Contract] form
// where BOTH operands have the contracting dimension last (sequential access for both).
func isContractLastOrder(lhsShape, rhsShape shapes.Shape, lhsContractingAxes, rhsContractingAxes, lhsBatchAxes, rhsBatchAxes []int) bool {
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

// isBatchCrossContractOrder checks if the tensor axes are ordered as:
// [Batch..., Cross..., Contract...]
//
// This ordering means:
//   - Batch axes come first (axis 0, 1, ...)
//   - Cross axes come next (the "output" dimensions)
//   - Contracting axes come last (enabling sequential memory access for dot products)
//
// When axes are in this order, no transpose is needed before the normalized
// DotGeneral computation.
func isBatchCrossContractOrder(shape shapes.Shape, contractingAxes, batchAxes []int) bool {
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

// DirectPathMaxContractingSize is the maximum contracting dimension size for which
// the direct (no-transpose) path is beneficial. Beyond this size, the strided RHS
// access pattern causes too many cache misses, and the normalized path (which
// transposes RHS for sequential access) becomes faster despite the transpose overhead.
//
// This threshold was chosen based on typical L1 cache sizes (32-64KB) and cache line
// sizes (64 bytes). For float32, this allows ~4K elements which fits comfortably in L1.
const DirectPathMaxContractingSize = 4096

// canUseDirectPath determines if we can use the direct (no-transpose) execution path.
//
// The direct path skips normalization/transpose but has strided RHS access.
// It's only beneficial when:
//  1. The dtype is float32 (currently the only optimized implementation)
//  2. The axes are already in contract-last order for LHS
//  3. The matrix is small enough that strided access doesn't cause excessive cache misses
//
// For larger matrices, use execDotGeneralNormalized or execDotGeneralBlocked instead.
func canUseDirectPath(lhs, rhs *Buffer, params *dotGeneralNodeData) bool {
	// Only support float32 direct path for now (most common)
	if lhs.shape.DType != dtypes.Float32 {
		return false
	}

	// Check if axes are in contract-last order (standard matmul layout)
	if !isContractLastOrder(lhs.shape, rhs.shape,
		params.lhsContractingAxes, params.rhsContractingAxes,
		params.lhsBatchAxes, params.rhsBatchAxes) {
		return false
	}

	// Only use direct path for small matrices where strided RHS access
	// doesn't cause excessive cache misses. For larger matrices, the
	// normalized path (with transpose) is faster despite the transpose cost.
	if params.contractingSize > DirectPathMaxContractingSize {
		return false
	}

	return true
}

// execDotGeneralDirect executes matrix multiplication directly without transpose/normalization.
//
// This path is optimal for SMALL matrices where the transpose overhead exceeds the
// cache miss penalty from strided RHS access. For large matrices, use execDotGeneralNormalized
// or execDotGeneralBlocked instead.
//
// Returns true if direct path was used, false if caller should use another path.
func execDotGeneralDirect(backend *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, output *Buffer) bool {
	if !canUseDirectPath(lhs, rhs, params) {
		return false
	}

	// Execute the optimized float32 path
	execDotGeneralDirectFloat32(backend, lhs, rhs, params, output)
	return true
}

// execDotGeneralDirectFloat32 executes float32 matrix multiplication without transpose.
//
// Memory layout for row-major tensors [M, K] × [K, N] → [M, N]:
//
//   LHS [M, K]: element [m, k] at index m*K + k
//     → Row m is CONTIGUOUS: [m*K, m*K+1, ..., m*K+K-1] ✓ Good cache locality
//
//   RHS [K, N]: element [k, n] at index k*N + n
//     → Column n is STRIDED: [n, N+n, 2N+n, ...] with stride N ✗ Poor cache locality
//
//   Output [M, N]: element [m, n] at index m*N + n
//
// The strided RHS access is the key limitation of this path. For large K or N,
// each RHS element access may cause a cache miss. This is why we limit this path
// to small matrices (see DirectPathMaxContractingSize).
//
// For large matrices, execDotGeneralNormalized transposes RHS to [N, K] form where
// "row" n (the original column) becomes contiguous, enabling efficient vectorization.
func execDotGeneralDirectFloat32(backend *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, output *Buffer) {
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
