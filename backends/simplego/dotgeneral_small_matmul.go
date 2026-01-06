package simplego

import (
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// isMatMulOrder checks if the DotGeneral operands are in standard matrix multiplication order:
// LHS: [Batch..., M, K] (contracting dimension last)
// RHS: [Batch..., K, N] (contracting dimension first after batch)
//
// This is the familiar [M, K] × [K, N] → [M, N] layout.
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
// Supported patterns (generalized for any number of batch dimensions):
//   - Matrix × Matrix: [M, K] × [K, N] → [M, N]
//   - Matrix × Vector: [M, K] × [K] → [M]
//   - Batched: [B..., M, K] × [B..., K, N] → [B..., M, N]
//
// Requirements:
//   - Single contracting axis only
//   - Batch axes must be leading and sequential (0, 1, 2, ...)
//   - LHS contracting axis must be last
//   - RHS contracting axis must be first after batch axes
//
// See also: execDotGeneralSmallNormalized which transposes to [Batch, Cross, Contract] form
// where BOTH operands have the contracting dimension last (sequential access for both).
func isMatMulOrder(lhsShape, rhsShape shapes.Shape, lhsContractingAxes, rhsContractingAxes, lhsBatchAxes, rhsBatchAxes []int) bool {
	lhsRank := lhsShape.Rank()
	rhsRank := rhsShape.Rank()

	// Only support single contracting axis for SmallMatMul
	if len(lhsContractingAxes) != 1 || len(rhsContractingAxes) != 1 {
		return false
	}

	// Batch axes must match in count
	numBatchAxes := len(lhsBatchAxes)
	if len(rhsBatchAxes) != numBatchAxes {
		return false
	}

	// Batch axes must be leading and sequential: 0, 1, 2, ...
	for i := 0; i < numBatchAxes; i++ {
		if lhsBatchAxes[i] != i || rhsBatchAxes[i] != i {
			return false
		}
	}

	// LHS: [Batch..., M, K] - contracting axis must be last
	if lhsContractingAxes[0] != lhsRank-1 {
		return false
	}

	// LHS must have shape [Batch..., M, K] where M is the cross dimension
	// rank = numBatchAxes + 1 (cross) + 1 (contracting) = numBatchAxes + 2
	if lhsRank != numBatchAxes+2 {
		return false
	}

	// RHS: [Batch..., K, N] or [Batch..., K] (vector case)
	// Contracting axis must be first after batch axes
	if rhsContractingAxes[0] != numBatchAxes {
		return false
	}

	// RHS can be:
	// - [Batch..., K, N] where rank = numBatchAxes + 2 (matrix)
	// - [Batch..., K] where rank = numBatchAxes + 1 (vector, no cross dimension)
	if rhsRank != numBatchAxes+2 && rhsRank != numBatchAxes+1 {
		return false
	}

	return true
}

// smallMatMulMaxContractingSize is the maximum contracting dimension size for which
// the small matmul (no-transpose) path is beneficial. Beyond this size, the strided RHS
// access pattern causes too many cache misses, and the normalized path (which
// transposes RHS for sequential access) becomes faster despite the transpose overhead.
//
// This threshold was determined by benchmarking (BenchmarkSmallMatMulThreshold):
//   - For [256, K] × [K, 256]: SmallMatMul wins at K≤128, NormalizedPath wins at K≥256
//   - Crossover point is between K=128 and K=256
//
// Exception: For single-row operations (M=1), SmallMatMul is always faster because
// the transpose overhead dominates when there's only one output row to compute.
const smallMatMulMaxContractingSize = 128

// smallMatMulMaxBatchSize is the maximum batch size for which the small matmul path is beneficial.
// For larger batch sizes, the normalized path with batch parallelism is faster.
// The small matmul path processes batches sequentially, while the normalized path can parallelize
// across batches using multiple workers.
const smallMatMulMaxBatchSize = 64

// smallMatMulMaxRhsCrossSize is the maximum RHS cross dimension (N) for which
// the small matmul path is beneficial. In [M, K] × [K, N] → [M, N], the RHS is
// accessed with stride N during the contracting loop. When N is large, each
// iteration causes a cache line miss, making the normalized path faster despite
// the transpose overhead.
//
// This threshold is important because the RHS stride equals N, so large N causes
// more cache misses per contracting step than large K does.
const smallMatMulMaxRhsCrossSize = 256

// smallMatMulMaxRhsCrossSizeM1 is the maximum RHS cross dimension (N) for M=1 cases.
// For single-row operations, transpose overhead is more significant relative to
// computation, so we use a higher threshold. However, we still need a cap to avoid
// catastrophic cache behavior with very large N (e.g., [1, K] × [K, 100000]).
// The strided access pattern with stride N=100000 would cause a cache miss on
// virtually every RHS element access.
const smallMatMulMaxRhsCrossSizeM1 = 4096

// execDotGeneralSmallMatMulFloat32 executes float32 matrix multiplication without transpose.
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
// to small matrices (see smallMatMulMaxContractingSize).
//
// For large matrices, execDotGeneralSmallNormalized transposes RHS to [N, K] form where
// "row" n (the original column) becomes contiguous, enabling efficient vectorization.
func execDotGeneralSmallMatMulFloat32(_ *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, output *Buffer) {
	lhsFlat := lhs.flat.([]float32)
	rhsFlat := rhs.flat.([]float32)
	outputFlat := output.flat.([]float32)

	batchSize := params.batchSize
	lhsCrossSize := params.lhsCrossSize      // M
	rhsCrossSize := params.rhsCrossSize      // N
	contractingSize := params.contractingSize // K

	lhsBatchStride := lhsCrossSize * contractingSize  // M * K elements per batch
	rhsBatchStride := contractingSize * rhsCrossSize  // K * N elements per batch (for [B,K,N] layout)
	outputBatchStride := lhsCrossSize * rhsCrossSize  // M * N elements per batch

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
