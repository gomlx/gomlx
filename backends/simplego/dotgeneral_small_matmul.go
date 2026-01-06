package simplego

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
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

// useSmallMatMul determines if we should use the small matmul (no-transpose) execution path.
//
// The small matmul path skips normalization/transpose but has strided RHS access.
// It's beneficial when:
//  1. Not in checkPath mode (need deterministic paths for comparison)
//  2. The dtype is float32 (currently the only optimized implementation)
//  3. The axes are already in standard matmul order
//  4. The batch size is small (small matmul path doesn't parallelize across batches)
//  5. Either:
//     a. Single-row operation (lhsCrossSize=1) where transpose overhead dominates, OR
//     b. Small contracting dimension (K) AND small RHS cross dimension (N), where strided
//     access doesn't cause excessive cache misses. N matters because the RHS stride is N.
//
// For larger matrices or batch sizes, use execDotGeneralSmall (normalized) or execDotGeneralBlocked instead.
func useSmallMatMul(backend *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData) bool {
	// In checkPath mode, we need deterministic execution for comparison.
	// SmallMatMul would only run for normalizedPath anyway, so skip it during checkPath.
	if backend.dotGeneralForceExecutionPath == checkPath {
		return false
	}
	// Only support float32 small matmul path for now (most common)
	if lhs.shape.DType != dtypes.Float32 {
		return false
	}

	// Check if axes are in standard matmul order
	if !isMatMulOrder(lhs.shape, rhs.shape,
		params.lhsContractingAxes, params.rhsContractingAxes,
		params.lhsBatchAxes, params.rhsBatchAxes) {
		return false
	}

	// For large batch sizes, the normalized path with batch parallelism is faster.
	// The small matmul path processes batches sequentially without parallelization.
	if params.batchSize > smallMatMulMaxBatchSize {
		return false
	}

	// For single-row operations (M=1) with small batch sizes, small matmul path is faster
	// because transpose overhead dominates when computing just one output row per batch.
	// Benchmarks show SmallMatMul is 10-15x faster for M=1, batchSize=1 cases.
	// Note: Large batch sizes are handled above (parallelization wins).
	if params.lhsCrossSize == 1 {
		return true
	}

	// For multi-row operations, check both contracting and RHS cross dimensions.
	// The RHS is accessed with stride N (rhsCrossSize), so large N causes more cache
	// misses per contracting step. Both dimensions affect cache behavior.
	if params.contractingSize > smallMatMulMaxContractingSize {
		return false
	}

	// Check RHS cross size (N) - large N means large stride in RHS access
	if params.rhsCrossSize > smallMatMulMaxRhsCrossSize {
		return false
	}

	return true
}

// execDotGeneralSmallMatMul executes matrix multiplication directly without transpose/normalization.
//
// This path is optimal for SMALL matrices where the transpose overhead exceeds the
// cache miss penalty from strided RHS access. For large matrices, use execDotGeneralSmallNormalized
// or execDotGeneralBlocked instead.
//
// Returns true if small matmul path was used, false if caller should use another path.
func execDotGeneralSmallMatMul(backend *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, output *Buffer) bool {
	if !useSmallMatMul(backend, lhs, rhs, params) {
		return false
	}

	// Execute the optimized float32 path
	execDotGeneralSmallMatMulFloat32(backend, lhs, rhs, params, output)
	return true
}

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
func execDotGeneralSmallMatMulFloat32(backend *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, output *Buffer) {
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
