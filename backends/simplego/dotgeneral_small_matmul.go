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
//	For [M, K] × [K, N] → [M, N]:
//	- LHS row m: elements at [m*K, m*K+1, ..., m*K+K-1] → SEQUENTIAL (good cache locality)
//	- RHS col n: elements at [n, N+n, 2N+n, ...] → STRIDED with stride N (poor cache locality)
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

	// Batch axes must match in count and must precede other dimensions.
	numBatchAxes := len(lhsBatchAxes)
	if len(rhsBatchAxes) != numBatchAxes {
		return false
	}
	for i := range numBatchAxes {
		if lhsBatchAxes[i] != i || rhsBatchAxes[i] != i {
			return false
		}
	}

	// LHS: [Batch..., M, K] - contracting axis must be last
	if lhsContractingAxes[0] != lhsRank-1 {
		return false
	}

	// LHS must have at most one cross dimension "M": [Batch..., M, K] or [Batch..., K]
	// rank = numBatchAxes + 1/0 (cross) + 1 (contracting) = numBatchAxes + 2
	if lhsRank > numBatchAxes+2 {
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
const smallMatMulMaxRhsCrossSize = 64

// smallMatMulMaxRhsCrossSizeM1 is the maximum RHS cross dimension (N) for M=1 cases.
// For single-row operations, transpose overhead is more significant relative to
// computation, so we use a higher threshold. However, we still need a cap to avoid
// catastrophic cache behavior with very large N (e.g., [1, K] × [K, 100000]).
// The strided access pattern with stride N=100000 would cause a cache miss on
// virtually every RHS element access.
const smallMatMulMaxRhsCrossSizeM1 = 4096

// smallMatMulMaxContractingSizeM1 is the maximum contracting dimension (K) for M=1 cases.
// For single-row operations, transpose overhead is more significant, so we use a higher
// threshold than smallMatMulMaxContractingSize. However, very large K values (e.g., 10000)
// still cause cache thrashing due to strided RHS access, so we cap it.
const smallMatMulMaxContractingSizeM1 = 1024

// smallMatMulMaxSize is the maximum size in bytes of the output for which the small matmul
// path is beneficial. This is a sanity check to avoid using the small matmul path for
// very large outputs -- which usually will do better with normalized/blocked paths
const smallMatMulMaxSize = 256 * 1024 // 256Kb

// dgUseSmallMatMul checks whether the SmallMatMul fast path is beneficial.
// SmallMatMul skips transpose overhead but has strided RHS access, so it's only
// beneficial for small matrices in standard [M,K]×[K,N] order.
// Supports all numeric dtypes (POD types + BFloat16 + Float16).
func dgUseSmallMatMul(dtype dtypes.DType, lhsShape, rhsShape shapes.Shape, params *dotGeneralNodeData) bool {
	// Check if dtype has a registered SmallMatMul implementation
	if dtype >= MaxDTypes || dotGeneralSmallMatMulDTypeMap.Map[dtype] == nil {
		return false
	}

	// Check if axes are in standard matmul order
	if !isMatMulOrder(lhsShape, rhsShape,
		params.lhsContractingAxes, params.rhsContractingAxes,
		params.lhsBatchAxes, params.rhsBatchAxes) {
		return false
	}

	// For large batch sizes, the normalized path with batch parallelism is faster.
	// The small matmul path processes batches sequentially without parallelization.
	if params.batchSize > smallMatMulMaxBatchSize {
		return false
	}

	// For single-row operations (M=1), SmallMatMul is faster because transpose overhead
	// dominates when computing just one output row per batch.
	// BUT we still need to check rhsCrossSize and contractingSize - for M=1 with huge N or K,
	// the strided access causes cache thrashing.
	if params.lhsCrossSize == 1 {
		// For M=1, use larger thresholds since transpose overhead is more significant
		// But still cap to avoid catastrophic cache behavior with very large dimensions
		if params.rhsCrossSize > smallMatMulMaxRhsCrossSizeM1 {
			return false
		}
		if params.contractingSize > smallMatMulMaxContractingSizeM1 {
			return false
		}
		return true
	}

	// For multi-row operations, check both contracting and RHS cross dimensions.
	// The RHS is accessed with stride N (rhsCrossSize), so large N causes more cache
	// misses per contracting step.
	if params.contractingSize > smallMatMulMaxContractingSize {
		return false
	}

	// Check RHS cross size (N) - large N means large stride in RHS access
	if params.rhsCrossSize > smallMatMulMaxRhsCrossSize {
		return false
	}

	// Larger data size benefit from the blocking done by the blocked and normalized paths.
	problemSize := /* LHS size */ params.lhsCrossSize*params.contractingSize +
		/* RHS size */ params.rhsCrossSize*params.contractingSize +
		/* Output size */ params.lhsCrossSize*params.rhsCrossSize
	problemSize *= params.batchSize
	problemSize *= dtype.Size()
	if problemSize > smallMatMulMaxSize {
		return false
	}
	return true
}

// dotGeneralSmallMatMulDTypeMap holds the dtype-specific implementations for SmallMatMul.
var dotGeneralSmallMatMulDTypeMap = NewDTypeMap("DotGeneralSmallMatMul")

// Auto-generate alternate specialized versions of execDotGeneralSmallMatMul for BFloat16/Float16
// (these need float32 accumulation for numerical stability)
//go:generate go run ../../internal/cmd/alternates_generator -base=dotgeneral_small_matmul_alt_base.go -tags=bf16,f16

func init() {
	// Optimized Float32 implementation (from alt_base, with 4-way loop unrolling)
	dotGeneralSmallMatMulDTypeMap.Register(dtypes.Float32, priorityGeneric, execDotGeneralSmallMatMulFloat32)

	// Generic implementation for other POD numeric types
	dotGeneralSmallMatMulDTypeMap.Register(dtypes.Float64, priorityGeneric, execDotGeneralSmallMatMulGeneric[float64])
	dotGeneralSmallMatMulDTypeMap.Register(dtypes.Int8, priorityGeneric, execDotGeneralSmallMatMulGeneric[int8])
	dotGeneralSmallMatMulDTypeMap.Register(dtypes.Int16, priorityGeneric, execDotGeneralSmallMatMulGeneric[int16])
	dotGeneralSmallMatMulDTypeMap.Register(dtypes.Int32, priorityGeneric, execDotGeneralSmallMatMulGeneric[int32])
	dotGeneralSmallMatMulDTypeMap.Register(dtypes.Int64, priorityGeneric, execDotGeneralSmallMatMulGeneric[int64])
	dotGeneralSmallMatMulDTypeMap.Register(dtypes.Uint8, priorityGeneric, execDotGeneralSmallMatMulGeneric[uint8])
	dotGeneralSmallMatMulDTypeMap.Register(dtypes.Uint16, priorityGeneric, execDotGeneralSmallMatMulGeneric[uint16])
	dotGeneralSmallMatMulDTypeMap.Register(dtypes.Uint32, priorityGeneric, execDotGeneralSmallMatMulGeneric[uint32])
	dotGeneralSmallMatMulDTypeMap.Register(dtypes.Uint64, priorityGeneric, execDotGeneralSmallMatMulGeneric[uint64])

	// Specialized BFloat16 and Float16 implementations (need float32 accumulation)
	dotGeneralSmallMatMulDTypeMap.Register(dtypes.BFloat16, priorityTyped, execDotGeneralSmallMatMulBFloat16)
	dotGeneralSmallMatMulDTypeMap.Register(dtypes.Float16, priorityTyped, execDotGeneralSmallMatMulFloat16)
}

// execDotGeneralSmallMatMulGeneric is a generic implementation for POD numeric types.
// It uses the same algorithm as Float32 but works with any numeric type that supports
// direct arithmetic operations.
func execDotGeneralSmallMatMulGeneric[T PODNumericConstraints](
	_ *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, output *Buffer,
) {
	lhsFlat := lhs.flat.([]T)
	rhsFlat := rhs.flat.([]T)
	outputFlat := output.flat.([]T)

	batchSize := params.batchSize
	lhsCrossSize := params.lhsCrossSize       // M
	rhsCrossSize := params.rhsCrossSize       // N
	contractingSize := params.contractingSize // K

	lhsBatchStride := lhsCrossSize * contractingSize // M * K elements per batch
	rhsBatchStride := contractingSize * rhsCrossSize // K * N elements per batch (for [B,K,N] layout)
	outputBatchStride := lhsCrossSize * rhsCrossSize // M * N elements per batch

	// For row-major RHS [K, N], the stride between elements in the same column is N
	rhsColStride := rhsCrossSize // N

	for batchIdx := range batchSize {
		lhsBaseIdx := batchIdx * lhsBatchStride
		rhsBaseIdx := batchIdx * rhsBatchStride
		outputBaseIdx := batchIdx * outputBatchStride

		for m := range lhsCrossSize {
			lhsRowStart := lhsBaseIdx + m*contractingSize
			outputRowStart := outputBaseIdx + m*rhsCrossSize

			for n := range rhsCrossSize {
				// For column n in row-major [K,N], element [k,n] is at k*N + n
				rhsColStart := rhsBaseIdx + n
				var sum T

				// Scalar loop with strided RHS access and 4-way unrolling
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
