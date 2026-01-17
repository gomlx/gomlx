// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import ( //alt:base
	_ "github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16" //alt:base
	_ "github.com/x448/float16"                         //alt:base
) //alt:base
//alt:bf16 import	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
//alt:f16 import	"github.com/x448/float16"

// execDotGeneralSmallMatMul* executes matrix multiplication without transpose.
//
// Memory layout for row-major tensors [M, K] × [K, N] → [M, N]:
//
//	LHS [M, K]: element [m, k] at index m*K + k
//	  → Row m is CONTIGUOUS: [m*K, m*K+1, ..., m*K+K-1] - Good cache locality
//
//	RHS [K, N]: element [k, n] at index k*N + n
//	  → Column n is STRIDED: [n, N+n, 2N+n, ...] with stride N - Poor cache locality
//
//	Output [M, N]: element [m, n] at index m*N + n
//
// The strided RHS access is the key limitation of this path. For large K or N,
// each RHS element access may cause a cache miss. This is why we limit this path
// to small matrices (see smallMatMulMaxContractingSize).
//
// For large matrices, execDotGeneralSmallNormalized transposes RHS to [N, K] form where
// "row" n (the original column) becomes contiguous, enabling efficient vectorization.
//
// BFloat16/Float16 variants accumulate in float32 for numerical stability, then
// convert to the native dtype when writing to output (fused conversion).
func execDotGeneralSmallMatMulGeneric[T PODNumericConstraints]( //alt:base
	//alt:bf16 func execDotGeneralSmallMatMulBFloat16(
	//alt:f16 func execDotGeneralSmallMatMulFloat16(
	_ *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, output *Buffer) {

	lhsFlat := lhs.flat.([]T)       //alt:base
	rhsFlat := rhs.flat.([]T)       //alt:base
	outputFlat := output.flat.([]T) //alt:base
	//alt:bf16 lhsFlat := lhs.flat.([]bfloat16.BFloat16)
	//alt:bf16 rhsFlat := rhs.flat.([]bfloat16.BFloat16)
	//alt:bf16 outputFlat := output.flat.([]bfloat16.BFloat16)
	//alt:f16 lhsFlat := lhs.flat.([]float16.Float16)
	//alt:f16 rhsFlat := rhs.flat.([]float16.Float16)
	//alt:f16 outputFlat := output.flat.([]float16.Float16)

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
				var sum T //alt:base
				//alt:bf16|f16 var sum float32

				// Scalar loop with strided RHS access
				// We cannot use NEON here because RHS column elements are not contiguous
				k := 0
				for ; k+3 < contractingSize; k += 4 {
					//alt:base{
					sum += lhsFlat[lhsRowStart+k]*rhsFlat[rhsColStart+k*rhsColStride] +
						lhsFlat[lhsRowStart+k+1]*rhsFlat[rhsColStart+(k+1)*rhsColStride] +
						lhsFlat[lhsRowStart+k+2]*rhsFlat[rhsColStart+(k+2)*rhsColStride] +
						lhsFlat[lhsRowStart+k+3]*rhsFlat[rhsColStart+(k+3)*rhsColStride]
					//alt:base}
					/* //alt:bf16|f16{
					sum += lhsFlat[lhsRowStart+k].Float32()*rhsFlat[rhsColStart+k*rhsColStride].Float32() +
						lhsFlat[lhsRowStart+k+1].Float32()*rhsFlat[rhsColStart+(k+1)*rhsColStride].Float32() +
						lhsFlat[lhsRowStart+k+2].Float32()*rhsFlat[rhsColStart+(k+2)*rhsColStride].Float32() +
						lhsFlat[lhsRowStart+k+3].Float32()*rhsFlat[rhsColStart+(k+3)*rhsColStride].Float32()
					*/ //alt:bf16|f16}
				}
				for ; k < contractingSize; k++ {
					sum += lhsFlat[lhsRowStart+k] * rhsFlat[rhsColStart+k*rhsColStride] //alt:base
					//alt:bf16|f16 sum += lhsFlat[lhsRowStart+k].Float32() * rhsFlat[rhsColStart+k*rhsColStride].Float32()
				}

				outputFlat[outputRowStart+n] = sum //alt:base
				//alt:bf16 outputFlat[outputRowStart+n] = bfloat16.FromFloat32(sum)
				//alt:f16 outputFlat[outputRowStart+n] = float16.Fromfloat32(sum)
			}
		}
	}
}
