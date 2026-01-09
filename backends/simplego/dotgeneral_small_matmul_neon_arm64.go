// Copyright 2025 The GoMLX Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !noasm && arm64

package simplego

import (
	"runtime"
	"unsafe"
)

// smallMatMulRow4_neon_asm computes 4 output columns for one LHS row using NEON.
//
// For [M, K] x [K, N] -> [M, N], this computes output[m, n:n+4] for a single row m.
//
// The key insight is that by processing 4 output columns together, we can access
// RHS row-wise (contiguous) instead of column-wise (strided):
//   - For each k: load lhs[m,k] (scalar), load rhs[k, n:n+4] (4 contiguous floats)
//   - Accumulate: output[n:n+4] += lhs[m,k] * rhs[k, n:n+4]
//
// Parameters:
//   - lhs: pointer to lhs[m, 0] (start of LHS row)
//   - rhs: pointer to rhs[0, n] (start of first RHS column in group)
//   - output: pointer to output[m, n] (start of output for this row/column group)
//   - K: contracting dimension (number of elements in LHS row)
//   - N: RHS cross size (stride between RHS rows, i.e., rhs[k,n] to rhs[k+1,n])
//
//go:noescape
func smallMatMulRow4_neon_asm(lhs, rhs, output unsafe.Pointer, K, N int64)

// execDotGeneralSmallMatMulFloat32NEON is the NEON-accelerated version of SmallMatMul.
//
// Instead of the original loop order (m -> n -> k) with strided RHS column access,
// we use register blocking to process 4 output columns at once:
//
// Original (cannot use NEON):
//
//	for m in M:
//	  for n in N:
//	    for k in K:
//	      output[m,n] += lhs[m,k] * rhs[k,n]  // rhs[k,n] is strided!
//
// NEON-friendly (this implementation):
//
//	for m in M:
//	  for n in 0..N step 4:
//	    for k in K:
//	      lhs_val = lhs[m,k]                    // scalar
//	      rhs_row = rhs[k, n:n+4]               // 4 CONTIGUOUS floats!
//	      output[m, n:n+4] += lhs_val * rhs_row // vectorized
//
// This converts strided column access into contiguous row access, enabling NEON.
func execDotGeneralSmallMatMulFloat32NEON(_ *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, output *Buffer) {
	lhsFlat := lhs.flat.([]float32)
	rhsFlat := rhs.flat.([]float32)
	outputFlat := output.flat.([]float32)

	batchSize := params.batchSize
	lhsCrossSize := params.lhsCrossSize       // M
	rhsCrossSize := params.rhsCrossSize       // N
	contractingSize := params.contractingSize // K

	lhsBatchStride := lhsCrossSize * contractingSize // M * K elements per batch
	rhsBatchStride := contractingSize * rhsCrossSize // K * N elements per batch
	outputBatchStride := lhsCrossSize * rhsCrossSize // M * N elements per batch

	for batchIdx := 0; batchIdx < batchSize; batchIdx++ {
		lhsBaseIdx := batchIdx * lhsBatchStride
		rhsBaseIdx := batchIdx * rhsBatchStride
		outputBaseIdx := batchIdx * outputBatchStride

		for m := 0; m < lhsCrossSize; m++ {
			lhsRowStart := lhsBaseIdx + m*contractingSize
			outputRowStart := outputBaseIdx + m*rhsCrossSize

			// Process 4 columns at a time with NEON
			n := 0
			for ; n+3 < rhsCrossSize; n += 4 {
				// rhs[0, n] is at rhsBaseIdx + n (start of column group)
				// For row k, rhs[k, n:n+4] is at rhsBaseIdx + k*N + n
				rhsColGroupStart := rhsBaseIdx + n

				smallMatMulRow4_neon_asm(
					unsafe.Pointer(&lhsFlat[lhsRowStart]),
					unsafe.Pointer(&rhsFlat[rhsColGroupStart]),
					unsafe.Pointer(&outputFlat[outputRowStart+n]),
					int64(contractingSize),
					int64(rhsCrossSize), // N = stride between RHS rows
				)
			}

			// Scalar fallback for remainder columns (0-3 columns)
			for ; n < rhsCrossSize; n++ {
				rhsColStart := rhsBaseIdx + n
				outputFlat[outputRowStart+n] = smallMatMulScalarDotColumn(
					lhsFlat, rhsFlat, lhsRowStart, rhsColStart, contractingSize, rhsCrossSize)
			}
		}
	}

	// Keep slices alive until after all assembly calls complete
	runtime.KeepAlive(lhsFlat)
	runtime.KeepAlive(rhsFlat)
	runtime.KeepAlive(outputFlat)
}
