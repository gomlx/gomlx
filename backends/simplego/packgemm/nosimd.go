// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package packgemm

import (
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/gomlx/gomlx/internal/workerspool"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/x448/float16"
	"k8s.io/klog/v2"
)

var (
	// NoSIMD32Params are generic assumptions for L1/L2/L3 cache sizes for 32 bits dtypes (float32, int32, uint32)
	//
	// These values are somewhat arbitrary, assuming "standard" modern cache sizes.
	// They are parameterized so they can be tuned or determined dynamically later.
	NoSIMD32Params = CacheParams{
		// Do not change these 2 values: they are hard-coded by the allocated registers in basicSymmetricMicroKernel8x8.
		LHSL1KernelRows: 2, // Mr: Rows of LHS in local registers.
		RHSL1KernelCols: 4, // Nr: Cols of RHS in local registers.

		PanelContractingSize: 512, // Kc: L1 Block contracting "depth".
		LHSPanelCrossSize:    2,   // Mc: Block Height fitting L2/L3 cache.
		RHSPanelCrossSize:    512, // Nc: Block Width fitting L2/L3 cache.
	}

	// Threshold in byte size for switching to the small matrix multiplication kernel.
	// If the total number of operations is below this threshold, the small
	// matrix multiplication kernel is used instead of the tiled implementation.
	// This is a heuristic and may need to be tuned for different architectures.
	// Expressed in number of bytes.
	nosimdSmallMatMulSizeThreshold = 4 * 1024 * 1024

	// Minimum number of flops per worker: above this number, if possible we should
	// parallelize computation on separate goroutines.
	nosimdMinMatMulFlopsPerWorker = 1024
)

func init() {
	RegisterGEMM("Basic(non-SIMD)", basicSymmetricGeneric[float32], &NoSIMD32Params, PriorityBase)
	RegisterGEMM("Basic(non-SIMD)", basicSymmetricGeneric[float64], &NoSIMD32Params, PriorityBase)
	RegisterGEMM("Basic(non-SIMD)", basicSymmetricFloat16, &NoSIMD32Params, PriorityBase)

	knownParams["no-SIMD"] = &NoSIMD32Params
}

func convertFloat16ToHighway(src []float16.Float16) []hwy.Float16 {
	if len(src) == 0 {
		return nil
	}
	// SliceData returns a pointer to the first element
	// Slice creates a new slice header pointing to that data
	return unsafe.Slice((*hwy.Float16)(unsafe.Pointer(unsafe.SliceData(src))), len(src))
}

func convertBFloat16ToHighway(src []bfloat16.BFloat16) []hwy.BFloat16 {
	if len(src) == 0 {
		return nil
	}
	// SliceData returns a pointer to the first element
	// Slice creates a new slice header pointing to that data
	return unsafe.Slice((*hwy.BFloat16)(unsafe.Pointer(unsafe.SliceData(src))), len(src))
}

// Wrapper to be able to use go-highway types.
func basicSymmetricFloat16(alpha, beta float16.Float16, lhsFlat, rhsFlat []float16.Float16,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	outputFlat []float16.Float16,
	bufAllocFn BufAllocFn[float16.Float16], bufReleaseFn BufReleaseFn, pool *workerspool.Pool) error {
	convertedBufAllocFn := func(size int) (any, []hwy.Float16) {
		ref, data := bufAllocFn(size)
		return ref, convertFloat16ToHighway(data)
	}
	return basicSymmetricGeneric(hwy.Float16(alpha), hwy.Float16(beta),
		convertFloat16ToHighway(lhsFlat), convertFloat16ToHighway(rhsFlat),
		batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
		convertFloat16ToHighway(outputFlat), convertedBufAllocFn, bufReleaseFn, pool)
}

// highwayToDType converts a go-highway type to a dtypes.DType.
func highwayToDType[T hwy.Floats]() dtypes.DType {
	switch any(T(0)) {
	case hwy.Float16(0):
		return dtypes.Float16
	case hwy.BFloat16(0):
		return dtypes.BFloat16
	default:
		return dtypes.Float32
	}
}

// basicSymmetricGeneric implements basic symmetric (input and output dtypes are the same) non-SIMD
// GEMM for various types of inputs and outputs.
//
// It is used when no SIMD-optimized implementation is available.
func basicSymmetricGeneric[T hwy.Floats](alpha, beta T, lhsFlat, rhsFlat []T,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	outputFlat []T,
	bufAllocFn BufAllocFn[T], bufReleaseFn BufReleaseFn, pool *workerspool.Pool) error {

	// 1. Resolve Strides
	lhsBatchStride := lhsCrossSize * contractingSize
	rhsBatchStride := contractingSize * rhsCrossSize
	outputBatchStride := lhsCrossSize * rhsCrossSize
	dtype := highwayToDType[T]()
	gemmSize := (lhsBatchStride + rhsBatchStride + outputBatchStride) * dtype.Size()
	// gemmFlops := lhsCrossSize * rhsCrossSize * contractingSize

	// 2. Check if small matrix multiplication kernel can be used.
	if (forceVariant == VariantNone && gemmSize < nosimdSmallMatMulSizeThreshold) || forceVariant == VariantSmall {
		klog.V(1).Infof("Using small variant for GEMM kernel")
		return basicSymmetricGenericSmallGEMMParallel(
			alpha, beta,
			lhsFlat, rhsFlat, outputFlat,
			batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
			lhsBatchStride, rhsBatchStride, outputBatchStride,
			pool)
	}

	klog.V(1).Infof("Using large variant for GEMM kernel")
	return basicSymmetricGenericLargeGEMMParallel(
		alpha, beta,
		lhsFlat, rhsFlat, outputFlat,
		batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
		lhsBatchStride, rhsBatchStride, outputBatchStride,
		bufAllocFn, bufReleaseFn,
		pool)
}

// basicSymmetricGenericSmallGEMMParallel implements basic symmetric (input and output dtypes are the same) non-SIMD
// GEMM for various types of inputs and outputs for **small matrices** (not counting the batch size).
//
// This function will attempt to parallelize the computation on the batch dimension, if it evaluate it as
// worth parallelizing.
//
// It is used when no SIMD-optimized implementation is available.
func basicSymmetricGenericSmallGEMMParallel[T hwy.Floats](
	alpha, beta T,
	lhsFlat, rhsFlat []T, outputFlat []T,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	lhsBatchStride, rhsBatchStride, outputBatchStride int,
	pool *workerspool.Pool) error {

	gemmFlops := lhsCrossSize * rhsCrossSize * contractingSize
	var maxWorkers int
	if pool != nil {
		maxWorkers = pool.AdjustedMaxParallelism()
	}
	if maxWorkers <= 1 || batchSize == 1 || batchSize*gemmFlops < nosimdMinMatMulFlopsPerWorker {
		// Not worth parallelizing: just run the small matmul kernel sequentially.
		basicSymmetricGenericSmallGEMM(
			alpha, beta,
			lhsFlat, rhsFlat, outputFlat,
			batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
		)
		return nil
	}

	// Parallelize on the batch dimension:
	batchCountPerTask := nosimdMinMatMulFlopsPerWorker / gemmFlops
	if maxWorkers > 0 {
		// Make parallelization more fine-grained if there are enough workers
		batchCountPerTask = min(batchCountPerTask, batchSize/maxWorkers)
	}
	batchCountPerTask = max(batchCountPerTask, 1)

	// Crate work that needs doing in a buffered channel.
	type chunkData struct {
		batchIdx, batchCount int
	}
	numChunks := (batchSize + batchCountPerTask - 1) / batchCountPerTask
	work := make(chan chunkData, numChunks)
	for batchIdx := 0; batchIdx < batchSize; batchIdx += batchCountPerTask {
		batchCount := min(batchCountPerTask, batchSize-batchIdx)
		work <- chunkData{batchIdx, batchCount}
	}
	close(work)

	// Execute the work in as many workers as available.
	pool.Saturate(func() {
		for w := range work {
			batchLhs := lhsFlat[w.batchIdx*lhsBatchStride : (w.batchIdx+w.batchCount)*lhsBatchStride]
			batchRhs := rhsFlat[w.batchIdx*rhsBatchStride : (w.batchIdx+w.batchCount)*rhsBatchStride]
			batchOutput := outputFlat[w.batchIdx*outputBatchStride : (w.batchIdx+w.batchCount)*outputBatchStride]
			basicSymmetricGenericSmallGEMM(
				alpha, beta,
				batchLhs, batchRhs, batchOutput,
				w.batchCount, lhsCrossSize, rhsCrossSize, contractingSize,
			)
		}
	})
	return nil
}

func basicSymmetricGenericSmallGEMM[T hwy.Floats](
	alpha, beta T,
	lhs, rhs, output []T,
	batchCount, lhsCrossSize, rhsCrossSize, contractingSize int,
) {
	lhsStride := contractingSize * lhsCrossSize
	rhsStride := rhsCrossSize * contractingSize
	outputStride := rhsCrossSize * lhsCrossSize

	// Bounds check hint for the compiler
	if len(lhs) < lhsStride*batchCount || len(rhs) < rhsStride*batchCount || len(output) < outputStride*batchCount {
		return
	}

	for batchIdx := range batchCount {
		lhsBase := batchIdx * lhsStride
		rhsBase := batchIdx * rhsStride
		outputBase := batchIdx * outputStride

		row := 0
		// Main Loop: Process 3 rows at a time
		for ; row+2 < lhsCrossSize; row += 3 {
			// Pre-calculate base indices for the 3 LHS rows
			lRow0Base := lhsBase + row*contractingSize
			lRow1Base := lRow0Base + contractingSize
			lRow2Base := lRow1Base + contractingSize

			col := 0
			// Main Tile: Process 4 columns at a time
			for ; col+3 < rhsCrossSize; col += 4 {
				var c00, c01, c02, c03 T
				var c10, c11, c12, c13 T
				var c20, c21, c22, c23 T

				// rIdx tracks the current row in the RHS for these 4 columns
				rIdx := rhsBase + col

				for k := range contractingSize {
					// Load RHS row segment
					r0, r1, r2, r3 := rhs[rIdx], rhs[rIdx+1], rhs[rIdx+2], rhs[rIdx+3]

					// Row 0
					l0 := lhs[lRow0Base+k]
					c00 += l0 * r0
					c01 += l0 * r1
					c02 += l0 * r2
					c03 += l0 * r3
					// Row 1
					l1 := lhs[lRow1Base+k]
					c10 += l1 * r0
					c11 += l1 * r1
					c12 += l1 * r2
					c13 += l1 * r3
					// Row 2
					l2 := lhs[lRow2Base+k]
					c20 += l2 * r0
					c21 += l2 * r1
					c22 += l2 * r2
					c23 += l2 * r3

					rIdx += rhsCrossSize
				}

				// Write 3x4 tile results
				basicWriteCol4(output, outputBase+row*rhsCrossSize+col, alpha, beta, c00, c01, c02, c03)
				basicWriteCol4(output, outputBase+(row+1)*rhsCrossSize+col, alpha, beta, c10, c11, c12, c13)
				basicWriteCol4(output, outputBase+(row+2)*rhsCrossSize+col, alpha, beta, c20, c21, c22, c23)
			}

			// Columns-fringe: handle remaining columns for the current 3 rows
			for ; col < rhsCrossSize; col++ {
				var c0, c1, c2 T
				rIdx := rhsBase + col
				for k := range contractingSize {
					rk := rhs[rIdx]
					c0 += lhs[lRow0Base+k] * rk
					c1 += lhs[lRow1Base+k] * rk
					c2 += lhs[lRow2Base+k] * rk
					rIdx += rhsCrossSize
				}
				outputIdx := outputBase + row*rhsCrossSize + col
				basicWriteScalar(output, outputIdx, alpha, beta, c0)
				basicWriteScalar(output, outputIdx+rhsCrossSize, alpha, beta, c1)
				basicWriteScalar(output, outputIdx+2*rhsCrossSize, alpha, beta, c2)
			}
		}

		// Row-Fringe: Handle remaining rows (fewer than 3)
		outputIdx := outputBase + row*rhsCrossSize
		for ; row < lhsCrossSize; row++ {
			for col := range rhsCrossSize {
				var acc T
				lhsIdx := lhsBase + row*contractingSize
				rhsIdx0 := rhsBase + col
				rhsIdx1 := rhsBase + col + rhsCrossSize
				rhsIdx2 := rhsBase + col + 2*rhsCrossSize
				rhsIdx3 := rhsBase + col + 3*rhsCrossSize
				rhsStride := rhsCrossSize * 4
				var contractingIdx int
				for ; contractingIdx+3 < contractingSize; contractingIdx += 4 {
					v0 := lhs[lhsIdx] *
						rhs[rhsIdx0]
					v1 := lhs[lhsIdx+1] * rhs[rhsIdx1]
					v2 := lhs[lhsIdx+2] * rhs[rhsIdx2]
					v3 := lhs[lhsIdx+3] * rhs[rhsIdx3]
					acc += v0 + v1 + v2 + v3
					lhsIdx += 4
					rhsIdx0 += rhsStride
					rhsIdx1 += rhsStride
					rhsIdx2 += rhsStride
					rhsIdx3 += rhsStride
				}
				for ; contractingIdx < contractingSize; contractingIdx++ {
					acc += lhs[lhsIdx] * rhs[rhsIdx0]
					lhsIdx++
					rhsIdx0 += rhsCrossSize
				}
				basicWriteScalar(output, outputIdx, alpha, beta, acc)
				outputIdx++
			}
		}
	}
}

// basicWriteCol4 handles a single row of 4 columns to maximize store-throughput
func basicWriteCol4[T hwy.Floats](out []T, offset int, alpha, beta T, v0, v1, v2, v3 T) {
	if beta != 0 {
		out[offset+0] = beta*out[offset+0] + alpha*v0
		out[offset+1] = beta*out[offset+1] + alpha*v1
		out[offset+2] = beta*out[offset+2] + alpha*v2
		out[offset+3] = beta*out[offset+3] + alpha*v3
	} else {
		out[offset+0] = alpha * v0
		out[offset+1] = alpha * v1
		out[offset+2] = alpha * v2
		out[offset+3] = alpha * v3
	}
}

// basicWriteScalar handles a single scalar write to maximize store-throughput
func basicWriteScalar[T hwy.Floats](out []T, idx int, alpha, beta T, value T) {
	if beta != 0 {
		out[idx] = beta*out[idx] + alpha*value
	} else {
		out[idx] = alpha * value
	}
}

func basicSymmetricGenericLargeGEMMParallel[T hwy.Floats](
	alpha, beta T,
	lhsFlat, rhsFlat []T, outputFlat []T,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	lhsBatchStride, rhsBatchStride, outputBatchStride int,
	bufAllocFn BufAllocFn[T], bufReleaseFn BufReleaseFn,
	pool *workerspool.Pool) error {

	params := &NoSIMD32Params

	// Split work in reasonable number of "chunks".
	maxWorkers := 1
	if pool != nil {
		maxWorkers = pool.AdjustedMaxParallelism()
	}
	if maxWorkers <= 1 {
		// Do everything sequentially.
		packedLhsRef, packedLHS := bufAllocFn(params.LHSPanelCrossSize * params.PanelContractingSize)
		packedRhsRef, packedRHS := bufAllocFn(params.PanelContractingSize * params.RHSPanelCrossSize)
		packedOutRef, packedOutput := bufAllocFn(params.LHSPanelCrossSize * params.RHSPanelCrossSize)
		defer func() {
			bufReleaseFn(packedLhsRef)
			bufReleaseFn(packedRhsRef)
			bufReleaseFn(packedOutRef)
		}()
		for batchIdx := range batchSize {
			batchLhs := lhsFlat[batchIdx*lhsBatchStride : (batchIdx+1)*lhsBatchStride]
			batchRhs := rhsFlat[batchIdx*rhsBatchStride : (batchIdx+1)*rhsBatchStride]
			batchOutput := outputFlat[batchIdx*outputBatchStride : (batchIdx+1)*outputBatchStride]
			basicSymmetricLargeGemmSlice(
				alpha, beta,
				batchLhs, batchRhs, batchOutput,
				/*lhsCrossSize,*/ rhsCrossSize, contractingSize,
				NoSIMD32Params,
				0, lhsCrossSize, 0, rhsCrossSize,
				packedLHS, packedRHS, packedOutput,
			)
		}
		return nil
	}

	// 1. Split work in workItems.
	workChan := make(chan workItem, max(2000, 2*maxWorkers))
	go feedWorkItems(
		batchSize, lhsCrossSize, rhsCrossSize,
		params, maxWorkers, workChan)

	// 2. Saturate (fan-out workers) on workItems.
	pool.Saturate(func() {
		packedLhsRef, packedLHS := bufAllocFn(params.LHSPanelCrossSize * params.PanelContractingSize)
		packedRhsRef, packedRHS := bufAllocFn(params.PanelContractingSize * params.RHSPanelCrossSize)
		packedOutRef, packedOutput := bufAllocFn(params.LHSPanelCrossSize * params.RHSPanelCrossSize)
		defer func() {
			bufReleaseFn(packedLhsRef)
			bufReleaseFn(packedRhsRef)
			bufReleaseFn(packedOutRef)
		}()

		for item := range workChan {
			for batchIdx := item.batchStart; batchIdx < item.batchEnd; batchIdx++ {
				batchLhs := lhsFlat[batchIdx*lhsBatchStride : (batchIdx+1)*lhsBatchStride]
				batchRhs := rhsFlat[batchIdx*rhsBatchStride : (batchIdx+1)*rhsBatchStride]
				batchOutput := outputFlat[batchIdx*outputBatchStride : (batchIdx+1)*outputBatchStride]
				basicSymmetricLargeGemmSlice(
					alpha, beta,
					batchLhs, batchRhs, batchOutput,
					/*lhsCrossSize,*/ rhsCrossSize, contractingSize,
					NoSIMD32Params,
					item.lhsRowStart, item.lhsRowEnd, item.rhsColStart, item.rhsColEnd,
					packedLHS, packedRHS, packedOutput,
				)
			}
		}
	})
	return nil
}

// basicSymmetricLargeGemmSlice performs a slice of the matrix multiplication on one example: lhs, rhs an output
// must already have sliced one example of the batch dimension.
//
// packedLHS and packedRHS must be pre-allocated buffers of appropriate size.
func basicSymmetricLargeGemmSlice[T hwy.Floats](
	alpha, beta T,
	lhs, rhs, output []T,
	/*lhsCrossSize,*/ rhsCrossSize, contractingSize int,
	params CacheParams,
	rowStart, rowEnd, colStart, colEnd int,
	packedLHS, packedRHS, packedOutput []T,
) {
	// Loop 5 (jc): Tiling N (Output Columns)
	for rhsPanelColIdx := colStart; rhsPanelColIdx < colEnd; rhsPanelColIdx += params.RHSPanelCrossSize {
		rhsPanelWidth := min(params.RHSPanelCrossSize, colEnd-rhsPanelColIdx)

		// Loop 4 (p): Tiling K (Depth)
		for contractingPanelIdx := 0; contractingPanelIdx < contractingSize; contractingPanelIdx += params.PanelContractingSize {
			contractingPanelWidth := min(params.PanelContractingSize, contractingSize-contractingPanelIdx)
			PackRHS(rhs, packedRHS, contractingPanelIdx, rhsPanelColIdx, rhsCrossSize, contractingPanelWidth, rhsPanelWidth, params.RHSL1KernelCols)

			// Loop 3 (ic): Tiling M (Output Rows)
			for lhsPanelRowIdx := rowStart; lhsPanelRowIdx < rowEnd; lhsPanelRowIdx += params.LHSPanelCrossSize {
				lhsPanelHeight := min(params.LHSPanelCrossSize, rowEnd-lhsPanelRowIdx)

				// PACK LHS
				packLHS(lhs, packedLHS, lhsPanelRowIdx, contractingPanelIdx, contractingSize, lhsPanelHeight, contractingPanelWidth, params.LHSL1KernelRows)

				basicSymmetricPanel(
					packedLHS, packedRHS, packedOutput,
					params.LHSPanelCrossSize, params.RHSPanelCrossSize,
					contractingPanelWidth,
					lhsPanelHeight, rhsPanelWidth,
				)

				// Accumulate (or write) packedOutput to output.
				effectiveBeta := beta
				if contractingPanelIdx > 0 {
					effectiveBeta = 1
				}
				ApplyPackedOutput(
					packedOutput, output,
					alpha, effectiveBeta,
					params.RHSPanelCrossSize,
					lhsPanelRowIdx, rhsPanelColIdx,
					rhsCrossSize,
					lhsPanelHeight, rhsPanelWidth)
			}
		}
	}
}

// basicSymmetricPanel implements the gemm for a lhs and rhs packed panels
// into an output panel, using packedOutput as intermediate.
//
// It uses register blocking: it divides the 4x4 matrix in 4 4x4 sub-matrices.
// For each sub-matrix it iterates over k (contracting dim), accumulating the results
// in local variables (registers).
// finally it writes the results to output.
//
// It assumes lhsL1KernelRows=4 and rhsL1KernelCols=4.
//
// See basicSymmetricMicroKernel for documentation on arguments.
func basicSymmetricPanel[T hwy.Floats](
	packedLHS, packedRHS []T,
	packedOutput []T,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
) {
	const kernelRows = 2
	const kernelCols = 4

	// BCE hints
	_ = packedLHS[contractingLen]
	_ = packedRHS[contractingLen]
	_ = packedOutput[lhsPanelRows*rhsPanelCols-1]

	// Strides in the packed buffers for one block.
	lhsBlockStride := kernelRows * contractingLen
	rhsBlockStride := kernelCols * contractingLen
	lhsOffset := 0

	// Write active part of 4x4 block to output
	// Helper to write a row
	// Write active part of 4x4 block to output
	// Bounds check is not needed as packedOutput is allocated to panel size, and we will discard
	// whatever is written beyond the active part.

	for rowIdx := 0; rowIdx < lhsActiveRows; rowIdx += kernelRows {
		rhsOffset := 0
		for colIdx := 0; colIdx < rhsActiveCols; colIdx += kernelCols {
			// Process 2x4 block at (r, c)
			// Accumulators for 2x4 block
			var c00, c01, c02, c03 T
			var c10, c11, c12, c13 T

			idxLhs := lhsOffset
			idxRhs := rhsOffset

			// K-Loop unrolled by 4
			k := 0
			for ; k+3 < contractingLen; k += 4 {
				// We need 4 steps.
				// For each step (l is k offset):
				//   load lhs (2 vals), load rhs (4 vals), fma.

				// --- Step 0 ---
				// BCE hint
				_ = packedLHS[idxLhs+1]
				_ = packedRHS[idxRhs+3]
				l0 := packedLHS[idxLhs]
				l1 := packedLHS[idxLhs+1]

				r0 := packedRHS[idxRhs]
				r1 := packedRHS[idxRhs+1]
				r2 := packedRHS[idxRhs+2]
				r3 := packedRHS[idxRhs+3]

				c00 += l0 * r0
				c01 += l0 * r1
				c02 += l0 * r2
				c03 += l0 * r3
				c10 += l1 * r0
				c11 += l1 * r1
				c12 += l1 * r2
				c13 += l1 * r3

				idxLhs += kernelRows
				idxRhs += kernelCols

				// --- Step 1 ---
				_ = packedLHS[idxLhs+1]
				_ = packedRHS[idxRhs+3]
				l0 = packedLHS[idxLhs]
				l1 = packedLHS[idxLhs+1]
				r0 = packedRHS[idxRhs]
				r1 = packedRHS[idxRhs+1]
				r2 = packedRHS[idxRhs+2]
				r3 = packedRHS[idxRhs+3]

				c00 += l0 * r0
				c01 += l0 * r1
				c02 += l0 * r2
				c03 += l0 * r3
				c10 += l1 * r0
				c11 += l1 * r1
				c12 += l1 * r2
				c13 += l1 * r3

				idxLhs += kernelRows
				idxRhs += kernelCols

				// --- Step 2 ---
				_ = packedLHS[idxLhs+1]
				_ = packedRHS[idxRhs+3]
				l0 = packedLHS[idxLhs]
				l1 = packedLHS[idxLhs+1]
				r0 = packedRHS[idxRhs]
				r1 = packedRHS[idxRhs+1]
				r2 = packedRHS[idxRhs+2]
				r3 = packedRHS[idxRhs+3]

				c00 += l0 * r0
				c01 += l0 * r1
				c02 += l0 * r2
				c03 += l0 * r3
				c10 += l1 * r0
				c11 += l1 * r1
				c12 += l1 * r2
				c13 += l1 * r3

				idxLhs += kernelRows
				idxRhs += kernelCols

				// --- Step 3 ---
				_ = packedLHS[idxLhs+1]
				_ = packedRHS[idxRhs+3]
				l0 = packedLHS[idxLhs]
				l1 = packedLHS[idxLhs+1]
				r0 = packedRHS[idxRhs]
				r1 = packedRHS[idxRhs+1]
				r2 = packedRHS[idxRhs+2]
				r3 = packedRHS[idxRhs+3]

				c00 += l0 * r0
				c01 += l0 * r1
				c02 += l0 * r2
				c03 += l0 * r3
				c10 += l1 * r0
				c11 += l1 * r1
				c12 += l1 * r2
				c13 += l1 * r3

				idxLhs += kernelRows
				idxRhs += kernelCols
			}

			// K-Loop Tail
			for ; k < contractingLen; k++ {
				l0 := packedLHS[idxLhs]
				l1 := packedLHS[idxLhs+1]

				r0 := packedRHS[idxRhs]
				r1 := packedRHS[idxRhs+1]
				r2 := packedRHS[idxRhs+2]
				r3 := packedRHS[idxRhs+3]

				c00 += l0 * r0
				c01 += l0 * r1
				c02 += l0 * r2
				c03 += l0 * r3
				c10 += l1 * r0
				c11 += l1 * r1
				c12 += l1 * r2
				c13 += l1 * r3

				idxLhs += kernelRows
				idxRhs += kernelCols
			}

			// Optimization: write full 2x4 block directly to packedOutput.
			// The buffer is large enough even for fringe blocks.
			// Row 0
			rowOffset := rowIdx*rhsPanelCols + colIdx
			packedOutput[rowOffset] = c00
			packedOutput[rowOffset+1] = c01
			packedOutput[rowOffset+2] = c02
			packedOutput[rowOffset+3] = c03

			// Row 1
			rowOffset1 := rowOffset + rhsPanelCols
			packedOutput[rowOffset1] = c10
			packedOutput[rowOffset1+1] = c11
			packedOutput[rowOffset1+2] = c12
			packedOutput[rowOffset1+3] = c13

			rhsOffset += rhsBlockStride
		}
		lhsOffset += lhsBlockStride
	}
}
