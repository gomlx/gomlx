// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package packgemm

import (
	"simd/archsimd"
	"sync"
	"unsafe"

	"github.com/gomlx/gomlx/internal/workerspool"
	"k8s.io/klog/v2"
)

var avx512Float32Params = CacheParams{
	LHSL1KernelRows:      4,   // Mr: Uses 4 ZMM registers for accumulation rows, this number must be a multiple of 4
	RHSL1KernelCols:      32,  // Nr: Uses 2 ZMM registers for accumulation cols, each holds 16 values
	PanelContractingSize: 128, // Kc: A strip fits in L1 cache
	LHSPanelCrossSize:    4,   // Mc: Fits in L2 cache (multiple of LHSL1KernelRows)
	RHSPanelCrossSize:    512, // Nc: Fits in L3 cache (multiple of RHSL1KernelCols)
}

func init() {
	if archsimd.X86.AVX512() {
		RegisterGEMM("AVX512", avx512Float32, &avx512Float32Params, PriorityDTypeSIMD)
	}
}

var avx512WarningOnce sync.Once

// avx512Float32 implements generic matrix multiplication for float32 inputs and outputs.
// output = alpha * (lhs x rhs) + beta * output
func avx512Float32(alpha, beta float32, lhsFlat, rhsFlat []float32, batchSize, lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat []float32,
	bufAllocFn BufAllocFn[float32], bufReleaseFn BufReleaseFn, pool *workerspool.Pool) error {
	avx512WarningOnce.Do(func() {
		klog.Infof("AVX512 GEMM (General Matrix Multiplication) algorithm still experimental!")
	})

	// 1. Resolve Strides
	params := &avx512Float32Params
	lhsBatchStride := lhsCrossSize * contractingSize
	rhsBatchStride := contractingSize * rhsCrossSize
	outputBatchStride := lhsCrossSize * rhsCrossSize

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
			avx512Float32GemmChunk(
				alpha, beta,
				batchLhs, batchRhs, batchOutput,
				lhsCrossSize, rhsCrossSize, contractingSize,
				params, 0, lhsCrossSize, 0, rhsCrossSize,
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
				avx512Float32GemmChunk(
					alpha, beta,
					batchLhs, batchRhs, batchOutput,
					lhsCrossSize, rhsCrossSize, contractingSize,

					params, item.lhsRowStart, item.lhsRowEnd, item.rhsColStart, item.rhsColEnd,
					packedLHS, packedRHS, packedOutput,
				)
			}
		}
	})
	return nil
}

// avx512Float32GemmChunk performs the 5-loop GotoBLAS algorithm on a slice of a single batch matrix.
func avx512Float32GemmChunk(
	alpha, beta float32,
	lhs, rhs, output []float32,
	lhsCrossSize, rhsCrossSize, contractingSize int,
	params *CacheParams, lhsRowStart, lhsRowEnd, rhsColStart, rhsColEnd int,
	packedLhs, packedRhs, packedOutput []float32,
) {
	// fmt.Printf("gemmChunk(colStart=%d, colEnd=%d)\n", colStart, colEnd)

	// Loop 5 (jc): Tiling N (Output Columns) - Fits in L3
	// Iterates over the assigned strip [colStart, colEnd) in chunks of rhsL3PanelCrossSize.
	for rhsPanelColIdx := rhsColStart; rhsPanelColIdx < rhsColEnd; rhsPanelColIdx += params.RHSPanelCrossSize {

		// The width of the current panel is limited by the L3 block size (Nc)
		// AND the end of our assigned chunk (colEnd).
		rhsPanelWidth := min(params.RHSPanelCrossSize, rhsColEnd-rhsPanelColIdx)

		// Loop 4 (p): Tiling K (Depth) - Fits in L1
		// Iterates over the contracting dimension in chunks of contractingPanelSize
		for contractingPanelIdx := 0; contractingPanelIdx < contractingSize; contractingPanelIdx += params.PanelContractingSize {
			// fmt.Printf("- contractingPanelIdx=%d\n", contractingPanelIdx)

			contractingPanelWidth := min(params.PanelContractingSize, contractingSize-contractingPanelIdx)

			// ---------------------------------------------------------
			// PACK RHS (Bit) -> ~B
			// We pack a [contractingPanelWidth, rhsPanelWidth] block of RHS into contiguous memory.
			// Format: Vertical strips of width rhsL1KernelCols (Nr).
			// ---------------------------------------------------------
			avx512Float32PackRHS(rhs, packedRhs, contractingPanelIdx, rhsPanelColIdx, rhsCrossSize, contractingPanelWidth,
				rhsPanelWidth, params.RHSL1KernelCols)

			// Loop 3 (ic): Tiling M (Output Rows) - Fits in L2
			// Iterates over the LHS height in chunks of lhsL2PanelCrossSize
			for lhsPanelRowIdx := lhsRowStart; lhsPanelRowIdx < lhsRowEnd; lhsPanelRowIdx += params.LHSPanelCrossSize {
				lhsPanelHeight := min(params.LHSPanelCrossSize, lhsRowEnd-lhsPanelRowIdx)

				// -----------------------------------------------------
				// PACK LHS (Ait) -> ~A
				// We pack a [lhsPanelHeight, contractingPanelWidth] block of LHS into contiguous memory.
				// Format: Horizontal strips of height lhsL1KernelRows (Mr).
				// -----------------------------------------------------
				packLHS(lhs, packedLhs, lhsPanelRowIdx, contractingPanelIdx, contractingSize, lhsPanelHeight,
					contractingPanelWidth, params.LHSL1KernelRows)

				// ---------------------------------------------
				// PANEL KERNEL
				// Computes a [lhsPanelHeight, rhsPanelWidth] block of Output
				// by iterating over micro-kernels.
				// ---------------------------------------------
				avx512Float32Panel(
					contractingPanelWidth,
					packedLhs, packedRhs, packedOutput,
					params,
					lhsPanelHeight, rhsPanelWidth,
				)

				// Accumulate (or write) packedOutput to output.
				effectiveBeta := beta
				if contractingPanelIdx > 0 {
					effectiveBeta = 1
				}
				avx512Float32ApplyPackedOutput(
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

// avx512Float32Panel computes a [lhsPanelHeight, rhsPanelWidth] block of the output matrix.
// It iterates over micro-kernels of size [params.LHSL1KernelRows, params.RHSL1KernelCols].
func avx512Float32Panel(
	activeContractingLen int,
	packedLHS, packedRHS, packedOutput []float32, // Packed Buffers
	params *CacheParams,
	lhsActivePanelHeight, rhsActivePanelWidth int,
) {
	// BCE hints
	_ = packedLHS[activeContractingLen*lhsActivePanelHeight-1]
	_ = packedRHS[activeContractingLen*rhsActivePanelWidth-1]
	_ = packedOutput[lhsActivePanelHeight*rhsActivePanelWidth-1]

	// Loop 1 (ir): Micro-Kernel Rows (Mr == lhsL1BlockRows)
	for lhsRowIdx := 0; lhsRowIdx < lhsActivePanelHeight; lhsRowIdx += params.LHSL1KernelRows {
		// Loop 2 (jr): Micro-Kernel Columns (Nr == rhsL1BlockCols)
		idxRHS := 0
		for rhsColIdx := 0; rhsColIdx < rhsActivePanelWidth; rhsColIdx += params.RHSL1KernelCols {
			// Output index calculation (relative to panel)
			outputRowStart := lhsRowIdx
			outputColStart := rhsColIdx
			outputStride := params.RHSPanelCrossSize

			// ---------------------------------------------------------
			// MICRO KERNEL BODY
			// ---------------------------------------------------------

			lhsKernelRows := params.LHSL1KernelRows // Alias for clarity/compatibility with old code structure

			// ---------------------------------------------------------
			// 2. Initialize Accumulators (Registers) to 0.0
			// ---------------------------------------------------------
			// We use 4 rows (Mr) worth of registers at a time.
			accum_lhs0_rhs0 := archsimd.BroadcastFloat32x16(0.0)
			accum_lhs0_rhs1 := archsimd.BroadcastFloat32x16(0.0)
			accum_lhs1_rhs0 := archsimd.BroadcastFloat32x16(0.0)
			accum_lhs1_rhs1 := archsimd.BroadcastFloat32x16(0.0)
			accum_lhs2_rhs0 := archsimd.BroadcastFloat32x16(0.0)
			accum_lhs2_rhs1 := archsimd.BroadcastFloat32x16(0.0)
			accum_lhs3_rhs0 := archsimd.BroadcastFloat32x16(0.0)
			accum_lhs3_rhs1 := archsimd.BroadcastFloat32x16(0.0)

			// ---------------------------------------------------------
			// 3. The K-Loop (Dot Product)
			// ---------------------------------------------------------
			idxLHS := lhsRowIdx * activeContractingLen
			for range activeContractingLen {
				// Load RHS (Broadcasting/Streaming)
				rhsVec0 := archsimd.LoadFloat32x16(castToArray16(&packedRHS[idxRHS]))
				rhsVec1 := archsimd.LoadFloat32x16(castToArray16(&packedRHS[idxRHS+16]))
				idxRHS += 32

				// Row 0
				lhsVal0 := packedLHS[idxLHS+0]
				lhsVec0 := archsimd.BroadcastFloat32x16(lhsVal0)
				accum_lhs0_rhs0 = rhsVec0.MulAdd(lhsVec0, accum_lhs0_rhs0)
				accum_lhs0_rhs1 = rhsVec1.MulAdd(lhsVec0, accum_lhs0_rhs1)

				// Row 1
				lhsVal1 := packedLHS[idxLHS+1]
				lhsVec1 := archsimd.BroadcastFloat32x16(lhsVal1)
				accum_lhs1_rhs0 = rhsVec0.MulAdd(lhsVec1, accum_lhs1_rhs0)
				accum_lhs1_rhs1 = rhsVec1.MulAdd(lhsVec1, accum_lhs1_rhs1)

				// Row 2
				lhsVal2 := packedLHS[idxLHS+2]
				lhsVec2 := archsimd.BroadcastFloat32x16(lhsVal2)
				accum_lhs2_rhs0 = rhsVec0.MulAdd(lhsVec2, accum_lhs2_rhs0)
				accum_lhs2_rhs1 = rhsVec1.MulAdd(lhsVec2, accum_lhs2_rhs1)

				// Row 3
				lhsVal3 := packedLHS[idxLHS+3]
				lhsVec3 := archsimd.BroadcastFloat32x16(lhsVal3)
				accum_lhs3_rhs0 = rhsVec0.MulAdd(lhsVec3, accum_lhs3_rhs0)
				accum_lhs3_rhs1 = rhsVec1.MulAdd(lhsVec3, accum_lhs3_rhs1)

				idxLHS += lhsKernelRows
			}

			// ---------------------------------------------------------
			// 4. Write Back to Output
			// ---------------------------------------------------------
			outputIdx0 := outputRowStart*outputStride + outputColStart
			outputIdx1 := outputIdx0 + params.RHSPanelCrossSize
			outputIdx2 := outputIdx0 + 2*params.RHSPanelCrossSize
			outputIdx3 := outputIdx0 + 3*params.RHSPanelCrossSize

			accum_lhs0_rhs0.Store(castToArray16(&packedOutput[outputIdx0]))
			accum_lhs0_rhs1.Store(castToArray16(&packedOutput[outputIdx0+16]))
			accum_lhs1_rhs0.Store(castToArray16(&packedOutput[outputIdx1]))
			accum_lhs1_rhs1.Store(castToArray16(&packedOutput[outputIdx1+16]))
			accum_lhs2_rhs0.Store(castToArray16(&packedOutput[outputIdx2]))
			accum_lhs2_rhs1.Store(castToArray16(&packedOutput[outputIdx2+16]))
			accum_lhs3_rhs0.Store(castToArray16(&packedOutput[outputIdx3]))
			accum_lhs3_rhs1.Store(castToArray16(&packedOutput[outputIdx3+16]))
		}
	}
}

func castToArray16[T float32](ptr *T) *[16]T {
	return (*[16]T)(unsafe.Pointer(ptr))
}

// applyPackedOutput applies the computed packedOutput to the final output.
func avx512Float32ApplyPackedOutput(
	packedOutput, output []float32,
	alpha, beta float32,
	packedOutputRowStride int,
	lhsRowOffset, rhsColOffset int, // Global output offsets
	outputRowStride int,
	height, width int, // actual amount of data to copy
) {
	// Vectorized constants
	alphaVec := archsimd.BroadcastFloat32x16(alpha)
	betaVec := archsimd.BroadcastFloat32x16(beta)

	for r := range height {
		packedIdx := r * packedOutputRowStride
		outputIdx := (lhsRowOffset+r)*outputRowStride + rhsColOffset

		c := 0
		// Vectorized loop
		for ; c+16 <= width; c += 16 {
			packedVal := archsimd.LoadFloat32x16(castToArray16(&packedOutput[packedIdx]))
			outputVal := archsimd.LoadFloat32x16(castToArray16(&output[outputIdx]))

			// output = alpha * packed + beta * output
			newVal := alphaVec.MulAdd(packedVal, betaVec.Mul(outputVal))

			newVal.Store(castToArray16(&output[outputIdx]))

			packedIdx += 16
			outputIdx += 16
		}

		// Scalar tail
		for ; c < width; c++ {
			val := packedOutput[packedIdx]
			output[outputIdx] = beta*output[outputIdx] + alpha*val
			packedIdx++
			outputIdx++
		}
	}
}

// avx512Float32PackRHS is the AVX512/Flaot32 version of the generic packRHS.
// It packs a slice of size [contractingRows, rhsCols] block from RHS into
// the panel reshaped+transposed to [ceil(rhsCols/RHSL1KernelCols), contractingRows, RHSL1KernelCols],
// padding the cols of the last strip with zeros if necessary.
//
//   - src: [contractingSize, rhsCrossSize]
//   - dst: a slice with enough size to hold the panel
//   - srcRowStart: start row in src
//   - srcColStart: start col in src
//   - srcStrideCol: stride of src
//   - contractingRows: number of rows to be copied in the panel (must fit total panel allocated size)
//   - rhsCols: number of columns to be copied in the panel (excluding padding), will be padded to a RHSL1KernelCols
//     multiple with zeros.
//   - RHSL1KernelCols: number of columns in each "L1 kernel"
func avx512Float32PackRHS(src, dst []float32, srcRowStart, srcColStart, srcStrideCol,
	contractingRows, rhsCols, RHSL1KernelCols int) {
	dstIdx := 0
	// Iterate over strips of width nr
	for stripColIdx := 0; stripColIdx < rhsCols; stripColIdx += RHSL1KernelCols {
		// How many columns valid in this strip?
		validCols := min(RHSL1KernelCols, rhsCols-stripColIdx)

		if validCols == 32 && RHSL1KernelCols == 32 {
			// Fast path for full AVX512 strip (32 floats = 2x ZMM).
			// We hoist srcIdx calculation.
			srcIdx := (srcRowStart * srcStrideCol) + (srcColStart + stripColIdx)
			for range contractingRows {
				// Load 2 vectors (unaligned loads)
				v0 := archsimd.LoadFloat32x16(castToArray16(&src[srcIdx]))
				v1 := archsimd.LoadFloat32x16(castToArray16(&src[srcIdx+16]))

				// Advance src to next row
				srcIdx += srcStrideCol

				// Store to packed destination (guaranteed valid size)
				v0.Store(castToArray16(&dst[dstIdx]))
				v1.Store(castToArray16(&dst[dstIdx+16]))

				dstIdx += 32
			}
			continue
		}

		// Fallback for partial strips or non-32 kernel size
		// Iterate over rows (k)
		for row := range contractingRows {
			srcRow := srcRowStart + row
			srcColBase := srcColStart + stripColIdx
			srcIdx := (srcRow * srcStrideCol) + srcColBase
			// Copy valid columns
			copy(dst[dstIdx:], src[srcIdx:srcIdx+validCols])
			dstIdx += validCols
			// Zero-pad if strip is incomplete (edge of matrix)
			for c := validCols; c < RHSL1KernelCols; c++ {
				dst[dstIdx] = 0
				dstIdx++
			}
		}
	}
}
