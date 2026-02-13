// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package packgemm

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/gomlx/gomlx/internal/workerspool"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/x448/float16"
)

//go:generate go tool github.com/ajroetker/go-highway/cmd/hwygen -input gemm_16reg_base.go -output_prefix=gen_gemm16reg_impl -dispatch gen_gemm16reg_dispatch -targets avx2,avx512,fallback

var (
	// simd16RegistersParams are the parameters to use if there are 16 SIMD registers available.
	// It still needs to be adjusted to the size of the registers (in terms )
	simd16RegistersParams = CacheParams{
		LHSL1KernelRows:      4,   // Mr: Uses 4 registers for accumulation rows.
		RHSL1KernelCols:      2,   // Nr: Uses 2 registers for accumulation cols: this must be multiplied by the number of lanes.
		PanelContractingSize: 128, // Kc: A strip fits in L1 cache
		LHSPanelCrossSize:    4,   // Mc: Fits in L2 cache (multiple of LHSL1KernelRows), multiple of LHSL1KernelRows, but usually just LHSL1KernelRows.
		RHSPanelCrossSize:    512, // Nc: Fits in L3 cache (multiple of RHSL1KernelCols), multiple of RHSL1KernelRows.
	}
)

func init() {
	initGen_gemm16reg_dispatchAll()

	if hwy.HasSIMD() {
		knownVariations["hwy-16regs"] = &simd16RegistersParams // Testing purpose only.
		RegisterGEMM("hwy-16regs", GEMMSymmetric16RegistersFloat32, &simd16RegistersParams, PriorityDTypeSIMD+1)
		RegisterGEMM("hwy-16regs", GEMMSymmetric16RegistersFloat64, &simd16RegistersParams, PriorityDTypeSIMD+1)

		// Float16 version requires casting to/from the different versions of float16.
		RegisterGEMM("hwy-16regs",
			func(
				alpha, beta float16.Float16,
				lhsFlat, rhsFlat []float16.Float16,
				batchSize int, lhsCrossSize int, rhsCrossSize int, contractingSize int,
				outputFlat []float16.Float16,
				bufAllocFn BufAllocFn[float16.Float16],
				bufReleaseFn BufReleaseFn, pool *workerspool.Pool) error {
				return GEMMSymmetric16RegistersFloat16(
					hwy.Float16(alpha), hwy.Float16(beta),
					castHalfPrecisionSlice[hwy.Float16](lhsFlat),
					castHalfPrecisionSlice[hwy.Float16](rhsFlat),
					batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
					castHalfPrecisionSlice[hwy.Float16](outputFlat),
					castBufAllocFn[hwy.Float16](bufAllocFn),
					bufReleaseFn, pool)
			},
			&simd16RegistersParams, PriorityDTypeSIMD+1)

		// BFloat16 version requires casting to/from the different versions of BFloat16.
		RegisterGEMM("hwy-16regs",
			func(
				alpha, beta bfloat16.BFloat16,
				lhsFlat, rhsFlat []bfloat16.BFloat16,
				batchSize int, lhsCrossSize int, rhsCrossSize int, contractingSize int,
				outputFlat []bfloat16.BFloat16,
				bufAllocFn BufAllocFn[bfloat16.BFloat16],
				bufReleaseFn BufReleaseFn, pool *workerspool.Pool) error {
				return GEMMSymmetric16RegistersBFloat16(
					hwy.BFloat16(alpha), hwy.BFloat16(beta),
					castHalfPrecisionSlice[hwy.BFloat16](lhsFlat),
					castHalfPrecisionSlice[hwy.BFloat16](rhsFlat),
					batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
					castHalfPrecisionSlice[hwy.BFloat16](outputFlat),
					castBufAllocFn[hwy.BFloat16](bufAllocFn),
					bufReleaseFn, pool)
			},
			&simd16RegistersParams, PriorityDTypeSIMD+1)
	}
}

func BaseGEMMSymmetric16Registers[T hwy.Floats](
	alpha, beta T, lhsFlat, rhsFlat []T, batchSize, lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat []T,
	bufAllocFn BufAllocFn[T], bufReleaseFn BufReleaseFn, pool *workerspool.Pool) error {

	// Adjust params to current architecture.
	numLanes := hwy.NumLanes[T]()
	params := simd16RegistersParams // Copy.
	params.RHSL1KernelCols *= numLanes

	// 1. Resolve Strides
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
			GEMMSymmetric16RegistersGemmChunk(
				alpha, beta,
				batchLhs, batchRhs, batchOutput,
				lhsCrossSize, rhsCrossSize, contractingSize,
				&params, 0, lhsCrossSize, 0, rhsCrossSize,
				packedLHS, packedRHS, packedOutput,
			)
		}
		return nil
	}

	// 1. Split work in workItems.
	workChan := make(chan workItem, max(2000, 2*maxWorkers))
	go feedWorkItems(
		batchSize, lhsCrossSize, rhsCrossSize,
		&params, maxWorkers, workChan)

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
				GEMMSymmetric16RegistersGemmChunk(
					alpha, beta,
					batchLhs, batchRhs, batchOutput,
					lhsCrossSize, rhsCrossSize, contractingSize,
					&params, item.lhsRowStart, item.lhsRowEnd, item.rhsColStart, item.rhsColEnd,
					packedLHS, packedRHS, packedOutput,
				)
			}
		}
	})
	return nil
}

// BaseGEMMSymmetric16RegistersGemmChunk performs the 5-loop GotoBLAS algorithm on a slice of a single batch matrix.
func BaseGEMMSymmetric16RegistersGemmChunk[T hwy.Floats](
	alpha, beta T,
	lhs, rhs, output []T,
	lhsCrossSize, rhsCrossSize, contractingSize int,
	params *CacheParams, lhsRowStart, lhsRowEnd, rhsColStart, rhsColEnd int,
	packedLhs, packedRhs, packedOutput []T,
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
			PackRHS(rhs, packedRhs, contractingPanelIdx, rhsPanelColIdx, rhsCrossSize, contractingPanelWidth,
				rhsPanelWidth, params.RHSL1KernelCols)
			// PackRHS(rhs, packedRhs, contractingPanelIdx, rhsPanelColIdx, rhsCrossSize, contractingPanelWidth,
			// 	rhsPanelWidth, params.RHSL1KernelCols)

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
				GEMMSymmetric16RegistersPanel(
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
				ApplyPackedOutput(
					packedOutput, output,
					alpha, T(effectiveBeta),
					params.RHSPanelCrossSize,
					lhsPanelRowIdx, rhsPanelColIdx,
					rhsCrossSize,
					lhsPanelHeight, rhsPanelWidth)
			}
		}
	}
}

// BaseGEMMSymmetric16RegistersPanel computes a [lhsPanelHeight, rhsPanelWidth] block of the output matrix.
// It iterates over micro-kernels of size [params.LHSL1KernelRows, params.RHSL1KernelCols].
func BaseGEMMSymmetric16RegistersPanel[T hwy.Floats](
	activeContractingLen int,
	packedLHS, packedRHS, packedOutput []T, // Packed Buffers
	params *CacheParams,
	lhsActivePanelHeight, rhsActivePanelWidth int,
) {
	// BCE hints
	_ = packedLHS[activeContractingLen*lhsActivePanelHeight-1]
	_ = packedRHS[activeContractingLen*rhsActivePanelWidth-1]
	_ = packedOutput[lhsActivePanelHeight*rhsActivePanelWidth-1]
	numLanes := hwy.NumLanes[T]()

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
			accum_lhs0_rhs0 := hwy.Zero[T]()
			accum_lhs0_rhs1 := hwy.Zero[T]()
			accum_lhs1_rhs0 := hwy.Zero[T]()
			accum_lhs1_rhs1 := hwy.Zero[T]()
			accum_lhs2_rhs0 := hwy.Zero[T]()
			accum_lhs2_rhs1 := hwy.Zero[T]()
			accum_lhs3_rhs0 := hwy.Zero[T]()
			accum_lhs3_rhs1 := hwy.Zero[T]()

			// ---------------------------------------------------------
			// 3. The K-Loop (Dot Product)
			// ---------------------------------------------------------
			idxLHS := lhsRowIdx * activeContractingLen
			for range activeContractingLen {
				// Load RHS (Broadcasting/Streaming)
				rhsVec0 := hwy.Load(packedRHS[idxRHS:])
				rhsVec1 := hwy.Load(packedRHS[idxRHS+numLanes:])
				idxRHS += 2 * numLanes

				// Row 0
				lhsVal0 := packedLHS[idxLHS+0]
				lhsVec0 := hwy.Set[T](lhsVal0)
				accum_lhs0_rhs0 = hwy.MulAdd(rhsVec0, lhsVec0, accum_lhs0_rhs0)
				accum_lhs0_rhs1 = hwy.MulAdd(rhsVec1, lhsVec0, accum_lhs0_rhs1)

				// Row 1
				lhsVal1 := packedLHS[idxLHS+1]
				lhsVec1 := hwy.Set[T](lhsVal1)
				accum_lhs1_rhs0 = hwy.MulAdd(rhsVec0, lhsVec1, accum_lhs1_rhs0)
				accum_lhs1_rhs1 = hwy.MulAdd(rhsVec1, lhsVec1, accum_lhs1_rhs1)

				// Row 2
				lhsVal2 := packedLHS[idxLHS+2]
				lhsVec2 := hwy.Set[T](lhsVal2)
				accum_lhs2_rhs0 = hwy.MulAdd(rhsVec0, lhsVec2, accum_lhs2_rhs0)
				accum_lhs2_rhs1 = hwy.MulAdd(rhsVec1, lhsVec2, accum_lhs2_rhs1)

				// Row 3
				lhsVal3 := packedLHS[idxLHS+3]
				lhsVec3 := hwy.Set[T](lhsVal3)
				accum_lhs3_rhs0 = hwy.MulAdd(rhsVec0, lhsVec3, accum_lhs3_rhs0)
				accum_lhs3_rhs1 = hwy.MulAdd(rhsVec1, lhsVec3, accum_lhs3_rhs1)

				idxLHS += lhsKernelRows
			}

			// ---------------------------------------------------------
			// 4. Write Back to Output
			// ---------------------------------------------------------
			outputIdx0 := outputRowStart*outputStride + outputColStart
			outputIdx1 := outputIdx0 + params.RHSPanelCrossSize
			outputIdx2 := outputIdx0 + 2*params.RHSPanelCrossSize
			outputIdx3 := outputIdx0 + 3*params.RHSPanelCrossSize

			hwy.Store(accum_lhs0_rhs0, packedOutput[outputIdx0:])
			hwy.Store(accum_lhs0_rhs1, packedOutput[outputIdx0+numLanes:])
			hwy.Store(accum_lhs1_rhs0, packedOutput[outputIdx1:])
			hwy.Store(accum_lhs1_rhs1, packedOutput[outputIdx1+numLanes:])
			hwy.Store(accum_lhs2_rhs0, packedOutput[outputIdx2:])
			hwy.Store(accum_lhs2_rhs1, packedOutput[outputIdx2+numLanes:])
			hwy.Store(accum_lhs3_rhs0, packedOutput[outputIdx3:])
			hwy.Store(accum_lhs3_rhs1, packedOutput[outputIdx3+numLanes:])
		}
	}
}
