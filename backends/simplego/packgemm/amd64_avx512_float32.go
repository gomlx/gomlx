// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package packgemm

import (
	"runtime"
	"simd/archsimd"
	"sync"
	"unsafe"

	"github.com/gomlx/gomlx/pkg/support/xsync"
	"k8s.io/klog/v2"
)

var avx512Float32Params = CacheParams{
	LHSL1KernelRows:      16,   // Mr: Uses 4 ZMM registers for accumulation rows, this number must be a multiple of 4
	RHSL1KernelCols:      32,   // Nr: Uses 2 ZMM registers for accumulation cols
	PanelContractingSize: 512,  // Kc: A strip fits in L1 cache
	LHSL2PanelCrossSize:  512,  // Mc: Fits in L2 cache (multiple of LHSL1KernelRows)
	RHSL3PanelCrossSize:  4096, // Nc: Fits in L3 cache (multiple of RHSL1KernelCols)
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
	bufAllocFn BufAllocFn[float32], bufReleaseFn BufReleaseFn, starter GoroutineStarter) error {

	avx512WarningOnce.Do(func() {
		klog.Infof("AVX512 GEMM (General Matrix Multiplication) algorithm still experimental!")
	})

	// 1. Resolve Strides
	lhsBatchStride := lhsCrossSize * contractingSize
	rhsBatchStride := contractingSize * rhsCrossSize
	outputBatchStride := lhsCrossSize * rhsCrossSize

	// 2. Determine "Quantum" Size (Splitting Strategy)
	// We want enough tasks to fill the machine, but not so many that we trash the cache/packing.
	// Target parallelism: Number of physical cores (or default GOMAXPROCS).
	targetParallelism := runtime.GOMAXPROCS(0)

	// Default: Treat the whole N (rhsCrossSize) as one quantum.
	// This minimizes redundant packing of Matrix A.
	rhsColSplitSize := rhsCrossSize

	// Heuristic: If we don't have enough batches to saturate the cores,
	// we MUST split the columns to get parallelism.
	if batchSize < targetParallelism {
		// Example: 16 cores, Batch=2. We want ~8 tasks per batch.
		// tasksPerBatch := targetParallelism / batchSize
		// This is a rough estimate.

		// Ensure we don't split too small (avoid overhead for tiny strips).
		// Minimum strip size e.g., 256 columns or the L1 Block Size.
		minSplit := 256
		if rhsColSplitSize > minSplit {
			neededSplits := (targetParallelism + batchSize - 1) / batchSize
			calculatedSplit := rhsCrossSize / neededSplits

			// Clamp to valid range
			if calculatedSplit < minSplit {
				calculatedSplit = minSplit
			}

			// Align to 32 (Nr) for SIMD efficiency
			rhsColSplitSize = (calculatedSplit + 31) &^ 31
		}
	}

	// 3. The Work Loop
	// We iterate sequentially. If the pool is full, we do the work ourselves.
	wg := xsync.NewDynamicWaitGroup() // Control workers started.
	for batchIdx := range batchSize {
		// Capture batch offsets once per batch
		batchLhs := lhsFlat[batchIdx*lhsBatchStride : (batchIdx+1)*lhsBatchStride]
		batchRhs := rhsFlat[batchIdx*rhsBatchStride : (batchIdx+1)*rhsBatchStride]
		batchOutput := outputFlat[batchIdx*outputBatchStride : (batchIdx+1)*outputBatchStride]

		// Iterate over Column Strips -- a "chunk" of work that can be parallelized.
		for colStart := 0; colStart < rhsCrossSize; colStart += rhsColSplitSize {
			colEnd := colStart + rhsColSplitSize
			if colEnd > rhsCrossSize {
				colEnd = rhsCrossSize
			}

			// Define the task closure
			wg.Add(1)
			task := func() {
				avx512Float32GemmChunk(
					alpha, beta,
					batchLhs, batchRhs, batchOutput,
					lhsCrossSize, rhsCrossSize, contractingSize,
					avx512Float32Params, colStart, colEnd,
					bufAllocFn, bufReleaseFn,
				)
				wg.Done()
			}

			// 4. Try to Offload
			if !starter(task) {
				// Pool is busy/full.
				// Execute immediately on this thread to prevent starvation.
				task()
			}
		}
	}
	wg.Wait()
	return nil
}

// avx512Float32GemmChunk performs the 5-loop GotoBLAS algorithm on a slice of a single batch matrix.
func avx512Float32GemmChunk(
	alpha, beta float32,
	lhs, rhs, output []float32,
	lhsCrossSize, rhsCrossSize, contractingSize int,
	params CacheParams, colStart, colEnd int,
	bufAllocFn BufAllocFn[float32], bufReleaseFn BufReleaseFn,
) {
	// fmt.Printf("gemmChunk(colStart=%d, colEnd=%d)\n", colStart, colEnd)
	packedLhsRef, packedLhs := bufAllocFn(params.LHSL2PanelCrossSize * params.PanelContractingSize)
	packedRhsRef, packedRhs := bufAllocFn(params.PanelContractingSize * params.RHSL3PanelCrossSize)
	defer func() {
		bufReleaseFn(packedLhsRef)
		bufReleaseFn(packedRhsRef)
	}()

	// Loop 5 (jc): Tiling N (Output Columns) - Fits in L3
	// Iterates over the assigned strip [colStart, colEnd) in chunks of rhsL3PanelCrossSize.
	for rhsPanelColIdx := colStart; rhsPanelColIdx < colEnd; rhsPanelColIdx += params.RHSL3PanelCrossSize {

		// The width of the current panel is limited by the L3 block size (Nc)
		// AND the end of our assigned chunk (colEnd).
		rhsPanelWidth := min(params.RHSL3PanelCrossSize, colEnd-rhsPanelColIdx)

		// Loop 4 (p): Tiling K (Depth) - Fits in L1
		// Iterates over the contracting dimension in chunks of contractingPanelSize
		for contractingPanelIdx := 0; contractingPanelIdx < contractingSize; contractingPanelIdx += params.PanelContractingSize {
			// fmt.Printf("- contractingPanelIdx=%d\n", contractingPanelIdx)
			effectiveBeta := beta
			if contractingPanelIdx > 0 {
				// We only apply (multiply) the current output by beta the first time we touch the output buffer
				// at this panel, after that the output is already accumulating the results of the matmul.
				effectiveBeta = 1
			}
			contractingPanelWidth := min(params.PanelContractingSize, contractingSize-contractingPanelIdx)

			// ---------------------------------------------------------
			// PACK RHS (Bit) -> ~B
			// We pack a [contractingPanelWidth, rhsPanelWidth] block of RHS into contiguous memory.
			// Format: Vertical strips of width rhsL1KernelCols (Nr).
			// ---------------------------------------------------------
			packRHS(rhs, packedRhs, contractingPanelIdx, rhsPanelColIdx, rhsCrossSize, contractingPanelWidth, rhsPanelWidth, params.RHSL1KernelCols)

			// Loop 3 (ic): Tiling M (Output Rows) - Fits in L2
			// Iterates over the LHS height in chunks of lhsL2PanelCrossSize
			for lhsPanelRowIdx := 0; lhsPanelRowIdx < lhsCrossSize; lhsPanelRowIdx += params.LHSL2PanelCrossSize {
				lhsPanelHeight := min(params.LHSL2PanelCrossSize, lhsCrossSize-lhsPanelRowIdx)

				// -----------------------------------------------------
				// PACK LHS (Ait) -> ~A
				// We pack a [lhsPanelHeight, contractingPanelWidth] block of LHS into contiguous memory.
				// Format: Horizontal strips of height lhsL1KernelRows (Mr).
				// -----------------------------------------------------
				packLHS(lhs, packedLhs, lhsPanelRowIdx, contractingPanelIdx, contractingSize, lhsPanelHeight, contractingPanelWidth, params.LHSL1KernelRows)

				// Loop 2 (jr): Micro-Kernel Columns (Nr == rhsL1BlockCols)
				for microColIdx := 0; microColIdx < rhsPanelWidth; microColIdx += params.RHSL1KernelCols {
					// Actual width to process (might be < Nr at matrix edge)
					microKernelActualWidth := min(params.RHSL1KernelCols, rhsPanelWidth-microColIdx)

					// Loop 1 (ir): Micro-Kernel Rows (Mr == lhsL1BlockRows)
					for microRowIdx := 0; microRowIdx < lhsPanelHeight; microRowIdx += params.LHSL1KernelRows {
						microKernelActualHeight := min(params.LHSL1KernelRows, lhsPanelHeight-microRowIdx)

						// ---------------------------------------------
						// MICRO KERNEL
						// Computes a [Mr, Nr] tile of Output
						// ---------------------------------------------

						// Calculate pointers into packed buffers.
						// PackRhs is organized in strips of Nr. We need the current strip (microColIdx / Nr).
						// Each strip has size (contractingPanelWidth * Nr).
						offsetRhs := (microColIdx / params.RHSL1KernelCols) * (contractingPanelWidth * params.RHSL1KernelCols)

						// PackLhs is organized in strips of Mr. We need the current strip (microRowIdx / Mr).
						// Each strip has size (contractingPanelWidth * Mr).
						offsetLhs := (microRowIdx / params.LHSL1KernelRows) * (contractingPanelWidth * params.LHSL1KernelRows)

						// Output index calculation (Absolute coordinates)
						outputRow := lhsPanelRowIdx + microRowIdx
						outputCol := rhsPanelColIdx + microColIdx

						avx512Float32MicroKernel(
							contractingPanelWidth,
							alpha, effectiveBeta,
							packedLhs[offsetLhs:],
							packedRhs[offsetRhs:],
							output,
							outputRow, outputCol,
							rhsCrossSize,           // Stride of Output
							params.LHSL1KernelRows, // Kernel size.
							microKernelActualHeight, microKernelActualWidth,
						)
					}
				}
			}
		}
	}
}

// avx512Float32MicroKernel computes a [lhaL1KernelRows, rhsKernelCols] tile.
//
// It uses simulated SIMD logic based on the user's pseudo-code.
// This assumes lhsCrossSize multiple of 4 (4 registers) and rhsKernelCols=32 (2 registers),
// so it uses 4x2 = 8 total accumulator registers.
// Go only seems to make use of the first 16 registers (AVX512 has 32 in total though).
//
// lhsActiveRows and rhsActiveCols are the number of rows/cols that should be
// written back to the output. Notice the inputs (ptrLhs and ptrRhs) are padded.
//
// -ptrLhs: is the slice of the packed LHS, organized as [contractingPanelSize, lhsL1KernelRows], zero padded.
// -ptrRhs: is the slice of the packed RHS, organized as [contractingPanelSize, rhsL1KernelCols==32], zero padded.
func avx512Float32MicroKernel(
	contractingLen int,
	alpha, beta float32,
	ptrLhs, ptrRhs []float32, // Packed Buffers
	output []float32, // Output Matrix
	outputRowStart, outputColStart int, // Coordinates
	outputStride int,
	lhsKernelRows int,
	lhsActiveRows, rhsActiveCols int, // Active rows/cols (for edge handling)
) {
	// fmt.Printf("\t- microKernelFloat32(beta=%g, contractingLen=%d, lhsActiveRows=%d, rhsActiveCols=%d)\n",
	// 	beta, contractingLen, lhsActiveRows, rhsActiveCols)

	// ---------------------------------------------------------
	// 1. Write Back Setup
	// ---------------------------------------------------------
	betaBroadcast := archsimd.BroadcastFloat32x16(beta)
	cols0Bits := min(16, rhsActiveCols)
	cols1Bits := min(16, max(0, rhsActiveCols-16))
	maskForCols0 := archsimd.Mask32x16FromBits(uint16(uint64(1<<cols0Bits) - 1))
	maskForCols1 := archsimd.Mask32x16FromBits(uint16(uint64(1<<cols1Bits) - 1))

	// writeRow is used in the end to store the accumulator registers back into the output.
	writeRow := func(row int, acc0, acc1 archsimd.Float32x16) {
		outputIdx := (outputRowStart+row)*outputStride + outputColStart
		if rhsActiveCols >= 16 {
			// Store first strip of columns.
			outSlice := output[outputIdx : outputIdx+16]
			curValue := archsimd.LoadFloat32x16Slice(outSlice)
			curValue = curValue.Mul(betaBroadcast)
			curValue = curValue.Add(acc0)
			curValue.StoreSlice(outSlice)

			// Store second strip of columns.
			if rhsActiveCols > 16 {
				if rhsActiveCols == 32 {
					// Full second strip of columns.
					outSlice = output[outputIdx+16 : outputIdx+32]
					curValue = archsimd.LoadFloat32x16Slice(outSlice)
					curValue = curValue.Mul(betaBroadcast)
					curValue = curValue.Add(acc1)
					curValue.StoreSlice(outSlice)
				} else {
					// Partial second strip of columns.
					outSlice := output[outputIdx+16 : outputIdx+rhsActiveCols]
					curValue = archsimd.LoadFloat32x16SlicePart(outSlice)
					curValue = curValue.Mul(betaBroadcast)
					curValue = curValue.Add(acc1)
					curValue.StoreMasked(castToArray16(&outSlice[0]), maskForCols1)
				}
			}

		} else {
			// Store partial first strip of columns.
			outSlice := output[outputIdx : outputIdx+rhsActiveCols]
			curValue := archsimd.LoadFloat32x16SlicePart(outSlice)
			curValue = curValue.Mul(betaBroadcast)
			curValue = curValue.Add(acc0)
			curValue.StoreMasked(castToArray16(&outSlice[0]), maskForCols0)
		}
	}

	for rowOffset := 0; rowOffset < lhsActiveRows; rowOffset += 4 {
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
		idxLhs := rowOffset
		idxRhs := 0
		for range contractingLen {
			// Load RHS (Broadcasting/Streaming)
			rhsVec0 := archsimd.LoadFloat32x16Slice(ptrRhs[idxRhs : idxRhs+16])
			rhsVec1 := archsimd.LoadFloat32x16Slice(ptrRhs[idxRhs+16 : idxRhs+32])
			idxRhs += 32

			// Row rowOffset+0
			lhsVal0 := ptrLhs[idxLhs+0]
			lhsVec0 := archsimd.BroadcastFloat32x16(lhsVal0)
			accum_lhs0_rhs0 = rhsVec0.MulAdd(lhsVec0, accum_lhs0_rhs0)
			accum_lhs0_rhs1 = rhsVec1.MulAdd(lhsVec0, accum_lhs0_rhs1)

			// Row rowOffset+1
			lhsVal1 := ptrLhs[idxLhs+1]
			lhsVec1 := archsimd.BroadcastFloat32x16(lhsVal1)
			accum_lhs1_rhs0 = rhsVec0.MulAdd(lhsVec1, accum_lhs1_rhs0)
			accum_lhs1_rhs1 = rhsVec1.MulAdd(lhsVec1, accum_lhs1_rhs1)

			// Row rowOffset+2
			lhsVal2 := ptrLhs[idxLhs+2]
			lhsVec2 := archsimd.BroadcastFloat32x16(lhsVal2)
			accum_lhs2_rhs0 = rhsVec0.MulAdd(lhsVec2, accum_lhs2_rhs0)
			accum_lhs2_rhs1 = rhsVec1.MulAdd(lhsVec2, accum_lhs2_rhs1)

			// Row rowOffset+3
			lhsVal3 := ptrLhs[idxLhs+3]
			lhsVec3 := archsimd.BroadcastFloat32x16(lhsVal3)
			accum_lhs3_rhs0 = rhsVec0.MulAdd(lhsVec3, accum_lhs3_rhs0)
			accum_lhs3_rhs1 = rhsVec1.MulAdd(lhsVec3, accum_lhs3_rhs1)

			idxLhs += lhsKernelRows
		}

		// Apply alpha factor.
		if alpha != 1 {
			alphaBroadcast := archsimd.BroadcastFloat32x16(alpha)
			accum_lhs0_rhs0 = accum_lhs0_rhs0.Mul(alphaBroadcast)
			accum_lhs0_rhs1 = accum_lhs0_rhs1.Mul(alphaBroadcast)
			accum_lhs1_rhs0 = accum_lhs1_rhs0.Mul(alphaBroadcast)
			accum_lhs1_rhs1 = accum_lhs1_rhs1.Mul(alphaBroadcast)
			accum_lhs2_rhs0 = accum_lhs2_rhs0.Mul(alphaBroadcast)
			accum_lhs2_rhs1 = accum_lhs2_rhs1.Mul(alphaBroadcast)
			accum_lhs3_rhs0 = accum_lhs3_rhs0.Mul(alphaBroadcast)
			accum_lhs3_rhs1 = accum_lhs3_rhs1.Mul(alphaBroadcast)
		}

		// ---------------------------------------------------------
		// 4. Write Back to Output
		// ---------------------------------------------------------
		remainingRows := lhsActiveRows - rowOffset
		switch {
		case remainingRows > 3:
			writeRow(rowOffset+3, accum_lhs3_rhs0, accum_lhs3_rhs1)
			fallthrough
		case remainingRows > 2:
			writeRow(rowOffset+2, accum_lhs2_rhs0, accum_lhs2_rhs1)
			fallthrough
		case remainingRows > 1:
			writeRow(rowOffset+1, accum_lhs1_rhs0, accum_lhs1_rhs1)
			fallthrough
		case remainingRows > 0:
			writeRow(rowOffset+0, accum_lhs0_rhs0, accum_lhs0_rhs1)
		}
	}
}

func castToArray16[T float32](ptr *T) *[16]T {
	return (*[16]T)(unsafe.Pointer(ptr))
}
