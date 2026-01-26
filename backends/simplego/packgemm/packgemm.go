// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// packgemm implements General Matrix Multiplication (GEMM) using a tuned/slightly
// changed GotoBLAS' GEMM algorithm, using packing of slices of the input matrices
// into temporary panels
//
// It also include parallelization and it uses temporary packed output buffers.
//
// Finally, it uses go-highway to generate versions for the different SIMD architectures.
package packgemm

import (
	"slices"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/gomlx/gomlx/internal/workerspool"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/pkg/errors"
)

// Generate the GEMMDynamic dispatcher.
//go:generate go run ../../../internal/cmd/packgemm_generator

// BufAllocFn is a function that allocates a buffer (a slice) of type T, of the given size.
type BufAllocFn[T any] func(size int) (ref any, data []T)

// BufAllocAnyFn is a function that allocates a buffer (a slice) of some pre-agreed type.
type BufAllocAnyFn func(size int) (ref any, data any)

// BufReleaseFn is a function that releases a buffer allocated with BufAllocFn.

type BufReleaseFn func(ref any)

// Block/packs parameters for current architecture.
type CacheParams struct {
	LHSL1KernelRows int // or Mr: number of lhs kernel rows going to registers.
	RHSL1KernelCols int // or Nr: Register Block Width

	PanelContractingSize int // Kc: LHS cols or RHS rows to fit in L2/L3
	LHSPanelCrossSize    int // Mc: L2 rows
	RHSPanelCrossSize    int // Nc: L3 cols
}

// Priority is used to determine the priority of a gemm version, when setting the
// DTypeToGEMM map.
type Priority int

const (
	PriorityBase      Priority = 0
	PriorityDType     Priority = 10 // Version for a specific dtype (instead of generic).
	PrioritySIMD      Priority = 20 // Version specialized for a SIMD architecture.
	PriorityDTypeSIMD Priority = 30 // Version specialized for a dtype and SIMD architecture.
)

// DTypePair represents the input/output types.
type DTypePair struct {
	Input, Output dtypes.DType
}

// GetDTypePair returns the DTypePair for the given types.
func GetDTypePair[TInput, TOutput dtypes.Supported]() DTypePair {
	return DTypePair{Input: dtypes.FromGenericsType[TInput](), Output: dtypes.FromGenericsType[TOutput]()}
}

var (
	// DTypeToGEMM is a map of DType to GEMM function.
	// Used for registration, use the generic GEMM[TInput, TOutput] to actually call it.
	DTypeToGEMM = make(map[DTypePair][]GEMMRegistration, 100)

	forceVariant Variant = VariantNone
)

// Variant of algorithms: usually just one for small matrices and the other for large matrices.
type Variant int

const (
	VariantNone Variant = iota
	VariantSmall
	VariantLarge
)

// HasDTypeSupport returns true if a GEMM function is registered for the given dtypes.
func HasDTypeSupport(input, output dtypes.DType) bool {
	return len(DTypeToGEMM[DTypePair{input, output}]) > 0
}

// ForceVariant forces the use of the small/large variant.
// Used for testing only.
func ForceVariant(v Variant) {
	forceVariant = v
}

// GEMMRegistration is a registration of a GEMM function for the given dtype pair.
type GEMMRegistration struct {
	Name      string
	DTypePair DTypePair
	GEMMFn    any // Typed GEMM function
	Priority  Priority
	Params    *CacheParams
}

// RegisterGEMM registers a GEMM function for the given dtypes with the given priority.
// If the priority is lower than the one already registered, it does nothing.
func RegisterGEMM[TInput, TOutput dtypes.Supported](
	name string,
	gemmFn func(alpha, beta TOutput, lhsFlat, rhsFlat []TInput, batchSize,
		lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat []TOutput,
		bufAllocFn BufAllocFn[TInput], bufReleaseFn BufReleaseFn, pool *workerspool.Pool) error,
	params *CacheParams,
	priority Priority) {
	dtypePair := GetDTypePair[TInput, TOutput]()
	DTypeToGEMM[dtypePair] = append(DTypeToGEMM[dtypePair], GEMMRegistration{
		Name:      name,
		DTypePair: dtypePair,
		GEMMFn:    gemmFn,
		Params:    params,
		Priority:  priority,
	})
	// Sort the GEMM registrations by priority, highest priority first.
	slices.SortFunc(DTypeToGEMM[dtypePair], func(a, b GEMMRegistration) int {
		return int(b.Priority - a.Priority)
	})
}

// GEMM[TInput, TOutput dtypes.DType] implements the matrix multiplication for the given dtypes.
// It returns an error if a GEMM function is not registered for the given dtypes.
func GEMM[TInput, TOutput dtypes.Supported](alpha, beta TOutput, lhsFlat, rhsFlat []TInput, batchSize,
	lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat []TOutput,
	bufAllocFn BufAllocFn[TInput], bufReleaseFn BufReleaseFn, pool *workerspool.Pool) error {
	dtypePair := GetDTypePair[TInput, TOutput]()
	gemmRegs := DTypeToGEMM[dtypePair]
	if len(gemmRegs) == 0 {
		return errors.Errorf("no GEMM function registered for dtypes input=%s, output=%s",
			dtypePair.Input, dtypePair.Output)
	}
	gemmFn, ok := gemmRegs[0].GEMMFn.(func(alpha, beta TOutput, lhsFlat, rhsFlat []TInput, batchSize,
		lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat []TOutput,
		bufAllocFn BufAllocFn[TInput], bufReleaseFn BufReleaseFn, pool *workerspool.Pool) error)
	if !ok {
		return errors.Errorf("Registered GEMM function invalid for dtypes input=%s, output=%s!? This is a bug, we got"+
			"instead %T as the registered function",
			dtypePair.Input, dtypePair.Output, gemmRegs[0].GEMMFn)
	}
	return gemmFn(alpha, beta, lhsFlat, rhsFlat, batchSize,
		lhsCrossSize, rhsCrossSize, contractingSize, outputFlat,
		bufAllocFn, bufReleaseFn, pool)
}

// packRHS packs a slice of size [contractingRows, rhsCols] block from RHS into
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
func packRHS[T hwy.Floats](src, dst []T, srcRowStart, srcColStart, srcStrideCol,
	contractingRows, rhsCols, RHSL1KernelCols int) {
	dstIdx := 0
	// Iterate over strips of width nr
	for stripColIdx := 0; stripColIdx < rhsCols; stripColIdx += RHSL1KernelCols {
		// How many columns valid in this strip?
		validCols := min(RHSL1KernelCols, rhsCols-stripColIdx)

		// Iterate over rows (k)
		srcColBase := srcColStart + stripColIdx
		for row := range contractingRows {
			srcRow := srcRowStart + row
			srcIdx := (srcRow * srcStrideCol) + srcColBase
			// Copy valid columns
			copy(dst[dstIdx:], src[srcIdx:srcIdx+validCols])
			dstIdx += validCols
			// Zero-pad if strip is incomplete (edge of matrix)
			for c := validCols; c < RHSL1KernelCols; c++ {
				dst[dstIdx] = T(0)
				dstIdx++
			}
		}
	}
}

// packLHS packs a slice of size [lhsRows, contractingCols] block from LHS into
// a [ceil(lhsRows/lhsL1KernelRows), contractingCols, lhsL1KernelRows] "panel"
// (a block of size Mr x Kc) from LHS.
// It rearranges data into horizontal strips of height Mr (lhsL1BlockRows).
//
// How it is called:
//
//	packLHS(lhs, packedLhs, lhsPanelRowIdx, contractingPanelIdx, contractingSize,
//		lhsPanelHeight, contractingPanelWidth,
//		params.LHSL1KernelRows)
func packLHS[T hwy.Floats](src, dst []T,
	srcRowStart, srcColStart, srcRowStride,
	lhsRows, contractingCols, lhsL1KernelRows int) {
	dstIdx := 0
	// Iterate over strips of height mr
	for stripRowIdx := 0; stripRowIdx < lhsRows; stripRowIdx += lhsL1KernelRows {
		validRows := min(lhsL1KernelRows, lhsRows-stripRowIdx)

		// Iterate over columns (contracting size k), we want LHS to be traversed K-first in the kernel
		for col := range contractingCols {
			srcCol := srcColStart + col
			srcRowBase := srcRowStart + stripRowIdx

			// Copy valid "rows" (they are the last axis in the returned panel)
			for row := range validRows {
				srcIdx := ((srcRowBase + row) * srcRowStride) + srcCol
				dst[dstIdx] = src[srcIdx]
				dstIdx++
			}

			// Zero-pad
			for r := validRows; r < lhsL1KernelRows; r++ {
				dst[dstIdx] = T(0)
				dstIdx++
			}
		}
	}
}

// workItem is used when parallelizing the GEMM into batch/lhs/rhs slices.
type workItem struct {
	batchStart, batchEnd,
	lhsRowStart, lhsRowEnd,
	rhsColStart, rhsColEnd int
}

// feedWorkItems split the GEMM tasks is "workItems" optimized (as large as possible, prioritizing whole batch items)
// for maxWokers (>=1).
// It closes workChan on exit.
//
// feedWorkItems is typically called on a separate goroutine, and it uses almost no CPU.
func feedWorkItems(
	batchSize, lhsCrossSize, rhsCrossSize int,
	params *CacheParams,
	maxWorkers int,
	workChan chan<- workItem) {
	defer func() {
		// Invariant: it closes the channel on exit.
		close(workChan)
	}()
	if batchSize >= 2*maxWorkers {
		// Split the work on the batch dimension only.
		batchStep := batchSize / maxWorkers
		for batchIdx := 0; batchIdx < batchSize; batchIdx += batchStep {
			workChan <- workItem{
				batchIdx, batchIdx + min(batchStep, batchSize-batchIdx),
				0, lhsCrossSize,
				0, rhsCrossSize}
		}
		return
	}

	// First maxWorkers batch examples are handled as one at a time:
	batchIdx := 0
	if batchSize >= maxWorkers {
		for ; batchIdx < maxWorkers; batchIdx++ {
			workChan <- workItem{
				batchIdx, batchIdx + 1,
				0, lhsCrossSize,
				0, rhsCrossSize}
		}
	}

	// The remaining work is split into RHS or LHS slices.
	batchCountRemaining := batchSize - batchIdx
	if batchCountRemaining == 0 {
		return // We are finished.
	}
	splitFactor := (maxWorkers + batchCountRemaining - 1) / batchCountRemaining
	if lhsCrossSize > rhsCrossSize {
		// Split on the LHS dimension, in multiples of LHSPanelCrossSize.
		lhsSplitSize := (lhsCrossSize + splitFactor - 1) / splitFactor
		lhsSplitSize = max(1, lhsSplitSize/params.LHSPanelCrossSize) * params.LHSPanelCrossSize
		batchStart := batchIdx
		for lhsRowIdx := 0; lhsRowIdx < lhsCrossSize; lhsRowIdx += lhsSplitSize {
			for batchIdx = batchStart; batchIdx < batchSize; batchIdx++ {
				workChan <- workItem{
					batchIdx, batchIdx + 1,
					lhsRowIdx, lhsRowIdx + min(lhsSplitSize, lhsCrossSize-lhsRowIdx),
					0, rhsCrossSize}
			}
		}
	} else {
		// Split on the RHS dimension, in multiples of RHSPanelCrossSize.
		rhsSplitSize := (rhsCrossSize + splitFactor - 1) / splitFactor
		rhsSplitSize = max(1, rhsSplitSize/params.RHSPanelCrossSize) * params.RHSPanelCrossSize
		batchStart := batchIdx
		for rhsColIdx := 0; rhsColIdx < rhsCrossSize; rhsColIdx += rhsSplitSize {
			for batchIdx = batchStart; batchIdx < batchSize; batchIdx++ {
				workChan <- workItem{
					batchIdx, batchIdx + 1,
					0, lhsCrossSize,
					rhsColIdx, rhsColIdx + min(rhsSplitSize, rhsCrossSize-rhsColIdx)}
			}
		}
	}
}
