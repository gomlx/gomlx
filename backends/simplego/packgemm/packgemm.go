// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package packgemm

import (
	"slices"

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

// GoroutineStarter is a function that starts a goroutine, if available from the global pool.
// It returns false if no goroutine was started.
type GoroutineStarter func(work func()) bool

// Block/packs parameters for current architecture.
type CacheParams struct {
	LHSL1KernelRows int // or Mr: number of lhs kernel rows going to registers.
	RHSL1KernelCols int // or Nr: Register Block Width

	ContractingPanelSize int // Kc: L1 Block Depth
	LHSL2PanelCrossSize  int // Mc: L2 Block Height
	RHSL3PanelCrossSize  int // Nc: L3 Block Width
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
	// Float32Params is set by the architecture files.
	Float32Params CacheParams

	// DTypeToGEMM is a map of DType to GEMM function.
	// Used for registration, use the generic GEMM[TInput, TOutput] to actually call it.
	DTypeToGEMM = make(map[DTypePair][]GEMMRegistration, 100)
)

// HasDTypeSupport returns true if a GEMM function is registered for the given dtypes.
func HasDTypeSupport(input, output dtypes.DType) bool {
	return len(DTypeToGEMM[DTypePair{input, output}]) > 0
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
		bufAllocFn BufAllocFn[TInput], bufReleaseFn BufReleaseFn, starter GoroutineStarter) error,
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
	bufAllocFn BufAllocFn[TInput], bufReleaseFn BufReleaseFn, starter GoroutineStarter) error {
	dtypePair := GetDTypePair[TInput, TOutput]()
	gemmRegs := DTypeToGEMM[dtypePair]
	if len(gemmRegs) == 0 {
		return errors.Errorf("no GEMM function registered for dtypes input=%s, output=%s",
			dtypePair.Input, dtypePair.Output)
	}
	gemmFn, ok := gemmRegs[0].GEMMFn.(func(alpha, beta TOutput, lhsFlat, rhsFlat []TInput, batchSize,
		lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat []TOutput,
		bufAllocFn BufAllocFn[TInput], bufReleaseFn BufReleaseFn, starter GoroutineStarter) error)
	if !ok {
		return errors.Errorf("Registered GEMM function invalid for dtypes input=%s, output=%s!? This is a bug, we got"+
			"instead %T as the registered function",
			dtypePair.Input, dtypePair.Output, gemmRegs[0].GEMMFn)
	}
	return gemmFn(alpha, beta, lhsFlat, rhsFlat, batchSize,
		lhsCrossSize, rhsCrossSize, contractingSize, outputFlat,
		bufAllocFn, bufReleaseFn, starter)
}

// packRhs packs a [depth, width] block from RHS into packedRhs.
// It rearranges data into vertical strips of width Nr (rhsL1BlockCols).
// If the block is smaller than Nr, it ZERO-PADS.
func packRhs(src, dst []float32, rowStart, colStart, strideCol, depth, width, nr int) {
	dstIdx := 0
	// Iterate over strips of width nr
	for stripColIdx := 0; stripColIdx < width; stripColIdx += nr {
		// How many columns valid in this strip?
		validCols := min(nr, width-stripColIdx)

		// Iterate over rows (k)
		for row := range depth {
			srcRow := rowStart + row
			srcColBase := colStart + stripColIdx
			// Copy valid columns
			srcIdx := (srcRow * strideCol) + srcColBase
			copy(dst[dstIdx:], src[srcIdx:srcIdx+validCols])
			dstIdx += validCols
			// Zero-pad if strip is incomplete (edge of matrix)
			for c := validCols; c < nr; c++ {
				dst[dstIdx] = 0.0
				dstIdx++
			}
		}
	}
}

// packLhs packs a [lhsPanelHeight/lhsL1KernelRows, contractingPanelWidth, lhsL1KernelRows] "panel"
// (a block of size Mr x Kc) from LHS.
// It rearranges data into horizontal strips of height Mr (lhsL1BlockRows).
// packLhs(lhs, packedLhs, lhsPanelRowIdx, contractingPanelIdx, contractingSize, lhsPanelHeight, contractingPanelWidth,
// params.LHSL1KernelRows)
func packLhs(src, dst []float32, rowStart,
	colStart, rowStride, lhsPanelHeight, contractingPanelWidth, lhsL1KernelRows int) {
	dstIdx := 0
	// Iterate over strips of height mr
	for stripRowIdx := 0; stripRowIdx < lhsPanelHeight; stripRowIdx += lhsL1KernelRows {
		validRows := min(lhsL1KernelRows, lhsPanelHeight-stripRowIdx)

		// Iterate over columns (contracting size k), we want LHS to be traversed K-first in the kernel
		for col := range contractingPanelWidth {
			srcCol := colStart + col
			srcRowBase := rowStart + stripRowIdx

			// Copy valid "rows" (they are the last axis in the returned panel)
			for row := range validRows {
				srcIdx := ((srcRowBase + row) * rowStride) + srcCol
				dst[dstIdx] = src[srcIdx]
				dstIdx++
			}

			// Zero-pad
			for r := validRows; r < lhsL1KernelRows; r++ {
				dst[dstIdx] = 0.0
				dstIdx++
			}
		}
	}
}
