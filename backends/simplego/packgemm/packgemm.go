// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package packgemm

// BufAllocFn is a function that allocates a buffer of type T, of the given size.
type BufAllocFn[T any] func(size int) (ref any, data []T)

// BufReleaseFn is a function that releases a buffer allocated with BufAllocFn.
type BufReleaseFn func(ref any)

// GoroutineStarter is a function that starts a goroutine, if available from the global pool.
// It returns false if no goroutine was started.
type GoroutineStarter func(work func()) bool

// Block/packs parameters for current architecture.
type CacheParams struct {
	lhsL1KernelRows int // or Mr: number of lhs kernel rows going to registers.
	rhsL1KernelCols int // or Nr: Register Block Width

	contractingPanelSize int // Kc: L1 Block Depth
	lhsL2PanelCrossSize  int // Mc: L2 Block Height
	rhsL3PanelCrossSize  int // Nc: L3 Block Width
}

var (
	// Set by the architecture files.
	DefaultCacheParams CacheParams

	// Float32 implements generic matrix multiplication for float32 inputs and outputs.
	// output = alpha * (lhs x rhs) + beta * output
	//
	// Check it is not nil in your platform before using.
	Float32 func(alpha, beta float32, lhsFlat, rhsFlat []float32, batchSize, lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat []float32,
		bufAllocFn BufAllocFn[float32], bufReleaseFn BufReleaseFn, starter GoroutineStarter)
)
