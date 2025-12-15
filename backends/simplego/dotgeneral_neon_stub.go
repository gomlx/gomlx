// Copyright 2025 The GoMLX Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build noasm || !arm64

package simplego

// dotProduct_neon stub for non-ARM64 platforms.
// Signature matches dotgeneral_neon_arm64.go for consistency.
func dotProduct_neon(aSlice, bSlice []float32, aIdx, bIdx int, n int64) float32 {
	return 0
}

// dotProductInnerLoopNEON stub for non-ARM64 platforms
func dotProductInnerLoopNEON(lhsFlat, rhsFlat, outputFlat []float32,
	lhsIdx, rhsIdx, outputIdx, blockDim int) (sum0, sum1, sum2, sum3 float32) {
	// Should never be called since hasNEON will be false
	panic("NEON not available")
}
