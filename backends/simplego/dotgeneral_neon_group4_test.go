// Copyright 2025 The GoMLX Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !noasm && arm64

package simplego

import (
	"math/rand"
	"testing"
)

// TestDotProductGroup4NEON_Padding tests the kernel with padding (zeros)
// to mimic the behavior in DotGeneral_large_version where valid data < blockDim.
func TestDotProductGroup4NEON_Padding(t *testing.T) {
	if !hasNEON {
		t.Skip("NEON not available")
	}

	blockDim := 128
	validSize := 32 // Like LLM_1 case

	// Setup data
	stride := blockDim
	// RHS: 4 vectors, spaced by stride (blockDim)
	// Total size: 4 * blockDim (since we access rhs[0], rhs[blockDim], etc.)
	// But wait, if we access rhs[0..127] and rhs[128..255].
	// We need buffer of size 4 * blockDim.
	rhsData := make([]float32, 4*blockDim)
	lhsData := make([]float32, blockDim)

	// Fill valid data with random values
	rng := rand.New(rand.NewSource(42))
	
	for i := 0; i < validSize; i++ {
		lhsData[i] = rng.Float32()
		// Fill 4 RHS vectors
		rhsData[i] = rng.Float32()              // Vec 0
		rhsData[i+stride] = rng.Float32()       // Vec 1
		rhsData[i+stride*2] = rng.Float32()     // Vec 2
		rhsData[i+stride*3] = rng.Float32()     // Vec 3
	}
	// Remaining 32..127 are 0.0 by make() default.

	// Calculate expected
	var exp0, exp1, exp2, exp3 float32
	for i := 0; i < validSize; i++ { // Only sum valid part
		v := lhsData[i]
		exp0 += v * rhsData[i]
		exp1 += v * rhsData[i+stride]
		exp2 += v * rhsData[i+stride*2]
		exp3 += v * rhsData[i+stride*3]
	}
	// Summing the padding (0*0) changes nothing.

	output := make([]float32, 4)
	// Run Kernel
	s0, s1, s2, s3 := dotProductInnerLoopNEON(
		lhsData, rhsData, output,
		0, 0, 0, blockDim)

	// Check
	epsilon := float32(1e-4)
	if diff := s0 - exp0; diff > epsilon || diff < -epsilon { t.Errorf("Sum0 mismatch: got %f, want %f", s0, exp0) }
	if diff := s1 - exp1; diff > epsilon || diff < -epsilon { t.Errorf("Sum1 mismatch: got %f, want %f", s1, exp1) }
	if diff := s2 - exp2; diff > epsilon || diff < -epsilon { t.Errorf("Sum2 mismatch: got %f, want %f", s2, exp2) }
	if diff := s3 - exp3; diff > epsilon || diff < -epsilon { t.Errorf("Sum3 mismatch: got %f, want %f", s3, exp3) }
}