// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package bfloat16 is an alias to compute/dtypes/bfloat16 package. It's here for historical reasons.
//
// Deprecated: use the [github.com/gomlx/compute/dtypes/bfloat16] package instead.
package bfloat16

import "github.com/gomlx/compute/dtypes/bfloat16"

// BFloat16 (brain floating point) floating-point format is a computer number format occupying 16 bits in
// computer memory.
//
// Deprecated: it's just an alias to [bfloat16.BFloat16], use that instead.
type BFloat16 = bfloat16.BFloat16

// FromFloat32 converts a float32 to a BFloat16.
//
// Deprecated: use [bfloat16.FromFloat32] instead.
func FromFloat32(x float32) BFloat16 {
	return bfloat16.FromFloat32(x)
}

// FromFloat64 converts a float64 to a BFloat16.
//
// Deprecated: use [bfloat16.FromFloat64] instead.
func FromFloat64(x float64) BFloat16 {
	return bfloat16.FromFloat64(x)
}

// FromBits convert an uint16 to a BFloat16.
//
// Deprecated: use [bfloat16.FromBits] instead.
func FromBits(uint16 uint16) BFloat16 {
	return bfloat16.FromBits(uint16)
}

// Inf returns a BFloat16 with an infinity value with the specified sign.
//
// Deprecated: use [bfloat16.Inf] instead.
func Inf(sign int) BFloat16 {
	return bfloat16.Inf(sign)
}

// NaN returns a BFloat16 with a NaN value.
//
// Deprecated: use [bfloat16.NaN] instead.
func NaN() BFloat16 {
	return bfloat16.NaN()
}

// SmallestNonzero is the smallest nonzero denormal value for bfloat16.
//
// Deprecated: use [bfloat16.SmallestNonzero] instead.
const SmallestNonzero = bfloat16.SmallestNonzero
