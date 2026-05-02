// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package float16 is an alias to compute/dtypes/float16 package. It's here for historical reasons.
//
// Deprecated: use the [github.com/gomlx/compute/dtypes/float16] package instead.
package float16

import "github.com/gomlx/compute/dtypes/float16"

// Float16 (half-precision floating-point) format is a computer number format occupying 16 bits in
// computer memory.
//
// Deprecated: it's just an alias to [float16.Float16], use that instead.
type Float16 = float16.Float16

// FromFloat32 converts a float32 to a Float16.
//
// Deprecated: use [float16.FromFloat32] instead.
func FromFloat32(x float32) Float16 {
	return float16.FromFloat32(x)
}

// FromFloat64 converts a float64 to a Float16.
//
// Deprecated: use [float16.FromFloat64] instead.
func FromFloat64(x float64) Float16 {
	return float16.FromFloat64(x)
}

// FromBits convert an uint16 to a Float16.
//
// Deprecated: use [float16.FromBits] instead.
func FromBits(uint16 uint16) Float16 {
	return float16.FromBits(uint16)
}

// Inf returns a Float16 with an infinity value with the specified sign.
//
// Deprecated: use [float16.Inf] instead.
func Inf(sign int) Float16 {
	return float16.Inf(sign)
}

// NaN returns a Float16 with a NaN value.
//
// Deprecated: use [float16.NaN] instead.
func NaN() Float16 {
	return float16.NaN()
}

// SmallestNonzero is the smallest nonzero denormal value for float16.
//
// Deprecated: use [float16.SmallestNonzero] instead.
const SmallestNonzero = float16.SmallestNonzero
