// Package bfloat16 is a trivial implementation for the bfloat16 type,
// based on https://github.com/x448/float16 and the pending issue in
// https://github.com/x448/float16/issues/22
package bfloat16

import (
	"math"
	"strconv"
)

// BFloat16 (brain floating point)[1][2] floating-point format is a computer number format occupying 16 bits in
// computer memory; it represents a wide dynamic range of numeric values by using a floating radix point.
// This format is a shortened (16-bit) version of the 32-bit IEEE 754 single-precision floating-point format
// (binary32) with the intent of accelerating machine learning and near-sensor computing.
//
// bfloat16 and patents:
//
//   - https://en.wikipedia.org/wiki/Tensor_Processing_Unit#Lawsuit
//   - https://www.reddit.com/r/MachineLearning/comments/193zpyi/d_does_patent_lawsuit_against_googles_tpu_imperil/
type BFloat16 uint16

func (f BFloat16) Float32() float32 {
	return math.Float32frombits(uint32(f) << 16)
}

// FromFloat32 converts a float32 to a BFloat16.
func FromFloat32(x float32) BFloat16 {
	return BFloat16(math.Float32bits(x) >> 16)
}

// FromFloat64 converts a float32 to a BFloat16.
func FromFloat64(x float64) BFloat16 {
	return FromFloat32(float32(x))
}

// FromBits convert an uint16 to a BFloat16.
func FromBits(uint16 uint16) BFloat16 {
	return BFloat16(uint16)
}

// Bits convert BFloat16 to an uint16.
func (f BFloat16) Bits() uint16 {
	return uint16(f)
}

// String implements fmt.Stringer, and prints a float representation of the BFloat16.
func (f BFloat16) String() string {
	return strconv.FormatFloat(float64(f.Float32()), 'f', -1, 32)
}

// Inf returns a BFloat16 with an infinity value with the specified sign.
// A sign >= returns positive infinity.
// A sign < 0 returns negative infinity.
func Inf(sign int) BFloat16 {
	return FromFloat32(float32(math.Inf(sign)))
}

// SmallestNonzero is the smallest nonzero denormal value for bfloat16 (9.1835e-41).
// It's the float16 equivalent for [math.SmallestNonzeroFloat32] and [math.SmallestNonzeroFloat64].
// For context, [math.SmallestNonzeroFloat32] used the formula 1 / 2**(127 - 1 + 23) to produce
// the smallest denormal value for float32 (1.401298464324817070923729583289916131280e-45).
// The equivalent formula for float16 is 1 / 2**(15 - 1 + 10). We use Float16(0x0001) to compile as const.
const SmallestNonzero = BFloat16(0x0001) // 5.9604645e-08 (effectively 0x1p-14 * 0x1p-10)
