// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package dtypes is an alias to compute/dtypes package. It's here for historical reasons.
//
// Deprecated: use the [github.com/gomlx/compute/dtypes] package instead.
package dtypes

import (
	"reflect"
	"unsafe"

	"github.com/gomlx/compute/dtypes"
)

// DType is an enum represents the data type of a buffer or a scalar.
//
// Deprecated: it's just an alias to [dtypes.DType], use that instead.
type DType = dtypes.DType

const (
	InvalidDType = dtypes.InvalidDType
	Bool         = dtypes.Bool
	Int8         = dtypes.Int8
	Int16        = dtypes.Int16
	Int32        = dtypes.Int32
	Int64        = dtypes.Int64
	Uint8        = dtypes.Uint8
	Uint16       = dtypes.Uint16
	Uint32       = dtypes.Uint32
	Uint64       = dtypes.Uint64
	Float16      = dtypes.Float16
	Float32      = dtypes.Float32
	Float64      = dtypes.Float64
	BFloat16     = dtypes.BFloat16
	Complex64    = dtypes.Complex64
	Complex128   = dtypes.Complex128
	Int4         = dtypes.Int4
	Uint4        = dtypes.Uint4
	Int2         = dtypes.Int2
	Uint2        = dtypes.Uint2
)

// Aliases from PJRT C API.
const (
	INVALID = dtypes.INVALID
	PRED    = dtypes.PRED
	S8      = dtypes.S8
	S16     = dtypes.S16
	S32     = dtypes.S32
	S64     = dtypes.S64
	U8      = dtypes.U8
	U16     = dtypes.U16
	U32     = dtypes.U32
	U64     = dtypes.U64
	F16     = dtypes.F16
	F32     = dtypes.F32
	F64     = dtypes.F64
	BF16    = dtypes.BF16
	C64     = dtypes.C64
	C128    = dtypes.C128
	S4      = dtypes.S4
	U4      = dtypes.U4
	S2      = dtypes.S2
	U2      = dtypes.U2
)

// MapOfNames to their dtypes.
//
// Deprecated: use [dtypes.MapOfNames] instead.
var MapOfNames = dtypes.MapOfNames

// MaxDType is the maximum number of DTypes that there can be.
const MaxDType = dtypes.MaxDType

// DTypeSet represents a set of DTypes.
//
// Deprecated: it's just an alias to [dtypes.DTypeSet], use that instead.
type DTypeSet = dtypes.DTypeSet

var (
	FloatDTypes     = dtypes.FloatDTypes
	Float16DTypes   = dtypes.Float16DTypes
	ComplexDTypes   = dtypes.ComplexDTypes
	IntDTypes       = dtypes.IntDTypes
	UnsignedDTypes  = dtypes.UnsignedDTypes
	SupportedDTypes = dtypes.SupportedDTypes
)

// Supported lists the Go types that GoMLX knows how to convert.
//
// Deprecated: it's just an alias to [dtypes.Supported], use that instead.
type Supported = dtypes.Supported

// Number represents the Go numeric types corresponding to supported DType's.
//
// Deprecated: it's just an alias to [dtypes.Number], use that instead.
type Number = dtypes.Number

// NumberNotComplex represents the Go numeric types corresponding to supported DType's.
//
// Deprecated: it's just an alias to [dtypes.NumberNotComplex], use that instead.
type NumberNotComplex = dtypes.NumberNotComplex

// GoFloat represent a continuous Go numeric type, supported by GoMLX.
//
// Deprecated: it's just an alias to [dtypes.GoFloat], use that instead.
type GoFloat = dtypes.GoFloat

// FromGenericsType returns the DType enum for the given type that this package knows about.
//
// Deprecated: use [dtypes.FromGenericsType] instead.
func FromGenericsType[T Supported]() DType {
	return dtypes.FromGenericsType[T]()
}

// FromGoType returns the DType for the given "reflect.Type".
//
// Deprecated: use [dtypes.FromGoType] instead.
func FromGoType(t reflect.Type) DType {
	return dtypes.FromGoType(t)
}

// FromAny introspects the underlying type of any and returns the corresponding DType.
//
// Deprecated: use [dtypes.FromAny] instead.
func FromAny(value any) DType {
	return dtypes.FromAny(value)
}

// UnsafeByteSliceFromAny casts a slice of any of the supported Go types (feed as type any) to a slice of bytes.
//
// Deprecated: use [dtypes.UnsafeByteSliceFromAny] instead.
func UnsafeByteSliceFromAny(flatAny any) []byte {
	return dtypes.UnsafeByteSliceFromAny(flatAny)
}

// UnsafeByteSlice casts a slice of any of the supported Go types to a slice of bytes.
//
// Deprecated: use [dtypes.UnsafeByteSlice] instead.
func UnsafeByteSlice[E Supported](flat []E) []byte {
	return dtypes.UnsafeByteSlice(flat)
}

// UnsafeAnySliceFromBytes casts a pointer to a buffer of bytes to a slice of the given dtype and length.
//
// Deprecated: use [dtypes.UnsafeAnySliceFromBytes] instead.
func UnsafeAnySliceFromBytes(bytesPtr unsafe.Pointer, dtype DType, length int) any {
	return dtypes.UnsafeAnySliceFromBytes(bytesPtr, dtype, length)
}

// UnsafeSliceFromBytes casts a pointer to a buffer of bytes to a slice of the given E type and length.
//
// Deprecated: use [dtypes.UnsafeSliceFromBytes] instead.
func UnsafeSliceFromBytes[E Supported](bytesPtr unsafe.Pointer, length int) []E {
	return dtypes.UnsafeSliceFromBytes[E](bytesPtr, length)
}

// MakeAnySlice creates a slice of the given dtype and length, casted to any.
//
// Deprecated: use [dtypes.MakeAnySlice] instead.
func MakeAnySlice(dtype DType, length int) any {
	return dtypes.MakeAnySlice(dtype, length)
}

// CopyAnySlice copies the contents of src to dst, both should be slices of the same DType.
//
// Deprecated: use [dtypes.CopyAnySlice] instead.
func CopyAnySlice(dst, src any) {
	dtypes.CopyAnySlice(dst, src)
}
