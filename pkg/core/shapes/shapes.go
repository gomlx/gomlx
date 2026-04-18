// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package shapes is an alias to compute/shapes package. It's here for historical reasons.
//
// Deprecated: use the [github.com/gomlx/compute/shapes] package instead.
package shapes

import (
	"encoding/gob"
	"unsafe"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

// Shape represents the shape of either a Tensor or the expected shape
// of the value from a computation node.
//
// Deprecated: it's just an alias to [shapes.Shape], use that instead.
type Shape = shapes.Shape

// UncheckedAxis can be used in CheckDims or AssertDims functions for an axis
// whose dimension doesn't matter.
const UncheckedAxis = shapes.UncheckedAxis

// HasShape is an interface for objects that have an associated Shape.
//
// Deprecated: it's just an alias to [shapes.HasShape], use that instead.
type HasShape = shapes.HasShape

// Make returns a Shape structure filled with the values given.
//
// Deprecated: use [shapes.Make] instead.
func Make(dtype dtypes.DType, dimensions ...int) Shape {
	return shapes.Make(dtype, dimensions...)
}

// Scalar returns a scalar Shape for the given type.
//
// Deprecated: use [shapes.Scalar] instead.
func Scalar[T dtypes.Number]() Shape {
	return shapes.Scalar[T]()
}

// Invalid returns an invalid shape.
//
// Deprecated: use [shapes.Invalid] instead.
func Invalid() Shape {
	return shapes.Invalid()
}

// MakeTuple returns a shape representing a tuple of elements with the given shapes.
//
// Deprecated: use [shapes.MakeTuple] instead.
func MakeTuple(elements []Shape) Shape {
	return shapes.MakeTuple(elements)
}

// GobDeserialize a Shape. Returns new Shape or an error.
//
// Deprecated: use [shapes.GobDeserialize] instead.
func GobDeserialize(decoder *gob.Decoder) (Shape, error) {
	return shapes.GobDeserialize(decoder)
}

// ConcatenateDimensions of two shapes.
//
// Deprecated: use [shapes.ConcatenateDimensions] instead.
func ConcatenateDimensions(s1, s2 Shape) Shape {
	return shapes.ConcatenateDimensions(s1, s2)
}

// FromAnyValue attempts to convert a Go "any" value to its expected shape.
//
// Deprecated: use [shapes.FromAnyValue] instead.
func FromAnyValue(v any) (Shape, error) {
	return shapes.FromAnyValue(v)
}

// ConvertTo converts any scalar to T.
//
// Deprecated: use [shapes.ConvertTo] instead.
func ConvertTo[T dtypes.NumberNotComplex](value any) T {
	return shapes.ConvertTo[T](value)
}

// UnsafeSliceForDType creates a slice of the corresponding dtype and casts it to any.
//
// Deprecated: use [shapes.UnsafeSliceForDType] instead.
func UnsafeSliceForDType(dtype dtypes.DType, unsafePtr unsafe.Pointer, len int) any {
	return shapes.UnsafeSliceForDType(dtype, unsafePtr, len)
}

// CastAsDType casts a numeric value to the corresponding for the DType.
//
// Deprecated: use [shapes.CastAsDType] instead.
func CastAsDType(value any, dtype dtypes.DType) any {
	return shapes.CastAsDType(value, dtype)
}

// CheckDims checks that the shape has the given dimensions and rank.
//
// Deprecated: use [shapes.CheckDims] instead.
func CheckDims(shaped HasShape, dimensions ...int) error {
	return shapes.CheckDims(shaped, dimensions...)
}

// AssertDims checks that the shape has the given dimensions and rank.
//
// Deprecated: use [shapes.AssertDims] instead.
func AssertDims(shaped HasShape, dimensions ...int) {
	shapes.AssertDims(shaped, dimensions...)
}

// Assert checks that the shape has the given dtype, dimensions and rank.
//
// Deprecated: use [shapes.Assert] instead.
func Assert(shaped HasShape, dtype dtypes.DType, dimensions ...int) {
	shapes.Assert(shaped, dtype, dimensions...)
}

// CheckRank checks that the shape has the given rank.
//
// Deprecated: use [shapes.CheckRank] instead.
func CheckRank(shaped HasShape, rank int) error {
	return shapes.CheckRank(shaped, rank)
}

// AssertRank checks that the shape has the given rank.
//
// Deprecated: use [shapes.AssertRank] instead.
func AssertRank(shaped HasShape, rank int) {
	shapes.AssertRank(shaped, rank)
}

// CheckScalar checks that the shape is a scalar.
//
// Deprecated: use [shapes.CheckScalar] instead.
func CheckScalar(shaped HasShape) error {
	return shapes.CheckScalar(shaped)
}

// AssertScalar checks that the shape is a scalar.
//
// Deprecated: use [shapes.AssertScalar] instead.
func AssertScalar(shaped HasShape) {
	shapes.AssertScalar(shaped)
}
