// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package shapes

import (
	"slices"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/pkg/errors"
)

// DynamicDim is the sentinel value used in Shape.Dimensions to indicate an axis whose
// size is unknown at graph build time and will be resolved at execution time via AxisBindings.
//
// Note: this is the same numeric value as UncheckedAxis (-1), but used in a different context.
// UncheckedAxis is used in assertion arguments (CheckDims, AssertDims) to mean "don't check this axis".
// DynamicDim is used in Shape.Dimensions to mean "this axis has an unknown size".
const DynamicDim = -1

// MakeDynamic creates a Shape with named axes and optional dynamic dimensions.
//
// dimensions can contain DynamicDim (-1) for axes whose size is unknown at graph build time.
// Dynamic axes must have a non-empty name in axisNames. Static axes (>= 0) may have names too.
//
// axisNames must have the same length as dimensions. Use "" for unnamed axes.
//
// Example:
//
//	shapes.MakeDynamic(dtypes.F32, []int{-1, 512}, []string{"batch", ""})
//	// Creates shape (Float32)[batch=-1 512]
func MakeDynamic(dtype dtypes.DType, dimensions []int, axisNames []string) Shape {
	if len(dimensions) != len(axisNames) {
		panic(errors.Errorf("shapes.MakeDynamic: len(dimensions)=%d must equal len(axisNames)=%d", len(dimensions), len(axisNames)))
	}
	for i, dim := range dimensions {
		if dim == DynamicDim {
			if axisNames[i] == "" {
				panic(errors.Errorf("shapes.MakeDynamic: dynamic axis %d (dimension=-1) must have a non-empty axis name", i))
			}
		} else if dim < 0 {
			panic(errors.Errorf("shapes.MakeDynamic: dimension %d has invalid value %d (must be >= 0 or DynamicDim)", i, dim))
		}
	}
	return Shape{
		DType:      dtype,
		Dimensions: slices.Clone(dimensions),
		AxisNames:  slices.Clone(axisNames),
	}
}

// HasDynamicDims returns true if any dimension is dynamic (DynamicDim).
func (s Shape) HasDynamicDims() bool {
	for _, d := range s.Dimensions {
		if d == DynamicDim {
			return true
		}
	}
	return false
}

// IsDynamicDim returns true if the given axis has a dynamic dimension (DynamicDim).
// axis supports negative indexing (e.g., -1 for the last axis).
func (s Shape) IsDynamicDim(axis int) bool {
	adjustedAxis := axis
	if adjustedAxis < 0 {
		adjustedAxis += s.Rank()
	}
	if adjustedAxis < 0 || adjustedAxis >= s.Rank() {
		panic(errors.Errorf("Shape.IsDynamicDim(%d) out-of-bounds for rank %d (shape=%s)", axis, s.Rank(), s))
	}
	return s.Dimensions[adjustedAxis] == DynamicDim
}

// AxisName returns the name of the given axis, or "" if unnamed or if AxisNames is nil.
// axis supports negative indexing.
func (s Shape) AxisName(axis int) string {
	if s.AxisNames == nil {
		return ""
	}
	adjustedAxis := axis
	if adjustedAxis < 0 {
		adjustedAxis += s.Rank()
	}
	if adjustedAxis < 0 || adjustedAxis >= s.Rank() {
		panic(errors.Errorf("Shape.AxisName(%d) out-of-bounds for rank %d (shape=%s)", axis, s.Rank(), s))
	}
	return s.AxisNames[adjustedAxis]
}

// WithAxisNames returns a copy of the shape with the given axis names set.
// The number of names must equal the rank.
func (s Shape) WithAxisNames(names ...string) Shape {
	if len(names) != s.Rank() {
		panic(errors.Errorf("Shape.WithAxisNames: len(names)=%d must equal rank=%d", len(names), s.Rank()))
	}
	s2 := s.Clone()
	s2.AxisNames = slices.Clone(names)
	return s2
}

// axisNamesEqual compares two axis name slices for equality.
// nil is considered equal to a slice of all empty strings.
func axisNamesEqual(a, b []string) bool {
	if a == nil && b == nil {
		return true
	}
	aLen := len(a)
	bLen := len(b)
	if a == nil {
		aLen = bLen
	}
	if b == nil {
		bLen = aLen
	}
	if aLen != bLen {
		return false
	}
	for i := range aLen {
		aName := ""
		if a != nil {
			aName = a[i]
		}
		bName := ""
		if b != nil {
			bName = b[i]
		}
		if aName != bName {
			return false
		}
	}
	return true
}
