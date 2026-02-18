// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"github.com/gomlx/gomlx/pkg/core/shapes"
	. "github.com/gomlx/gomlx/pkg/support/exceptions"
)

// MustAdjustAxis returns the positive axis to the operand shapes, adjusting in case the axis given is negative.
//
// It panics if axis given is not in the operand's rank range.
func MustAdjustAxis(axis int, operand shapes.HasShape) int {
	adjustedAxis := axis
	if axis < 0 {
		adjustedAxis = operand.Shape().Rank() + axis
	}
	if adjustedAxis < 0 || adjustedAxis >= operand.Shape().Rank() {
		Panicf("invalid axis %d, operand rank is %d", axis, operand.Shape().Rank())
	}
	return adjustedAxis
}

// adjustAxisToRank converts negative axes to a value starting from the end.
// Similar to AdjustAxisToOperandRank, but not specific to an operand and it doesn't panic if out-of-bounds:
// for some operations it is valid to go out-of-bounds.
func adjustAxisToRank(axis, rank int) int {
	if axis < 0 {
		axis += rank
	}
	return axis
}

