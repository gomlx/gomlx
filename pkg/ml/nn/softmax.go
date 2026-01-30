// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package nn provides higher-level neural network building blocks built on top of the graph package.
package nn

import (
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// Softmax computes softmax activations. It's the equivalent to
//
//	Exp(logits) / ReduceAndKeep(Exp(logits), ReduceSum, axes...)
//
// But implemented in a numerical stable way.
//
// The list axes defines which axes is it supposed to run the softmax over
// (the axes that will be summed over).
//
// If no axes are given, it is assumed to be [-1], meaning, the last axes.
//
// If the backend supports fused softmax (single axis), it will use the
// optimized native implementation instead of decomposing into primitives.
func Softmax(logits *Node, axes ...int) *Node {
	if !logits.DType().IsFloat() {
		Panicf("invalid logits dtype (%s), it must be float", logits.DType())
	}
	if len(axes) == 0 {
		axes = []int{-1}
	}

	// Try native softmax for the single-axis case.
	if len(axes) == 1 {
		if logits.Graph().Backend().Capabilities().Operations[backends.OpTypeFusedSoftmax] {
			return FusedSoftmax(logits, axes[0])
		}
	}

	// Fall back to decomposition.
	normalizingMax := StopGradient(ReduceAndKeep(logits, ReduceMax, axes...))
	normalizedLogits := Sub(logits, normalizingMax)
	numerator := Exp(normalizedLogits)
	denominator := ReduceAndKeep(numerator, ReduceSum, axes...)
	return Div(numerator, denominator)
}

// MaskedSoftmax computes softmax activations. It's the equivalent to
// ```
//
//	Exp(logits) / InsertAxes(ReduceSum(Exp(logits), -1), -1)
//
// ```
//
// But implemented in a numerical stable way.
//
// It takes a mask that is true on the values to be considered, and false for the values
// not to be considered.
//
// The list axes defines which axes is it supposed to run the softmax over
// (the axes that will be summed over). If no axes are given, it is assumed to
// be [-1], meaning, the last axes.
//
// It ignores values for which the corresponding mask is false, and will return 0 for
// those fields. mask and logits must have the same shape.
func MaskedSoftmax(logits, mask *Node, axes ...int) *Node {
	if mask == nil {
		return Softmax(logits, axes...)
	}
	if !logits.DType().IsFloat() {
		Panicf("invalid logits dtype (%s), it must be float", logits.DType())
	}
	if len(axes) == 0 {
		axes = []int{-1}
	}
	normalizingMax := StopGradient(MaskedReduceAndKeep(logits, mask, MaskedReduceMax, axes...))
	zeros := ZerosLike(logits)
	normalizedLogits := Sub(logits, normalizingMax)
	normalizedLogits = Where(mask, normalizedLogits, zeros)
	numerator := Exp(normalizedLogits)
	numerator = Where(mask, numerator, zeros)
	// Apply mask on numerator, setting softmax to zero where masked.
	denominator := ReduceAndKeep(numerator, ReduceSum, axes...)
	result := Div(numerator, denominator)
	result = Where(mask, result, zeros)
	return result
}

// LogSoftmax computes the logarithm of the Softmax function, which rescales
// elements to the range $[-\infty, 0)$.
//
//	$$
//	\mathrm{log\_softmax}(x)_i = \log \left( \frac{\exp(x_i)}{\sum_j \exp(x_j)}
//	\right)
//	$$
//
// The axes define over which axes the LogSoftmax should be computed. If missing it is assumed to be -1.
//
// If any input values are "+inf", the result will be all "NaN": this reflects the
// fact that "inf / inf" is not well-defined in the context of floating-point math.
func LogSoftmax(logits *Node, axes ...int) *Node {
	if !logits.DType().IsFloat() {
		Panicf("invalid logits dtype (%s), it must be float", logits.DType())
	}
	if len(axes) == 0 {
		axes = []int{-1}
	}
	adjustedAxes := AdjustAxesToRankAndSort(logits.Rank(), axes, "logits")
	normalizingMax := StopGradient(ReduceAndKeep(logits, ReduceMax, adjustedAxes...))
	shiftedLogits := Sub(logits, normalizingMax)
	shiftedLogSumExp := Log(ReduceAndKeep(Exp(shiftedLogits), ReduceSum, adjustedAxes...))
	return Sub(shiftedLogits, shiftedLogSumExp)
}

// MaskedLogSoftmax computes the logarithm of the MaskedSoftmax function, which rescales
// elements to the range $[-\infty, 0)$.
//
// It takes a mask that is true on the values to be considered, and false for the values
// not to be considered.
// If mask is nil, it behaves like LogSoftmax.
//
// See LogSoftmax for details.
func MaskedLogSoftmax(logits, mask *Node, axes ...int) *Node {
	if mask == nil {
		return LogSoftmax(logits, axes...)
	}
	dtype := logits.DType()
	if !dtype.IsFloat() {
		Panicf("invalid logits dtype (%s), it must be float", logits.DType())
	}
	if len(axes) == 0 {
		axes = []int{-1}
	}
	adjustedAxes := AdjustAxesToRankAndSort(logits.Rank(), axes, "logits")
	normalizingMax := StopGradient(MaskedReduceAndKeep(logits, mask, MaskedReduceMax, adjustedAxes...))
	shiftedLogits := Sub(logits, normalizingMax)
	shiftedLogSumExp := Log(MaskedReduceAndKeep(Exp(shiftedLogits), mask, MaskedReduceSum, adjustedAxes...))
	return Where(mask, Sub(shiftedLogits, shiftedLogSumExp), Infinity(logits.Graph(), dtype, -1))
}
