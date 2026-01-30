// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package nn provides higher-level neural network building blocks built on top of the graph package.
package nn

import (
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/graph"
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
func Softmax(logits *graph.Node, axes ...int) *graph.Node {
	if !logits.DType().IsFloat() {
		Panicf("invalid logits dtype (%s), it must be float", logits.DType())
	}
	if len(axes) == 0 {
		axes = []int{-1}
	}

	// Try native softmax for the single-axis case.
	if len(axes) == 1 {
		if logits.Graph().Backend().Capabilities().Operations[backends.OpTypeSoftmax] {
			return graph.Softmax(logits, axes[0])
		}
	}

	// Fall back to decomposition.
	normalizingMax := graph.StopGradient(graph.ReduceAndKeep(logits, graph.ReduceMax, axes...))
	normalizedLogits := graph.Sub(logits, normalizingMax)
	numerator := graph.Exp(normalizedLogits)
	denominator := graph.ReduceAndKeep(numerator, graph.ReduceSum, axes...)
	return graph.Div(numerator, denominator)
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
func MaskedSoftmax(logits, mask *graph.Node, axes ...int) *graph.Node {
	if mask == nil {
		return Softmax(logits, axes...)
	}
	if !logits.DType().IsFloat() {
		Panicf("invalid logits dtype (%s), it must be float", logits.DType())
	}
	if len(axes) == 0 {
		axes = []int{-1}
	}
	normalizingMax := graph.StopGradient(graph.MaskedReduceAndKeep(logits, mask, graph.MaskedReduceMax, axes...))
	zeros := graph.ZerosLike(logits)
	normalizedLogits := graph.Sub(logits, normalizingMax)
	normalizedLogits = graph.Where(mask, normalizedLogits, zeros)
	numerator := graph.Exp(normalizedLogits)
	numerator = graph.Where(mask, numerator, zeros)
	// Apply mask on numerator, setting softmax to zero where masked.
	denominator := graph.ReduceAndKeep(numerator, graph.ReduceSum, axes...)
	result := graph.Div(numerator, denominator)
	result = graph.Where(mask, result, zeros)
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
func LogSoftmax(logits *graph.Node, axes ...int) *graph.Node {
	if !logits.DType().IsFloat() {
		Panicf("invalid logits dtype (%s), it must be float", logits.DType())
	}
	if len(axes) == 0 {
		axes = []int{-1}
	}
	adjustedAxes := graph.AdjustAxesToRankAndSort(logits.Rank(), axes, "logits")
	normalizingMax := graph.StopGradient(graph.ReduceAndKeep(logits, graph.ReduceMax, adjustedAxes...))
	shiftedLogits := graph.Sub(logits, normalizingMax)
	shiftedLogSumExp := graph.Log(graph.ReduceAndKeep(graph.Exp(shiftedLogits), graph.ReduceSum, adjustedAxes...))
	return graph.Sub(shiftedLogits, shiftedLogSumExp)
}

// MaskedLogSoftmax computes the logarithm of the MaskedSoftmax function, which rescales
// elements to the range $[-\infty, 0)$.
//
// It takes a mask that is true on the values to be considered, and false for the values
// not to be considered.
// If mask is nil, it behaves like LogSoftmax.
//
// See LogSoftmax for details.
func MaskedLogSoftmax(logits, mask *graph.Node, axes ...int) *graph.Node {
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
	adjustedAxes := graph.AdjustAxesToRankAndSort(logits.Rank(), axes, "logits")
	normalizingMax := graph.StopGradient(graph.MaskedReduceAndKeep(logits, mask, graph.MaskedReduceMax, adjustedAxes...))
	shiftedLogits := graph.Sub(logits, normalizingMax)
	shiftedLogSumExp := graph.Log(graph.MaskedReduceAndKeep(graph.Exp(shiftedLogits), mask, graph.MaskedReduceSum, adjustedAxes...))
	return graph.Where(mask, graph.Sub(shiftedLogits, shiftedLogSumExp), graph.Infinity(logits.Graph(), dtype, -1))
}
