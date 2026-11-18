// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/internal/graphimpl"
	. "github.com/gomlx/gomlx/support/exceptions"
)

// FFT computes a forward 1D fast-fourier transformation of the operand, which is expected
// to be complex.
// The FFT is computed on the last dimension, in case `operand.Rank() > 1`.
//
// If no lengths are specified, the resulting tensor (Node) has the same shapes as the operand and has the
// values on the frequency domain calculated at the last axis.
//
// Optionally, one can provide the FFT lengths: the default being the last dimension, over which the FFT is calculated.
// Note: gradients are not defined for different FFT lengths yet.
//
// Use InverseFFT to reverse the result.
func FFT(operand *Node, fftLengths ...int) *Node {
	_ = validateBuildingGraphFromInputs(operand)
	if !operand.DType().IsComplex() {
		Panicf("FFT requires a complex input, got %s for dtype instead", operand.DType())
	}
	if operand.Shape().IsScalar() {
		Panicf("FFT requires a complex input with rank > 1, got scalar %s instead", operand.DType())
	}
	if len(fftLengths) == 0 {
		fftLengths = []int{operand.Shape().Dim(-1)}
	}
	return wrapNode(graphimpl.BackendFFT(unwrapNode(operand), compute.FFTForward, fftLengths))
}

// InverseFFT computes an inverse fast-fourier transformation of the operand, which is expected to be complex.
// The InverseFFT is computed on the last dimension, in case `operand.Rank() > 1`.
//
// Optionally, one can provide the FFT lengths: the default being the last dimension, over which the FFT is calculated.
// Note: gradients are not defined for different FFT lengths yet.
//
// The resulting tensor (Node) has the same shapes as the input and has the values on the frequency
// domain.
func InverseFFT(operand *Node, fftLengths ...int) *Node {
	_ = validateBuildingGraphFromInputs(operand)
	if !operand.DType().IsComplex() {
		Panicf("InverseFFT requires a complex input, got %s for dtype instead", operand.DType())
	}
	if operand.Shape().IsScalar() {
		Panicf("InverseFFT requires a complex input with rank > 1, got scalar %s instead", operand.DType())
	}
	if len(fftLengths) == 0 {
		fftLengths = []int{operand.Shape().Dim(-1)}
	}
	return wrapNode(graphimpl.BackendFFT(unwrapNode(operand), compute.FFTInverse, fftLengths))
}

// RealFFT computes a forward 1D fast-fourier transformation on a real (float) input.
// The FFT is computed on the last dimension, in case `operand.Rank() > 1`.
//
// The resulting tensor (Node) has the shapes equal to the input, except the last dimension (where the FFT is computed)
// which has dimension `dim/2 + 1`, where `dim` is the operand's last axis dimension.
//
// Note that because of the last dimension change in RealFFT, this cannot be perfectly reversed if
// `operand.Shape().Dimensions[-1]` is odd.
// Preferably, use with an even last axis dimension.
//
// Optionally, one can provide the FFT lengths: the default being the last dimension, over which the FFT is calculated.
// Note: gradients are not defined for different FFT lengths yet.
func RealFFT(operand *Node, fftLengths ...int) *Node {
	if !operand.DType().IsFloat() {
		Panicf("FFT requires a real (float) input, got %s for dtype instead", operand.DType())
	}
	if operand.Shape().IsScalar() {
		Panicf("RealFFT requires a real (float) input with rank > 1, got scalar %s instead", operand.DType())
	}
	if len(fftLengths) == 0 {
		fftLengths = []int{operand.Shape().Dim(-1)}
	}
	return wrapNode(graphimpl.BackendFFT(unwrapNode(operand), compute.FFTForwardReal, fftLengths))
}

// InverseRealFFT computes the inverse of a forward 1D fast-fourier transformation.
// The inverse FFT is computed on the last dimension, in case `operand.Rank() > 1`.
//
// The resulting tensor (Node) has the shapes equal to the input, except the last dimension (where the FFT is computed)
// which is reversed back to the original, `(dim-1)*2`, where `dim` is the operand's last axis dimension.
//
// Optionally, one can provide the FFT lengths. The default being the last dimension of the original RealFFT
// (so 2*(x.Dims(-1)-1)).
// Note: gradients are not defined for different FFT lengths yet.
func InverseRealFFT(operand *Node, fftLengths ...int) *Node {
	if !operand.DType().IsComplex() {
		Panicf("InverseRealFFT requires a complex input, got %s for dtype instead", operand.DType())
	}
	if operand.Shape().IsScalar() {
		Panicf("RealFFT requires a real (float) input with rank > 1, got scalar %s instead", operand.DType())
	}
	if len(fftLengths) == 0 {
		lastDim := operand.Shape().Dim(-1)
		fftLengths = []int{(lastDim - 1) * 2}
	}
	return wrapNode(graphimpl.BackendFFT(unwrapNode(operand), compute.FFTInverseReal, fftLengths))
}
