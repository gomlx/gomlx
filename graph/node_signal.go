package graph

import (
	. "github.com/gomlx/gomlx/types/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/xla"
)

// FFT computes a forward 1D fast-fourier transformation of the operand, which is expected
// to be complex.
// The FFT is computed on the last dimension, in case `operand.Rank() > 1`.
//
// The resulting tensor (Node) has the same shape as the input, and has the values on the frequency
// domain. Use InverseFFT to reverse the result.
func FFT(operand *Node) *Node {
	_ = validateGraphFromInputs(operand)
	if !operand.DType().IsComplex() {
		Panicf("FFT requires a complex input, got %s for dtype instead", operand.DType())
	}
	if operand.Shape().IsScalar() {
		Panicf("FFT requires a complex input with rank > 1, got scalar %s instead", operand.DType())
	}
	return fftXLA(operand, xla.FftForward, []int{slices.At(operand.Shape().Dimensions, -1)})
}

// InverseFFT computes an inverse fast-fourier transformation of the operand, which is expected to be complex.
// The InverseFFT is computed on the last dimension, in case `operand.Rank() > 1`.
//
// The resulting tensor (Node) has the same shape as the input, and has the values on the frequency
// domain.
func InverseFFT(operand *Node) *Node {
	_ = validateGraphFromInputs(operand)
	if !operand.DType().IsComplex() {
		Panicf("InverseFFT requires a complex input, got %s for dtype instead", operand.DType())
	}
	if operand.Shape().IsScalar() {
		Panicf("InverseFFT requires a complex input with rank > 1, got scalar %s instead", operand.DType())
	}
	return fftXLA(operand, xla.FftInverse, []int{slices.At(operand.Shape().Dimensions, -1)})
}

// RealFFT computes a forward 1D fast-fourier transformation on a real (float) input.
// The FFT is computed on the last dimension, in case `operand.Rank() > 1`.
//
// The resulting tensor (Node) has the shape equal to the input, except the last dimension (where the FFT is computed)
// which has dimension `dim/2 + 1`, where `dim` is the last dimensions of `operand`.
func RealFFT(operand *Node) *Node {
	if !operand.DType().IsFloat() {
		Panicf("FFT requires a real (float) input, got %s for dtype instead", operand.DType())
	}
	if operand.Shape().IsScalar() {
		Panicf("RealFFT requires a real (float) input with rank > 1, got scalar %s instead", operand.DType())
	}
	return fftXLA(operand, xla.FftForwardReal, []int{slices.At(operand.Shape().Dimensions, -1)})
}

// InverseRealFFT computes the inverse of a forward 1D fast-fourier transformation.
// The inverse FFT is computed on the last dimension, in case `operand.Rank() > 1`.
//
// The resulting tensor (Node) has the shape equal to the input, except the last dimension (where the FFT is computed)
// which is reversed back to the original, `(dim-1)*2`, where `dim` is the last dimensions of `operand`.
func InverseRealFFT(operand *Node) *Node {
	if !operand.DType().IsComplex() {
		Panicf("InverseRealFFT requires a complex input, got %s for dtype instead", operand.DType())
	}
	if operand.Shape().IsScalar() {
		Panicf("RealFFT requires a real (float) input with rank > 1, got scalar %s instead", operand.DType())
	}
	lastDim := (slices.At(operand.Shape().Dimensions, -1) - 1) * 2
	return fftXLA(operand, xla.FftInverseReal, []int{lastDim})
}

// fftVJP implements the auto-grad for all the FFT variations.
func fftVJP(_, _ *Node, _ shapes.Shape) []*Node {
	return nil
}
