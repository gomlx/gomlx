package graph

import (
	. "github.com/gomlx/gomlx/types/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/xla"
)

// FFT runs a fast-forward-fourier transformation of the operand, which is expected to be complex.
//
// The resulting tensor (Node) has the same shape as the input, and have the values on the frequency
// domain. Use InverseFFT to reverse the result.
func FFT(operand *Node) *Node {
	_ = validateGraphFromInputs(operand)
	if !operand.DType().IsComplex() {
		Panicf("FFT requires a complex input, got %s for dtype instead", operand.DType())
	}
	return fftXLA(operand, xla.FftForward, operand.Shape().Dimensions)
}

// InverseFFT runs an inverse fast-forward-fourier transformation of the operand, which is expected to be complex.
//
// The resulting tensor (Node) has the same shape as the input, and have the values on the frequency
// domain.
func InverseFFT(operand *Node) *Node {
	_ = validateGraphFromInputs(operand)
	if !operand.DType().IsComplex() {
		Panicf("FFT requires a complex input, got %s for dtype instead", operand.DType())
	}
	return fftXLA(operand, xla.FftInverse, operand.Shape().Dimensions)
}

// fftVJP implements the auto-grad for all the FFT variations.
func fftVJP(_, _ *Node, _ shapes.Shape) []*Node {
	return nil
}
