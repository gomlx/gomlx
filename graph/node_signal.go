package graph

import (
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gomlx/xla"
	"github.com/gomlx/gopjrt/dtypes"
)

// FFT computes a forward 1D fast-fourier transformation of the operand, which is expected
// to be complex.
// The FFT is computed on the last dimension, in case `operand.Rank() > 1`.
//
// The resulting tensor (Node) has the same shape as the input, and has the values on the frequency
// domain. Use InverseFFT to reverse the result.
func FFT(operand *Node) *Node {
	_ = validateBuildingGraphFromInputs(operand)
	if !operand.DType().IsComplex() {
		Panicf("FFT requires a complex input, got %s for dtype instead", operand.DType())
	}
	if operand.Shape().IsScalar() {
		Panicf("FFT requires a complex input with rank > 1, got scalar %s instead", operand.DType())
	}
	return backendFFT(operand, xla.FftForward, []int{xslices.Last(operand.Shape().Dimensions)})
}

// InverseFFT computes an inverse fast-fourier transformation of the operand, which is expected to be complex.
// The InverseFFT is computed on the last dimension, in case `operand.Rank() > 1`.
//
// The resulting tensor (Node) has the same shape as the input, and has the values on the frequency
// domain.
func InverseFFT(operand *Node) *Node {
	_ = validateBuildingGraphFromInputs(operand)
	if !operand.DType().IsComplex() {
		Panicf("InverseFFT requires a complex input, got %s for dtype instead", operand.DType())
	}
	if operand.Shape().IsScalar() {
		Panicf("InverseFFT requires a complex input with rank > 1, got scalar %s instead", operand.DType())
	}
	return fftXLA(operand, xla.FftInverse, []int{xslices.At(operand.Shape().Dimensions, -1)})
}

// RealFFT computes a forward 1D fast-fourier transformation on a real (float) input.
// The FFT is computed on the last dimension, in case `operand.Rank() > 1`.
//
// The resulting tensor (Node) has the shape equal to the input, except the last dimension (where the FFT is computed)
// which has dimension `dim/2 + 1`, where `dim` is the last dimensions of `operand`.
//
// Note that because of the last dimension change in `RealFFT`, this cannot be perfectly reversed if
// `operand.Shape().Dimensions[-1]` is odd.
// Preferably use with even numbers.
func RealFFT(operand *Node) *Node {
	if !operand.DType().IsFloat() {
		Panicf("FFT requires a real (float) input, got %s for dtype instead", operand.DType())
	}
	if operand.Shape().IsScalar() {
		Panicf("RealFFT requires a real (float) input with rank > 1, got scalar %s instead", operand.DType())
	}
	return fftXLA(operand, xla.FftForwardReal, []int{xslices.At(operand.Shape().Dimensions, -1)})
}

// InverseRealFFT computes the inverse of a forward 1D fast-fourier transformation.
// The inverse FFT is computed on the last dimension, in case `operand.Rank() > 1`.
//
// The resulting tensor (Node) has the shape equal to the input, except the last dimension (where the FFT is computed)
// which is reversed back to the original, `(dim-1)*2`, where `dim` is the last dimensions of `operand`.
//
// Note that because of the last dimension change in `RealFFT`, this cannot be perfectly reversed if
// `operand.Shape().Dimensions[-1]` is odd.
// Preferably use with even numbers.
func InverseRealFFT(operand *Node) *Node {
	if !operand.DType().IsComplex() {
		Panicf("InverseRealFFT requires a complex input, got %s for dtype instead", operand.DType())
	}
	if operand.Shape().IsScalar() {
		Panicf("RealFFT requires a real (float) input with rank > 1, got scalar %s instead", operand.DType())
	}
	lastDim := (xslices.At(operand.Shape().Dimensions, -1) - 1) * 2
	return fftXLA(operand, xla.FftInverseReal, []int{lastDim})
}

// fftVJP implements the auto-grad for all the FFT variations.
func fftVJP(node, v *Node, _ shapes.Shape) []*Node {
	fftType := xla.FftType(node.serializedNode.Int)
	switch fftType {
	case xla.FftForward:
		size := float64(xslices.At(v.Shape().Dimensions, -1))
		return []*Node{MulScalar(InverseFFT(v), size)}
	case xla.FftInverse:
		invSize := 1.0 / float64(xslices.At(v.Shape().Dimensions, -1))
		return []*Node{MulScalar(FFT(v), invSize)}
	case xla.FftForwardReal:
		return realFftVJP(node, v)
	case xla.FftInverseReal:
		return inverseRealFftVJP(node, v)
	}

	Panicf("Sorry, gradient for FFT of type %s not implemented, please add an issue in github.com/gomlx/gomlx if you need it!",
		fftType)
	return nil
}

// realFftVJP is called from fftVJP.
// It is implemented based on the TensorFlow function `_rfft_grad_helper`, in the file
// `tensorflow/python/ops/signal/fft_ops.pyfft_ops.py`.
//
// Experimental, not well tested yet.
func realFftVJP(node, v *Node) []*Node {
	fftLength := node.serializedNode.Ints
	isEven := 1.0 - float64(xslices.At(fftLength, -1)%2)
	operand := node.inputNodes[0]
	rank := operand.Rank()
	complexDType := v.DType()
	operandLastDim := operand.Shape().Dimensions[rank-1]

	// Create yMask with a sequence of alternating signs +1, -1, +1, -1 for the length of
	// the operand's last dimension. So yMask is shaped `[1, 1, ..., operandLastDim]`.
	g := v.Graph()
	yMaskShape := operand.Shape().Clone()
	yMaskShape.DType = dtypes.Int32 // For now.
	for ii := 0; ii < rank-1; ii++ {
		yMaskShape.Dimensions[ii] = 1
	}
	// yMask = -1 * (2*(Iota()%2)-1) -> +1, -1, +1, -1, etc.
	yMask := Iota(g, yMaskShape, -1)
	yMask = ModScalar(yMask, 2.0)
	yMask = Neg(AddScalar(MulScalar(yMask, 2), -1))
	yMask = ConvertType(yMask, complexDType)

	y0 := Slice(v, AxisRange().Spacer(), AxisElem(0))
	yLast := Slice(v, AxisRange().Spacer(), AxisElem(-1))
	extraTerms := Add(y0, MulScalar(Mul(yLast, yMask), isEven))

	// The gradient of RFFT is the IRFFT of the incoming gradient times a scaling
	// factor, plus some additional terms to make up for the components dropped
	// due to Hermitian symmetry.
	irFft := fftXLA(v, xla.FftInverseReal, fftLength)
	newVJP := MulScalar(Add(MulScalar(irFft, float64(operandLastDim)), Real(extraTerms)), 0.5)
	return []*Node{newVJP}
}

// inverseRealFftVJP is called from fftVJP.
// It is implemented based on the TensorFlow function `_irfft_grad_helper`, in the file
// `tensorflow/python/ops/signal/fft_ops.pyfft_ops.py`.
//
// Experimental, not well tested yet.
func inverseRealFftVJP(node, v *Node) []*Node {
	g := node.Graph()
	realDType := v.DType()
	fftValue := node.inputNodes[0]
	complexDType := fftValue.DType()
	fftValueLastDim := xslices.Last(fftValue.Shape().Dimensions)
	fftLength := node.serializedNode.Ints
	isFftLengthOdd := xslices.Last(fftLength) % 2 // 1 if fftLength is odd, 0 if even.

	// Create a simple mask like [1.0, 2.0, 2.0, ..., 2.0, 2.0, 1.0] for even-length FFTs
	// or [1.0, 2.0, ..., 2.0] for odd-length FFTs -- same length as fftValueLastDim.
	innerMask := AddScalar(Ones(g, shapes.Make(realDType, fftValueLastDim-2+isFftLengthOdd)), 1.0)
	edgeMask := Ones(g, shapes.Make(realDType, 1))
	var mask *Node
	if isFftLengthOdd == 1 {
		mask = Concatenate([]*Node{edgeMask, innerMask}, 0)
	} else {
		mask = Concatenate([]*Node{edgeMask, innerMask, edgeMask}, 0)
	}
	mask.AssertDims(fftValueLastDim) // Our mask will apply to the fftValue (the input to the InverseRealFFT node).
	mask = ExpandLeftToRank(mask, fftValue.Rank())
	sizeNormalization := 1.0 / float64(xslices.Last(v.Shape().Dimensions))
	normalizedMask := ConvertType(MulScalar(mask, sizeNormalization), complexDType)
	vjp := Mul(RealFFT(v), normalizedMask)
	return []*Node{vjp}
}
