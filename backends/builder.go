package backends

import (
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/pkg/errors"
	"slices"
)

// Op represents the output of an operation, during the computation graph building time.
// It is opaque from GoMLX perspective, but one of the backend methods take this value as input, and needs
// to be able to implement Backend.OpShape to return its shape.
type Op any

// NotImplemented panics with a not implemented error, for backends that don't implement all ops.
// It allows users of the backend to capture the exception and handle it differently.
func NotImplemented() {
	panic(errors.New("not implemented"))
}

// Builder is the minimal set of ops to support building an interface. is the sub-interface that defines the operations that the backend must support.
type Builder interface {
	// Compile the computation built. This immediately invalidates the Builder and returns an Executable that
	// can now be used to run the computation.
	//
	// It is given the list of outputs.
	Compile(outputs ...Op) Executable

	// Name of the computation being built.
	Name() string

	// OpShape returns the shape of a computation Op.
	OpShape(op Op) shapes.Shape

	// Parameter creates an input parameter for the computation.
	// During execution of the computation this value will need to be fed, in the same order it is created.
	Parameter(name string, shape shapes.Shape) Op

	// Constant creates a constant in the graph with the given flat values, and the shape defined by dims.
	//
	// flat must be a slice of a basic type supported -- that can be converted to a DType.
	//
	// The value is copied into the graph. It's recommended that for very large tensors,
	// even if constants, that they are passed as side inputNodes (or variables, see context package) instead.
	Constant(flat any, dims ...int) Op

	// Identity returns an Op whose output is the same as its input.
	// It's a no-op that can serve as a place-holder.
	Identity(x Op) Op

	// ReduceWindow runs a reduction function of the type given by reductionType,
	// it can be either ReduceMaxNode, ReduceSumNode or ReduceMultiplyNode.
	//
	// The parameter windowDimensions must be set and have a value for each axis.
	// If strides is nil, it's assumed to be the same as windowDimensions -- that is, the strides jump a window at a time.
	// If baseDilations, windowDilations are nil, they are assumed to be 1 (no dilation).
	// If paddings are nil they are assumed to be 0.
	ReduceWindow(x Op, reductionType ReduceOpType, windowDimensions, strides, baseDilations, windowDilations []int, paddings [][2]int) Op

	// RngBitGenerator generates the given shape filled with random bits.
	// It takes as input the current random number generator (RNG) state, see RngState or RngStateFromSeed.
	// The algorithm is hard-coded to use Philox algorithm for now.
	//
	// It returns the new state of the RNG and the generated values (with random bits) with the given shape.
	RngBitGenerator(state Op, shape shapes.Shape) (newState, values Op)

	// BatchNormForInference implements Batch Norm for inference. See details in
	// https://www.tensorflow.org/xla/operation_semantics#batchnorminference.
	//
	// Based on paper "Batch Normalization: Accelerating Deep Network Training by Reducing
	// Internal Covariate Shift" (Sergey Ioffe, Christian Szegedy), https://arxiv.org/abs/1502.03167.
	BatchNormForInference(operand, scale, offset, mean, variance Op, epsilon float32, axis int) Op

	// BatchNormForTraining implements Batch Norm for training. See details in
	// https://www.tensorflow.org/xla/operation_semantics#batchnormtraining.
	//
	// It returns the normalized tensor, the batchMean and the batchVariance.
	//
	// Based on paper "Batch Normalization: Accelerating Deep Network Training by Reducing
	// Internal Covariate Shift" (Sergey Ioffe, Christian Szegedy), https://arxiv.org/abs/1502.03167.
	BatchNormForTraining(operand, scale, offset Op, epsilon float32, axis int) (normalized, batchMean, batchVariance Op)

	// BatchNormGradient calculates the BatchNorm gradient. See details in
	// https://openxla.org/xla/operation_semantics#batchnormgrad
	//
	// The gradOutput is the adjoint gradient, that is, the gradient with respect to the output of the
	// batch normalization.
	//
	// It returns  as a tuple with the 3 elements.
	//
	// Based on paper "Batch Normalization: Accelerating Deep Network Training by Reducing
	// Internal Covariate Shift" (Sergey Ioffe, Christian Szegedy), https://arxiv.org/abs/1502.03167.
	BatchNormGradient(operand, scale, mean, variance, gradOutput Op, epsilon float32, axis int) (gradOperand, gradScale, gradOffset Op)

	// BitCount returns the number of bits that are set to one.
	BitCount(operand Op) Op

	// StandardOps include automatically generated list of operations for the Builder.
	StandardOps
}

// ConvolveAxesConfig defines the interpretation of the input/kernel/output tensor axes.
// There must be the same number of spatial dimensions (axes) for each of the 3 tensors.
// Input and output has batch and channel axes. Kernel has inputChannel and outputChannel axes.
//
// See Builder.ConvGeneralDilated
type ConvolveAxesConfig struct {
	InputBatch, InputChannel int
	InputSpatial             []int

	KernelInputChannel, KernelOutputChannel int
	KernelSpatial                           []int

	OutputBatch, OutputChannel int
	OutputSpatial              []int
}

// Clone returns a deep copy of the structure.
func (c ConvolveAxesConfig) Clone() ConvolveAxesConfig {
	var c2 ConvolveAxesConfig
	c2 = c
	c2.InputSpatial = slices.Clone(c.InputSpatial)
	c2.KernelSpatial = slices.Clone(c.KernelSpatial)
	c2.OutputSpatial = slices.Clone(c.OutputSpatial)
	return c2
}

// PadAxis defines the amount of padding preceding one axis (Start), at the end of axis (End)
// or in between the inputs (Interior).
// This is used as a parameter for the Pad operation.
type PadAxis struct {
	Start, End, Interior int
}

type FFTType int

const (
	// FFTForward - complex in, complex out.
	FFTForward FFTType = iota

	// FFTInverse - complex in, complex out.
	FFTInverse

	// FFTForwardReal - real in, fft_length / 2 + 1 complex out
	FFTForwardReal

	// FFTInverseReal - fft_length / 2 + 1 complex in
	FFTInverseReal
)

// Enumer can be installed with:
//
//	go install github.com/dmarkham/enumer@latest
//
// With go 1.24 we will add this to go tools support.

//go:generate enumer -type FFTType -trimprefix=FFT

// ReduceOpType select among the basic types of reduction supported, see XlaBuilder.ReduceComputation.
type ReduceOpType int

const (
	// ReduceOpUndefined is an undefined value.
	ReduceOpUndefined ReduceOpType = iota

	// ReduceOpSum reduces by summing all elements being reduced.
	ReduceOpSum

	// ReduceOpProduct reduces by multiplying all elements being reduced.
	ReduceOpProduct

	// ReduceOpMax reduces by taking the maximum value.
	ReduceOpMax

	// ReduceOpMin reduces by taking the minimum value.
	ReduceOpMin
)

// Enumer can be installed with:
//
//	go install github.com/dmarkham/enumer@latest
//
// With go 1.24 we will add this to go tools support.

//go:generate enumer -type ReduceOpType -trimprefix=ReduceOp
