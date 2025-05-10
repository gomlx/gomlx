package xla

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/protos/xla_data"
	"github.com/gomlx/gopjrt/xlabuilder"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"reflect"
)

// Builder implements the backends.Builder interface using github.com/gomlx/gopjrt/xlabuilder
type Builder struct {
	name    string
	backend *Backend
	builder *xlabuilder.XlaBuilder

	parameterNames  []string
	parameterShapes []shapes.Shape
}

var _ backends.Builder = (*Builder)(nil)

// Builder creates a new builder used to define a new computation.
func (backend *Backend) Builder(name string) backends.Builder {
	if err := backend.CheckValid(); err != nil {
		klog.Error(err)
		return nil
	}
	return &Builder{
		backend: backend,
		builder: xlabuilder.New(name),
		name:    name,
	}
}

// Name of the computation being built.
func (b *Builder) Name() string {
	return b.name
}

// castToXlaOp casts the op to xlabuilder.Op and panics if not possible.
func castToXlaOp(op backends.Op) *xlabuilder.Op {
	xop, ok := op.(*xlabuilder.Op)
	if !ok {
		exceptions.Panicf("buffer given is not a %q backend (pjrt) buffer", BackendName)
	}
	return xop
}

// CheckValid panics if the backend or the builder are not ok -- e.g.: if they have been finalized or the builder
// has already been compiled.
func (b *Builder) CheckValid() error {
	if b == nil || b.builder == nil {
		return errors.Errorf("builder is nil or undefined for %q", BackendName)
	}
	return b.backend.CheckValid()

}

func xshapeToShape(xshape xlabuilder.Shape) shapes.Shape {
	return shapes.Make(xshape.DType, xshape.Dimensions...)
}

func shapeToXShape(shape shapes.Shape) xlabuilder.Shape {
	return xlabuilder.MakeShape(shape.DType, shape.Dimensions...)
}

// verifyAndCastOp sanity checks that the op is valid and created with this builder.
func (b *Builder) verifyAndCastOp(op backends.Op, paramName string) (*xlabuilder.Op, error) {
	if err := b.CheckValid(); err != nil {
		return nil, err
	}
	xlaOp, ok := op.(*xlabuilder.Op)
	if !ok {
		return nil, errors.Errorf("nil or invalid Op (%T: %v) given as an input %s, it must be an Op created by the same backend builder (%s:%s)",
			op, op, paramName, b.backend.Name(), b.name)
	}
	if xlaOp.Builder() != b.builder {
		return nil, errors.Errorf("op given to parameter %s was created with a different builder (%s) than the builder (%s) it is being used in -- Ops cannot cross to different builders",
			paramName, xlaOp.Builder().Name(), b.Name())
	}
	return xlaOp, nil
}

// verifyAndCastOps verify each of the ops are valid and created with this builder.
func (b *Builder) verifyAndCastOps(ops []backends.Op, paramName string) ([]*xlabuilder.Op, error) {
	if err := b.CheckValid(); err != nil {
		return nil, err
	}
	xlaOps := make([]*xlabuilder.Op, len(ops))
	var err error
	for ii, op := range ops {
		xlaOps[ii], err = b.verifyAndCastOp(op, paramName)
		if err != nil {
			return nil, err
		}
	}
	return xlaOps, nil
}

// OpShape returns the shape of a computation Op.
func (b *Builder) OpShape(op backends.Op) (shapes.Shape, error) {
	if err := b.CheckValid(); err != nil {
		return shapes.Invalid(), err
	}
	xOp := castToXlaOp(op)
	return xshapeToShape(xOp.Shape), nil
}

// Parameter creates an input parameter for the computation.
// During execution of the computation this value will need to be fed, in the same order it is created.
func (b *Builder) Parameter(name string, shape shapes.Shape) (backends.Op, error) {
	op, err := xlabuilder.Parameter(b.builder, name, len(b.parameterNames), shapeToXShape(shape))
	if err != nil {
		return nil, errors.WithMessagef(err, "backend %q: Parameter(%q, %s)", BackendName, name, shape)
	}
	b.parameterNames = append(b.parameterNames, name)
	b.parameterShapes = append(b.parameterShapes, shape)
	return op, nil
}

// Constant creates a constant in the graph with the given flat values, and the shape defined by dims.
//
// flat must be a slice of a basic type supported -- that can be converted to a DType.
//
// The value is copied into the graph. It's recommended that for very large tensors,
// even if constants, that they are passed as side inputNodes (or variables, see context package) instead.
func (b *Builder) Constant(flat any, dims ...int) (backends.Op, error) {
	if err := b.CheckValid(); err != nil {
		return nil, err
	}
	flatV := reflect.ValueOf(flat)
	if flatV.Kind() != reflect.Slice {
		return nil, errors.Errorf("Constant expects a slice, got %T instead", flat)
	}
	dtype := dtypes.FromGoType(flatV.Type().Elem())
	if dtype == dtypes.InvalidDType {
		return nil, errors.Errorf("Constant expects a slice of valid DTypes, got %T instead", flat)
	}
	literal, err := xlabuilder.NewArrayLiteralFromAny(flat, dims...)
	if err != nil {
		return nil, errors.WithMessagef(err, "backend %q builder %q: Constant(%T, dims=%v)", b.backend.Name(), b.name, flat, dims)
	}
	op, err := xlabuilder.Constant(b.builder, literal)
	if err != nil {
		return nil, errors.WithMessagef(err, "backend %q builder %q: Constant(%T, dims=%v)", b.backend.Name(), b.name, flat, dims)
	}
	return op, nil
}

// Identity returns an Op whose output is the same as its input.
// It's a no-op that can serve as a place-holder.
func (b *Builder) Identity(x backends.Op) (backends.Op, error) {
	xlaX, err := b.verifyAndCastOp(x, "x")
	if err != nil {
		return nil, err
	}
	return xlabuilder.Identity(xlaX), nil
}

func (b *Builder) ReduceWindow(x backends.Op, reductionType backends.ReduceOpType, windowDimensions, strides, baseDilations, windowDilations []int, paddings [][2]int) (backends.Op, error) {
	xlaX, err := b.verifyAndCastOp(x, "x")
	if err != nil {
		return nil, err
	}
	cfg := xlabuilder.ReduceWindow(xlaX, windowDimensions).
		WithStrides(strides).
		WithBaseDilations(baseDilations).
		WithWindowDilations(windowDilations).
		WithPadding(paddings)
	switch reductionType {
	case backends.ReduceOpSum:
		cfg = cfg.Sum()
	case backends.ReduceOpMax:
		cfg = cfg.Max()
	case backends.ReduceOpMin:
		cfg = cfg.Min()
	case backends.ReduceOpProduct:
		cfg = cfg.Product()
	default:
		return nil, errors.Errorf("unknown reduction type %s given to ReduceWindow (building %q)", reductionType, b.name)
	}
	op, err := cfg.Done()
	if err != nil {
		return nil, errors.WithMessagef(err, "backend %q builder %q: ReduceWindow(reductionType=%s)", b.backend.Name(), b.name, reductionType)
	}
	return op, nil
}

// RngBitGenerator generates the given shape filled with random bits.
// It takes as input the current random number generator (RNG) state, see RngState or RngStateFromSeed.
// The algorithm is hard-coded to use Philox algorithm for now.
//
// It returns the new state of the RNG and the generated values (with random bits) with the given shape.
func (b *Builder) RngBitGenerator(state backends.Op, shape shapes.Shape) (newState, values backends.Op, err error) {
	xlaState, err := b.verifyAndCastOp(state, "x")
	if err != nil {
		return nil, nil, err
	}
	xlaShape := shapeToXShape(shape)
	newState, values, err = xlabuilder.RngBitGenerator(xlaState, xlaShape)
	if err != nil {
		return nil, nil, errors.WithMessagef(err, "backend %q builder %q: RngBitGenerator(shape=%s)", b.backend.Name(), b.name, shape)
	}
	return
}

// BatchNormForTraining implements Batch Norm for training. See details in
// https://www.tensorflow.org/xla/operation_semantics#batchnormtraining.
//
// It returns the normalized tensor, the batchMean and the batchVariance.
//
// Based on paper "Batch Normalization: Accelerating Deep Network Training by Reducing
// Internal Covariate Shift" (Sergey Ioffe, Christian Szegedy), https://arxiv.org/abs/1502.03167.
func (b *Builder) BatchNormForTraining(operand, scale, offset backends.Op, epsilon float32, axis int) (normalized, batchMean, batchVariance backends.Op, err error) {
	xlaOperand, err := b.verifyAndCastOp(operand, "operand")
	if err != nil {
		return
	}
	xlaScale, err := b.verifyAndCastOp(scale, "scale")
	if err != nil {
		return
	}
	xlaOffset, err := b.verifyAndCastOp(offset, "offset")
	if err != nil {
		return
	}
	normalized, batchMean, batchVariance, err = xlabuilder.BatchNormForTraining(xlaOperand, xlaScale, xlaOffset, epsilon, axis)
	if err != nil {
		err = errors.WithMessagef(err, "backend %q builder %q: BatchNormForTraining(operand=%s)", b.backend.Name(), b.name, xlaOperand.Shape)
		return
	}
	return
}

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
func (b *Builder) BatchNormGradient(operand, scale, mean, variance, gradOutput backends.Op, epsilon float32, axis int) (gradOperand, gradScale, gradOffset backends.Op, err error) {
	xlaOperand, err := b.verifyAndCastOp(operand, "operand")
	if err != nil {
		err = errors.WithMessagef(err, "calling BatchNormGradient()")
		return
	}
	xlaScale, err := b.verifyAndCastOp(scale, "scale")
	if err != nil {
		err = errors.WithMessagef(err, "calling BatchNormGradient()")
		return
	}
	xlaMean, err := b.verifyAndCastOp(mean, "mean")
	if err != nil {
		err = errors.WithMessagef(err, "calling BatchNormGradient()")
		return
	}
	xlaVariance, err := b.verifyAndCastOp(variance, "variance")
	if err != nil {
		err = errors.WithMessagef(err, "calling BatchNormGradient()")
		return
	}
	xlaGradOutput, err := b.verifyAndCastOp(gradOutput, "gradOutput")
	if err != nil {
		err = errors.WithMessagef(err, "calling BatchNormGradient()")
		return
	}
	gradOperand, gradScale, gradOffset, err = xlabuilder.BatchNormGradient(xlaOperand, xlaScale, xlaMean, xlaVariance, xlaGradOutput, epsilon, axis)
	if err != nil {
		err = errors.WithMessagef(err, "backend %q builder %q: BatchNormGradient(operand=%s)", b.backend.Name(), b.name, xlaOperand.Shape)
		return
	}
	return
}

func convertConvolveAxesConfig(c backends.ConvolveAxesConfig) (xlaConfig xlabuilder.ConvolveAxesConfig) {
	xlaConfig = xlabuilder.ConvolveAxesConfig{
		InputBatch:          c.InputBatch,
		InputChannel:        c.InputChannel,
		InputSpatial:        c.InputSpatial,
		KernelInputChannel:  c.KernelInputChannel,
		KernelOutputChannel: c.KernelOutputChannel,
		KernelSpatial:       c.KernelSpatial,
		OutputBatch:         c.OutputBatch,
		OutputChannel:       c.OutputChannel,
		OutputSpatial:       c.OutputSpatial,
	}
	return
}

func convertPadAxis(pad backends.PadAxis) (xlaPad xlabuilder.PadAxis) {
	xlaPad = xlabuilder.PadAxis{
		Start:    pad.Start,
		End:      pad.End,
		Interior: pad.Interior,
	}
	return
}

func convertFFTType(fftType backends.FFTType) xla_data.FftType {
	switch fftType {
	case backends.FFTForward:
		return xla_data.FftType_FFT
	case backends.FFTInverse:
		return xla_data.FftType_IFFT
	case backends.FFTForwardReal:
		return xla_data.FftType_RFFT
	case backends.FFTInverseReal:
		return xla_data.FftType_IRFFT
	default:
		exceptions.Panicf("fft type %s is not supported", fftType)
		panic(nil) // To quiet IDE warning.
	}
}

// BitCount returns the number of bits that are set to one.
func (b *Builder) BitCount(x backends.Op) (backends.Op, error) {
	xlaX, err := b.verifyAndCastOp(x, "x")
	if err != nil {
		return nil, errors.WithMessagef(err, "Backend %q: failed BitCount", BackendName)
	}
	xlaResult, err := xlabuilder.PopulationCount(xlaX)
	if err != nil {
		return nil, errors.WithMessagef(err, "Backend %q: failed BitCount", BackendName)
	}
	return xlaResult, nil
}
