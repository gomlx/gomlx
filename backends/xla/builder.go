package xla

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/xlabuilder"
	"github.com/pkg/errors"
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

// Builder creates a new builder used to define a new computation.
func (backend *Backend) Builder(name string) backends.Builder {
	backend.AssertValid()
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

// AssertValid panics if the backend or the builder are not ok -- e.g.: if they have been finalized or the builder
// has already been compiled.
func (b *Builder) AssertValid() {
	b.backend.AssertValid()
}

func xshapeToShape(xshape xlabuilder.Shape) shapes.Shape {
	return shapes.Make(xshape.DType, xshape.Dimensions...)
}

func shapeToXShape(shape shapes.Shape) xlabuilder.Shape {
	return xlabuilder.MakeShape(shape.DType, shape.Dimensions...)
}

// OpShape returns the shape of a computation Op.
func (b *Builder) OpShape(op backends.Op) shapes.Shape {
	b.AssertValid()
	xOp := castToXlaOp(op)
	return xshapeToShape(xOp.Shape)
}

// Parameter creates an input parameter for the computation.
// During execution of the computation this value will need to be fed, in the same order it is created.
func (b *Builder) Parameter(name string, shape shapes.Shape) backends.Op {
	op, err := xlabuilder.Parameter(b.builder, name, len(b.parameterNames), shapeToXShape(shape))
	if err != nil {
		panic(errors.WithMessagef(err, "backend %q: Parameter(%q, %s)", BackendName, name, shape))
	}
	b.parameterNames = append(b.parameterNames, name)
	b.parameterShapes = append(b.parameterShapes, shape)
	return op
}

// Constant creates a constant in the graph with the given flat values, and the shape defined by dims.
//
// flat must be a slice of a basic type supported -- that can be converted to a DType.
//
// The value is copied into the graph. It's recommended that for very large tensors,
// even if constants, that they are passed as side inputNodes (or variables, see context package) instead.
func (b *Builder) Constant(flat any, dims ...int) backends.Op {
	flatV := reflect.ValueOf(flat)
	if flatV.Kind() != reflect.Slice {
		exceptions.Panicf("Constant expects a slice, got %T instead", flat)
	}
	dtype := dtypes.FromGoType(flatV.Type().Elem())
	if dtype == dtypes.InvalidDType {
		exceptions.Panicf("Constant expects a slice of valid DTypes, got %T instead", flat)
	}
	literal := xlabuilder.NewArrayLiteralFromAny(flat, dims...)
	op, err := xlabuilder.Constant(b.builder, literal)
	if err != nil {
		panic(errors.WithMessagef(err, "backend %q: Constant(%T, dims=%v)", b.name, flat, dims))
	}
	return op
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
