package xla

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/xlabuilder"
	"github.com/pkg/errors"
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
	xop := castToXlaOp(op)
	return xshapeToShape(xop.Shape)
}

// Parameter creates an input parameter for the computation.
// During execution of the computation this value will need to be fed, in the same order it is created.
func (b *Builder) Parameter(name string, shape shapes.Shape) backends.Op {
	xop, err := xlabuilder.Parameter(b.builder, name, len(b.parameterNames), shapeToXShape(shape))
	if err != nil {
		panic(errors.WithMessagef(err, "backend %q: Parameter(%q, %s)", BackendName, name, shape))
	}
	b.parameterNames = append(b.parameterNames, name)
	b.parameterShapes = append(b.parameterShapes, shape)
	return xop
}
