package stablehlo

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/notimplemented"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/xlabuilder"
	"github.com/gomlx/stablehlo"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// Builder keeps track of the computation graph being defined.
type Builder struct {
	notimplemented.Builder

	name     string
	backend  *Backend
	compiled bool

	builder *stablehlo.Builder
	fn      *stablehlo.Function
}

var _ backends.Builder = (*Builder)(nil)

// Builder creates a new builder used to define a new computation.
func (backend *Backend) Builder(name string) backends.Builder {
	if err := backend.CheckValid(); err != nil {
		klog.Error(err)
		return nil
	}
	b := &Builder{
		backend: backend,
		builder: stablehlo.New(name),
		name:    name,
	}
	b.fn = b.builder.Main()
	return b
}

// Node represents the output of an operation and implements a backends.Op
type Node struct {
	value   *stablehlo.Value
	shape   shapes.Shape
	builder *Builder
}

// CheckValid returns an error if the backend or the builder are not ok.
//
// E.g.: they have been finalized or the builder has already been compiled.
func (b *Builder) CheckValid() error {
	if b == nil || b.builder == nil {
		return errors.Errorf("builder is nil or undefined for %q", BackendName)
	}
	return b.backend.CheckValid()
}

// verifyAndCastOp sanity checks that the op is valid and created with this builder.
func (b *Builder) verifyAndCastOp(op backends.Op, opName string) (*Node, error) {
	if err := b.CheckValid(); err != nil {
		return nil, err
	}
	if node == nil {
		return nil, errors.Errorf("nil Op given as an input to %q", opName)
	}
	node, ok := op.(*Node)
	if !ok {
		return nil, errors.Errorf("nil or invalid Op (%T: %v) given as an input to %q, it must be an Op created by the same backend builder (%s:%s)",
			op, op, opName, b.backend.Name(), b.name)
	}
	if node.builder != b {
		return nil, errors.Errorf("op given to parameter %s was created with a different builder (%s) than the builder (%s) it is being used in -- Ops cannot cross to different builders",
			opName, node.builder.Name(), b.Name())
	}
	return node, nil
}

// OpShape returns the shape of a computation Op.
func (b *Builder) OpShape(op backends.Op) (shapes.Shape, error) {
	if err := b.CheckValid(); err != nil {
		return shapes.Invalid(), err
	}
	node, err := b.verifyAndCastOp(op, "OpShape")
	if err != nil {
		return shapes.Invalid(), err
	}
	return node.shape, nil
}

func (b *Builder) newNode(value *stablehlo.Value) *Node {
	return &Node{
		value:   value,
		shape:   ShapeFromStableHLO(value.Shape()),
		builder: b,
	}
}

// Parameter creates an input parameter for the computation.
//
// During the computation's execution this value will need to be fed, in the same order it is created.
func (b *Builder) Parameter(name string, shape shapes.Shape) (backends.Op, error) {
	if err := b.CheckValid(); err != nil {
		return nil, err
	}
	value := b.fn.NewNamedInput(name, ShapeToStableHLO(shape))
	return b.newNode(value), nil
}

// Identity returns an Op whose output is the same as its input.
// It's a no-op that can serve as a place-holder.
func (b *Builder) Identity(x backends.Op) (backends.Op, error) {
	if err := b.CheckValid(); err != nil {
		return nil, err
	}
	node, err := b.verifyAndCastOp(x, "OpShape")
	if err != nil {
		return nil, err
	}
	return node, nil
}

// Constant creates a constant in the graph with the given flat values, and the shape defined by dims.
//
// The flat value must be a slice of a basic type supported -- that can be converted to a DType.
//
// The value is copied into the graph. It's recommended that for very large tensors,
// even if constants, that they are passed as side inputNodes (or variables, see context package) instead.
func (b *Builder) Constant(flat any, dims ...int) (backends.Op, error) {
	if err := b.CheckValid(); err != nil {
		return nil, err
	}
	if flat == nil {
		return nil, errors.Errorf("nil value given to Constant")
	}
	b.fn.NewConstant()
}
