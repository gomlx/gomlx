package simplego

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
)

// nodeParameter data.
type nodeParameter struct {
	name     string
	inputIdx int
}

// Parameter creates an input parameter for the computation.
// During execution of the computation this value will need to be fed, in the same order it is created.
func (b *Builder) Parameter(name string, shape shapes.Shape) backends.Op {
	dtype := shape.DType
	if dtype == dtypes.InvalidDType {
		exceptions.Panicf("invalid shape %s for Parameter", shape)
	}
	if supported, ok := Capabilities.DTypes[dtype]; !ok || !supported {
		exceptions.Panicf("Parameter: data type (DType) %s not supported for backend %q, try using "+
			"a different backend, or open an issue in github.com/gomlx/gomlx", dtype, b.backend.Name())
	}
	n := b.newNode(backends.OpTypeParameter, shape)
	n.data = &nodeParameter{
		name:     name,
		inputIdx: len(b.inputs),
	}
	b.inputs = append(b.inputs, n)
	return n
}

// Constant creates a constant in the graph with the given flat values, and the shape defined by dims.
//
// flat must be a slice of a basic type supported -- that can be converted to a DType.
//
// The value is copied into the graph. It's recommended that for very large tensors,
// even if constants, that they are passed as side inputNodes (or variables, see context package) instead.
func (b *Builder) Constant(flat any, dims ...int) backends.Op {
	_ = b.checkOps("Constant")
	dtype, flatLen := checkFlat(flat)
	if supported, ok := Capabilities.DTypes[dtype]; !ok || !supported {
		exceptions.Panicf("Constant: data type (DType) %s not supported for backend %q, try using "+
			"a different backend, or open an issue in github.com/gomlx/gomlx", dtype, b.backend.Name())
	}
	shape := shapes.Make(dtype, dims...)
	if shape.Size() != flatLen {
		exceptions.Panicf("flat ([%d]%s) and shape size (%d) mismatch for constant value",
			flatLen, dtype, shape.Size())
	}
	n := b.newNode(backends.OpTypeConstant, shape)
	n.data = &Buffer{
		shape: shape,
		flat:  flat,
	}
	return n
}

// Neg implements backends.Builder interface.
func (b *Builder) Neg(operand backends.Op) backends.Op {
	return b.addUnaryOp(backends.OpTypeNeg, operand)
}
