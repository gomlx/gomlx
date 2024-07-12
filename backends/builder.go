package backends

import (
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/pkg/errors"
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

	// OpShape returns the shape of a computation Op.
	OpShape(op Op) shapes.Shape

	// Parameter creates an input parameter for the computation.
	// During execution of the computation this value will need to be fed, in the same order it is created.
	Parameter(name string, shape shapes.Shape) Op
}
