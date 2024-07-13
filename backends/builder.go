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
