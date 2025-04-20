package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
)

type Executable struct {
	// builder must have Builder.compiled set to true, so it is no longer active.
	builder *Builder
}

// Compile time check.
var _ backends.Executable = (*Executable)(nil)

// Finalize immediately frees resources associated to the executable.
func (e *Executable) Finalize() {
	e.builder.Finalize()
	e.builder = nil
	return
}

// Inputs returns the list of parameters names and shapes, in order created by the Builder.Parameter calls.
func (e *Executable) Inputs() (names []string, inputShapes []shapes.Shape) {
	numInputs := len(e.builder.inputs)
	if numInputs == 0 {
		return
	}
	names = make([]string, numInputs)
	inputShapes = make([]shapes.Shape, numInputs)
	for ii, node := range e.builder.inputs {
		parameter := e.builder.inputs[ii].data.(*nodeParameter)
		names[ii] = parameter.name
		inputShapes[ii] = node.shape
	}
	return
}

// Outputs returns the list of the shapes of the outputs of the computation, in order given to the Builder.Compile call.
func (e *Executable) Outputs() (outputShapes []shapes.Shape) {
	numOutputs := len(e.builder.outputs)
	if numOutputs == 0 {
		return
	}
	outputShapes = make([]shapes.Shape, numOutputs)
	for ii, node := range e.builder.outputs {
		outputShapes[ii] = node.shape
	}
	return outputShapes
}

// Execute the executable on the default device (0).
// The number and shapes of the inputs must match those returned by Inputs.
//
// The inputs marked in donate will become invalid after use.
// This is useful if the input buffer is no longer needed or if updating a variable
// so its Buffer space can be reused as an output Buffer.
//
// Donated buffers are no longer valid after the call.
// If donate is nil, it is assumed to be false for all buffers, and no buffer is donated.
func (e *Executable) Execute(inputs []backends.Buffer, donate []bool) []backends.Buffer {
	return nil
}
