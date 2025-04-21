package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"sync"
)

// Executable holds a frozen Builder. It assumes the graph in Builder is valid and has been properly
// checked that all the shapes and data types are valid.
//
// If any inconsistencies are found, please fix in the Builder, so Executable can be written without need
// of any duplicate checks.
type Executable struct {
	// builder must have Builder.compiled set to true, so it is no longer active.
	builder *Builder

	// numUses is the number of times each Node is used during the calculation.
	// It has the same length as builder.Nodes.
	numUses []int

	// executionBuffersPool allow for re-use of executionBuffers.
	executionBuffersPool sync.Pool
}

// executionBuffers holds the intermediate results during the execution of the graph.
// One is created per execution of Executable.
type executionBuffers struct {
	// results hold the calculated computations at each step.
	// It has the same length as builder.Nodes.
	results []*Buffer

	// numUsed hold the number of times each node has been used already. Once they match numUses, the results buffer can
	// be released or re-used.
	// It has the same length as builder.Nodes.
	numUsed []int
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

// newExecutable creates an Executable ready to run the graph built with builder.
func newExecutable(builder *Builder) *Executable {
	e := &Executable{
		builder: builder,
		numUses: make([]int, len(builder.nodes)),
		executionBuffersPool: sync.Pool{
			New: func() interface{} {
				return &executionBuffers{
					results: make([]*Buffer, len(builder.nodes)),
					numUsed: make([]int, len(builder.nodes)),
				}
			},
		},
	}

	// Count uses for each node starting from outputs
	for _, output := range builder.outputs {
		countNodeUses(output, e.numUses)
	}

	return e
}

// countNodeUses recursively counts how many times a node is used.
func countNodeUses(node *Node, numUses []int) {
	numUses[node.builderIdx]++
	if numUses[node.builderIdx] == 1 {
		// On the first visit, recursively, traverse inputs of node.
		for _, input := range node.inputs {
			countNodeUses(input, numUses)
		}
	}
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
