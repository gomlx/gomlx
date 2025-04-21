package simplego

import (
	"github.com/gomlx/exceptions"
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

	// numNodesToProcess is the max(outputs). We don't need to look or store
	// information above that.
	numNodesToProcess int

	// numUses is the number of times each Node is used during the calculation.
	// It has the length of numNodesToProcess.
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

	// owned indicates whether the corresponding buffer in results is owned by the executor:
	// in which case it's either a temporary buffer or was donated by the caller.
	// That means after use the corresponding buffer and can be reused or freed after it's no longer used.
	owned []bool
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
	var numNodesToProcess int
	for _, output := range builder.outputs {
		numNodesToProcess = max(numNodesToProcess, output.builderIdx+1)
	}

	e := &Executable{
		builder:           builder,
		numNodesToProcess: numNodesToProcess,
		numUses:           make([]int, numNodesToProcess),
		executionBuffersPool: sync.Pool{
			New: func() interface{} {
				return &executionBuffers{
					results: make([]*Buffer, numNodesToProcess),
					numUsed: make([]int, numNodesToProcess),
					owned:   make([]bool, numNodesToProcess),
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

// nodeExecutor for the given operation type.
//
// It is given the buffers for its inputs, and a reserved buffer where to store its output, already
// with the shape pre-calculated.
type nodeExecutor func(node *Node, inputs []*Buffer, output *Buffer)

var (
	// nodeExecutors should be populated during initialization (`init` functions) for the ops implemented.
	// For the nodes not implemented, leave it as nil, and it will throw and exception.
	nodeExecutors [backends.OpTypeLast]nodeExecutor
)

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
	// Check inputs length
	if len(inputs) != len(e.builder.inputs) {
		exceptions.Panicf("Execute: expected %d inputs, got %d", len(e.builder.inputs), len(inputs))
	}

	// Check input shapes
	for ii, input := range inputs {
		inputBuffer, ok := input.(*Buffer)
		if !ok {
			exceptions.Panicf("Execute: input buffer #%d is not from SimpleGo backend", ii)
		}
		expectedShape := e.builder.inputs[ii].shape
		if !inputBuffer.shape.Equal(expectedShape) {
			exceptions.Panicf("Execute: input #%d expected shape %s, got %s",
				ii, expectedShape, inputBuffer.shape)
		}
	}

	// Get execution buffers from pool and reset numUsed.
	execBuf := e.executionBuffersPool.Get().(*executionBuffers)
	for ii := range e.numNodesToProcess {
		execBuf.numUsed[ii] = 0
		execBuf.owned[ii] = false
	}

	// Initialize "parameters" results with input buffers.
	for ii, input := range inputs {
		inputBuffer := input.(*Buffer)
		inputNodeIdx := e.builder.inputs[ii].builderIdx
		execBuf.results[inputNodeIdx] = inputBuffer
		execBuf.owned[inputNodeIdx] = donate[ii]
	}

	// The e.buffer.nodes are stored in a natural order for the Graph order, so
	// we can sequentially execute each one of them, and their node inputs will be
	// already calculated.
	for nodeIdx := range e.numNodesToProcess {
		if execBuf.results[nodeIdx] != nil {
			// Parameters will have its results pre-filled, and
			continue
		}
		if execBuf.numUsed[nodeIdx] == 0 {
			// This node is not used by any of the outputs of this executable.
			// TODO: these nodes can simply be removed in Builder.Compile().
			continue
		}
		node := e.builder.nodes[nodeIdx]

		// Constants have a special treatment, since they have no inputs and their outputs are not owned by
		// the execBuf.
		if node.opType == backends.OpTypeConstant {
			execBuf.results[nodeIdx] = node.data.(*Buffer)
			execBuf.owned[nodeIdx] = false
			continue
		}

		// Reserve a buffer for the node:
		outputBuf := getBuffer(node.shape.DType, node.shape.Size())
		outputBuf.shape = node.shape.Clone()
		execBuf.results[nodeIdx] = outputBuf
		execBuf.owned[nodeIdx] = true

		// Call node executor:
		executor := nodeExecutors[node.opType]
		if executor == nil {
			exceptions.Panicf("Execute: node executor for op type %s not implemented!?", node.opType)
		}
		inputBuffers := make([]*Buffer, len(node.inputs))
		for ii, input := range node.inputs {
			inputBuffer := execBuf.results[input.builderIdx]
			if inputBuffer == nil {
				exceptions.Panicf("Execute: input #%d of node #%d is not calculated yet (!?) -- "+
					"this is a bug, it should never have happened", ii, nodeIdx)
			}
			inputBuffers[ii] = inputBuffer
		}
		executor(node, execBuf.results, outputBuf)

		// See if corresponding inputs can be freed.
		for ii, input := range node.inputs {
			inputNodeIdx := input.builderIdx
			execBuf.numUsed[inputNodeIdx]++
			if execBuf.numUsed[inputNodeIdx] < e.numUses[inputNodeIdx] {
				// input node will still be used.
				continue
			}
			if !execBuf.owned[inputNodeIdx] {
				// we don't own the buffer.
				continue
			}
			// inputBuffer no longer used, we can return it to the pool.
			// Notice that the final outputs will always have numUses >= 1, and they
			// will never be completely used by the executor, hence they don't get freed here.
			inputBuf := execBuf.results[inputNodeIdx]
			putBuffer(inputBuf)
			inputBuf.shape = shapes.Invalid()
			execBuf.results[inputNodeIdx] = nil
		}
	}

	// Return outputs, copying them if not owned by the executor
	outputs := make([]backends.Buffer, len(e.builder.outputs))
	for ii, output := range e.builder.outputs {
		outIdx := output.builderIdx
		outBuf := execBuf.results[outIdx]
		if !execBuf.owned[outIdx] {
			// Make a copy of the buffer since we don't own it
			newBuf := getBuffer(outBuf.shape.DType, outBuf.shape.Size())
			newBuf.shape = outBuf.shape.Clone()
			copy(newBuf.flat.([]byte), outBuf.flat.([]byte))
			outBuf = newBuf
		}
		outputs[ii] = outBuf
	}

	// Return buffers to pool
	e.executionBuffersPool.Put(execBuf)
	return outputs
}
