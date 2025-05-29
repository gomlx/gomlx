package simplego

import (
	"fmt"
	"sync"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/pkg/errors"
)

var _ backends.Executable = (*Executable)(nil)

// Executable holds a frozen Builder. It assumes the graph in Builder is valid and has been properly
// checked that all the shapes and data types are valid.
//
// If any inconsistencies are found, please fix in the Builder, so Executable can be written without need
// of any duplicate checks.
type Executable struct {
	backend *Backend

	// builder must have Builder.compiled set to true, so it is no longer active.
	builder *Builder

	// numNodesToProcess is the max(outputs). We don't need to look or store
	// information above that.
	numNodesToProcess int

	// numUses is the number of times each Node is used during the calculation.
	// It has the length of numNodesToProcess.z
	numUses []int

	// executionBuffersPool allow for re-use of executionBuffers.
	executionBuffersPool sync.Pool

	// maxInputs of all nodes used in the graph.
	maxInputs int

	// dependents maps each node to the list of nodes that depend on it -- only count nodes that are used
	// by this executable.
	dependents [][]int
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

	// remainingDeps is the number of remaining dependencies for each node.
	remainingDeps []int
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

// Outputs returns the output shapes of the computation, in order given to the Builder.Compile call.
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
		backend:           builder.backend,
		builder:           builder,
		numNodesToProcess: numNodesToProcess,
		numUses:           make([]int, numNodesToProcess),
		executionBuffersPool: sync.Pool{
			New: func() interface{} {
				return &executionBuffers{
					results:       make([]*Buffer, numNodesToProcess),
					numUsed:       make([]int, numNodesToProcess),
					owned:         make([]bool, numNodesToProcess),
					remainingDeps: make([]int, numNodesToProcess),
				}
			},
		},
		dependents: make([][]int, numNodesToProcess),
	}

	// Find the largest number of inputs needed.
	for nodeIdx := range numNodesToProcess {
		e.maxInputs = max(e.maxInputs, len(builder.nodes[nodeIdx].inputs))
	}

	// Count uses for each node starting from outputs
	for _, output := range builder.outputs {
		e.countNodeUsesAndDependants(output)
	}

	return e
}

// countNodeUsesAndDependants recursively counts how many times a node is used.
func (e *Executable) countNodeUsesAndDependants(node *Node) {
	thisNodeIdx := node.builderIdx
	e.numUses[thisNodeIdx]++
	if e.numUses[thisNodeIdx] == 1 {
		// On the first visit, recursively, traverse inputs of node.
		for _, input := range node.inputs {
			e.dependents[input.builderIdx] = append(e.dependents[input.builderIdx], thisNodeIdx)
			e.countNodeUsesAndDependants(input)
		}
	}
}

// nodeExecutor for the given operation type.
//
// It is given the buffers for its inputs, and a reserved buffer where to store its output, already
// with the shape pre-calculated.
type nodeExecutor func(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error)

// nodeMultiOutputExecutor is a version of a node executor when it returns multiple outputs.
type nodeMultiOutputExecutor func(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) ([]*Buffer, error)

var (
	// nodeExecutors should be populated during initialization (`init` functions) for the ops implemented.
	// For the nodes not implemented, leave it as nil, and it will return an error.
	nodeExecutors [backends.OpTypeLast]nodeExecutor

	// multiOutputsNodeExecutors should be populated during initialization for the multi-output ops
	// implemented. E.g.: RngBitGenerator.
	multiOutputsNodeExecutors [backends.OpTypeLast]nodeMultiOutputExecutor
)

var execForceSequential bool

// Execute the executable on the default device (0).
// The number and shapes of the inputs must match those returned by Inputs.
//
// The inputs marked in donate will become invalid after use.
// This is useful if the input buffer is no longer needed or if updating a variable
// so its Buffer space can be reused as an output Buffer.
//
// Donated buffers are no longer valid after the call.
// If donate is nil, it is assumed to be false for all buffers, and no buffer is donated.
func (e *Executable) Execute(inputs []backends.Buffer, donate []bool) ([]backends.Buffer, error) {
	// Check inputs length
	if len(inputs) != len(e.builder.inputs) {
		return nil, errors.Errorf("Execute: expected %d inputs, got %d", len(e.builder.inputs), len(inputs))
	}

	// Check input shapes
	for ii, input := range inputs {
		if input == nil {
			return nil, errors.Errorf("Execute: input buffer #%d is nil!?", ii)
		}
		inputBuffer, ok := input.(*Buffer)
		if !ok {
			return nil, errors.Errorf("Execute: input buffer #%d is not from SimpleGo backend", ii)
		}
		if !inputBuffer.valid {
			return nil, errors.Errorf("Execute: input buffer (%p) #%d is not valid, likely it is being used after being finalized", inputBuffer, ii)
		}
		if inputBuffer.flat == nil {
			return nil, errors.Errorf("Execute: input buffer #%d flat data is set to nil (!?)", ii)
		}
		nodeInput := e.builder.inputs[ii]
		if !inputBuffer.shape.Equal(nodeInput.shape) {
			paramName := nodeInput.data.(*nodeParameter).name
			return nil, errors.Errorf("Execute: parameter %q (input #%d) for %q: expected shape %s, got %s",
				paramName, ii, e.builder.name, nodeInput.shape, inputBuffer.shape)
		}
	}

	// Get execution buffers from pool and reset numUsed.
	execBuf := e.executionBuffersPool.Get().(*executionBuffers)
	for ii := range e.numNodesToProcess {
		execBuf.numUsed[ii] = 0
		execBuf.owned[ii] = false
		execBuf.results[ii] = nil
		execBuf.remainingDeps[ii] = 0
	}

	// Initialize channel for nodes ready to execute and dependency tracking
	var (
		readyToExecute chan int // protected by execMu
		collectErrors  []error  // protected by execMu
		execMu         sync.Mutex
	)
	readyToExecute = make(chan int, e.numNodesToProcess)
	completed := 0
	expected := 0

	// Count expected nodes and initialize dependencies
	for nodeIdx := range e.numNodesToProcess {
		if e.numUses[nodeIdx] > 0 && execBuf.results[nodeIdx] == nil {
			expected++
			execBuf.remainingDeps[nodeIdx] = len(e.builder.nodes[nodeIdx].inputs)
			if execBuf.remainingDeps[nodeIdx] == 0 {
				readyToExecute <- nodeIdx
			}
		}
	}

	// Initialize "parameters" results with input buffers.
	for ii, input := range inputs {
		inputBuffer := input.(*Buffer)
		inputNodeIdx := e.builder.inputs[ii].builderIdx
		execBuf.results[inputNodeIdx] = inputBuffer
		execBuf.owned[inputNodeIdx] = donate[ii]
	}

	// Pre-allocate slices:
	var (
		inputBuffersPool = make([]*Buffer, e.maxInputs)
		inputsOwnedPool  = make([]bool, e.maxInputs)
	)

	// Execute nodes as they become available
	appendError := func(err error) {
		execMu.Lock()
		defer execMu.Unlock()
		collectErrors = append(collectErrors, err)
		close(readyToExecute)
	}

	for nodeIdx := range readyToExecute {
		nodeExecFn := func() {
			node := e.builder.nodes[nodeIdx]

			// On return, update dependencies and check if all outputs are complete.
			defer func() {
				execMu.Lock()
				defer execMu.Unlock()
				if len(collectErrors) > 0 {
					// Interrupted anyway.
					return
				}
				// Update dependencies and schedule ready nodes
				completed++
				if completed == expected {
					close(readyToExecute)
					return
				}

				// Check dependent nodes: for current node and for multi-output nodes.
				for _, depIdx := range e.dependents[nodeIdx] {
					execBuf.remainingDeps[depIdx]--
					if execBuf.remainingDeps[depIdx] == 0 && execBuf.results[depIdx] == nil {
						readyToExecute <- depIdx
					}
				}
				for _, outputNode := range node.multiOutputsNodes {
					outputIdx := outputNode.builderIdx
					if e.numUses[outputIdx] > 0 {
						// Mark this output as completed as well.
						completed++
						for _, depIdx := range e.dependents[outputIdx] {
							execBuf.remainingDeps[depIdx]--
							if execBuf.remainingDeps[depIdx] == 0 && execBuf.results[depIdx] == nil {
								readyToExecute <- depIdx
							}
						}
					}
				}
			}()

			if execBuf.results[nodeIdx] != nil {
				// Parameters and multi-output nodes will have their results pre-filled.
				return
			}
			if e.numUses[nodeIdx] == 0 {
				// This node is not used by any of the outputs of this executable.
				// TODO: these nodes can simply be removed in Builder.Compile().
				return
			}

			// Constants have a special treatment, since they have no inputs and their outputs are not owned by
			// the execBuf.
			if node.opType == backends.OpTypeConstant {
				execBuf.results[nodeIdx] = node.data.(*Buffer)
				execBuf.owned[nodeIdx] = false
				return
			}

			// Call node executor:
			var nodeExecutor nodeExecutor
			var multiNodeExecutor nodeMultiOutputExecutor
			if node.IsMultiOutputs() {
				multiNodeExecutor = multiOutputsNodeExecutors[node.opType]
				if multiNodeExecutor == nil {
					appendError(errors.Errorf("Execute: multi-outputs node executor for op type %s not implemented!?", node.opType))
					return
				}

			} else {
				nodeExecutor = nodeExecutors[node.opType]
				if nodeExecutor == nil {
					appendError(errors.Errorf("Execute: node executor for op type %s not implemented!?", node.opType))
					return
				}
			}
			var (
				inputBuffers []*Buffer
				inputsOwned  []bool
			)

			inputBuffers = inputBuffersPool[:len(node.inputs)]
			inputsOwned = inputsOwnedPool[:len(node.inputs)]
			for ii, input := range node.inputs {
				inputNodeIdx := input.builderIdx
				inputBuffers[ii] = execBuf.results[inputNodeIdx]
				if inputBuffers[ii] == nil {
					appendError(errors.Errorf("Execute: input #%d of node #%d is not calculated yet (!?) -- "+
						"this is a bug, it should never have happened", ii, nodeIdx))
					return
				}
				execBuf.numUsed[inputNodeIdx]++
				inputsOwned[ii] = execBuf.owned[inputNodeIdx] && execBuf.numUsed[inputNodeIdx] == e.numUses[inputNodeIdx]
			}

			// Execute op.
			if nodeExecutor != nil {
				// Single output operation:
				var err error
				execBuf.results[nodeIdx], err = nodeExecutor(e.backend, node, inputBuffers, inputsOwned)
				if err != nil {
					appendError(errors.WithMessagef(err, "while executing %q", node.opType))
					return
				}
				execBuf.owned[nodeIdx] = true

			} else {
				// Multi-output operation.
				outputs, err := multiNodeExecutor(e.backend, node, inputBuffers, inputsOwned)
				if err != nil {
					appendError(errors.WithMessagef(err, "while executing %q", node.opType))
					return
				}
				for outputIdx, outputBuf := range outputs {
					outputNodeIdx := node.multiOutputsNodes[outputIdx].builderIdx
					execBuf.results[outputNodeIdx] = outputBuf
					execBuf.owned[outputNodeIdx] = true
				}
			}

			// See if corresponding inputs can be freed.
			for ii, input := range node.inputs {
				if inputBuffers[ii] == nil {
					// Input was re-used (or consumed) by the nodeExecutor, remove it from results.
					// This is not strictly necessary, since the input can only be re-used if there are no more
					// uses of it in the graph, but just in case.
					execBuf.results[input.builderIdx] = nil
					continue
				}

				// If executor doesn't own buffer, we don't need to free it.
				if !inputsOwned[ii] || inputBuffers[ii] == nil {
					continue
				}

				// inputBuffer no longer used, we can return it to the pool.
				// Notice that the final outputs will always have numUses >= 1, and they
				// will never be completely used by the executor, hence they don't get freed here.
				//e.backend.putBuffer(inputBuffers[ii])
				//execBuf.results[input.builderIdx] = nil
			}
		}

		if execForceSequential {
			nodeExecFn()
		} else {
			e.backend.workers.WaitToStart(nodeExecFn)
		}
	}

	// If there were errors, return the first.
	if len(collectErrors) > 0 {
		return nil, collectErrors[0]
	}

	// Return outputs, copying them if not owned by the executor
	outputs := make([]backends.Buffer, len(e.builder.outputs))
	for ii, outputNode := range e.builder.outputs {
		outNodeIdx := outputNode.builderIdx
		outBuf := execBuf.results[outNodeIdx]
		if outBuf == nil {
			return nil, errors.Errorf("Execute: output #%d (%s, nodeIdx=%d) is not calculated yet (!?) -- "+
				"this is a bug, it should never have happened", ii, outputNode.opType, outNodeIdx)
		}
		if !outBuf.shape.Ok() {
			return nil, errors.Errorf("Execute: output #%d (%s, nodeIdx=%d) returned an invalid shape (!?) -- "+
				"this is a bug, it should never have happened", ii, outputNode.opType, outNodeIdx)
		}
		if !execBuf.owned[outNodeIdx] {
			// Make a copy of the buffer since we don't own it
			fmt.Printf("\tcloning output #%d (%s) @ %p\n", ii, outputNode.opType, outBuf)
			outBuf = e.backend.cloneBuffer(outBuf)
		}
		outputs[ii] = outBuf
	}

	// Return buffers to pool
	e.executionBuffersPool.Put(execBuf)
	return outputs, nil
}
