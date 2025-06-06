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

	// numNodesToProcess is the max(outputs) -- with the exceptions of multi-output nodes.
	// We generally don't need to look or store information above that.
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

type opsExecutionType int

const (
	opsExecutionDynamic opsExecutionType = iota
	opsExecutionParallel
	opsExecutionSequential
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
func (e *Executable) Execute(inputs []backends.Buffer, donate []bool) ([]backends.Buffer, error) {
	// Keep count of the number of live executions.
	e.backend.numLiveExecutions.Add(1)
	defer e.backend.numLiveExecutions.Add(-1)

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
			return nil, errors.Errorf("Execute: input buffer (%p) #%d is not valid, likely it is being used after being isFinalized", inputBuffer, ii)
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

	// Initialize "parameters" results with input buffers.
	for ii, input := range inputs {
		inputBuffer := input.(*Buffer)
		inputNodeIdx := e.builder.inputs[ii].builderIdx
		execBuf.results[inputNodeIdx] = inputBuffer
		execBuf.owned[inputNodeIdx] = donate[ii]
	}

	// Decide if we are going to execute ops in parallel or sequentially:
	executionMode := e.backend.opsExecutionType
	if executionMode == opsExecutionDynamic {
		if e.backend.numLiveExecutions.Load() == 1 {
			// Current execution is the only one, so execute ops in parallel:
			executionMode = opsExecutionParallel
		} else {
			// Parallel execution of ops, while executing parallel execution programs
			// is less efficient.
			executionMode = opsExecutionSequential
		}
	}

	if executionMode == opsExecutionSequential {
		// Pre-allocate inputBuffers and inputsOwned: they will be
		// reused by every op.
		inputBuffers := make([]*Buffer, e.maxInputs)
		inputsOwned := make([]bool, e.maxInputs)

		// Loop over nodes sequentially: they are already sorted by their dependencies,
		// so nodes should be always ready to execute.
		for nodeIdx := range e.numNodesToProcess {
			node := e.builder.nodes[nodeIdx]
			if execBuf.results[nodeIdx] != nil {
				// Inputs, parameters and multi-output nodes will have their results pre-filled.
				continue
			}
			if e.numUses[nodeIdx] == 0 {
				// This node is not used by any of the outputs of this executable.
				// TODO: these nodes can simply be removed in Builder.Compile().
				continue
			}

			// Constants have a special treatment, since they have no inputs and their outputs are not owned by
			// the execBuf.
			if node.opType == backends.OpTypeConstant {
				execBuf.owned[nodeIdx] = false
				execBuf.results[nodeIdx] = node.data.(*Buffer)
				continue
			}

			// Prepare inputs:
			numInputs := len(node.inputs)
			inputBuffers = inputBuffers[:numInputs]
			inputsOwned := inputsOwned[:numInputs]
			for ii, input := range node.inputs {
				inputNodeIdx := input.builderIdx
				inputBuffers[ii] = execBuf.results[inputNodeIdx]
				if inputBuffers[ii] == nil || !inputBuffers[ii].shape.Ok() {
					return nil, errors.Errorf("Execute: input #%d of node #%d is not calculated yet (!?) -- "+
						"this is a bug, it should never have happened", ii, nodeIdx)
				}
				execBuf.numUsed[inputNodeIdx]++
				inputsOwned[ii] = execBuf.owned[inputNodeIdx] && e.numUses[inputNodeIdx] == execBuf.numUsed[inputNodeIdx]
			}

			// Call node executor:
			if node.IsMultiOutputs() {
				// Multi-output node:
				multiNodeExecutor := multiOutputsNodeExecutors[node.opType]
				if multiNodeExecutor == nil {
					return nil, errors.Errorf("Execute: multi-outputs node executor for op type %s not implemented!?", node.opType)
				}
				outputs, err := multiNodeExecutor(e.backend, node, inputBuffers, inputsOwned)
				if err != nil {
					return nil, errors.WithMessagef(err, "while executing %q", node.opType)
				}
				for outputIdx, outputBuf := range outputs {
					outputNode := node.multiOutputsNodes[outputIdx]
					outputNodeIdx := outputNode.builderIdx
					if outputIdx >= e.numNodesToProcess || e.numUses[outputIdx] == 0 {
						// Ignore, this is a node that is not part of the computation.
						e.backend.putBuffer(outputBuf)
						continue
					}
					execBuf.results[outputNodeIdx] = outputBuf
					execBuf.owned[outputNodeIdx] = true
				}

			} else {
				// Single-output node:
				nodeExecutor := nodeExecutors[node.opType]
				if nodeExecutor == nil {
					return nil, errors.Errorf("Execute: node executor for op type %s not implemented!?", node.opType)
				}
				var err error
				execBuf.results[nodeIdx], err = nodeExecutor(e.backend, node, inputBuffers, inputsOwned)
				if err != nil {
					return nil, errors.WithMessagef(err, "while executing %q", node.opType)
				}
				execBuf.owned[nodeIdx] = true
			}

			// If input has been reused, erase it from results.
			for ii, inputNode := range node.inputs {
				inputNodeIdx := inputNode.builderIdx
				if inputBuffers[ii] == nil {
					execBuf.results[inputNodeIdx] = nil
					continue
				} else if inputsOwned[ii] {
					// Release immediately input result: we no longer need it.
					e.backend.putBuffer(inputBuffers[ii])
					execBuf.results[inputNodeIdx] = nil
				}
			}
		} // Sequential execution loop.

	} else {
		// Parallel execution: executionMode == opsExecutionParallel

		// Initialize chanel for nodes ready to execute and dependency tracking
		var (
			readyToExecute chan int // protected by execMu
			collectErrors  []error  // protected by execMu
			execMu         sync.Mutex
		)
		readyToExecute = make(chan int, e.numNodesToProcess+10)
		stopExecution := sync.OnceFunc(func() { close(readyToExecute) }) // Can be called concurrently.

		// expected is the number of nodes that needs executing to complete the computation.
		expected := 0
		// completed is the number of required nodes that have been executed.
		completed := 0

		// Count expected nodes and initialize dependencies
		for nodeIdx := range e.numNodesToProcess {
			if e.numUses[nodeIdx] > 0 {
				expected++
				execBuf.remainingDeps[nodeIdx] = len(e.builder.nodes[nodeIdx].inputs)
				if execBuf.remainingDeps[nodeIdx] == 0 {
					readyToExecute <- nodeIdx
				}
			}
		}

		// Execute nodes as they become available
		appendError := func(err error) {
			execMu.Lock()
			defer execMu.Unlock()
			collectErrors = append(collectErrors, err)
			stopExecution()
		}

		// Loop over nodes that are ready to execute:
		for nodeIdx := range readyToExecute {
			nodeExecFn := func() {
				node := e.builder.nodes[nodeIdx]

				// On return, it updates the dependencies and checks that all outputs are complete.
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
						//fmt.Printf("\t- node #%d: all nodes completed, closing channel\n", nodeIdx)
						stopExecution()
						return
					}
					//if completed > expected {
					//	fmt.Printf("\t- node #%d: more nodes than expected (%d > %d), stopping execution\n", nodeIdx, completed, expected)
					//}

					// Check dependent nodes: for current node and for multi-output nodes.
					if node.IsMultiOutputs() {
						// Multi-output node: process each output, mark them as completed and enqueue their dependencies.
						for _, outputNode := range node.multiOutputsNodes {
							outputIdx := outputNode.builderIdx
							if outputIdx >= e.numNodesToProcess || e.numUses[outputIdx] == 0 {
								// Ignore, this is a node that is not part of the computation.
								continue
							}

							// Mark this output as completed as well.
							completed++
							if completed == expected {
								//fmt.Printf("\t- node #%d: all nodes completed, closing channel\n", nodeIdx)
								stopExecution()
								return
							}

							// Enqueue read dependencies:
							for _, depIdx := range e.dependents[outputIdx] {
								execBuf.remainingDeps[depIdx]--
								if execBuf.remainingDeps[depIdx] == 0 && execBuf.results[depIdx] == nil {
									//fmt.Printf("\t- node #%d: enqueueing #%d (%s) -- a dependency from it's multi-output node #%d\n", nodeIdx, depIdx, e.builder.nodes[depIdx].opType, outputIdx)
									readyToExecute <- depIdx
								}
							}
						}

					} else {
						// Single output node: enqueue all dependent nodes that are ready to execute.
						for _, depIdx := range e.dependents[nodeIdx] {
							execBuf.remainingDeps[depIdx]--
							if execBuf.remainingDeps[depIdx] == 0 {
								//fmt.Printf("\t- node #%d: enqueueing #%d (%s), since it has no more dependencies and %d uses\n", nodeIdx, depIdx, e.builder.nodes[depIdx].opType, e.numUses[depIdx])
								readyToExecute <- depIdx
							}
						}
					}
				}()

				//fmt.Printf("Execute: starting to process node #%d (%s): inputs=%v\n", nodeIdx, node.opType,
				//	xslices.Map(node.inputs, func(input *Node) int { return input.builderIdx }))
				if execBuf.results[nodeIdx] != nil {
					// Parameters and multi-output nodes will have their results pre-filled.
					//fmt.Printf("\t- node #%d (%s) already calculated, skipping\n", nodeIdx, node.opType)
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
					execBuf.owned[nodeIdx] = false
					execBuf.results[nodeIdx] = node.data.(*Buffer)
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

				inputBuffers := make([]*Buffer, len(node.inputs))
				inputsOwned := make([]bool, len(node.inputs))
				for ii, input := range node.inputs {
					inputNodeIdx := input.builderIdx
					inputBuffers[ii] = execBuf.results[inputNodeIdx]
					if inputBuffers[ii] == nil || !inputBuffers[ii].shape.Ok() {
						//fmt.Printf("\t- node #%d (%s) input[%d]=#%d (%s -> %s) is not calculated yet!?\n", nodeIdx, node.opType, ii, inputNodeIdx, input.opType, input.shape)
						appendError(errors.Errorf("Execute: input #%d of node #%d is not calculated yet (!?) -- "+
							"this is a bug, it should never have happened", ii, nodeIdx))
						return
					}
					// Because of a possible race condition, we just allow the input to be reused if there is only one
					// node using it.
					inputsOwned[ii] = execBuf.owned[inputNodeIdx] && e.numUses[inputNodeIdx] == 1
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
						outputNode := node.multiOutputsNodes[outputIdx]
						outputNodeIdx := outputNode.builderIdx
						if outputIdx >= e.numNodesToProcess || e.numUses[outputIdx] == 0 {
							// Ignore, this is a node that is not part of the computation.
							e.backend.putBuffer(outputBuf)
							continue
						}
						execBuf.results[outputNodeIdx] = outputBuf
						execBuf.owned[outputNodeIdx] = true
					}
				}

				// If input has been reused, erase it from results.
				for ii, inputNode := range node.inputs {
					inputNodeIdx := inputNode.builderIdx
					if inputBuffers[ii] == nil {
						execBuf.results[inputNodeIdx] = nil
						continue
					}
				}
			}

			// Schedule this op (nodeIdx) for execution.
			e.backend.workers.WaitToStart(nodeExecFn)
		}

		// If there were errors, return the first.
		if len(collectErrors) > 0 {
			return nil, collectErrors[0]
		}

	} // If sequential/parallel execution.

	// Return outputs, copying them if not owned by the executor
	outputs := make([]backends.Buffer, len(e.builder.outputs))
	for ii, outputNode := range e.builder.outputs {
		outNodeIdx := outputNode.builderIdx
		outBuf := execBuf.results[outNodeIdx]
		execBuf.results[outNodeIdx] = nil // Make sure we don't return the same buffer twice.
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

	// Free other intermediate buffers that haven't been freed yet.
	for nodeIdx, buf := range execBuf.results {
		if buf == nil {
			continue
		}
		if !execBuf.owned[nodeIdx] {
			continue
		}
		e.backend.putBuffer(buf)
		execBuf.results[nodeIdx] = nil
	}

	// Return buffers to pool
	e.executionBuffersPool.Put(execBuf)
	return outputs, nil
}
