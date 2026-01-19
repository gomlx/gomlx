// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"sync"
	"sync/atomic"

	"github.com/gomlx/gomlx/backends"
	"github.com/pkg/errors"
)

// FunctionExecutable contains pre-compiled execution information for any function.
// This is used for both the main function and closures, unifying their execution model.
type FunctionExecutable struct {
	// function is the source Function this was compiled from.
	function *Function

	// numNodesToProcess is the max(outputs.builderIdx)+1.
	// Arrays are sized to this to allow direct builderIdx indexing.
	numNodesToProcess int

	// numUses tracks how many times each node's result is used (indexed by builderIdx).
	numUses []int

	// dependents maps each node (by builderIdx) to the list of dependent node builderIdxs.
	dependents [][]int

	// outputNodes are the nodes that produce the function's outputs.
	outputNodes []*Node

	// maxInputs is the maximum number of inputs any node has.
	maxInputs int

	// executionBuffersPool allows reuse of execution buffers.
	executionBuffersPool sync.Pool
}

// newFunctionExecutable creates a FunctionExecutable for the given function.
// The function must have Return() called (f.returned == true).
func newFunctionExecutable(f *Function) (*FunctionExecutable, error) {
	if !f.returned {
		return nil, errors.Errorf("function must have Return() called before compilation")
	}

	// Use total node count to handle graphs with unused nodes (dead code).
	// Previously this only considered output nodes, which caused index out of range
	// errors when the graph contained nodes with higher builderIdx than outputs.
	numNodesToProcess := len(f.builder.nodes)

	fe := &FunctionExecutable{
		function:          f,
		outputNodes:       f.outputs,
		numNodesToProcess: numNodesToProcess,
		numUses:           make([]int, numNodesToProcess),
		dependents:        make([][]int, numNodesToProcess),
	}

	// Find max inputs (including captured inputs) and count uses/dependents
	for nodeIdx := range numNodesToProcess {
		node := f.builder.nodes[nodeIdx]
		// Total inputs = regular inputs + captured inputs
		totalInputs := len(node.inputs) + len(node.capturedInputs)
		fe.maxInputs = max(fe.maxInputs, totalInputs)
	}

	// Count uses for each node starting from outputs
	for _, output := range f.outputs {
		fe.countNodeUsesAndDependents(output)
	}

	// Initialize execution buffers pool
	fe.executionBuffersPool = sync.Pool{
		New: func() interface{} {
			return &funcExecBuffers{
				results:       make([]*Buffer, numNodesToProcess),
				numUsed:       make([]atomic.Int32, numNodesToProcess),
				owned:         make([]bool, numNodesToProcess),
				remainingDeps: make([]int, numNodesToProcess),
			}
		},
	}

	return fe, nil
}

// countNodeUsesAndDependents recursively counts how many times a node is used.
// It tracks both regular inputs and captured inputs (for closure-calling ops).
func (fe *FunctionExecutable) countNodeUsesAndDependents(node *Node) {
	nodeIdx := node.builderIdx
	fe.numUses[nodeIdx]++
	if fe.numUses[nodeIdx] == 1 {
		// On the first visit, recursively traverse inputs of the node.
		for _, input := range node.inputs {
			fe.dependents[input.builderIdx] = append(fe.dependents[input.builderIdx], nodeIdx)
			fe.countNodeUsesAndDependents(input)
		}
		// Also track captured inputs for closure-calling ops (If, While, Sort, etc.).
		// This ensures captured values are properly tracked in the dependency graph
		// so they can be freed when no longer needed.
		for _, capturedInput := range node.capturedInputs {
			fe.dependents[capturedInput.builderIdx] = append(fe.dependents[capturedInput.builderIdx], nodeIdx)
			fe.countNodeUsesAndDependents(capturedInput)
		}
	}
}

// funcExecBuffers holds intermediate results during function execution.
type funcExecBuffers struct {
	// results hold the calculated computations at each step (indexed by builderIdx).
	results []*Buffer

	// numUsed tracks how many times each node has been used already.
	// Uses atomic.Int32 to allow safe concurrent reads in ownership checks.
	numUsed []atomic.Int32

	// owned indicates whether the corresponding buffer is owned by the executor.
	owned []bool

	// remainingDeps is the number of remaining dependencies for each node.
	remainingDeps []int

	// opsExecutionType can be sequential or parallel.
	opsExecutionType opsExecutionType

	// Sequential execution-only: reused for each op.
	opInputBuffers []*Buffer
	opInputsOwned  []bool

	// Parallel execution only: protects shared state.
	mu sync.Mutex
}

// Execute runs the compiled function with the given inputs.
// The inputs must match the function's parameters in count and shape.
// capturedInputs are the values captured from parent scopes (for closures).
// donateCaptures indicates which captured inputs can be donated to the closure.
// If donateCaptures is nil, no captured inputs will be donated.
func (fe *FunctionExecutable) Execute(backend *Backend, inputs []*Buffer, donate []bool, capturedInputs []*Buffer, donateCaptures []bool) ([]*Buffer, error) {
	// Use function's parameters (not builder.inputs) for proper function/closure support
	funcParams := fe.function.parameters
	if len(inputs) != len(funcParams) {
		return nil, errors.Errorf("function expects %d inputs, got %d",
			len(funcParams), len(inputs))
	}

	// Validate captured inputs count
	if len(capturedInputs) != len(fe.function.capturedLocalNodes) {
		return nil, errors.Errorf("function expects %d captured values, got %d",
			len(fe.function.capturedLocalNodes), len(capturedInputs))
	}

	// donate defaults to false
	if len(donate) == 0 {
		donate = make([]bool, len(inputs))
	}

	// donateCaptures defaults to false (no donation)
	if len(donateCaptures) == 0 {
		donateCaptures = make([]bool, len(capturedInputs))
	}

	// Get execution buffers from pool and reset
	execBuf := fe.executionBuffersPool.Get().(*funcExecBuffers)
	for i := range fe.numNodesToProcess {
		execBuf.numUsed[i].Store(0)
		execBuf.owned[i] = false
		execBuf.results[i] = nil
		execBuf.remainingDeps[i] = 0
	}

	// Set up parameters from inputs using builderIdx directly
	for i, inputNode := range funcParams {
		inputIdx := inputNode.builderIdx
		execBuf.results[inputIdx] = inputs[i]
		execBuf.owned[inputIdx] = donate[i]
	}

	// Set up captured values from parent scope.
	// If donateCaptures[i] is true, the closure takes ownership of the buffer.
	for i, captureNode := range fe.function.capturedLocalNodes {
		captureIdx := captureNode.builderIdx
		execBuf.results[captureIdx] = capturedInputs[i]
		execBuf.owned[captureIdx] = donateCaptures[i]
	}

	// Decide execution mode
	executionMode := backend.opsExecutionType
	if executionMode == opsExecutionDynamic {
		if backend.numLiveExecutions.Load() <= 1 {
			executionMode = opsExecutionParallel
		} else {
			executionMode = opsExecutionSequential
		}
	}
	execBuf.opsExecutionType = executionMode

	// Execute
	var err error
	if executionMode == opsExecutionSequential {
		err = fe.executeSequentially(backend, execBuf)
	} else {
		err = fe.executeParallel(backend, execBuf)
	}
	if err != nil {
		fe.executionBuffersPool.Put(execBuf)
		return nil, err
	}

	// Collect outputs
	outputs := make([]*Buffer, len(fe.outputNodes))
	for i, outNode := range fe.outputNodes {
		outIdx := outNode.builderIdx
		outputs[i] = execBuf.results[outIdx]
		if outputs[i] == nil {
			fe.executionBuffersPool.Put(execBuf)
			return nil, errors.Errorf("output %d not computed", i)
		}
		if !execBuf.owned[outIdx] {
			// Clone the buffer since we don't own it
			outputs[i] = backend.cloneBuffer(execBuf.results[outIdx])
		}
		execBuf.results[outIdx] = nil // Prevent double-free
	}

	// Free any remaining owned buffers that weren't outputs
	for idx, buf := range execBuf.results {
		if buf != nil && execBuf.owned[idx] {
			backend.putBuffer(buf)
		}
	}

	fe.executionBuffersPool.Put(execBuf)
	return outputs, nil
}

// executeSequentially executes nodes one after another in topological order.
func (fe *FunctionExecutable) executeSequentially(backend *Backend, execBuf *funcExecBuffers) error {
	// Pre-allocate input buffers for reuse
	execBuf.opInputBuffers = make([]*Buffer, fe.maxInputs)
	execBuf.opInputsOwned = make([]bool, fe.maxInputs)
	defer func() {
		execBuf.opInputBuffers = nil
		execBuf.opInputsOwned = nil
	}()

	for nodeIdx := range fe.numNodesToProcess {
		if execBuf.results[nodeIdx] != nil {
			// Already computed (parameter)
			continue
		}
		if fe.numUses[nodeIdx] == 0 {
			// Not used by any output
			continue
		}

		node := fe.function.builder.nodes[nodeIdx]
		if err := fe.executeNode(backend, node, execBuf); err != nil {
			return err
		}
	}
	return nil
}

// executeParallel executes nodes in parallel based on dependency graph.
func (fe *FunctionExecutable) executeParallel(backend *Backend, execBuf *funcExecBuffers) error {
	var (
		readyToExecute chan int
		collectErrors  []error
		execMu         sync.Mutex
	)
	readyToExecute = make(chan int, fe.numNodesToProcess+10)
	stopExecutionFn := sync.OnceFunc(func() { close(readyToExecute) })

	expected := 0
	completed := 0

	// Count expected nodes and initialize dependencies
	// Dependencies include both regular inputs and captured inputs
	for nodeIdx := range fe.numNodesToProcess {
		if fe.numUses[nodeIdx] > 0 {
			expected++
			node := fe.function.builder.nodes[nodeIdx]
			// Total dependencies = regular inputs + captured inputs
			execBuf.remainingDeps[nodeIdx] = len(node.inputs) + len(node.capturedInputs)
			if execBuf.remainingDeps[nodeIdx] == 0 {
				readyToExecute <- nodeIdx
			}
		}
	}

	appendErrorFn := func(err error) {
		execMu.Lock()
		defer execMu.Unlock()
		collectErrors = append(collectErrors, err)
		stopExecutionFn()
	}

	for nodeIdx := range readyToExecute {
		nodeIdx := nodeIdx // Capture loop variable
		nodeExecFn := func() {
			node := fe.function.builder.nodes[nodeIdx]

			defer func(nodeIdx int) {
				execMu.Lock()
				defer execMu.Unlock()
				if len(collectErrors) > 0 {
					return
				}
				completed++
				if completed == expected {
					stopExecutionFn()
					return
				}

				// Handle multi-output nodes
				if node.IsMultiOutputs() {
					for _, outputNode := range node.multiOutputsNodes {
						outputIdx := outputNode.builderIdx
						if outputIdx >= fe.numNodesToProcess || fe.numUses[outputIdx] == 0 {
							continue
						}
						completed++
						if completed == expected {
							stopExecutionFn()
							return
						}
						for _, depIdx := range fe.dependents[outputIdx] {
							execBuf.remainingDeps[depIdx]--
							if execBuf.remainingDeps[depIdx] == 0 {
								readyToExecute <- depIdx
							}
						}
					}
				} else {
					for _, depIdx := range fe.dependents[nodeIdx] {
						execBuf.remainingDeps[depIdx]--
						if execBuf.remainingDeps[depIdx] == 0 {
							readyToExecute <- depIdx
						}
					}
				}
			}(nodeIdx)

			if execBuf.results[nodeIdx] != nil {
				return
			}
			if fe.numUses[nodeIdx] == 0 {
				return
			}

			if err := fe.executeNode(backend, node, execBuf); err != nil {
				appendErrorFn(err)
				return
			}
		}

		backend.workers.WaitToStart(nodeExecFn)
	}

	if len(collectErrors) > 0 {
		return collectErrors[0]
	}
	return nil
}

// executeNode executes a single node and stores its result.
func (fe *FunctionExecutable) executeNode(backend *Backend, node *Node, execBuf *funcExecBuffers) error {
	nodeIdx := node.builderIdx

	// Handle constants specially
	if node.opType == backends.OpTypeConstant {
		execBuf.owned[nodeIdx] = false
		execBuf.results[nodeIdx] = node.data.(*Buffer)
		return nil
	}

	// Captured values are already set up in Execute() - nothing to do
	if node.opType == backends.OpTypeCapturedValue {
		// Result should already be set from Execute()
		if execBuf.results[nodeIdx] == nil {
			return errors.Errorf("captured value not set for node %d", nodeIdx)
		}
		return nil
	}

	// Prepare inputs
	numInputs := len(node.inputs)
	var (
		inputBuffers []*Buffer
		inputsOwned  []bool
	)
	if execBuf.opInputBuffers != nil {
		inputBuffers = execBuf.opInputBuffers[:numInputs]
		inputsOwned = execBuf.opInputsOwned[:numInputs]
	} else {
		inputBuffers = make([]*Buffer, numInputs)
		inputsOwned = make([]bool, numInputs)
	}

	// Gather inputs. In parallel mode, we do NOT hold a lock here - the dependency
	// tracking ensures inputs are ready. The lock is only used in cleanup.
	for i, input := range node.inputs {
		inputIdx := input.builderIdx
		inputBuffers[i] = execBuf.results[inputIdx]
		if inputBuffers[i] == nil {
			return errors.Errorf("input %d for node %s not computed yet", i, node.opType)
		}
		// Only "own" the input if this is the last use of it.
		// The atomic Load is safe for concurrent access - if we miss ownership,
		// the buffer just won't be reused in-place. The important thing
		// is we don't free the buffer until all users have finished (handled in cleanup).
		inputsOwned[i] = execBuf.owned[inputIdx] &&
			fe.numUses[inputIdx]-int(execBuf.numUsed[inputIdx].Load()) == 1
	}

	// Execute the node
	if node.IsMultiOutputs() {
		multiExecutor := multiOutputsNodeExecutors[node.opType]
		if multiExecutor == nil {
			return errors.Errorf("no multi-output executor for op %s", node.opType)
		}

		outputBuffers, err := multiExecutor(backend, node, inputBuffers, inputsOwned)
		if err != nil {
			return errors.WithMessagef(err, "executing multi-output %s", node.opType)
		}

		for outputIdx, outputBuf := range outputBuffers {
			outputNode := node.multiOutputsNodes[outputIdx]
			outputNodeIdx := outputNode.builderIdx
			if outputNodeIdx >= fe.numNodesToProcess || fe.numUses[outputNodeIdx] == 0 {
				backend.putBuffer(outputBuf)
				continue
			}
			execBuf.results[outputNodeIdx] = outputBuf
			execBuf.owned[outputNodeIdx] = true
		}
	} else {
		executor := nodeExecutors[node.opType]
		if executor == nil {
			return errors.Errorf("no executor for op %s", node.opType)
		}

		result, err := executor(backend, node, inputBuffers, inputsOwned)
		if err != nil {
			return errors.WithMessagef(err, "executing %s", node.opType)
		}
		execBuf.results[nodeIdx] = result
		execBuf.owned[nodeIdx] = true
	}

	// Update usage counts and free unused buffers.
	// The lock protects results in parallel mode; numUsed uses atomics for safe reads.
	if execBuf.opsExecutionType == opsExecutionParallel {
		execBuf.mu.Lock()
	}
	for i, input := range node.inputs {
		inputIdx := input.builderIdx
		newCount := execBuf.numUsed[inputIdx].Add(1) // Mark this input as used.
		if inputBuffers[i] == nil {
			execBuf.results[inputIdx] = nil
			continue
		}
		if int(newCount) == fe.numUses[inputIdx] && execBuf.owned[inputIdx] {
			// Release the input buffer - all users have finished.
			backend.putBuffer(inputBuffers[i])
			execBuf.results[inputIdx] = nil
		}
	}
	// Also update usage counts for captured inputs.
	// These are treated as additional inputs for lifetime tracking.
	for _, capturedInput := range node.capturedInputs {
		capturedIdx := capturedInput.builderIdx
		newCount := execBuf.numUsed[capturedIdx].Add(1)
		capturedBuf := execBuf.results[capturedIdx]
		if capturedBuf == nil {
			continue
		}
		if int(newCount) == fe.numUses[capturedIdx] && execBuf.owned[capturedIdx] {
			// Release the captured buffer - all users have finished.
			backend.putBuffer(capturedBuf)
			execBuf.results[capturedIdx] = nil
		}
	}
	if execBuf.opsExecutionType == opsExecutionParallel {
		execBuf.mu.Unlock()
	} else {
		execBuf.opInputBuffers = inputBuffers
		execBuf.opInputsOwned = inputsOwned
	}

	return nil
}
