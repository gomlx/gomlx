// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"slices"
	"sync"

	"github.com/gomlx/gomlx/backends"
	"github.com/pkg/errors"
)

// FunctionExecutable contains pre-compiled execution information for any function.
// This is used for both the main function and closures, unifying their execution model.
type FunctionExecutable struct {
	// function is the source Function this was compiled from.
	function *Function

	// sortedNodes are the nodes needed for execution, in topological order.
	sortedNodes []*Node

	// nodeToSortedIdx maps a node's builderIdx to its index in sortedNodes.
	// This allows O(1) lookup when executing.
	nodeToSortedIdx map[int]int

	// numUses tracks how many times each node's result is used (indexed by sortedNodes position).
	numUses []int

	// dependents maps each node (by sortedNodes position) to the list of dependent nodes.
	dependents [][]int

	// parameterIndices maps parameter nodes (by builderIdx) to their input index.
	parameterIndices map[int]int

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

	fe := &FunctionExecutable{
		function:         f,
		outputNodes:      f.outputs,
		parameterIndices: make(map[int]int),
		nodeToSortedIdx:  make(map[int]int),
	}

	// 1. Identify all nodes reachable from outputs using DFS
	neededNodes := make(map[int]bool)
	var findNeeded func(node *Node)
	findNeeded = func(node *Node) {
		if neededNodes[node.builderIdx] {
			return
		}
		neededNodes[node.builderIdx] = true
		for _, input := range node.inputs {
			findNeeded(input)
		}
	}
	for _, out := range f.outputs {
		findNeeded(out)
	}

	// 2. Collect and sort nodes topologically (by builderIdx order)
	for nodeIdx := range neededNodes {
		fe.sortedNodes = append(fe.sortedNodes, f.builder.nodes[nodeIdx])
	}
	slices.SortFunc(fe.sortedNodes, func(a, b *Node) int {
		return a.builderIdx - b.builderIdx
	})

	numNodes := len(fe.sortedNodes)

	// 3. Build reverse mapping from builderIdx to sortedNodes index
	for i, node := range fe.sortedNodes {
		fe.nodeToSortedIdx[node.builderIdx] = i
	}

	// 4. Map parameters to input indices
	for i, param := range f.parameters {
		fe.parameterIndices[param.builderIdx] = i
	}

	// 5. Count uses, find max inputs, and build dependents
	fe.numUses = make([]int, numNodes)
	fe.dependents = make([][]int, numNodes)

	for sortedIdx, node := range fe.sortedNodes {
		fe.maxInputs = max(fe.maxInputs, len(node.inputs))
		for _, input := range node.inputs {
			if inputSortedIdx, ok := fe.nodeToSortedIdx[input.builderIdx]; ok {
				fe.numUses[inputSortedIdx]++
				fe.dependents[inputSortedIdx] = append(fe.dependents[inputSortedIdx], sortedIdx)
			}
		}
	}

	// Count output uses
	for _, out := range f.outputs {
		if outSortedIdx, ok := fe.nodeToSortedIdx[out.builderIdx]; ok {
			fe.numUses[outSortedIdx]++
		}
	}

	// 6. Initialize execution buffers pool
	fe.executionBuffersPool = sync.Pool{
		New: func() interface{} {
			return &funcExecBuffers{
				results:       make([]*Buffer, numNodes),
				numUsed:       make([]int, numNodes),
				owned:         make([]bool, numNodes),
				remainingDeps: make([]int, numNodes),
			}
		},
	}

	return fe, nil
}

// funcExecBuffers holds intermediate results during function execution.
type funcExecBuffers struct {
	// results hold the calculated computations at each step.
	results []*Buffer

	// numUsed tracks how many times each node has been used already.
	numUsed []int

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
func (fe *FunctionExecutable) Execute(backend *Backend, inputs []*Buffer, donate []bool) ([]*Buffer, error) {
	// Validate input count
	if len(inputs) != len(fe.function.parameters) {
		return nil, errors.Errorf("function expects %d inputs, got %d",
			len(fe.function.parameters), len(inputs))
	}

	// donate defaults to false
	if len(donate) == 0 {
		donate = make([]bool, len(inputs))
	}

	numNodes := len(fe.sortedNodes)

	// Get execution buffers from pool and reset
	execBuf := fe.executionBuffersPool.Get().(*funcExecBuffers)
	for i := range numNodes {
		execBuf.numUsed[i] = 0
		execBuf.owned[i] = false
		execBuf.results[i] = nil
		execBuf.remainingDeps[i] = 0
	}

	// Set up parameters from inputs
	for paramBuilderIdx, inputIdx := range fe.parameterIndices {
		if sortedIdx, ok := fe.nodeToSortedIdx[paramBuilderIdx]; ok {
			execBuf.results[sortedIdx] = inputs[inputIdx]
			execBuf.owned[sortedIdx] = donate[inputIdx]
		}
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
		sortedIdx, ok := fe.nodeToSortedIdx[outNode.builderIdx]
		if !ok {
			fe.executionBuffersPool.Put(execBuf)
			return nil, errors.Errorf("output node %d not found in sorted nodes", outNode.builderIdx)
		}
		outputs[i] = execBuf.results[sortedIdx]
		if outputs[i] == nil {
			fe.executionBuffersPool.Put(execBuf)
			return nil, errors.Errorf("output %d not computed", i)
		}
		if !execBuf.owned[sortedIdx] {
			// Clone the buffer since we don't own it
			outputs[i] = backend.cloneBuffer(execBuf.results[sortedIdx])
		}
		execBuf.results[sortedIdx] = nil // Prevent double-free
	}

	// Free any remaining owned buffers that weren't outputs
	for sortedIdx, buf := range execBuf.results {
		if buf != nil && execBuf.owned[sortedIdx] {
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

	for sortedIdx, node := range fe.sortedNodes {
		if execBuf.results[sortedIdx] != nil {
			// Already computed (parameter or multi-output)
			continue
		}
		if fe.numUses[sortedIdx] == 0 {
			// Not used by any output
			continue
		}

		if err := fe.executeNode(backend, sortedIdx, node, execBuf); err != nil {
			return err
		}
	}
	return nil
}

// executeParallel executes nodes in parallel based on dependency graph.
func (fe *FunctionExecutable) executeParallel(backend *Backend, execBuf *funcExecBuffers) error {
	numNodes := len(fe.sortedNodes)

	var (
		readyToExecute chan int
		collectErrors  []error
		execMu         sync.Mutex
	)
	readyToExecute = make(chan int, numNodes+10)
	stopExecutionFn := sync.OnceFunc(func() { close(readyToExecute) })

	expected := 0
	completed := 0

	// Count expected nodes and initialize dependencies
	for sortedIdx := range numNodes {
		if fe.numUses[sortedIdx] > 0 {
			expected++
			node := fe.sortedNodes[sortedIdx]
			// Count only inputs that are in our sorted nodes
			depCount := 0
			for _, input := range node.inputs {
				if _, ok := fe.nodeToSortedIdx[input.builderIdx]; ok {
					depCount++
				}
			}
			execBuf.remainingDeps[sortedIdx] = depCount
			if depCount == 0 {
				readyToExecute <- sortedIdx
			}
		}
	}

	appendErrorFn := func(err error) {
		execMu.Lock()
		defer execMu.Unlock()
		collectErrors = append(collectErrors, err)
		stopExecutionFn()
	}

	for sortedIdx := range readyToExecute {
		nodeExecFn := func() {
			node := fe.sortedNodes[sortedIdx]

			defer func(sortedIdx int) {
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
						outputSortedIdx, ok := fe.nodeToSortedIdx[outputNode.builderIdx]
						if !ok || fe.numUses[outputSortedIdx] == 0 {
							continue
						}
						completed++
						if completed == expected {
							stopExecutionFn()
							return
						}
						for _, depIdx := range fe.dependents[outputSortedIdx] {
							execBuf.remainingDeps[depIdx]--
							if execBuf.remainingDeps[depIdx] == 0 {
								readyToExecute <- depIdx
							}
						}
					}
				} else {
					for _, depIdx := range fe.dependents[sortedIdx] {
						execBuf.remainingDeps[depIdx]--
						if execBuf.remainingDeps[depIdx] == 0 {
							readyToExecute <- depIdx
						}
					}
				}
			}(sortedIdx)

			if execBuf.results[sortedIdx] != nil {
				return
			}
			if fe.numUses[sortedIdx] == 0 {
				return
			}

			if err := fe.executeNode(backend, sortedIdx, node, execBuf); err != nil {
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
func (fe *FunctionExecutable) executeNode(backend *Backend, sortedIdx int, node *Node, execBuf *funcExecBuffers) error {
	// Handle constants specially
	if node.opType == backends.OpTypeConstant {
		execBuf.owned[sortedIdx] = false
		execBuf.results[sortedIdx] = node.data.(*Buffer)
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

	for i, input := range node.inputs {
		inputSortedIdx, ok := fe.nodeToSortedIdx[input.builderIdx]
		if !ok {
			return errors.Errorf("input node %d not found in sorted nodes", input.builderIdx)
		}
		inputBuffers[i] = execBuf.results[inputSortedIdx]
		if inputBuffers[i] == nil {
			return errors.Errorf("input %d for node %s not computed yet", i, node.opType)
		}
		// Only own the input if this is the last use
		inputsOwned[i] = execBuf.owned[inputSortedIdx] &&
			fe.numUses[inputSortedIdx]-execBuf.numUsed[inputSortedIdx] == 1
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
			outputSortedIdx, ok := fe.nodeToSortedIdx[outputNode.builderIdx]
			if !ok || fe.numUses[outputSortedIdx] == 0 {
				backend.putBuffer(outputBuf)
				continue
			}
			execBuf.results[outputSortedIdx] = outputBuf
			execBuf.owned[outputSortedIdx] = true
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
		execBuf.results[sortedIdx] = result
		execBuf.owned[sortedIdx] = true
	}

	// Update usage counts and free unused buffers
	if execBuf.opsExecutionType == opsExecutionParallel {
		execBuf.mu.Lock()
	}
	for i, input := range node.inputs {
		inputSortedIdx := fe.nodeToSortedIdx[input.builderIdx]
		execBuf.numUsed[inputSortedIdx]++
		if execBuf.numUsed[inputSortedIdx] == fe.numUses[inputSortedIdx] &&
			execBuf.owned[inputSortedIdx] &&
			inputBuffers[i] != nil {
			backend.putBuffer(inputBuffers[i])
			execBuf.results[inputSortedIdx] = nil
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
