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

	// numNodesToProcess is the max(outputs.idx)+1.
	// Arrays are sized to this to allow direct idx indexing.
	numNodesToProcess int

	// numUses tracks how many times each node's result is used (indexed by idx).
	numUses []int

	// dependents maps each node (by idx) to the list of dependent node idxs.
	dependents [][]int

	// outputNodes are the nodes that produce the function's outputs.
	outputNodes []*Node

	// maxInputs is the maximum number of inputs any node has.
	maxInputs int

	// executionBuffersPool allows reuse of execution buffers.
	executionBuffersPool sync.Pool

	// seqInputBuffersPool pools input buffer slices for sequential execution only.
	// Parallel execution must allocate per-node to avoid races.
	seqInputBuffersPool sync.Pool

	// seqInputOwnedPool pools input ownership slices for sequential execution only.
	seqInputOwnedPool sync.Pool
}

// newFunctionExecutable creates a FunctionExecutable for the given function.
// The function must have Return() called (f.returned == true).
func newFunctionExecutable(f *Function) (*FunctionExecutable, error) {
	if !f.returned {
		return nil, errors.Errorf("function must have Return() called before compilation")
	}

	// Calculate numNodesToProcess from outputs.
	// This has the benefit of immediately discarding nodes with idx > max(outputs.idx),
	// meaning nodes that outputs don't depend on.
	var numNodesToProcess int
	for _, output := range f.outputs {
		numNodesToProcess = max(numNodesToProcess, output.idx+1)
	}

	fe := &FunctionExecutable{
		function:          f,
		outputNodes:       f.outputs,
		numNodesToProcess: numNodesToProcess,
		numUses:           make([]int, numNodesToProcess),
		dependents:        make([][]int, numNodesToProcess),
	}

	// Find max inputs (including captured inputs) and count uses/dependents
	for nodeIdx := range numNodesToProcess {
		node := f.nodes[nodeIdx]
		// Total inputs = regular inputs + all captured inputs across closures
		totalCaptured := 0
		for _, closureCaptures := range node.capturedInputs {
			totalCaptured += len(closureCaptures)
		}
		totalInputs := len(node.inputs) + totalCaptured
		fe.maxInputs = max(fe.maxInputs, totalInputs)
	}

	// Count uses for each node starting from outputs
	for _, output := range f.outputs {
		fe.countNodeUsesAndDependents(output)
	}

	// Initialize execution buffers pool with pre-allocated slices to avoid per-execution allocations.
	numOutputs := len(f.outputs)
	maxInputs := fe.maxInputs
	fe.executionBuffersPool = sync.Pool{
		New: func() interface{} {
			return &funcExecBuffers{
				results:       make([]*Buffer, numNodesToProcess),
				numUsed:       make([]atomic.Int32, numNodesToProcess),
				owned:         make([]bool, numNodesToProcess),
				remainingDeps: make([]int, numNodesToProcess),
				outputs:       make([]*Buffer, numOutputs),
			}
		},
	}

	// Initialize pools for sequential execution input slices.
	// These are separate because parallel execution must allocate per-node to avoid races.
	fe.seqInputBuffersPool = sync.Pool{
		New: func() interface{} {
			return make([]*Buffer, maxInputs)
		},
	}
	fe.seqInputOwnedPool = sync.Pool{
		New: func() interface{} {
			return make([]bool, maxInputs)
		},
	}

	return fe, nil
}

// countNodeUsesAndDependents recursively counts how many times a node is used.
// It tracks both regular inputs and captured inputs (for closure-calling ops).
func (fe *FunctionExecutable) countNodeUsesAndDependents(node *Node) {
	nodeIdx := node.idx
	fe.numUses[nodeIdx]++
	if fe.numUses[nodeIdx] == 1 {
		// On the first visit, recursively traverse inputs of the node.
		for _, input := range node.inputs {
			fe.dependents[input.idx] = append(fe.dependents[input.idx], nodeIdx)
			fe.countNodeUsesAndDependents(input)
		}
		// Also track captured inputs for closure-calling ops (If, While, Sort, etc.).
		// This ensures captured values are properly tracked in the dependency graph
		// so they can be freed when no longer needed.
		for _, closureCaptures := range node.capturedInputs {
			for _, capturedInput := range closureCaptures {
				fe.dependents[capturedInput.idx] = append(fe.dependents[capturedInput.idx], nodeIdx)
				fe.countNodeUsesAndDependents(capturedInput)
			}
		}
	}
}

// funcExecBuffers holds intermediate results during function execution.
type funcExecBuffers struct {
	// results hold the calculated computations at each step (indexed by idx).
	results []*Buffer

	// numUsed tracks how many times each node has been used already.
	// Uses atomic.Int32 to allow safe concurrent reads in ownership checks.
	numUsed []atomic.Int32

	// owned indicates whether the corresponding buffer is owned by the executor.
	owned []bool

	// remainingDeps is the number of remaining dependencies for each node.
	remainingDeps []int

	// outputs is pre-allocated to hold output buffers, avoiding allocation per execution.
	outputs []*Buffer

	// opsExecutionType can be sequential or parallel.
	opsExecutionType opsExecutionType

	// Sequential execution-only: reused for each op, pre-allocated to maxInputs size.
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

	// donate and donateCaptures default to nil (treated as all-false).
	// We avoid allocating slices here by checking for nil in the loop below.

	// Get execution buffers from pool and reset
	execBuf := fe.executionBuffersPool.Get().(*funcExecBuffers)
	for i := range fe.numNodesToProcess {
		execBuf.numUsed[i].Store(0)
		execBuf.owned[i] = false
		execBuf.results[i] = nil
		execBuf.remainingDeps[i] = 0
	}

	// Set up parameters from inputs using idx directly.
	// donate may be nil (meaning all false), so we check before indexing.
	for i, inputNode := range funcParams {
		inputIdx := inputNode.idx
		execBuf.results[inputIdx] = inputs[i]
		execBuf.owned[inputIdx] = donate != nil && donate[i]
	}

	// Set up captured values from parent scope.
	// If donateCaptures[i] is true, the closure takes ownership of the buffer.
	// donateCaptures may be nil (meaning all false), so we check before indexing.
	for i, captureNode := range fe.function.capturedLocalNodes {
		captureIdx := captureNode.idx
		execBuf.results[captureIdx] = capturedInputs[i]
		execBuf.owned[captureIdx] = donateCaptures != nil && donateCaptures[i]
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

	// Collect outputs using pre-allocated slice from pool.
	outputs := execBuf.outputs
	for i, outNode := range fe.outputNodes {
		outIdx := outNode.idx
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

	// Create a new slice to return (we can't return the pooled one directly).
	// This single allocation replaces the previous per-execution allocation.
	result := make([]*Buffer, len(outputs))
	copy(result, outputs)

	// Free any remaining owned buffers that weren't outputs
	for idx, buf := range execBuf.results {
		if buf != nil && execBuf.owned[idx] {
			backend.putBuffer(buf)
		}
	}

	fe.executionBuffersPool.Put(execBuf)
	return result, nil
}

// executeSequentially executes nodes one after another in topological order.
func (fe *FunctionExecutable) executeSequentially(backend *Backend, execBuf *funcExecBuffers) error {
	// Get input slices from pool for reuse during sequential execution.
	execBuf.opInputBuffers = fe.seqInputBuffersPool.Get().([]*Buffer)
	execBuf.opInputsOwned = fe.seqInputOwnedPool.Get().([]bool)
	clear(execBuf.opInputBuffers)
	clear(execBuf.opInputsOwned)
	defer func() {
		fe.seqInputBuffersPool.Put(execBuf.opInputBuffers)
		fe.seqInputOwnedPool.Put(execBuf.opInputsOwned)
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

		node := fe.function.nodes[nodeIdx]
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
			node := fe.function.nodes[nodeIdx]
			// Total dependencies = regular inputs + all captured inputs across closures
			totalCaptured := 0
			for _, closureCaptures := range node.capturedInputs {
				totalCaptured += len(closureCaptures)
			}
			execBuf.remainingDeps[nodeIdx] = len(node.inputs) + totalCaptured
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
		nodeExecFn := func() {
			node := fe.function.nodes[nodeIdx]

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
						outputIdx := outputNode.idx
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
	nodeIdx := node.idx

	// Handle constants specially
	if node.opType == backends.OpTypeConstant {
		execBuf.owned[nodeIdx] = false
		execBuf.results[nodeIdx] = node.data.(*Buffer)
		return nil
	}

	// Note: OpTypeParameter and OpTypeCapturedValue nodes have their results
	// set up in Execute() and should never reach executeNode.
	// We don't check for them here for performance (this is the inner execution loop).

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
		inputIdx := input.idx
		inputBuffers[i] = execBuf.results[inputIdx]
		if inputBuffers[i] == nil {
			return errors.Errorf("input %d for node %s not computed yet", i, node.opType)
		}
		if !inputBuffers[i].inUse {
			return errors.Errorf("input %d for node %s has been released already!?", i, node.opType)
		}
		// Only "own" the input if this is the last use of it.
		// The atomic Load is safe for concurrent access - if we miss ownership,
		// the buffer just won't be reused in-place. The important thing
		// is we don't free the buffer until all users have finished (handled in cleanup).
		inputsOwned[i] = execBuf.owned[inputIdx] &&
			fe.numUses[inputIdx]-int(execBuf.numUsed[inputIdx].Load()) == 1
	}

	// Check for closure executor first (If, While, Sort).
	// Closure executors receive captured inputs separately with explicit ownership tracking.
	closureExecutor := nodeClosureExecutors[node.opType]
	if closureExecutor != nil {
		// Build capture counts for workspace allocation.
		// Use stack-allocated array for common cases (If/While have 2 closures, Sort has 1).
		numClosures := len(node.capturedInputs)
		var captureCountsBuf [4]int
		var captureCounts []int
		if numClosures <= len(captureCountsBuf) {
			captureCounts = captureCountsBuf[:numClosures]
		} else {
			captureCounts = make([]int, numClosures)
		}
		for closureIdx, closureCaptures := range node.capturedInputs {
			captureCounts[closureIdx] = len(closureCaptures)
		}

		// Get pooled workspace for ClosureInputs
		ciWorkspace := getClosureInputsWorkspace(captureCounts)
		closureInputs := ciWorkspace.closureInputs

		// Fill in the buffer pointers and ownership flags
		for closureIdx, closureCaptures := range node.capturedInputs {
			for i, capturedNode := range closureCaptures {
				capturedIdx := capturedNode.idx
				closureInputs[closureIdx].Buffers[i] = execBuf.results[capturedIdx]
				if closureInputs[closureIdx].Buffers[i] == nil {
					putClosureInputsWorkspace(ciWorkspace)
					return errors.Errorf("captured input %d for closure %d of node %s not computed yet", i, closureIdx, node.opType)
				}
				// Only "own" the captured input if this is the last use of it.
				closureInputs[closureIdx].Owned[i] = execBuf.owned[capturedIdx] &&
					fe.numUses[capturedIdx]-int(execBuf.numUsed[capturedIdx].Load()) == 1
			}
		}

		outputBuffers, err := closureExecutor(backend, node, inputBuffers, inputsOwned, closureInputs)
		if err != nil {
			putClosureInputsWorkspace(ciWorkspace)
			return errors.WithMessagef(err, "executing closure op %s", node.opType)
		}

		// Check if any captured inputs were consumed (set to nil by the executor).
		// If so, mark execBuf.results as nil to indicate they're no longer available.
		for closureIdx, closureCaptures := range node.capturedInputs {
			for i, capturedNode := range closureCaptures {
				if closureInputs[closureIdx].Buffers[i] == nil {
					execBuf.results[capturedNode.idx] = nil
				}
			}
		}

		// Return workspace to pool
		putClosureInputsWorkspace(ciWorkspace)

		// Handle outputs (closure ops are always multi-output style)
		for outputIdx, outputBuf := range outputBuffers {
			outputNode := node.multiOutputsNodes[outputIdx]
			outputNodeIdx := outputNode.idx
			if outputNodeIdx >= fe.numNodesToProcess || fe.numUses[outputNodeIdx] == 0 {
				backend.putBuffer(outputBuf)
				continue
			}
			execBuf.results[outputNodeIdx] = outputBuf
			execBuf.owned[outputNodeIdx] = true
		}
	} else if node.IsMultiOutputs() {
		// Execute the node
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
			outputNodeIdx := outputNode.idx
			if outputNodeIdx >= fe.numNodesToProcess || fe.numUses[outputNodeIdx] == 0 {
				// Output of node is not used by any other node, we can immediately release it.
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
		inputIdx := input.idx
		newCount := execBuf.numUsed[inputIdx].Add(1) // Mark this input as used.
		if inputBuffers[i] == nil {
			// Input buffer is nil, means it has been consumed by the operation.
			// Mark that the associated results is no longer available.
			execBuf.results[inputIdx] = nil
			continue
		}
		if !inputBuffers[i].inUse {
			return errors.Errorf("input #%d for node %s has been released, but not marked as consumed!?",
				i, node.opType)
		}
		if int(newCount) == fe.numUses[inputIdx] && execBuf.owned[inputIdx] {
			// Check if it is reused as one of the outputs -- common for in-place operations, like in exec_binary.go.
			// The contract is that if the input is reused, the operator must set the input buffer to nil in the input slice.
			// If we find the input buffer reused as an output but it is not nil here, it is a bug in the operator implementation.
			if node.IsMultiOutputs() {
				for outIdx, outputNode := range node.multiOutputsNodes {
					if execBuf.results[outputNode.idx] == inputBuffers[i] {
						return errors.Errorf("op %s (output %d) reused input %d as output but didn't set input to nil in buffer slice", node.opType, outIdx, i)
					}
				}
			} else {
				if execBuf.results[nodeIdx] == inputBuffers[i] {
					return errors.Errorf("op %s reused input %d as output but didn't set input to nil in buffer slice", node.opType, i)
				}
			}

			// Release the input buffer - all users have finished.
			backend.putBuffer(inputBuffers[i])
			execBuf.results[inputIdx] = nil
		}
	}
	// Also update usage counts for captured inputs.
	// These are treated as additional inputs for lifetime tracking.
	for _, closureCaptures := range node.capturedInputs {
		for _, capturedInput := range closureCaptures {
			capturedIdx := capturedInput.idx
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
	}
	if execBuf.opsExecutionType == opsExecutionParallel {
		execBuf.mu.Unlock()
	} else {
		execBuf.opInputBuffers = inputBuffers
		execBuf.opInputsOwned = inputsOwned
	}

	return nil
}
