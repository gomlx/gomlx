// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"sort"

	"github.com/gomlx/gomlx/backends"
	"github.com/pkg/errors"
)

func init() {
	// Register multi-output executors for control flow operations.
	multiOutputsNodeExecutors[backends.OpTypeSort] = execSort
	multiOutputsNodeExecutors[backends.OpTypeWhile] = execWhile
	multiOutputsNodeExecutors[backends.OpTypeIf] = execIf
	multiOutputsNodeExecutors[backends.OpTypeCall] = execCall
}

// closureExecutor provides the ability to execute closures (sub-computations) during runtime.
type closureExecutor struct {
	backend *Backend
	closure *Function
}

// newClosureExecutor creates a new closure executor for the given closure.
func newClosureExecutor(backend *Backend, closure *Function) *closureExecutor {
	return &closureExecutor{
		backend: backend,
		closure: closure,
	}
}

// execute runs the closure with the given input buffers and returns the output buffers.
// The inputs must match the closure's parameters in order and shape.
func (ce *closureExecutor) execute(inputs []*Buffer) ([]*Buffer, error) {
	closure := ce.closure
	if len(inputs) != len(closure.parameters) {
		return nil, errors.Errorf("closure execution: expected %d inputs, got %d",
			len(closure.parameters), len(inputs))
	}

	// Build a mini execution context for the closure.
	// We need to:
	// 1. Map parameter nodes to input buffers
	// 2. Execute nodes in topological order
	// 3. Return output buffers

	// Find all nodes needed for this closure's outputs.
	builder := closure.builder
	neededNodes := make(map[int]bool)
	var markNeeded func(node *Node)
	markNeeded = func(node *Node) {
		if node == nil || neededNodes[node.builderIdx] {
			return
		}
		neededNodes[node.builderIdx] = true
		for _, input := range node.inputs {
			markNeeded(input)
		}
	}

	for _, output := range closure.outputs {
		markNeeded(output)
	}

	// Sort nodes by builder index for topological order.
	sortedNodes := make([]*Node, 0, len(neededNodes))
	for nodeIdx := range neededNodes {
		sortedNodes = append(sortedNodes, builder.nodes[nodeIdx])
	}
	sort.Slice(sortedNodes, func(i, j int) bool {
		return sortedNodes[i].builderIdx < sortedNodes[j].builderIdx
	})

	// Create results map.
	results := make(map[int]*Buffer)

	// Map parameters to input buffers.
	for i, param := range closure.parameters {
		results[param.builderIdx] = inputs[i]
	}

	// Execute nodes in order.
	for _, node := range sortedNodes {
		if results[node.builderIdx] != nil {
			// Already computed (parameter or constant).
			continue
		}

		// Handle constants.
		if node.opType == backends.OpTypeConstant {
			results[node.builderIdx] = node.data.(*Buffer)
			continue
		}

		// Gather inputs for this node.
		nodeInputs := make([]*Buffer, len(node.inputs))
		inputsOwned := make([]bool, len(node.inputs))
		for i, input := range node.inputs {
			nodeInputs[i] = results[input.builderIdx]
			if nodeInputs[i] == nil {
				return nil, errors.Errorf("closure execution: input %d for node %s not computed",
					i, node.opType)
			}
			// We don't own closure inputs, so don't allow reuse.
			inputsOwned[i] = false
		}

		// Execute the node.
		if node.IsMultiOutputs() {
			multiExecutor := multiOutputsNodeExecutors[node.opType]
			if multiExecutor == nil {
				return nil, errors.Errorf("closure execution: multi-output executor for %s not implemented",
					node.opType)
			}
			outputs, err := multiExecutor(ce.backend, node, nodeInputs, inputsOwned)
			if err != nil {
				return nil, errors.WithMessagef(err, "closure execution: while executing %s", node.opType)
			}
			for i, outputNode := range node.multiOutputsNodes {
				results[outputNode.builderIdx] = outputs[i]
			}
		} else {
			executor := nodeExecutors[node.opType]
			if executor == nil {
				return nil, errors.Errorf("closure execution: executor for %s not implemented", node.opType)
			}
			result, err := executor(ce.backend, node, nodeInputs, inputsOwned)
			if err != nil {
				return nil, errors.WithMessagef(err, "closure execution: while executing %s", node.opType)
			}
			results[node.builderIdx] = result
		}
	}

	// Collect outputs.
	outputs := make([]*Buffer, len(closure.outputs))
	for i, output := range closure.outputs {
		outputs[i] = results[output.builderIdx]
		if outputs[i] == nil {
			return nil, errors.Errorf("closure execution: output %d not computed", i)
		}
		// Clone the output so we own it.
		outputs[i] = ce.backend.cloneBuffer(outputs[i])
	}

	return outputs, nil
}

// execSort implements the Sort operation.
// It sorts one or more tensors along the specified axis using a comparator function.
func execSort(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) ([]*Buffer, error) {
	sortData := node.data.(*sortNode)
	axis := sortData.axis
	isStable := sortData.isStable
	comparator := newClosureExecutor(backend, sortData.comparatorFn)

	numInputs := len(inputs)
	if numInputs == 0 {
		return nil, errors.New("Sort: no inputs provided")
	}

	// Get the shape from the first input (all inputs have the same shape).
	shape := inputs[0].shape
	axisSize := shape.Dimensions[axis]

	// Calculate the number of elements before and after the sort axis.
	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= shape.Dimensions[i]
	}
	innerSize := 1
	for i := axis + 1; i < shape.Rank(); i++ {
		innerSize *= shape.Dimensions[i]
	}

	// Create output buffers (clone inputs).
	outputs := make([]*Buffer, numInputs)
	for i, input := range inputs {
		if inputsOwned[i] {
			outputs[i] = input
			inputs[i] = nil
		} else {
			outputs[i] = backend.cloneBuffer(input)
		}
	}

	// For each outer/inner position, sort along the axis.
	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			// Build indices for this slice along the axis.
			indices := make([]int, axisSize)
			for i := range indices {
				indices[i] = i
			}

			// Sort indices using the comparator.
			var sortErr error
			sortFn := func(i, j int) bool {
				if sortErr != nil {
					return false
				}

				// Build comparator inputs: lhs values then rhs values.
				compInputs := make([]*Buffer, 2*numInputs)
				for k := 0; k < numInputs; k++ {
					// Get scalar values at positions i and j.
					idxI := outer*axisSize*innerSize + indices[i]*innerSize + inner
					idxJ := outer*axisSize*innerSize + indices[j]*innerSize + inner

					compInputs[k] = extractScalar(backend, outputs[k], idxI)
					compInputs[numInputs+k] = extractScalar(backend, outputs[k], idxJ)
				}

				// Execute comparator.
				compOutputs, err := comparator.execute(compInputs)
				if err != nil {
					sortErr = err
					return false
				}

				// Free comparator input buffers.
				for _, buf := range compInputs {
					backend.putBuffer(buf)
				}

				if len(compOutputs) != 1 {
					sortErr = errors.Errorf("Sort: comparator must return exactly 1 output, got %d", len(compOutputs))
					return false
				}

				result := compOutputs[0].flat.([]bool)[0]
				backend.putBuffer(compOutputs[0])
				return result
			}

			if isStable {
				sort.SliceStable(indices, sortFn)
			} else {
				sort.Slice(indices, sortFn)
			}

			if sortErr != nil {
				return nil, sortErr
			}

			// Apply the permutation to all outputs.
			applyPermutation(backend, outputs, indices, outer, innerSize, axisSize, inner)
		}
	}

	return outputs, nil
}

// extractScalar extracts a scalar value from a buffer at the given flat index.
func extractScalar(backend *Backend, buf *Buffer, flatIdx int) *Buffer {
	scalar := backend.getBuffer(buf.shape.DType, 1)
	scalar.shape = buf.shape.Clone()
	scalar.shape.Dimensions = nil // Make it a scalar.
	copyScalarAt(scalar.flat, buf.flat, flatIdx)
	return scalar
}

// copyScalarAt copies a single element from src[srcIdx] to dst[0].
func copyScalarAt(dst, src any, srcIdx int) {
	switch s := src.(type) {
	case []float32:
		dst.([]float32)[0] = s[srcIdx]
	case []float64:
		dst.([]float64)[0] = s[srcIdx]
	case []int32:
		dst.([]int32)[0] = s[srcIdx]
	case []int64:
		dst.([]int64)[0] = s[srcIdx]
	case []uint8:
		dst.([]uint8)[0] = s[srcIdx]
	case []uint16:
		dst.([]uint16)[0] = s[srcIdx]
	case []uint32:
		dst.([]uint32)[0] = s[srcIdx]
	case []uint64:
		dst.([]uint64)[0] = s[srcIdx]
	case []int8:
		dst.([]int8)[0] = s[srcIdx]
	case []int16:
		dst.([]int16)[0] = s[srcIdx]
	case []bool:
		dst.([]bool)[0] = s[srcIdx]
	default:
		panic("unsupported type in copyScalarAt")
	}
}

// applyPermutation applies the index permutation to all output buffers.
func applyPermutation(backend *Backend, outputs []*Buffer, indices []int, outer, innerSize, axisSize, inner int) {
	// Create temporary storage for the permuted values.
	for _, output := range outputs {
		temp := make([]int, axisSize)
		for i := range temp {
			temp[i] = indices[i]
		}

		// Apply permutation by copying values.
		tempBuf := backend.getBuffer(output.shape.DType, axisSize)
		for i := 0; i < axisSize; i++ {
			srcIdx := outer*axisSize*innerSize + temp[i]*innerSize + inner
			copyScalarAt(tempBuf.flat, output.flat, srcIdx)
			swapAtIndex(tempBuf.flat, i, 0)
		}

		// Copy back.
		for i := 0; i < axisSize; i++ {
			dstIdx := outer*axisSize*innerSize + i*innerSize + inner
			copyFromAt(output.flat, dstIdx, tempBuf.flat, i)
		}
		backend.putBuffer(tempBuf)
	}
}

// swapAtIndex moves element at index 0 to index i in the same slice.
func swapAtIndex(data any, i, j int) {
	switch d := data.(type) {
	case []float32:
		d[i], d[j] = d[j], d[i]
	case []float64:
		d[i], d[j] = d[j], d[i]
	case []int32:
		d[i], d[j] = d[j], d[i]
	case []int64:
		d[i], d[j] = d[j], d[i]
	case []uint8:
		d[i], d[j] = d[j], d[i]
	case []uint16:
		d[i], d[j] = d[j], d[i]
	case []uint32:
		d[i], d[j] = d[j], d[i]
	case []uint64:
		d[i], d[j] = d[j], d[i]
	case []int8:
		d[i], d[j] = d[j], d[i]
	case []int16:
		d[i], d[j] = d[j], d[i]
	case []bool:
		d[i], d[j] = d[j], d[i]
	}
}

// copyFromAt copies a single element from src[srcIdx] to dst[dstIdx].
func copyFromAt(dst any, dstIdx int, src any, srcIdx int) {
	switch d := dst.(type) {
	case []float32:
		d[dstIdx] = src.([]float32)[srcIdx]
	case []float64:
		d[dstIdx] = src.([]float64)[srcIdx]
	case []int32:
		d[dstIdx] = src.([]int32)[srcIdx]
	case []int64:
		d[dstIdx] = src.([]int64)[srcIdx]
	case []uint8:
		d[dstIdx] = src.([]uint8)[srcIdx]
	case []uint16:
		d[dstIdx] = src.([]uint16)[srcIdx]
	case []uint32:
		d[dstIdx] = src.([]uint32)[srcIdx]
	case []uint64:
		d[dstIdx] = src.([]uint64)[srcIdx]
	case []int8:
		d[dstIdx] = src.([]int8)[srcIdx]
	case []int16:
		d[dstIdx] = src.([]int16)[srcIdx]
	case []bool:
		d[dstIdx] = src.([]bool)[srcIdx]
	}
}

// execWhile implements the While operation.
// It executes bodyFn repeatedly while condFn returns true.
func execWhile(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) ([]*Buffer, error) {
	whileData := node.data.(*whileNode)
	condExecutor := newClosureExecutor(backend, whileData.condFn)
	bodyExecutor := newClosureExecutor(backend, whileData.bodyFn)

	// Initialize state with input buffers (clone them to own them).
	state := make([]*Buffer, len(inputs))
	for i, input := range inputs {
		if inputsOwned[i] {
			state[i] = input
			inputs[i] = nil
		} else {
			state[i] = backend.cloneBuffer(input)
		}
	}

	// Execute the while loop.
	const maxIterations = 1000000 // Safety limit.
	for iter := 0; iter < maxIterations; iter++ {
		// Evaluate condition.
		condOutputs, err := condExecutor.execute(state)
		if err != nil {
			return nil, errors.WithMessage(err, "While: condition evaluation failed")
		}

		if len(condOutputs) != 1 {
			return nil, errors.Errorf("While: condition must return exactly 1 output, got %d", len(condOutputs))
		}

		condResult := condOutputs[0].flat.([]bool)[0]
		backend.putBuffer(condOutputs[0])

		if !condResult {
			// Condition is false, exit loop.
			break
		}

		// Execute body.
		newState, err := bodyExecutor.execute(state)
		if err != nil {
			return nil, errors.WithMessage(err, "While: body execution failed")
		}

		// Free old state and use new state.
		for _, buf := range state {
			backend.putBuffer(buf)
		}
		state = newState
	}

	return state, nil
}

// execIf implements the If operation.
// It executes one of two branches based on a predicate.
func execIf(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) ([]*Buffer, error) {
	ifData := node.data.(*ifNode)
	predBuffer := inputs[0]

	// Get predicate value.
	pred := predBuffer.flat.([]bool)[0]

	var branchExecutor *closureExecutor
	if pred {
		branchExecutor = newClosureExecutor(backend, ifData.trueBranch)
	} else {
		branchExecutor = newClosureExecutor(backend, ifData.falseBranch)
	}

	// Execute the selected branch (branches have no parameters based on If's graph construction).
	outputs, err := branchExecutor.execute(nil)
	if err != nil {
		branchName := "trueBranch"
		if !pred {
			branchName = "falseBranch"
		}
		return nil, errors.WithMessagef(err, "If: %s execution failed", branchName)
	}

	return outputs, nil
}

// execCall implements the Call operation.
// It invokes a named function with the given arguments.
func execCall(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) ([]*Buffer, error) {
	callData := node.data.(*callNode)
	targetFn := callData.targetFn

	// Create an executor for the target function.
	// Note: Call targets named functions, not closures, so we execute them like closures
	// but they're actually standalone functions.
	fnExecutor := newClosureExecutor(backend, targetFn)

	// Execute the function with the provided inputs.
	outputs, err := fnExecutor.execute(inputs)
	if err != nil {
		return nil, errors.WithMessagef(err, "Call: execution of function %q failed", targetFn.name)
	}

	return outputs, nil
}
