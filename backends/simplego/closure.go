// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/pkg/errors"
)

// CompiledClosure holds pre-compiled execution information for a closure.
// This allows efficient execution without recomputing the execution order each time.
type CompiledClosure struct {
	// function is the source Function this was compiled from.
	function *Function

	// sortedNodes are the nodes needed for execution, in topological order.
	// These are indices into the builder's nodes slice.
	sortedNodes []*Node

	// nodeToSortedIdx maps a node's builderIdx to its index in sortedNodes.
	// This allows O(1) lookup when executing.
	nodeToSortedIdx map[int]int

	// parameterIndices maps parameter nodes (by builderIdx) to their input index.
	parameterIndices map[int]int

	// outputNodes are the nodes that produce the closure's outputs.
	outputNodes []*Node

	// numUses tracks how many times each node's result is used (indexed by sortedNodes position).
	numUses []int

	// maxInputs is the maximum number of inputs any node has.
	maxInputs int
}

// Execute runs the compiled closure with the given inputs.
// The inputs must match the closure's parameters in count and shape.
func (cc *CompiledClosure) Execute(backend *Backend, inputs []*Buffer) ([]*Buffer, error) {
	// Validate input count
	if len(inputs) != len(cc.function.parameters) {
		return nil, errors.Errorf("closure expects %d inputs, got %d",
			len(cc.function.parameters), len(inputs))
	}

	// Allocate result storage
	numNodes := len(cc.sortedNodes)
	results := make([]*Buffer, numNodes)
	numUsed := make([]int, numNodes)
	owned := make([]bool, numNodes)

	// Set up parameters from inputs
	for paramBuilderIdx, inputIdx := range cc.parameterIndices {
		if sortedIdx, ok := cc.nodeToSortedIdx[paramBuilderIdx]; ok {
			results[sortedIdx] = inputs[inputIdx]
			owned[sortedIdx] = false // Don't own input buffers
		}
	}

	// Pre-allocate input buffers for node execution
	inputBuffers := make([]*Buffer, cc.maxInputs)
	inputsOwned := make([]bool, cc.maxInputs)

	// Execute nodes in topological order
	for sortedIdx, node := range cc.sortedNodes {
		if results[sortedIdx] != nil {
			// Already computed (parameter or will be handled by special case)
			continue
		}

		// Handle constants specially - they have pre-computed buffers
		if node.opType == backends.OpTypeConstant {
			results[sortedIdx] = node.data.(*Buffer)
			owned[sortedIdx] = false
			continue
		}

		// Prepare inputs for this node
		numInputs := len(node.inputs)
		for j, inputNode := range node.inputs {
			inputSortedIdx, ok := cc.nodeToSortedIdx[inputNode.builderIdx]
			if !ok {
				return nil, errors.Errorf("input node %d not found in sorted nodes", inputNode.builderIdx)
			}
			inputBuffers[j] = results[inputSortedIdx]
			if inputBuffers[j] == nil {
				return nil, errors.Errorf("input %d for node %s not computed yet", j, node.opType)
			}
			// Only "own" the input if this is the last use of it
			inputsOwned[j] = owned[inputSortedIdx] &&
				cc.numUses[inputSortedIdx]-numUsed[inputSortedIdx] == 1
		}

		// Execute the node
		if node.IsMultiOutputs() {
			// Multi-output nodes
			multiExecutor := multiOutputsNodeExecutors[node.opType]
			if multiExecutor == nil {
				return nil, errors.Errorf("no multi-output executor for op %s", node.opType)
			}

			outputBuffers, err := multiExecutor(backend, node, inputBuffers[:numInputs], inputsOwned[:numInputs])
			if err != nil {
				return nil, errors.WithMessagef(err, "executing multi-output %s", node.opType)
			}

			// Store outputs in their respective positions
			for outputIdx, outputBuf := range outputBuffers {
				outputNode := node.multiOutputsNodes[outputIdx]
				outputSortedIdx, ok := cc.nodeToSortedIdx[outputNode.builderIdx]
				if !ok {
					// This output is not needed, free it
					backend.putBuffer(outputBuf)
					continue
				}
				results[outputSortedIdx] = outputBuf
				owned[outputSortedIdx] = true
			}
		} else {
			// Single-output nodes
			executor := nodeExecutors[node.opType]
			if executor == nil {
				return nil, errors.Errorf("no executor for op %s", node.opType)
			}

			result, err := executor(backend, node, inputBuffers[:numInputs], inputsOwned[:numInputs])
			if err != nil {
				return nil, errors.WithMessagef(err, "executing %s", node.opType)
			}
			results[sortedIdx] = result
			owned[sortedIdx] = true
		}

		// Update usage counts and free unused buffers
		for j, inputNode := range node.inputs {
			inputSortedIdx := cc.nodeToSortedIdx[inputNode.builderIdx]
			numUsed[inputSortedIdx]++
			if numUsed[inputSortedIdx] == cc.numUses[inputSortedIdx] &&
				owned[inputSortedIdx] &&
				inputBuffers[j] != nil {
				// This was the last use, free the buffer
				backend.putBuffer(inputBuffers[j])
				results[inputSortedIdx] = nil
			}
		}
	}

	// Collect outputs
	outputs := make([]*Buffer, len(cc.outputNodes))
	for i, outNode := range cc.outputNodes {
		sortedIdx, ok := cc.nodeToSortedIdx[outNode.builderIdx]
		if !ok {
			return nil, errors.Errorf("output node %d not found in sorted nodes", outNode.builderIdx)
		}
		outputs[i] = results[sortedIdx]
		if outputs[i] == nil {
			return nil, errors.Errorf("output %d not computed", i)
		}
		if !owned[sortedIdx] {
			// Clone the buffer since we don't own it
			outputs[i] = backend.cloneBuffer(results[sortedIdx])
		}
	}

	// Free any remaining owned buffers that weren't outputs
	for sortedIdx, buf := range results {
		if buf == nil {
			continue
		}
		// Check if this buffer is one of the outputs
		isOutput := false
		for i, outNode := range cc.outputNodes {
			outSortedIdx := cc.nodeToSortedIdx[outNode.builderIdx]
			if sortedIdx == outSortedIdx && outputs[i] == buf {
				isOutput = true
				break
			}
		}
		if !isOutput && owned[sortedIdx] {
			backend.putBuffer(buf)
		}
	}

	return outputs, nil
}
