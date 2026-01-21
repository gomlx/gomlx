// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"reflect"
	"sort"

	"github.com/gomlx/gomlx/backends"
	"github.com/pkg/errors"
)

func init() {
	multiOutputsNodeExecutors[backends.OpTypeIf] = execIf
	multiOutputsNodeExecutors[backends.OpTypeWhile] = execWhile
	multiOutputsNodeExecutors[backends.OpTypeSort] = execSort
}

// gatherCapturedValues looks up the captured values for a closure from the parent's execution buffers.
func gatherCapturedValues(fn *Function, parentExecBuf *funcExecBuffers) []*Buffer {
	if len(fn.capturedParentNodes) == 0 {
		return nil
	}

	captured := make([]*Buffer, len(fn.capturedParentNodes))
	for i, capturedNode := range fn.capturedParentNodes {
		captured[i] = parentExecBuf.results[capturedNode.idx]
	}
	return captured
}

// execIf executes the If operation by evaluating the predicate and running one branch.
// Inputs layout: [pred, trueBranch captured values..., falseBranch captured values...]
func execIf(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool, parentExecBuf *funcExecBuffers) ([]*Buffer, error) {
	_ = parentExecBuf // Captured values are now passed as inputs, not looked up from parent

	predBuffer := inputs[0]
	predFlat := predBuffer.flat.([]bool)
	if len(predFlat) != 1 {
		return nil, errors.Errorf("If: predicate must be scalar, got %d elements", len(predFlat))
	}
	pred := predFlat[0]

	data := node.data.(*ifNode)

	// Extract captured values from inputs based on which branch we're executing
	var branchFn *Function
	var capturedInputs []*Buffer
	if pred {
		branchFn = data.trueBranch
		// True branch captured values are at inputs[1:1+trueCapturedCount]
		if data.trueCapturedCount > 0 {
			capturedInputs = inputs[1 : 1+data.trueCapturedCount]
		}
	} else {
		branchFn = data.falseBranch
		// False branch captured values are at inputs[1+trueCapturedCount:]
		if data.falseCapturedCount > 0 {
			capturedInputs = inputs[1+data.trueCapturedCount:]
		}
	}

	// Execute the branch (no parameters, but may have captured values)
	outputs, err := branchFn.compiled.Execute(backend, nil, nil, capturedInputs, nil)
	if err != nil {
		return nil, errors.WithMessagef(err, "If: executing branch")
	}

	return outputs, nil
}

// execWhile executes the While operation by looping until condition returns false.
// Inputs layout: [state values..., cond captured values..., body captured values...]
func execWhile(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool, parentExecBuf *funcExecBuffers) ([]*Buffer, error) {
	_ = parentExecBuf // Captured values are now passed as inputs, not looked up from parent

	data := node.data.(*whileNode)
	condFn := data.cond
	bodyFn := data.body

	// Extract state values and captured values from inputs
	stateCount := data.stateCount
	stateInputs := inputs[:stateCount]
	stateOwned := inputsOwned[:stateCount]

	// Extract captured values for cond and body
	var condCaptured, bodyCaptured []*Buffer
	if data.condCapturedCount > 0 {
		condCaptured = inputs[stateCount : stateCount+data.condCapturedCount]
	}
	if data.bodyCapturedCount > 0 {
		bodyCaptured = inputs[stateCount+data.condCapturedCount:]
	}

	// Set up state buffers and ownership tracking
	// Per review: if we own the input buffer, take ownership and donate it
	// Otherwise, use it directly but don't donate
	state := make([]*Buffer, stateCount)
	copy(state, stateInputs)
	donateState := make([]bool, stateCount)
	donateAll := make([]bool, stateCount)
	for i := range donateAll {
		donateAll[i] = true
	}

	for i := range stateCount {
		if stateOwned[i] {
			stateInputs[i] = nil   // Take ownership of buffer
			donateState[i] = true  // Ownership will be transferred to condFn
		}
	}

	// Loop while condition is true (no iteration limit)
	for iter := 0; ; iter++ {
		// Evaluate condition - donate state buffers we own
		condOutputs, err := condFn.compiled.Execute(backend, state, donateState, condCaptured, nil)
		// After condFn, all donated buffers have been consumed - we no longer own them
		// But condFn returns new buffers that we now own implicitly (captured in condOutputs for single bool)

		if err != nil {
			// On error, we don't own any state buffers anymore (they were donated or never owned)
			return nil, errors.WithMessagef(err, "While: evaluating condition at iteration %d", iter)
		}

		// Check condition result
		condResult := condOutputs[0].flat.([]bool)[0]
		backend.putBuffer(condOutputs[0])

		if !condResult {
			// Condition is false, exit loop
			// We need to return owned buffers. After first iteration, donateState = donateAll
			// so we own all state buffers. On first iteration, we need to clone non-owned ones.
			for i, owned := range donateState {
				if !owned {
					state[i] = backend.cloneBuffer(state[i])
				}
			}
			return state, nil
		}

		// Execute body to get new state
		// Pass state and donate - after this call, state buffers are consumed
		newState, err := bodyFn.compiled.Execute(backend, state, donateState, bodyCaptured, nil)
		// After bodyFn, all donated state is consumed. If error, we own nothing.
		// If success, we own all of newState.
		donateState = donateAll // After first iteration, we always own everything

		if err != nil {
			// On error, we no longer own state (donated), and newState is empty
			return nil, errors.WithMessagef(err, "While: executing body at iteration %d", iter)
		}

		state = newState
	}
}

// execSort sorts tensors along the specified axis using the comparator closure.
// Inputs layout: [input tensors..., comparator captured values...]
func execSort(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool, parentExecBuf *funcExecBuffers) ([]*Buffer, error) {
	_ = parentExecBuf // Captured values are now passed as inputs, not looked up from parent

	data := node.data.(*sortNode)
	axis := data.axis
	isStable := data.isStable
	compFn := data.comparator

	// Extract input tensors and captured values
	inputCount := data.inputCount
	tensorInputs := inputs[:inputCount]
	tensorOwned := inputsOwned[:inputCount]

	// Extract captured values for comparator
	var compCaptured []*Buffer
	if data.compCapturedCount > 0 {
		compCaptured = inputs[inputCount:]
	}

	if inputCount == 0 {
		return nil, errors.Errorf("Sort: requires at least one input")
	}

	// Get shape info from first input
	shape := tensorInputs[0].shape
	rank := shape.Rank()
	axisSize := shape.Dimensions[axis]

	// Calculate sizes for iteration
	// We iterate over all positions except the sort axis
	outerSize := 1
	for i := range axis {
		outerSize *= shape.Dimensions[i]
	}
	innerSize := 1
	for i := axis + 1; i < rank; i++ {
		innerSize *= shape.Dimensions[i]
	}

	// Create output buffers (clones of input tensors)
	outputs := make([]*Buffer, inputCount)
	for i, input := range tensorInputs {
		if tensorOwned[i] {
			outputs[i] = input
			tensorInputs[i] = nil
		} else {
			outputs[i] = backend.cloneBuffer(input)
		}
	}

	// Create index array for sorting
	indices := make([]int, axisSize)
	for i := range indices {
		indices[i] = i
	}

	// Create temporary buffers for comparator inputs (2 scalars per input tensor)
	compInputs := make([]*Buffer, 2*len(outputs))
	for i, output := range outputs {
		compInputs[2*i] = backend.getBuffer(output.shape.DType, 1)
		compInputs[2*i].shape = output.shape.Clone()
		compInputs[2*i].shape.Dimensions = nil // scalar

		compInputs[2*i+1] = backend.getBuffer(output.shape.DType, 1)
		compInputs[2*i+1].shape = output.shape.Clone()
		compInputs[2*i+1].shape.Dimensions = nil // scalar
	}
	defer func() {
		for _, buf := range compInputs {
			backend.putBuffer(buf)
		}
	}()

	// Calculate strides for the axis
	axisStride := innerSize

	// Sort each "row" along the axis
	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			baseOffset := outer*axisSize*innerSize + inner

			// Reset indices
			for i := range indices {
				indices[i] = i
			}

			// Sort indices using comparator
			var sortErr error
			lessFunc := func(i, j int) bool {
				if sortErr != nil {
					return false
				}

				idxI := indices[i]
				idxJ := indices[j]
				offsetI := baseOffset + idxI*axisStride
				offsetJ := baseOffset + idxJ*axisStride

				// Set comparator inputs
				for k, output := range outputs {
					setScalarFromFlat(compInputs[2*k], output.flat, offsetI)
					setScalarFromFlat(compInputs[2*k+1], output.flat, offsetJ)
				}

				// Execute comparator (don't donate inputs, pass captured values)
				compOutputs, err := compFn.compiled.Execute(backend, compInputs, nil, compCaptured, nil)
				if err != nil {
					sortErr = err
					return false
				}

				result := compOutputs[0].flat.([]bool)[0]
				backend.putBuffer(compOutputs[0])
				return result
			}

			if isStable {
				sort.SliceStable(indices, lessFunc)
			} else {
				sort.Slice(indices, lessFunc)
			}

			if sortErr != nil {
				for _, buf := range outputs {
					backend.putBuffer(buf)
				}
				return nil, errors.WithMessagef(sortErr, "Sort: comparator failed")
			}

			// Apply permutation to outputs
			for _, output := range outputs {
				applyPermutation(output, indices, baseOffset, axisStride, axisSize)
			}
		}
	}

	return outputs, nil
}

// setScalarFromFlat sets a scalar buffer's value from a flat array at the given offset.
func setScalarFromFlat(scalar *Buffer, flat any, offset int) {
	value := reflect.ValueOf(flat).Index(offset)
	reflect.ValueOf(scalar.flat).Index(0).Set(value)
}

// applyPermutationDTypeMap dispatches applyPermutation by dtype.
var applyPermutationDTypeMap = NewDTypeMap("ApplyPermutation")

// applyPermutation reorders elements along the sort axis according to the given indices.
func applyPermutation(buf *Buffer, indices []int, baseOffset, axisStride, axisSize int) {
	fn := applyPermutationDTypeMap.Get(buf.shape.DType).(func(buf *Buffer, indices []int, baseOffset, axisStride, axisSize int))
	fn(buf, indices, baseOffset, axisStride, axisSize)
}

func applyPermutationGeneric[T SupportedTypesConstraints](buf *Buffer, indices []int, baseOffset, axisStride, axisSize int) {
	flat := buf.flat.([]T)
	// Extract values to temp slice
	temp := make([]T, axisSize)
	for i := range axisSize {
		temp[i] = flat[baseOffset+i*axisStride]
	}

	// Apply permutation
	for i, idx := range indices {
		flat[baseOffset+i*axisStride] = temp[idx]
	}
}
