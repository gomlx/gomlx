// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"reflect"
	"sort"

	"github.com/gomlx/gomlx/backends"
	"github.com/pkg/errors"
)

func init() {
	nodeClosureExecutors[backends.OpTypeIf] = execIf
	nodeClosureExecutors[backends.OpTypeWhile] = execWhile
	nodeClosureExecutors[backends.OpTypeSort] = execSort
}

// freeOwnedBuffers releases buffers that are owned, setting them to nil to prevent double-free.
// This is used to clean up captured inputs when they're no longer needed.
func freeOwnedBuffers(backend *Backend, buffers []*Buffer, owned []bool) {
	for i, isOwned := range owned {
		if isOwned && buffers[i] != nil {
			backend.putBuffer(buffers[i])
			buffers[i] = nil
		}
	}
}

// execIf executes the If operation by evaluating the predicate and running one branch.
// Captured inputs layout: [trueBranch captured values..., falseBranch captured values...]
func execIf(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool, closureInputs ClosureInputs) ([]*Buffer, error) {
	predBuffer := inputs[0]
	predFlat := predBuffer.flat.([]bool)
	if len(predFlat) != 1 {
		return nil, errors.Errorf("If: predicate must be scalar, got %d elements", len(predFlat))
	}
	pred := predFlat[0]

	data := node.data.(*ifNode)
	allCaptured := closureInputs.Buffers
	allCapturedOwned := closureInputs.Owned

	// Split captured values based on which branch we're executing
	var branchFn *Function
	var capturedInputs []*Buffer
	var donateCaptures []bool
	if pred {
		branchFn = data.trueBranch
		if data.trueCapturedCount > 0 {
			capturedInputs = allCaptured[:data.trueCapturedCount]
			donateCaptures = allCapturedOwned[:data.trueCapturedCount]
		}
		// Free the false branch captured inputs we own (they won't be used)
		if data.falseCapturedCount > 0 {
			freeOwnedBuffers(backend, allCaptured[data.trueCapturedCount:], allCapturedOwned[data.trueCapturedCount:])
		}
	} else {
		branchFn = data.falseBranch
		if data.falseCapturedCount > 0 {
			capturedInputs = allCaptured[data.trueCapturedCount:]
			donateCaptures = allCapturedOwned[data.trueCapturedCount:]
		}
		// Free the true branch captured inputs we own (they won't be used)
		if data.trueCapturedCount > 0 {
			freeOwnedBuffers(backend, allCaptured[:data.trueCapturedCount], allCapturedOwned[:data.trueCapturedCount])
		}
	}

	// Execute the branch with proper donation of captured values
	outputs, err := branchFn.compiled.Execute(backend, nil, nil, capturedInputs, donateCaptures)
	if err != nil {
		return nil, errors.WithMessagef(err, "If: executing branch")
	}

	return outputs, nil
}

// execWhile executes the While operation by looping until condition returns false.
// Regular inputs: [state values...]
// Captured inputs layout: [cond captured values..., body captured values...]
//
// Note on captured input donation: Captured values are reused across all iterations,
// so we cannot donate them until the loop exits. On the final iteration (when condition
// returns false), we free the owned captured inputs since they won't be used again.
func execWhile(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool, closureInputs ClosureInputs) ([]*Buffer, error) {
	data := node.data.(*whileNode)
	condFn := data.cond
	bodyFn := data.body

	// State values come from regular inputs
	stateCount := data.stateCount
	stateInputs := inputs[:stateCount]
	stateOwned := inputsOwned[:stateCount]

	// Get captured inputs and their ownership
	allCaptured := closureInputs.Buffers
	allCapturedOwned := closureInputs.Owned

	// Split captured values for cond and body
	var condCaptured, bodyCaptured []*Buffer
	var condCapturedOwned, bodyCapturedOwned []bool
	if data.condCapturedCount > 0 {
		condCaptured = allCaptured[:data.condCapturedCount]
		condCapturedOwned = allCapturedOwned[:data.condCapturedCount]
	}
	if data.bodyCapturedCount > 0 {
		bodyCaptured = allCaptured[data.condCapturedCount:]
		bodyCapturedOwned = allCapturedOwned[data.condCapturedCount:]
	}

	// Set up state buffers and ownership tracking
	state := make([]*Buffer, stateCount)
	copy(state, stateInputs)
	donateState := make([]bool, stateCount)
	donateAll := make([]bool, stateCount)
	for i := range donateAll {
		donateAll[i] = true
	}

	for i := range stateCount {
		if stateOwned[i] {
			stateInputs[i] = nil  // Take ownership of buffer
			donateState[i] = true // Ownership will be transferred to condFn
		}
	}

	// Helper to free all owned captured inputs
	freeAllCaptures := func() {
		freeOwnedBuffers(backend, condCaptured, condCapturedOwned)
		freeOwnedBuffers(backend, bodyCaptured, bodyCapturedOwned)
	}

	// Loop while condition is true
	for iter := 0; ; iter++ {
		// Evaluate condition - DON'T donate state or captured buffers since we may need them
		condOutputs, err := condFn.compiled.Execute(backend, state, nil, condCaptured, nil)

		if err != nil {
			// On error, clean up owned state and captured buffers
			for i, owned := range donateState {
				if owned && state[i] != nil {
					backend.putBuffer(state[i])
				}
			}
			freeAllCaptures()
			return nil, errors.WithMessagef(err, "While: evaluating condition at iteration %d", iter)
		}

		// Check condition result
		condResult := condOutputs[0].flat.([]bool)[0]
		backend.putBuffer(condOutputs[0])

		if !condResult {
			// Condition is false, exit loop. Free owned captured inputs.
			freeAllCaptures()

			// Return state buffers. Clone any we don't own.
			for i, owned := range donateState {
				if !owned {
					state[i] = backend.cloneBuffer(state[i])
				}
			}
			return state, nil
		}

		// Execute body to get new state
		// DON'T donate captured buffers - they're reused across iterations
		newState, err := bodyFn.compiled.Execute(backend, state, donateState, bodyCaptured, nil)
		// After bodyFn, all donated state is consumed.
		donateState = donateAll // After first iteration, we always own everything

		if err != nil {
			// On error, we no longer own state (donated), but we still own captured buffers
			freeAllCaptures()
			return nil, errors.WithMessagef(err, "While: executing body at iteration %d", iter)
		}

		state = newState
	}
}

// execSort sorts tensors along the specified axis using the comparator closure.
// Regular inputs: [input tensors...]
// Captured inputs: [comparator captured values...]
//
// Note on captured input donation: The comparator is called O(n log n) times during
// sorting, so we cannot donate captured inputs until after the sort completes.
func execSort(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool, closureInputs ClosureInputs) ([]*Buffer, error) {
	data := node.data.(*sortNode)
	axis := data.axis
	isStable := data.isStable
	compFn := data.comparator

	// Input tensors come from regular inputs
	inputCount := data.inputCount
	tensorInputs := inputs[:inputCount]
	tensorOwned := inputsOwned[:inputCount]

	// Get captured inputs and their ownership
	compCaptured := closureInputs.Buffers
	compCapturedOwned := closureInputs.Owned

	if inputCount == 0 {
		freeOwnedBuffers(backend, compCaptured, compCapturedOwned)
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
			// Use panic/recover to abort sort immediately on comparator error
			sortErr := func() (sortErr error) {
				defer func() {
					if r := recover(); r != nil {
						if err, ok := r.(error); ok {
							sortErr = err
						} else {
							panic(r) // Re-panic if not our error
						}
					}
				}()

				lessFunc := func(i, j int) bool {
					idxI := indices[i]
					idxJ := indices[j]
					offsetI := baseOffset + idxI*axisStride
					offsetJ := baseOffset + idxJ*axisStride

					// Set comparator inputs
					for k, output := range outputs {
						setScalarFromFlat(compInputs[2*k], output.flat, offsetI)
						setScalarFromFlat(compInputs[2*k+1], output.flat, offsetJ)
					}

					// Execute comparator - DON'T donate captured inputs, they're reused
					compOutputs, err := compFn.compiled.Execute(backend, compInputs, nil, compCaptured, nil)
					if err != nil {
						panic(err) // Abort sort immediately
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
				return nil
			}()

			if sortErr != nil {
				for _, buf := range outputs {
					backend.putBuffer(buf)
				}
				freeOwnedBuffers(backend, compCaptured, compCapturedOwned)
				return nil, errors.WithMessagef(sortErr, "Sort: comparator failed")
			}

			// Apply permutation to outputs
			for _, output := range outputs {
				applyPermutation(output, indices, baseOffset, axisStride, axisSize)
			}
		}
	}

	// Sort complete, free owned captured inputs
	freeOwnedBuffers(backend, compCaptured, compCapturedOwned)

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
