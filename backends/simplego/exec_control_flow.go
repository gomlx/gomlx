// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"sort"

	"github.com/gomlx/gomlx/backends"
	"github.com/pkg/errors"
)

func init() {
	multiOutputsNodeExecutors[backends.OpTypeIf] = execIf
	multiOutputsNodeExecutors[backends.OpTypeWhile] = execWhile
	multiOutputsNodeExecutors[backends.OpTypeSort] = execSort
}

// execIf executes the If operation by evaluating the predicate and running one branch.
func execIf(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) ([]*Buffer, error) {
	predBuffer := inputs[0]
	predFlat := predBuffer.flat.([]bool)
	if len(predFlat) != 1 {
		return nil, errors.Errorf("If: predicate must be scalar, got %d elements", len(predFlat))
	}
	pred := predFlat[0]

	data := node.data.(*ifNode)
	var branchFn *Function
	if pred {
		branchFn = data.trueBranch
	} else {
		branchFn = data.falseBranch
	}

	// Execute the branch (no inputs since branches have no parameters)
	outputs, err := branchFn.compiled.Execute(backend, nil, nil)
	if err != nil {
		return nil, errors.WithMessagef(err, "If: executing branch")
	}

	return outputs, nil
}

// execWhile executes the While operation by looping until condition returns false.
func execWhile(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) ([]*Buffer, error) {
	data := node.data.(*whileNode)
	condFn := data.cond.compiled
	bodyFn := data.body.compiled

	// Clone inputs for state (we don't want to modify the original inputs)
	state := make([]*Buffer, len(inputs))
	for i, input := range inputs {
		if inputsOwned[i] {
			state[i] = input
			inputs[i] = nil
		} else {
			state[i] = backend.cloneBuffer(input)
		}
	}

	// Loop while condition is true
	const maxIterations = 1_000_000 // Safety limit
	for iter := range maxIterations {
		// Evaluate condition
		condOutputs, err := condFn.Execute(backend, state, nil)
		if err != nil {
			for _, buf := range state {
				backend.putBuffer(buf)
			}
			return nil, errors.WithMessagef(err, "While: evaluating condition at iteration %d", iter)
		}

		// Check condition result
		condResult := condOutputs[0].flat.([]bool)[0]
		backend.putBuffer(condOutputs[0])

		if !condResult {
			// Condition is false, exit loop
			return state, nil
		}

		// Execute body to get new state
		newState, err := bodyFn.Execute(backend, state, nil)
		if err != nil {
			for _, buf := range state {
				backend.putBuffer(buf)
			}
			return nil, errors.WithMessagef(err, "While: executing body at iteration %d", iter)
		}

		// Free old state and use new state
		for _, buf := range state {
			backend.putBuffer(buf)
		}
		state = newState
	}

	// Cleanup on max iterations reached
	for _, buf := range state {
		backend.putBuffer(buf)
	}
	return nil, errors.Errorf("While: exceeded maximum iterations (%d)", maxIterations)
}

// execSort sorts tensors along the specified axis using the comparator closure.
func execSort(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) ([]*Buffer, error) {
	data := node.data.(*sortNode)
	axis := data.axis
	isStable := data.isStable
	compFn := data.comparator.compiled

	if len(inputs) == 0 {
		return nil, errors.Errorf("Sort: requires at least one input")
	}

	// Get shape info from first input
	shape := inputs[0].shape
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

	// Create output buffers (clones of inputs)
	outputs := make([]*Buffer, len(inputs))
	for i, input := range inputs {
		if inputsOwned[i] {
			outputs[i] = input
			inputs[i] = nil
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

				// Execute comparator
				compOutputs, err := compFn.Execute(backend, compInputs, nil)
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
	switch f := flat.(type) {
	case []float32:
		scalar.flat.([]float32)[0] = f[offset]
	case []float64:
		scalar.flat.([]float64)[0] = f[offset]
	case []int32:
		scalar.flat.([]int32)[0] = f[offset]
	case []int64:
		scalar.flat.([]int64)[0] = f[offset]
	case []int8:
		scalar.flat.([]int8)[0] = f[offset]
	case []int16:
		scalar.flat.([]int16)[0] = f[offset]
	case []uint8:
		scalar.flat.([]uint8)[0] = f[offset]
	case []uint16:
		scalar.flat.([]uint16)[0] = f[offset]
	case []uint32:
		scalar.flat.([]uint32)[0] = f[offset]
	case []uint64:
		scalar.flat.([]uint64)[0] = f[offset]
	case []bool:
		scalar.flat.([]bool)[0] = f[offset]
	default:
		panic(errors.Errorf("setScalarFromFlat: unsupported type %T", flat))
	}
}

// applyPermutation reorders elements along the sort axis according to the given indices.
func applyPermutation(buf *Buffer, indices []int, baseOffset, axisStride, axisSize int) {
	switch flat := buf.flat.(type) {
	case []float32:
		applyPermutationTyped(flat, indices, baseOffset, axisStride, axisSize)
	case []float64:
		applyPermutationTyped(flat, indices, baseOffset, axisStride, axisSize)
	case []int32:
		applyPermutationTyped(flat, indices, baseOffset, axisStride, axisSize)
	case []int64:
		applyPermutationTyped(flat, indices, baseOffset, axisStride, axisSize)
	case []int8:
		applyPermutationTyped(flat, indices, baseOffset, axisStride, axisSize)
	case []int16:
		applyPermutationTyped(flat, indices, baseOffset, axisStride, axisSize)
	case []uint8:
		applyPermutationTyped(flat, indices, baseOffset, axisStride, axisSize)
	case []uint16:
		applyPermutationTyped(flat, indices, baseOffset, axisStride, axisSize)
	case []uint32:
		applyPermutationTyped(flat, indices, baseOffset, axisStride, axisSize)
	case []uint64:
		applyPermutationTyped(flat, indices, baseOffset, axisStride, axisSize)
	case []bool:
		applyPermutationTyped(flat, indices, baseOffset, axisStride, axisSize)
	default:
		panic(errors.Errorf("applyPermutation: unsupported type %T", buf.flat))
	}
}

func applyPermutationTyped[T any](flat []T, indices []int, baseOffset, axisStride, axisSize int) {
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
