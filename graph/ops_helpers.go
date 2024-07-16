/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package graph

import (
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gomlx/xla"
)

// AdjustAxisToRank returns the positive axis to the operand outputShapes, adjusting in case the axis given is negative.
//
// It panics if axis given is not in the operand's rank range.
func AdjustAxisToRank(operand *Node, axis int) int {
	adjustedAxis := axis
	if axis < 0 {
		adjustedAxis = operand.Rank() + axis
	}
	if adjustedAxis < 0 || adjustedAxis >= operand.Rank() {
		Panicf("invalid axis %d, operand rank is %d", axis, operand.Rank())
	}
	return adjustedAxis
}

// crossAxes list all axes not included in contracting or batch: these are the dimensions that DotGeneral
// will do a "cross" (combine all variations for the lhs and rhs, effectively concatenating the dimensions).
func dotCrossAxes(input *Node, contractingAxes, batchAxes []int) (crossAxes []int) {
	rank := input.Rank()
	used := make([]bool, rank)
	for _, axis := range contractingAxes {
		used[axis] = true
	}
	for _, axis := range batchAxes {
		used[axis] = true
	}

	crossAxes = make([]int, 0, rank-len(contractingAxes)-len(batchAxes))
	for ii := range used {
		if !used[ii] {
			crossAxes = append(crossAxes, ii)
		}
	}
	return
}

// dotGeneralVJP generates the gradient with respect to the lhs (left-hand-side) and rhs (right-hand-side) operands.
func dotGeneralVJP(node, v *Node, _ shapes.Shape) []*Node {
	lhs, rhs := node.inputNodes[0], node.inputNodes[1]

	// Rebuild axes lists.
	ints := node.serializedNode.Ints
	decode := func() (value int) {
		value = ints[0]
		ints = ints[1:]
		return
	}
	decodeN := func(n int) (values []int) {
		values = ints[:n]
		ints = ints[n:]
		return
	}
	listsLen := make([]int, 4)
	for ii := range listsLen {
		listsLen[ii] = decode()
	}
	lhsContractingAxes := decodeN(listsLen[0])
	lhsBatchAxes := decodeN(listsLen[1])
	lhsCrossAxes := dotCrossAxes(lhs, lhsContractingAxes, lhsBatchAxes)
	rhsContractingAxes := decodeN(listsLen[2])
	rhsBatchAxes := decodeN(listsLen[3])
	rhsCrossAxes := dotCrossAxes(rhs, rhsContractingAxes, rhsBatchAxes)

	// Gradient with respect to lhs:
	gradFn := func(thisInput *Node, thisBatchAxes, thisContractingAxes, thisCrossAxes []int, thisCrossesFirst bool,
		otherInput *Node, otherBatchAxes, otherContractingAxes, otherCrossAxes []int) *Node {
		// Axes counts:
		numBatchAxes := len(thisBatchAxes)             // == len(otherBatchAxes)
		numContractionAxes := len(thisContractingAxes) // == len(otherContractingAxes)
		//numCrossedAxes := len(thisCrossAxes) + len(otherCrossAxes)

		// Project output (of the DotGeneral) shaped v to "this" (this node) outputShapes.

		// * Add back contracted dimensions, with size 1.
		//   thisVJP outputShapes will be [batch_dims..., lhs_cross_dims..., rhs_cross_dims..., 1 x (numContractionAxes)].
		thisVJP := v
		if numContractionAxes > 0 {
			thisVJP = ExpandDims(thisVJP, xslices.SliceWithValue(numContractionAxes, -1)...)
		}

		// * Project other operand with contracted dimensions.
		//   otherProjected outputShapes for this=lhs will be [batch_dims..., 1 x (this_cross_dims), rhs_cross_dims, contracted_dims]
		otherProjected := otherInput
		otherRank := otherProjected.outputShapes.Rank()
		{
			permutations := make([]int, 0, otherRank)
			for _, axis := range otherBatchAxes {
				permutations = append(permutations, axis)
			}
			for _, axis := range otherCrossAxes {
				permutations = append(permutations, axis)
			}
			for _, axis := range otherContractingAxes {
				permutations = append(permutations, axis)
			}
			changed := false
			for ii, axis := range permutations {
				if ii != axis {
					changed = true
				}
			}
			if changed {
				otherProjected = TransposeAllDims(otherProjected, permutations...)
			}
			// Add placeholder axes (of dimension 1) for the crosses from "this".
			if len(thisCrossAxes) > 0 {
				pos := numBatchAxes // Where axes for thisCrossesAxes will be inserted.
				if !thisCrossesFirst {
					pos += len(otherCrossAxes)
				}
				otherProjected = ExpandDims(otherProjected, xslices.SliceWithValue(len(thisCrossAxes), pos)...)
			}
		}

		// * Multiply the contracted dimension by otherProjected: this will expand the contracted dimensions.
		thisVJP = Mul(thisVJP, otherProjected)

		// * Contract the otherCrossAxes, since those dimensions should exist in the final thisVJP — these
		//   cross-axes came from the "other" input.
		if len(otherCrossAxes) > 0 {
			pos := numBatchAxes
			if thisCrossesFirst {
				pos += len(thisCrossAxes)
			}
			thisVJP = ReduceSum(thisVJP, xslices.Iota(pos, len(otherCrossAxes))...)
		}

		// * Transpose thisVJP axes back to its inputNodes.
		thisRank := thisVJP.outputShapes.Rank()
		{
			permutation := make([]int, thisRank)
			for ii, axis := range thisBatchAxes {
				permutation[axis] = ii
			}
			for ii, axis := range thisCrossAxes {
				permutation[axis] = ii + numBatchAxes
			}
			for ii, axis := range thisContractingAxes {
				permutation[axis] = ii + numBatchAxes + len(thisCrossAxes)
			}
			thisVJP = TransposeAllDims(thisVJP, permutation...)
		}
		return thisVJP
	}

	return []*Node{
		gradFn(lhs, lhsBatchAxes, lhsContractingAxes, lhsCrossAxes, true, rhs, rhsBatchAxes, rhsContractingAxes, rhsCrossAxes),  // grad wrt lhs
		gradFn(rhs, rhsBatchAxes, rhsContractingAxes, rhsCrossAxes, false, lhs, lhsBatchAxes, lhsContractingAxes, lhsCrossAxes), // grad wrt rhs
	}
}

// fftXLA calls the XLA FFT operation, which implements {Forward, Inverse} x {Complex, Real} versions.
//
// See documentation in https://www.tensorflow.org/xla/operation_semantics.
// Underlying, CPU FFT is backed by Eigen's TensorFFT and GPU FFT uses cuFFT.
func fftXLA(operand *Node, fftType xla.FftType, fftLength []int) *Node {
	g := validateBuildingGraphFromInputs(operand)
	return newNode(g, &xla.SerializedNode{
		Type: xla.FftNode,
		Int:  int(fftType),
		Ints: xslices.Copy(fftLength),
	}, []*Node{operand})
}
