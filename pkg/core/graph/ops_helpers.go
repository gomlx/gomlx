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
	. "github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/xslices"
)

// AdjustAxisToOperandRank returns the positive axis to the operand shapes, adjusting in case the axis given is negative.
//
// It panics if axis given is not in the operand's rank range.
func AdjustAxisToOperandRank(operand *Node, axis int) int {
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
	params := node.inputs.(*nodeInputsDotGeneral)
	lhs, rhs := params.lhs, params.rhs
	lhsCrossAxes := dotCrossAxes(lhs, params.lhsContractingAxes, params.lhsBatchAxes)
	rhsCrossAxes := dotCrossAxes(rhs, params.rhsContractingAxes, params.rhsBatchAxes)

	// Gradient with respect to lhs:
	gradFn := func(thisInput *Node, thisBatchAxes, thisContractingAxes, thisCrossAxes []int, thisCrossesFirst bool,
		otherInput *Node, otherBatchAxes, otherContractingAxes, otherCrossAxes []int) *Node {
		_ = thisInput
		// Axes counts:
		numBatchAxes := len(thisBatchAxes)             // == len(otherBatchAxes)
		numContractionAxes := len(thisContractingAxes) // == len(otherContractingAxes)
		//numCrossedAxes := len(thisCrossAxes) + len(otherCrossAxes)

		// Project output (of the DotGeneral) shaped v to "this" (this node) shapes.

		// * Add back contracted dimensions, with size 1.
		//   thisVJP shapes will be [batch_dims..., lhs_cross_dims..., rhs_cross_dims..., 1 x (numContractionAxes)].
		thisVJP := v
		if numContractionAxes > 0 {
			thisVJP = InsertAxes(thisVJP, xslices.SliceWithValue(numContractionAxes, -1)...)
		}

		// * Project other operand with contracted dimensions.
		//   otherProjected shapes for this=lhs will be [batch_dims..., 1 x (this_cross_dims), rhs_cross_dims, contracted_dims]
		otherProjected := otherInput
		otherRank := otherProjected.Rank()
		{
			permutations := make([]int, 0, otherRank)
			permutations = append(permutations, otherBatchAxes...)
			permutations = append(permutations, otherCrossAxes...)
			permutations = append(permutations, otherContractingAxes...)
			changed := false
			for ii, axis := range permutations {
				if ii != axis {
					changed = true
				}
			}
			if changed {
				otherProjected = TransposeAllAxes(otherProjected, permutations...)
			}
			// Add placeholder axes (of dimension 1) for the crosses from "this".
			if len(thisCrossAxes) > 0 {
				pos := numBatchAxes // Where axes for thisCrossesAxes will be inserted.
				if !thisCrossesFirst {
					pos += len(otherCrossAxes)
				}
				otherProjected = InsertAxes(otherProjected, xslices.SliceWithValue(len(thisCrossAxes), pos)...)
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
		thisRank := thisVJP.Rank()
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
			thisVJP = TransposeAllAxes(thisVJP, permutation...)
		}
		return thisVJP
	}

	return []*Node{
		gradFn(lhs, params.lhsBatchAxes, params.lhsContractingAxes, lhsCrossAxes, true, rhs, params.rhsBatchAxes, params.rhsContractingAxes, rhsCrossAxes),  // grad wrt lhs
		gradFn(rhs, params.rhsBatchAxes, params.rhsContractingAxes, rhsCrossAxes, false, lhs, params.lhsBatchAxes, params.lhsContractingAxes, lhsCrossAxes), // grad wrt rhs
	}
}
