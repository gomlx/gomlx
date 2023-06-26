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
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/xla"
	"github.com/pkg/errors"
)

// This file includes internal operations to wrap XLA ops. They are wrapped
// by generic public ops in node.go that are a more Go friendly API.

// SliceXLA the operand from the start indices to the limit indices; e.g.
//
//	     x
//	[ 0 1 2 3 ]
//
// y [ 4 5 6 7 ] => slice(start={1, 1}, limit={2, 3}) => [ 5 6 ]
//
//	[ 8 9 a b ]
//
// Note that "limit" means up-to-but-not-including; i.e. [start, limit) in 1D
// range notation.
//
// The length of starts and limits must match the rank of x.
func SliceXLA(x *Node, starts, limits []int) *Node {
	g := validateGraphFromInputs(x)
	if !g.Ok() {
		return g.InvalidNode()
	}
	rank := x.shape.Rank()
	if len(starts) != rank || len(limits) != rank {
		g.SetErrorf("in SliceXLA(x, starts, limits) passed %d start values and %d limits values, but x has rank %d", len(starts), len(limits), rank)
		return g.InvalidNode()
	}
	strides := make([]int, rank)
	for ii := range strides {
		strides[ii] = 1
	}
	return SliceWithStridesXLA(x, starts, limits, strides)
}

// SliceWithStridesXLA is identical to SliceXLA but allows one to define the strides in
// each dimension.
// The length of starts, limits and strides must match the rank of x.
func SliceWithStridesXLA(x *Node, starts, limits, strides []int) *Node {
	g := validateGraphFromInputs(x)
	if !g.Ok() {
		return g.InvalidNode()
	}
	rank := x.shape.Rank()
	if len(starts) != rank || len(limits) != rank || len(strides) != rank {
		g.SetErrorf("in SliceWithStridesXLA(x, starts, limits, strides) passed %d start values, %d limits values and %d stride values, but x has rank %d", len(starts), len(limits), len(strides), rank)
		return g.InvalidNode()
	}

	// Encode starts, limits and strides sequentially, since their size are the same,
	// it will be easy to separate them in Const++.
	ints := make([]int, 0, 3*rank)
	ints = append(ints, starts...)
	ints = append(ints, limits...)
	ints = append(ints, strides...)
	return newNode(g, &xla.SerializedNode{
		Type: xla.SliceNode,
		Ints: ints,
	}, []*Node{x})
}

// gatherXLA is a powerful but cumbersome Gather operation offered by XLA. See Gather for the simpler version.
// Full details in https://www.tensorflow.org/xla/operation_semantics#gather.
// Its gradient is not defined for every case, use Gather and GatherSlices below.
// indices_are_unique are always set to false.
//
// Not exported for now, hopefully Gather and GatherSlices will suffice.
func gatherXLA(operand, startIndices *Node, indexVectorDim int, offsetDims, collapsedSliceDims, startIndexMap, sliceSizes []int, indicesAreSorted bool) *Node {
	g := validateGraphFromInputs(operand, startIndices)
	if !g.Ok() {
		return g.InvalidNode()
	}

	//fmt.Printf("\tgatherXLA: operand=%s, start=%s, indexVectorDim=%d, offsetDims=%v, collapsedSliceDims=%v, startIndexMap=%v, sliceSizes=%v\n",
	//	operand.shape, startIndices.shape, indexVectorDim, offsetDims, collapsedSliceDims, startIndexMap, sliceSizes)

	// Encoding of the values as follows. IMPORTANT: this code needs to be in sync with corresponding
	// decoding code in c/gomlx/computation.cpp, in function ComputationAddOp, under GatherNode case,
	// and with deserializeGatherXLA below.
	//
	//  * 6 first elements store the various parameters and lengths:
	ints := make([]int, 6+len(offsetDims)+len(collapsedSliceDims)+len(startIndexMap)+len(sliceSizes))
	ints[0] = indexVectorDim
	ints[1] = len(offsetDims)
	ints[2] = len(collapsedSliceDims)
	ints[3] = len(startIndexMap)
	ints[4] = len(sliceSizes)
	ints[5] = boolToInt(indicesAreSorted)

	//  * Copy sequentially the contents of the 3 int arrays:
	pos := 6
	for _, slice := range [][]int{offsetDims, collapsedSliceDims, startIndexMap, sliceSizes} {
		if len(slice) > 0 {
			copy(ints[pos:], slice)
			pos += len(slice)
		}
	}
	return newNode(g, &xla.SerializedNode{
		Type: xla.GatherNode,
		Ints: ints,
	}, []*Node{operand, startIndices})
}

// deserializeGatherXLA unpacks the parameters passed to xlaGather.
func deserializeGatherXLA(serialized *xla.SerializedNode) (
	indexVectorDim int, offsetDims, collapsedSliceDims, startIndexMap,
	sliceSizes []int, indicesAreSorted bool, err error) {
	if serialized.Type != xla.GatherNode {
		err = errors.Errorf("wrong node type (%s) for unserlizeGatherXLA", serialized.Type)
		return
	}
	ints := serialized.Ints
	indexVectorDim = ints[0]
	indicesAreSorted = intToBool(ints[5])

	pos := 6
	extractSlice := func(lenIdx int) []int {
		length := serialized.Ints[lenIdx]
		from := pos
		pos += length
		return ints[from:pos]
	}
	offsetDims = extractSlice(1)
	collapsedSliceDims = extractSlice(2)
	startIndexMap = extractSlice(3)
	sliceSizes = extractSlice(4)
	return
}

// scatterXLA is a powerful but cumbersome Scatter operation offered by XLA. See Scatter for the simpler version.
// Full details in https://www.tensorflow.org/xla/operation_semantics#scatter.
// Its gradient is not defined for every case, prefer instead using Scatter and ScatterAdd below.
// Not exported for now, hopefully Scatter and ScatterAdd will suffice.
func scatterXLA(operand, scatterIndices, updates *Node,
	indexVectorDim int, updateWindowDims, insertedWindowDims, scatterDimsToOperandDims []int,
	indicesAreSorted, uniqueIndices bool) *Node {
	//fmt.Printf("\tscatterXLA: operand=%s, scatterIndices=%s, updates=%s, indexVectorDim=%d, updateWindowDims=%v, insertedWindowDims=%v, scatterDimsToOperandDims=%v, indicesAreSorted=%v, uniqueIndices=%v\n",
	//	operand.shape, scatterIndices.shape, updates.shape, indexVectorDim, updateWindowDims, insertedWindowDims, scatterDimsToOperandDims, indicesAreSorted, uniqueIndices)
	g := validateGraphFromInputs(operand, scatterIndices, updates)
	if !g.Ok() {
		return g.InvalidNode()
	}

	// Encoding of the values as follows. IMPORTANT: this code needs to be in sync with corresponding
	// decoding code in c/gomlx/computation.cpp, in function ComputationAddOp, under GatherNode case.
	//  * 6 first elements store the various parameters and lengths:
	ints := make([]int, 0, 6+len(updateWindowDims)+len(insertedWindowDims)+len(scatterDimsToOperandDims))
	ints = append(ints, indexVectorDim)
	ints = append(ints, boolToInt(indicesAreSorted))
	ints = append(ints, boolToInt(uniqueIndices))
	ints = append(ints, len(updateWindowDims))
	ints = append(ints, len(insertedWindowDims))
	ints = append(ints, len(scatterDimsToOperandDims))
	for _, slice := range [][]int{updateWindowDims, insertedWindowDims, scatterDimsToOperandDims} {
		ints = append(ints, slice...)
	}
	return newNode(g, &xla.SerializedNode{
		Type: xla.ScatterNode,
		Ints: ints,
	}, []*Node{operand, scatterIndices, updates})
}

// AdjustAxis returns the positive axis to the operand shape, adjusting in case the axis given is negative.
//
// It sets the graph to an error state (`operand.Graph()`) if the axis given is out of range for the operand
// shape.
func AdjustAxis(operand *Node, axis int) int {
	g := operand.Graph()
	adjustedAxis := axis
	if axis < 0 {
		adjustedAxis = operand.Rank() + axis
	}
	if adjustedAxis < 0 || adjustedAxis >= operand.Rank() {
		g.SetErrorf("invalid axis %d, operand rank is %d", axis, operand.Rank())
		return axis
	}
	return adjustedAxis
}

// BatchNormInferenceXLA implements Batch Norm for inference. See details in
// https://www.tensorflow.org/xla/operation_semantics#batchnorminference.
//
// The recommendation is not to use this function directly, instead rely on layers.BatchNorm(), which
// will create and maintain the necessary variables.
//
// Based on paper "Batch Normalization: Accelerating Deep Network Training by Reducing
// Internal Covariate Shift" (Sergey Ioffe, Christian Szegedy), https://arxiv.org/abs/1502.03167.
func BatchNormInferenceXLA(operand, scale, offset, mean, variance *Node, epsilon float32, axis int) *Node {
	g := validateGraphFromInputs(operand, scale, offset, mean, variance)
	if !g.Ok() {
		return g.InvalidNode()
	}
	axis = AdjustAxis(operand, axis)
	if !g.Ok() {
		return g.InvalidNode()
	}
	return newNode(g, &xla.SerializedNode{
		Type:  xla.BatchNormInferenceNode,
		Int:   axis,
		Float: epsilon,
	}, []*Node{operand, scale, offset, mean, variance})
}

// BatchNormTrainingXLA implements Batch Norm for training. See details in
// https://www.tensorflow.org/xla/operation_semantics#batchnormtraining.
//
// It returns the normalized tensor, the batchMean and the batchVariance.
//
// The recommendation is not to use this function directly, instead rely on layers.BatchNorm(), which
// will create and maintain the necessary variables.
//
// Based on paper "Batch Normalization: Accelerating Deep Network Training by Reducing
// Internal Covariate Shift" (Sergey Ioffe, Christian Szegedy), https://arxiv.org/abs/1502.03167.
func BatchNormTrainingXLA(operand, scale, offset *Node, epsilon float32, axis int) (normalized, batchMean, batchVariance *Node) {
	g := validateGraphFromInputs(operand, scale, offset)
	if !g.Ok() {
		return g.InvalidNode(), g.InvalidNode(), g.InvalidNode()
	}
	axis = AdjustAxis(operand, axis)
	if !g.Ok() {
		return g.InvalidNode(), g.InvalidNode(), g.InvalidNode()
	}
	tuple := newNode(g, &xla.SerializedNode{
		Type:       xla.BatchNormTrainingNode,
		NodeInputs: make([]int32, 3),
		Int:        axis,
		Float:      epsilon,
	}, []*Node{operand, scale, offset})
	if !g.Ok() {
		return g.InvalidNode(), g.InvalidNode(), g.InvalidNode()
	}
	parts := SplitTuple(tuple)
	if !g.Ok() {
		return g.InvalidNode(), g.InvalidNode(), g.InvalidNode()
	}
	normalized, batchMean, batchVariance = parts[0], parts[1], parts[2]
	return
}

// batchNormGradXLA implements the gradient of Batch Norm. See details in
// https://www.tensorflow.org/xla/operation_semantics#batchnormgrad
//
// The recommendation is not to use this function directly, instead rely on layers.BatchNorm(), which
// will create and maintain the necessary variables.
//
// Based on paper "Batch Normalization: Accelerating Deep Network Training by Reducing
// Internal Covariate Shift" (Sergey Ioffe, Christian Szegedy), https://arxiv.org/abs/1502.03167.
func batchNormGradXLA(operand, scale, mean, variance, gradOutput *Node, epsilon float32, axis int) (
	gradOperand, gradScale, gradOffset *Node) {
	g := validateGraphFromInputs(operand, scale, mean, variance, gradOutput)
	if !g.Ok() {
		return g.InvalidNode(), g.InvalidNode(), g.InvalidNode()
	}
	axis = AdjustAxis(operand, axis)
	if !g.Ok() {
		return g.InvalidNode(), g.InvalidNode(), g.InvalidNode()
	}
	tuple := newNode(g, &xla.SerializedNode{
		Type:       xla.BatchNormGradNode,
		NodeInputs: make([]int32, 5),
		Int:        axis,
		Float:      epsilon,
	}, []*Node{operand, scale, mean, variance, gradOutput})
	if !g.Ok() {
		return g.InvalidNode(), g.InvalidNode(), g.InvalidNode()
	}
	parts := SplitTuple(tuple)
	if !g.Ok() {
		return g.InvalidNode(), g.InvalidNode(), g.InvalidNode()
	}
	gradOperand, gradScale, gradOffset = parts[0], parts[1], parts[2]
	return
}

// dotGeneralXLA takes as input lhs (left-hand-side) and rhs (right-hand-side) specifications
// for a general vector product. Each axis can be:
//
//   - Just aligned (batch axes), so the output has the same axes as the inputs. The dimensions
//     must match in lhs and rhs.
//   - Crossed (default), in which case the output is the combination (concatenation) of the
//     dimensions.
//   - Contracted (contracting axes), where the output does multiply the values and reduce sum
//     those dimensions.
//
// It follows that the resulting dimension number starts with the batch dimension, then the 'lhs'
// non-contracting/non-batch dimension, and finally the 'rhs' non-contracting/non-batch dimension.
//
// It provides the basic means of implementing Einsum.
func dotGeneralXLA(lhs *Node, lhsContractingAxes, lhsBatchAxes []int,
	rhs *Node, rhsContractingAxes, rhsBatchAxes []int) *Node {
	g := validateGraphFromInputs(lhs, rhs)
	if !g.Ok() {
		return g.InvalidNode()
	}

	var lists = [][]int{lhsContractingAxes, lhsBatchAxes, rhsContractingAxes, rhsBatchAxes}
	intsLen := len(lists)
	for _, list := range lists {
		intsLen += len(list)
	}
	ints := make([]int, 0, intsLen)
	for _, list := range lists {
		ints = append(ints, len(list))
	}
	for _, list := range lists {
		ints = append(ints, list...)
	}
	return newNode(g, &xla.SerializedNode{
		Type: xla.DotGeneralNode,
		Ints: ints,
	}, []*Node{lhs, rhs})
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
	lhs, rhs := node.inputs[0], node.inputs[1]

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

		// Project output (of the DotGeneral) shaped v to "this" (this node) shape.

		// * Add back contracted dimensions, with size 1.
		//   thisVJP shape will be [batch_dims..., lhs_cross_dims..., rhs_cross_dims..., 1 x (numContractionAxes)].
		thisVJP := v
		if numContractionAxes > 0 {
			thisVJP = ExpandDims(thisVJP, slices.SliceWithValue(numContractionAxes, -1)...)
		}

		// * Project other operand with contracted dimensions.
		//   otherProjected shape for this=lhs will be [batch_dims..., 1 x (this_cross_dims), rhs_cross_dims, contracted_dims]
		otherProjected := otherInput
		otherRank := otherProjected.shape.Rank()
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
				otherProjected = ExpandDims(otherProjected, slices.SliceWithValue(len(thisCrossAxes), pos)...)
			}
		}

		// * Multiply the contracted dimension by otherProjected: this will expand the contracted dimensions.
		thisVJP = Mul(thisVJP, otherProjected)

		// * Contract the otherCrossAxes, since those dimensions should exist in the final thisVJP (these
		//   cross axes came from the "other" input.
		if len(otherCrossAxes) > 0 {
			pos := numBatchAxes
			if thisCrossesFirst {
				pos += len(thisCrossAxes)
			}
			thisVJP = ReduceSum(thisVJP, slices.Iota(pos, len(otherCrossAxes))...)
		}

		// * Transpose thisVJP axes back to its inputs.
		thisRank := thisVJP.shape.Rank()
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
