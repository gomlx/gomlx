/*
 *	Copyright 2024 Jan Pfeifer
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
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gopjrt/dtypes"
)

// Gather values in params from the pointers in indices.
// The outputs are slices of `params` selected by `indices`, stitched together.
//
// Let's assume params has outputShapes `[i_0, ..., i_M, s_0, ..., s_o]`, where:
//
//   - `i_0, ..., i_N` are the N "indexed dimensions", that is, the dimensions indexed by `indices`.
//   - `s_0, ..., s_S` are the S dimensions of the slices that are going to be "gathered" (copied over).
//
// And let's assume indices has outputShapes `[o_0,...,o_O, N]`, where:
//
//   - `o_0, ..., o_O` are enumerations of the slices from `params` to gather.
//     E.g.: let's say O=1, and o_0=3, that means there will be 3 slices to gather.
//   - Last dimension `N`: this is the number of indices in `params` to point to. `N` is the number of
//     dimensions indexed `i_0, ..., i_N` in `params` above.
//
// The output will have outputShapes `[o_0,...,o_O, s_0, ... s_S]`, where:
//
//   - `o_0, ..., o_O` come from indices, and are enumerations of the slices from params to gather.
//   - `s_0, ..., s_S` are the slice sizes copied from params.
//
// For example:
//
//	params := [][]float32{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}
//	indices := [][]int{{1}, {0}}
//	Gather(params, indices) would return {{3, 4, 5}, {0, 1, 2}}
//
// In the case above params outputShapes is interpreted as `[i_0=3, s_0=3]`, and indices' outputShapes is
// `[o_0=2, N=1]`. The output outputShapes is `[o_0=2, s_0=3]`.
func Gather(params, indices *Node) *Node {
	_ = validateBuildingGraphFromInputs(params, indices)
	if params.IsScalar() {
		Panicf("cannot Gather from scalar or tuple, params outputShapes is %s", params.Shape())
	}
	if !indices.DType().IsInt() {
		Panicf("Gather requires indices to have an integer type, got outputShapes %q instead", indices.Shape())
	}

	// If indices is a scalar, simply convert it to outputShapes `[1]`.
	if indices.IsScalar() {
		indices = ExpandDims(indices, 0)
	}

	// Check ranks are compatible.
	paramsRank := params.Rank()
	indicesRank := indices.Rank()
	indexedSubRank := indices.Shape().Dimensions[indicesRank-1] // N from documentation.
	slicesSubRank := paramsRank - indexedSubRank                // S from documentation, the slices dimensions.
	if slicesSubRank < 0 {
		Panicf("Gather params are \"over-indexed\": params has only rank %d and "+
			"indexed rank is %d (last dimension of indices)", paramsRank, indexedSubRank)
	}
	outputSubRank := indicesRank - 1

	// Construct call to gatherXLA:
	// * indexVectorDim is always the last one.
	indexVectorDim := indicesRank - 1
	// * startIndexMap is sequential and sorted
	startIndexMap := make([]int, indexedSubRank)
	for ii := 0; ii < indexedSubRank; ii++ {
		startIndexMap[ii] = ii
	}
	// * sliceSizes are 1 everywhere but on the sliced dimensions.
	// * collapsedSliceDims is set to collapse all dimensions set to 1.
	sliceSizes := make([]int, paramsRank)
	collapsedSliceDims := make([]int, indexedSubRank)
	for ii := 0; ii < paramsRank; ii++ {
		if ii < indexedSubRank {
			sliceSizes[ii] = 1
			collapsedSliceDims[ii] = ii
		} else {
			sliceSizes[ii] = params.Shape().Dimensions[ii]
		}
	}
	// * offsetDims are the dimensions indexed.
	offsetDims := make([]int, paramsRank-indexedSubRank)
	for ii := range offsetDims {
		offsetDims[ii] = outputSubRank + ii
	}

	// Make no assumptions about indices being sorted or unique.
	// TODO: add version where these can be set.
	return backendGather(params, indices, indexVectorDim, offsetDims, collapsedSliceDims, startIndexMap, sliceSizes,
		false)
}

// GatherSlices from inputNodes. Each axis listed in slicedAxes have corresponding start position and size for each
// slice indexed by `start` (a graph Node, can be dynamically generated in the graph) and `sizes`, which will
// define the output final outputShapes, and must be statically given.
//
// Axes in slicedAxes can be given as negative numbers, which are taken from the the end of the input rank --
// that is axis -1 means the last axis in the input. Axes not given in slicedAxes (and in `start` and `sizes`)
// are taken in full length.
//
// Axes in slicedAxes must be given sorted in increasing order.
//
// The output has a rank equal to the prefixing rank of `start` (== `start.Rank()-1`) plus the rank of `input`.
// And the outputShapes will depend on the sizes of the slices.
//
//   - TODO: Add an option to support batch axes, present in both the input and in the start indices.
//     This will need to automatically concatenate the batch index in the start Node as a iota of each
//     batch example, and add the size 1 slice.
//     This can be done manually today.
//
// Example:
//
//		x := IotaFull(g, shapes.Make(dtypes.Float64, 3, 10, 10))  // 300 in total.
//		start := Const(g, [][]int32{{0, 3}, {1, 2}})  // 2 slices
//		sizes := []int{1, 3}
//		slices := GatherSlices(x, []int{1,2}, start, sizes)  // Axis=0 is taken in full.
//	    slices.AssertDims(2, 3, 1, 2)  // 2 slices, Axis=0 taken in full (3), and each slice of dimensions (1, 2).
//		// Result would be [][][][]int32{{{0, 1, 2, 3, 4}}, {{30, 31, 32, 33, 34}}, {{40, 41, 42, 43, 44}}}
func GatherSlices(input *Node, slicedAxes []int, start *Node, sizes []int) (gathered *Node) {
	_ = validateBuildingGraphFromInputs(input, start)
	if input.Shape().IsScalar() || input.Shape().IsTuple() {
		Panicf("cannot GatherSlices from scalar or tuple, input outputShapes is %s", input.Shape())
	}
	if !start.Shape().DType.IsInt() {
		Panicf("GatherSlices requires start indices to have an integer type, got outputShapes %q instead",
			start.Shape())
	}
	if start.Shape().IsScalar() {
		start = ExpandDims(start, 0)
	}

	// Check ranks are compatible.
	inputRank := input.Rank()
	startRank := start.Rank()
	numSlicedAxes := len(slicedAxes)
	if len(sizes) != numSlicedAxes {
		Panicf("GatherSlices requires one value in sizes for each axis marked as slicedAxes -- slicedAxes=%v, sizes=%v",
			slicedAxes, sizes)
	}
	if start.Shape().Dimensions[startRank-1] != numSlicedAxes {
		Panicf("GatherSlices requires the last axis of `start` to be the same dimension as the slicedAxes, "+
			"so it takes one index value per axis to be sliced -- slicedAxes=%v, start.Shape()=%s",
			slicedAxes, start.Shape())
	}
	outputPrefixRank := startRank - 1

	// AssertValid slicedAxes and normalizes it (replacing negative axis to their corresponding ones).
	{
		seen := types.MakeSet[int](numSlicedAxes)
		normalized := make([]int, 0, numSlicedAxes)
		for ii, axis := range slicedAxes {
			if axis < 0 {
				axis = inputRank + axis
			}
			if axis < 0 || axis >= inputRank {
				Panicf("GatherSlices got an invalid axis (%d) selected for slicing, input.Shape()=%s, slicedAxes=%v",
					slicedAxes[ii], input.Shape(), slicedAxes)
			}
			if seen.Has(axis) {
				Panicf("GatherSlices got an axis (%d) selected twice for slicing, input.Shape()=%s, slicedAxes=%v",
					slicedAxes[ii], input.Shape(), slicedAxes)
			}
			seen.Insert(axis)
			if ii > 0 && axis < normalized[ii-1] {
				Panicf("GatherSlices got an axis (%d) out-of-order, slicedAxes (%v) must be given in increasing order "+
					"(and `sizes` and `start` must match that order)",
					slicedAxes[ii], slicedAxes)
			}
			normalized = append(normalized, axis)
		}
		slicedAxes = normalized
	}

	// Construct call to gatherXLA:
	// * indexVectorDim indicates the axis in the start that has the indices: it's always the last one.
	indexVectorDim := startRank - 1
	// * startIndexMap holds the axis in the input that are pointed by `start`. These are exactly the normalized slicedAxes.
	startIndexMap := slicedAxes
	// * sliceSizes must be defined for each input axis, and are either given in `sizes` or are assumed to be the full dimension
	//   of the input.
	sliceSizes := input.Shape().Clone().Dimensions // Start with a copy of the input's dimensions.
	for ii, size := range sizes {
		axis := slicedAxes[ii]
		sliceSizes[axis] = size
	}

	// * offsetDims must point for each input axis that is not collapsed, the output Node. Since we don't collapse any of the
	//   input dimensions, all the input axes need to be mapped. Notice that this preserves the order of the axis given by
	//   the input (the order in `slicedAxes` will be ignored).
	offsetDims := make([]int, 0, numSlicedAxes)
	var collapsedSliceDims []int // Left empty.
	for ii := 0; ii < inputRank; ii++ {
		axis := ii + outputPrefixRank
		offsetDims = append(offsetDims, axis)
	}

	// Make no assumptions about indices being sorted or unique.
	// TODO: add version where these can be set.
	return backendGather(input, start, indexVectorDim, offsetDims, collapsedSliceDims, startIndexMap, sliceSizes, false)
}

// GatherWithBatchDims values in params from pointers in indices.
// It works exactly the same as tensorflow's gather_nd operation, described in
// https://www.tensorflow.org/api_docs/python/tf/gather_nd.
//
// Let's assume params has outputShapes `[b_0,...,b_{batchDim}, i_0, ..., i_M, s_0, ..., s_o]`, where:
//
//   - `b_0, ..., b_{batchDim}` are the batchDims batch dimensions, dimensions that are shared
//     in params, indices and will also be present in the output.
//   - `i_0, ..., i_N` are the N "indexed dimensions," that is, the dimensions indexed by indices.
//   - `s_0, ..., s_S` are the `S` dimensions of the slices that are going to be "gathered" (copied over).
//
// And, let's assume indices has outputShapes `[b_0, ... b_{batchDim}, o_0,...,o_O, N]`, where:
//
//   - `b_0, ..., b_{batchDim}` are the batchDims batch dimensions, dimensions that are shared
//     in params, indices and will also be present in the output.
//   - `o_0, ..., o_O` are enumerations of the slices from params to gather.
//     E.g.: let's say O=1, and o_0=3, that means there will be 3 slices to gather.
//   - Last dimension N: this is the number of indices in params to point to. N is the same value as
//     the dimension `i_0, ..., i_N` in params above.
//
// The output will have outputShapes `[b_0, ... b_{batchDim}, o_0,...,o_O, s_0, ... s_S]`, where:
//
//   - `b_0, ..., b_{batchDim}` are the batchDims batch dimensions, dimensions that are shared
//     in params, indices and will also be present in the output.
//   - `o_0, ..., o_O` come from indices, and are enumerations of the slices from params to gather.
//   - `s_0, ..., s_S` are the slice sizes copied from params.
//
// See some examples in `node_test.go` function `TestGather`.
/*
func GatherWithBatchDims(params, indices *Node, batchDims int) *Node {
	g := validateBuildingGraphFromInputs(params, indices)
	if params.Shape().IsScalar() || params.Shape().IsTuple() {
		Panicf("cannot Gather from scalar or tuple, params outputShapes is %s", params.Shape())
	}
	if !indices.Shape().DType.IsInt() {
		Panicf("Gather requires indices to have an integer type, got outputShapes %q instead", indices.Shape())
	}

	// If indices is a scalar, simply convert it to outputShapes `[1]`.
	if indices.Shape().IsScalar() {
		indices = ReshapeWithShape(indices, shapes.Make(indices.DType(), 1))
	}

	// Check ranks are compatible.
	paramsRank := params.Rank()
	indicesRank := indices.Rank()
	indexedSubRank := indices.Shape().Dimensions[indicesRank-1]
	numIndices := indices.Shape().Size() / indexedSubRank
	slicesSubRank := paramsRank - batchDims - indexedSubRank
	if slicesSubRank < 0 {
		Panicf("Gather params are \"over-indexed\": params has only rank %d, batchDims=%d and "+
			"indexed rank is %d (last dimension of indices)", paramsRank, batchDims, indexedSubRank))
	}
	outputSubRank := indicesRank - 1 - batchDims
	if outputSubRank < 0 {
		Panicf("Gather indices don't have enough batch dimensions: indices rank is %d and "+
			"one dimension is needed for the indices themselves, but batchDims=%d", indicesRank, batchDims))
	}

	// Grow indices to include batch dimensions: "example" here mean one element of the batch
	// dimensions. This is because the underlying gatherXLA doesn't support batch dimensions.
	if batchDims > 0 {
		if !types.DeepSliceCmp(params.Shape().Dimensions[0:batchDims], indices.Shape().Dimensions[0:batchDims], types.Equal[int]) {
			Panicf("batch dimensions (first %d dimensions) from params (outputShapes=%s) and indices (outputShapes=%s) don't match",
				batchDims, params.outputShapes, indices.outputShapes))
		}
		batchIndices := IndicesForShape(g, shapes.Make(types.Int64, indices.Shape().Dimensions[0:batchDims]))
		// Now batchIndices need to be broadcast to each id for the gather.
		... TODO: flatten batchIndices, broadcast it, and concatenate to a flattenedIndices, and
		... then reshape back. After that, just call the simpler Gather().
		flatIndices := ReshapeWithShape(indices)
	}
	return Gather(params, indices)
}
*/

// IndicesForShape enumerates a list of indices for all elements of the given outputShapes. It will always
// return a node with outputShapes [outputShapes.Size(), outputShapes.Rank()].
// E.g: if outputShapes=[3, 2], it returns `[[0 0] [0 1] [1 0] [1 1] [2 0] [2 1]]`.
func IndicesForShape(g *Graph, shape shapes.Shape) *Node {
	if shape.IsScalar() {
		Panicf("can't generate IndicesForShape for scalars (outputShapes=%s)", shape)
	}
	indices := Iota(g, shapes.Make(dtypes.Int64, shape.Size(), 1), 0)
	indices = BroadcastToShape(indices, shapes.Make(dtypes.Int64, shape.Size(), shape.Rank()))
	// Example of indices' value here: for outputShapes=`[3, 2]`, indices=`{{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}}`

	dividers := make([]int, shape.Rank())
	dividers[shape.Rank()-1] = 1
	for ii := shape.Rank() - 2; ii >= 0; ii -= 1 {
		dividers[ii] = dividers[ii+1] * shape.Dimensions[ii+1]
	}
	//fmt.Printf("outputShapes=%s, dividers=%v, size=%v\n", outputShapes, dividers, outputShapes.Size())
	indices = Div(indices, Const(g, [][]int{dividers}))
	indices = Mod(indices, Const(g, [][]int{shape.Dimensions}))
	return indices
}

// Scatter sums up the slices in updates into a new tensor of the given outputShapes, at the locations pointed by indices.
// It does the opposite of Gather.
//
// In the simplest form, [indices] is shaped `[num_updates, 1]`, [updates] is shaped `[num_updates, update_size]` and
// [outputShapes] is of the form `[output_size, update_size]`. The indices values should be in between 0 and `output_size-1`.
func Scatter(indices, updates *Node, shape shapes.Shape) *Node {
	g := validateBuildingGraphFromInputs(indices, updates)
	zeros := Zeros(g, shape)
	return ScatterAdd(zeros, indices, updates, false, false)
}

// ScatterAdd adds up the slices in updates into the given operand tensor, at the locations pointed by indices.
// It does the opposite of Gather.
//
// Args:
// - [sorted]: the indices must be in order. In some cases it is faster, but if indices are not in order results may be unstable.
// - [unique]: the indices must be unique. In some cases it is faster, but if indices are not unique results may be unstable.
func ScatterAdd(operand, indices, updates *Node, sorted, unique bool) *Node {
	_ = validateBuildingGraphFromInputs(operand, indices, updates)

	if !indices.Shape().DType.IsInt() {
		Panicf("scatter operations require integer indices, instead got outputShapes %s", indices.outputShapes)
	}
	if operand.Shape().DType != updates.Shape().DType {
		Panicf(
			"scatter operations require operand and updates to have the same DType, instead got shapes %s (operand) and %s (updates)",
			operand.outputShapes, updates.outputShapes)
	}
	if indices.Shape().IsTuple() || operand.Shape().IsTuple() || updates.Shape().IsTuple() {
		Panicf("tuples are not supported in ScatterAdd, operand.outputShapes=%s, indices.outputShapes=%s, updates.outputShapes=%s",
			operand.outputShapes, indices.outputShapes, updates.outputShapes)
	}
	if indices.Shape().IsScalar() {
		indices = ExpandDims(indices, 0)
	}

	// Check shapes compatibility.
	indicesRank := indices.Rank()
	indexedRank := indices.Shape().Dimensions[indicesRank-1]
	updatesRank := updates.Rank()
	if updatesRank < indicesRank-1 || !xslices.DeepSliceCmp(updates.Shape().Dimensions[:indicesRank-1], indices.Shape().Dimensions[:indicesRank-1], xslices.Equal[int]) {
		Panicf("updates rank prefix (outputShapes=%s) must match the first n-1 dimensions of the indices (outputShapes=%s)",
			updates.outputShapes, indices.outputShapes)
	}
	slicesRank := updatesRank - (indicesRank - 1)
	slicesDims := updates.Shape().Dimensions[indicesRank-1:]
	operandRank := operand.Shape().Rank()
	if operandRank != indexedRank+slicesRank || !xslices.DeepSliceCmp(operand.Shape().Dimensions[indexedRank:], slicesDims, xslices.Equal[int]) {
		Panicf("operand outputShapes (%s) has to be a combination of the indexed rank (%d, the last dimension of indices outputShapes %s) and "+
			"the slices coming from updates (the last %d dimensions %v of the updates, shaped %s)",
			operand.outputShapes, indexedRank, indices.outputShapes, slicesRank, slicesDims, updates.outputShapes)
	}

	// Set scatterXLA parameters:
	updateWindowsDims := make([]int, 0, slicesRank)
	for ii := updatesRank - slicesRank; ii < updatesRank; ii++ {
		updateWindowsDims = append(updateWindowsDims, ii)
	}
	insertedWindowDims := make([]int, 0, indexedRank)
	for ii := 0; ii < indexedRank; ii++ {
		insertedWindowDims = append(insertedWindowDims, ii)
	}
	scatterDimsToOperandDims := make([]int, 0, 10)
	for ii := 0; ii < indexedRank; ii++ {
		scatterDimsToOperandDims = append(scatterDimsToOperandDims, ii)
	}
	return backendScatterAdd(operand, indices, updates, indicesRank-1, updateWindowsDims, insertedWindowDims, scatterDimsToOperandDims,
		sorted, unique)
}
