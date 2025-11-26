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
	"slices"

	. "github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/gomlx/gopjrt/dtypes"
)

// NormalizeIndices converts Python-style negative indices to positive indices
// for gathering along a single axis of data.
//
// Negative indices are converted by adding the dimension size of the specified axis.
// For example, if data has dimension 5 on the specified axis and an index is -1,
// it becomes 4 (i.e., 5 + (-1) = 4).
//
// This is useful for compatibility with ONNX and Python where -1 refers to the
// last element, -2 to the second-to-last, etc.
//
// Parameters:
//   - data: The tensor from which gathering will be done (used to get dimension size)
//   - indices: The values to normalize according to data dimensions, must be integer
//   - axis: The axis of data along which gathering will happen (supports negative axis)
//
// Returns normalized indices with the same shape and dtype as input indices.
//
// Notes:
//   - This function only converts negative indices to positive by adding the axis
//     dimension. It does NOT clamp values to valid bounds. Indices that remain
//     out-of-bounds after normalization (e.g., -6 for axis size 5 yields -1) will
//     be handled by the underlying Gather operation according to XLA/StableHLO
//     semantics, which clamps to valid range [0, dim-1].
//   - For ONNX compatibility, valid input indices should be in range [-dim, dim-1].
//   - Unsigned integer indices are technically supported but will never be modified
//     since they cannot be negative. Use signed integer types (int32, int64) for
//     indices that may contain negative values.
func NormalizeIndices(data, indices *Node, axis int) *Node {
	_ = validateBuildingGraphFromInputs(data, indices)
	if !indices.DType().IsInt() {
		Panicf("NormalizeIndices requires indices to have an integer type, got %s", indices.DType())
	}
	if indices.DType().IsUnsigned() {
		// Unsigned values cannot be negative, we are done.
		return indices
	}

	axis = AdjustAxisToOperandRank(data, axis)
	if axis < 0 || axis >= data.Rank() {
		Panicf("NormalizeIndices: axis %d out of range for data with rank %d", axis, data.Rank())
	}

	g := data.Graph()
	dim := data.Shape().Dimensions[axis]
	dimNode := Scalar(g, indices.DType(), dim)

	// Where indices < 0, add dim; otherwise keep original
	// normalized = Where(indices < 0, indices + dim, indices)
	zero := ScalarZero(g, indices.DType())
	isNegative := LessThan(indices, zero)
	normalizedIndices := Where(isNegative, Add(indices, dimNode), indices)
	return normalizedIndices
}

// Gather values in params from the pointers in indices.
// The outputs are slices of params selected by indices, stitched together.
//
// Let's assume params has shapes [i_1, ..., i_N, s_1, ..., s_S], where:
//
//   - `i_1, ..., i_N` are the N "indexed axes", that is, the axes that are indexed by indices.
//   - `s_1, ..., s_S` are the S dimensions of the slices that are going to be "gathered" (copied over).
//
// And let's assume indices has shapes [o_1,...,o_O, N], where:
//
//   - `o_1, ..., o_O` are "batch dimensions" of the slices from `params` to gather, that will be included in the output.
//     E.g.: let's say O=1, and o_1=3, that means there will be 3 slices to gather.
//   - Last dimension `N`: this is the number of indices in `params` to point to. `N` is the number of
//     dimensions indexed `i_1, ..., i_N` in `params` above.
//
// The output will have shapes [o_1,...,o_O, s_1, ... s_S], where:
//
//   - `o_1, ..., o_O` come from indices, and are enumerations of the slices from params to gather.
//   - `s_1, ..., s_S` are the slice sizes copied from params.
//
// indicesAreSorted can be set if you know the indices in start are sorted, in some backends this allows
// for optimizations. If not set it default to false.
//
// For example:
//
//	params := [][]float32{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}
//	indices := [][]int{{1}, {0}}
//	Gather(params, indices) would return {{3, 4, 5}, {0, 1, 2}}
//
// In the case above params shapes is interpreted as [i_1=3, s_1=3], and indices shapes is
// [o_1=2, N=1]. The output shapes is [o_1=2, s_1=3].
func Gather(params, indices *Node, indicesAreSorted ...bool) *Node {
	_ = validateBuildingGraphFromInputs(params, indices)
	if params.IsScalar() {
		Panicf("cannot Gather from scalar or tuple, params shapes is %s", params.Shape())
	}
	if !indices.DType().IsInt() {
		Panicf("Gather requires indices to have an integer type, got shapes %q instead", indices.Shape())
	}

	// If indices is a scalar, simply convert it to shapes `[1]`.
	if indices.IsScalar() {
		indices = InsertAxes(indices, 0)
	}
	if len(indicesAreSorted) > 1 {
		Panicf("Gather() optional indicesAreSorted takes only one value, %d were defined", len(indicesAreSorted))
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
	// * startIndexMap is sequential and sorted: it always points to the first axes of params.
	startIndexMap := make([]int, indexedSubRank)
	for ii := 0; ii < indexedSubRank; ii++ {
		startIndexMap[ii] = ii
	}
	// * sliceSizes are 1 where we indexed (indexedSubRank), and the full params size else where (the slices we are gathering).
	// * collapsedSliceDims sets to collapse all the indexed params dimensions (the first indexedSubRank).
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
	// * offsetOutputAxes are set for all params axes not collapsed, and they point to where those axes go in the output shape.
	offsetOutputAxes := make([]int, paramsRank-indexedSubRank)
	for ii := range offsetOutputAxes {
		offsetOutputAxes[ii] = outputSubRank + ii
	}

	// Make no assumptions about indices being sorted or unique.
	// TODO: add version where these can be set.
	return backendGather(params, indices, indexVectorDim, offsetOutputAxes, collapsedSliceDims, startIndexMap, sliceSizes,
		len(indicesAreSorted) == 1 && indicesAreSorted[0])
}

// GatherSlices from inputNodes. Each axis listed in slicedAxes have corresponding start position and size for each
// slice indexed by `start` (a graph Node, can be dynamically generated in the graph) and `sizes`, which will
// define the output final shapes, and must be statically given.
//
// Axes in slicedAxes can be given as negative numbers, which are taken from the the end of the input rank --
// that is axis -1 means the last axis in the input. Axes not given in slicedAxes (and in `start` and `sizes`)
// are taken in full length.
//
// Axes in slicedAxes must be given sorted in increasing order.
//
// The output has a rank equal to the prefixing rank of `start` (== `start.Rank()-1`) plus the rank of `input`.
// And the shapes will depend on the sizes of the slices.
//
//   - TODO: Add an option to support batch axes, present in both the input and in the start indices.
//     This will need to automatically concatenate the batch index in the start Node as a iota of each
//     batch example, and add the size 1 slice.
//     This can be done manually today.
//
// indicesAreSorted can be set if you know the indices in start are sorted, in some backends this allows
// for optimizations.
//
// Example:
//
//		x := IotaFull(g, shapes.Make(dtypes.Float64, 3, 10, 10))  // 300 in total.
//		start := Const(g, [][]int32{{0, 3}, {1, 2}})  // 2 slices
//		sizes := []int{1, 3}
//		slices := GatherSlices(x, []int{1,2}, start, sizes, true)  // Axis=0 is taken in full.
//	    slices.AssertDims(2, 3, 1, 2)  // 2 slices, Axis=0 taken in full (3), and each slice of dimensions (1, 2).
//		// Result would be [][][][]int32{{{0, 1, 2, 3, 4}}, {{30, 31, 32, 33, 34}}, {{40, 41, 42, 43, 44}}}
func GatherSlices(input *Node, slicedAxes []int, start *Node, sizes []int, indicesAreSorted bool) (gathered *Node) {
	_ = validateBuildingGraphFromInputs(input, start)
	if input.Shape().IsScalar() || input.Shape().IsTuple() {
		Panicf("cannot GatherSlices from scalar or tuple, input shapes is %s", input.Shape())
	}
	if !start.DType().IsInt() {
		Panicf("GatherSlices requires start indices to have an integer type, got shapes %q instead",
			start.Shape())
	}
	if start.Shape().IsScalar() {
		start = InsertAxes(start, 0)
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
	outputPrefixRank := startRank - 1 // The start axes will

	// AssertValid slicedAxes and normalizes it (replacing negative axis to their corresponding ones).
	{
		seen := sets.Make[int](numSlicedAxes)
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

	// Construct call to the backend.
	// * indexVectorDim indicates the axis in the start that has the indices: it's always the last one.
	indexVectorDim := startRank - 1
	// * startIndexMap is the list of input axes that are pointed by `start`. These are exactly the normalized slicedAxes.
	//   len(startIndexMap) == len(sizes)
	startIndexMap := slicedAxes
	// * sliceSizes must be defined for each input axis, and are either given in `sizes` or are assumed to be the full dimension
	//   of the input. The sliceSize is the full size (dimension) of the input for axes not being sliced.
	sliceSizes := slices.Clone(input.Shape().Dimensions) // Start with a copy of the input's dimensions.
	for ii, size := range sizes {
		axis := slicedAxes[ii]
		sliceSizes[axis] = size
	}

	// * offsetOutputAxes must have one output axis value for each input axis that is not collapsed. Since we don't collapse any of the
	//   input dimensions, all the input axes need to be mapped. Notice that this preserves the order of the axis given by
	//   the input (the order in `slicedAxes` will be ignored).
	offsetOutputAxes := make([]int, 0, numSlicedAxes)
	var collapsedSliceDims []int // Left empty.
	for ii := 0; ii < inputRank; ii++ {
		axis := ii + outputPrefixRank
		offsetOutputAxes = append(offsetOutputAxes, axis)
	}
	return backendGather(input, start, indexVectorDim, offsetOutputAxes, collapsedSliceDims, startIndexMap, sliceSizes, indicesAreSorted)
}

// BackendGather exposes the raw backend Gather operator.
//
// This is internal, and it is exposed only for debugging purposes, please don't rely on it.
// If it turns out you need some functionality here that is not provided in Gather or GatherSlices,
// open an issue in GoMLX and we'll figure a betterAPI.
//
// See convoluted and circular description in
// https://openxla.org/xla/operation_semantics#gather
func BackendGather(operand *Node, startIndices *Node, indexVectorAxis int, offsetAxes []int, collapsedSliceAxes []int, startIndexMap []int, sliceSizes []int, indicesAreSorted bool) (node *Node) {
	return backendGather(operand, startIndices, indexVectorAxis, offsetAxes, collapsedSliceAxes, startIndexMap, sliceSizes, indicesAreSorted)
}

// GatherWithBatchDims values in params from pointers in indices.
// It works exactly the same as tensorflow's gather_nd operation, described in
// https://www.tensorflow.org/api_docs/python/tf/gather_nd.
//
// Let's assume params has shapes `[b_0,...,b_{batchDim}, i_0, ..., i_M, s_0, ..., s_o]`, where:
//
//   - `b_0, ..., b_{batchDim}` are the batchDims batch dimensions, dimensions that are shared
//     in params, indices and will also be present in the output.
//   - `i_0, ..., i_N` are the N "indexed dimensions," that is, the dimensions indexed by indices.
//   - `s_0, ..., s_S` are the `S` dimensions of the slices that are going to be "gathered" (copied over).
//
// And, let's assume indices has shapes `[b_0, ... b_{batchDim}, o_0,...,o_O, N]`, where:
//
//   - `b_0, ..., b_{batchDim}` are the batchDims batch dimensions, dimensions that are shared
//     in params, indices and will also be present in the output.
//   - `o_0, ..., o_O` are enumerations of the slices from params to gather.
//     E.g.: let's say O=1, and o_0=3, that means there will be 3 slices to gather.
//   - Last dimension N: this is the number of indices in params to point to. N is the same value as
//     the dimension `i_0, ..., i_N` in params above.
//
// The output will have shapes `[b_0, ... b_{batchDim}, o_0,...,o_O, s_0, ... s_S]`, where:
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
		Panicf("cannot Gather from scalar or tuple, params shapes is %s", params.Shape())
	}
	if !indices.DType().IsInt() {
		Panicf("Gather requires indices to have an integer type, got shapes %q instead", indices.Shape())
	}

	// If indices is a scalar, simply convert it to shapes `[1]`.
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
		if !types.DeepSliceCmp(params.Shape().Dimensions[0:batchDims], indices.Shape().Dimensions[0:batchDims], types.EqualAny[int]) {
			Panicf("batch dimensions (first %d dimensions) from params (shapes=%s) and indices (shapes=%s) don't match",
				batchDims, params.Shape(), indices.Shape()))
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

// IndicesForShape enumerates a list of indices for all elements of the given shapes. It will always
// return a node with shapes [shapes.Size(), shapes.Rank()].
// E.g: if shapes=[3, 2], it returns `[[0 0] [0 1] [1 0] [1 1] [2 0] [2 1]]`.
func IndicesForShape(g *Graph, shape shapes.Shape) *Node {
	if shape.IsScalar() {
		Panicf("can't generate IndicesForShape for scalars (shapes=%s)", shape)
	}
	indices := Iota(g, shapes.Make(dtypes.Int64, shape.Size(), 1), 0)
	indices = BroadcastToShape(indices, shapes.Make(dtypes.Int64, shape.Size(), shape.Rank()))
	// Example of indices' value here: for shapes=`[3, 2]`, indices=`{{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}}`

	dividers := make([]int, shape.Rank())
	dividers[shape.Rank()-1] = 1
	for ii := shape.Rank() - 2; ii >= 0; ii -= 1 {
		dividers[ii] = dividers[ii+1] * shape.Dimensions[ii+1]
	}
	//fmt.Printf("shapes=%s, dividers=%v, size=%v\n", shapes, dividers, shapes.Size())
	indices = Div(indices, Const(g, [][]int{dividers}))
	indices = Mod(indices, Const(g, [][]int{shape.Dimensions}))
	return indices
}

// Scatter sums up the slices in updates into a new tensor of the given shapes, at the locations pointed by indices.
// It does the opposite of Gather.
//
// In the simplest form, [indices] is shaped `[num_updates, 1]`, [updates] is shaped `[num_updates, update_size]` and
// [shapes] is of the form `[output_size, update_size]`. The indices values should be in between 0 and `output_size-1`.
//
// Args:
//   - indices: the positions where to set the new values.
//   - updates: the values that are going to be set.
//   - shape: output shape, it must have the same dtype as updates.
//   - sorted: the indices must be in order.
//     If set to true for some backends it is faster, but if the indices are not sorted, results may be unstable.
//     If in doubt, leave it false.
//   - unique: Whether the indices are unique.
//     If set to true for some backends it is faster, but if the indices are not unique, results may be unstable.
//     If in doubt, leave it false.
func Scatter(indices, updates *Node, shape shapes.Shape, sorted, unique bool) *Node {
	g := validateBuildingGraphFromInputs(indices, updates)
	zeros := Zeros(g, shape)
	return ScatterSum(zeros, indices, updates, sorted, true)
}

// ScatterUpdate replaces values in the operand with values from updates, at the locations pointed by indices.
//
// Only implemented for `unique=true` for now: it doesn't handle the case with overlapping updates.
//
// Args:
//   - operand: input values that are going to be modified.
//   - indices: the positions where to set the new values.
//   - updates: the values that are going to be set.
//   - sorted: the indices must be in order.
//     If set to true for some backends it is faster, but if the indices are not sorted, results may be unstable.
//     If in doubt, leave it false.
//   - unique: Only **true** is implemented for now, if set to **false** it will panic.
//     Whether the indices are unique.
//     If set to true for some backends it is faster, but if the indices are not unique, results may be unstable.
//     If in doubt, leave it false.
func ScatterUpdate(operand, indices, updates *Node, sorted, unique bool) *Node {
	if !unique {
		Panicf("ScatterUpdate only implemented for unique indices -- ScatterSum/Max/Min support non-unique indices though")
	}
	g := operand.Graph()
	dtype := operand.DType()
	shape := operand.Shape()

	zero := ScalarZero(g, dtype)
	maskUpdates := OnesLike(updates)
	updateMask := Scatter(indices, maskUpdates, shape, sorted, unique)
	updateMaskBool := ConvertDType(updateMask, dtypes.Bool)
	operand = Where(updateMaskBool, zero, operand)
	return ScatterSum(operand, indices, updates, sorted, unique)
}

// ScatterSum adds up the slices in updates into the given operand tensor, at the locations pointed by indices.
// It does the opposite of Gather.
//
// Args:
//   - operand: input values to which new values will be added.
//   - indices: the positions where add the new values.
//   - updates: the values to add.
//   - sorted: the indices must be in order.
//     If set to true for some backends it is faster, but if the indices are not sorted, results may be unstable.
//     If in doubt, leave it false.
//   - unique: Whether the indices are unique.
//     If set to true for some backends it is faster, but if the indices are not unique, results may be unstable.
//     If in doubt, leave it false.
func ScatterSum(operand, indices, updates *Node, sorted, unique bool) *Node {
	_ = validateBuildingGraphFromInputs(operand, indices, updates)
	return genericScatter(operand, indices, updates, sorted, unique, backendScatterSum)
}

// ScatterAdd is a deprecated alias to ScatterSum.
//
// Deprecated: Please use ScatterSum instead.
func ScatterAdd(operand, indices, updates *Node, sorted, unique bool) *Node {
	return ScatterSum(operand, indices, updates, sorted, unique)
}

// ScatterMax updates the max value of operand, from the values in updates pointed by indices.
//
// The operand provides the initial values for the operation, and typically will be initialized
// with -inf. See Infinity and BroadcastToDims to create an arbitrarily shaped node filled with
// infinity.
//
// Args:
// - [sorted]: the indices must be in order. In some cases it is faster, but if indices are not in order results may be unstable.
// - [unique]: the indices must be unique. In some cases it is faster, but if indices are not unique results may be unstable.
func ScatterMax(operand, indices, updates *Node, sorted, unique bool) *Node {
	_ = validateBuildingGraphFromInputs(operand, indices, updates)
	return genericScatter(operand, indices, updates, sorted, unique, backendScatterMax)
}

// ScatterMin updates the min value of operand, from the values in updates pointed by indices.
//
// The operand provides the initial values for the operation, and typically will be initialized
// with +inf. See Infinity and BroadcastToDims to create an arbitrarily shaped node filled with
// infinity.
//
// Args:
// - [sorted]: the indices must be in order. In some cases it is faster, but if indices are not in order results may be unstable.
// - [unique]: the indices must be unique. In some cases it is faster, but if indices are not unique results may be unstable.
func ScatterMin(operand, indices, updates *Node, sorted, unique bool) *Node {
	_ = validateBuildingGraphFromInputs(operand, indices, updates)
	return genericScatter(operand, indices, updates, sorted, unique, backendScatterMin)
}

type scatterFn func(operand *Node, scatterIndices *Node, updates *Node, indexVectorAxis int, updateWindowAxes []int, insertedWindowAxes []int, scatterAxesToOperandAxes []int, indicesAreSorted bool, uniqueIndices bool) (node *Node)

func genericScatter(operand, indices, updates *Node, sorted, unique bool, fn scatterFn) *Node {
	if !indices.DType().IsInt() {
		Panicf("scatter operations require integer indices, instead got shapes %s", indices.Shape())
	}
	if operand.DType() != updates.DType() {
		Panicf(
			"scatter operations require operand and updates to have the same DType, instead got shapes %s (operand) and %s (updates)",
			operand.Shape(), updates.Shape())
	}
	if indices.Shape().IsTuple() || operand.Shape().IsTuple() || updates.Shape().IsTuple() {
		Panicf("tuples are not supported in ScatterSum, operand.Shape()=%s, indices.Shape()=%s, updates.Shape()=%s",
			operand.Shape(), indices.Shape(), updates.Shape())
	}
	if indices.Shape().IsScalar() {
		indices = InsertAxes(indices, 0)
	}

	// Check shapes compatibility.
	indicesRank := indices.Rank()
	indexedRank := indices.Shape().Dimensions[indicesRank-1]
	updatesRank := updates.Rank()
	if updatesRank < indicesRank-1 || !xslices.DeepSliceCmp(updates.Shape().Dimensions[:indicesRank-1], indices.Shape().Dimensions[:indicesRank-1], xslices.EqualAny[int]) {
		Panicf("updates rank prefix (shapes=%s) must match the first n-1 dimensions of the indices (shapes=%s)",
			updates.Shape(), indices.Shape())
	}
	slicesRank := updatesRank - (indicesRank - 1)
	slicesDims := updates.Shape().Dimensions[indicesRank-1:]
	operandRank := operand.Shape().Rank()
	if operandRank != indexedRank+slicesRank || !xslices.DeepSliceCmp(operand.Shape().Dimensions[indexedRank:], slicesDims, xslices.EqualAny[int]) {
		Panicf("operand shapes (%s) has to be a combination of the indexed rank (%d, the last dimension of indices shapes %s) and "+
			"the slices coming from updates (the last %d dimensions %v of the updates, shaped %s)",
			operand.Shape(), indexedRank, indices.Shape(), slicesRank, slicesDims, updates.Shape())
	}

	// Set scatterXLA parameters:
	updateWindowsAxes := make([]int, 0, slicesRank)
	for ii := updatesRank - slicesRank; ii < updatesRank; ii++ {
		updateWindowsAxes = append(updateWindowsAxes, ii)
	}
	insertedWindowsAxes := make([]int, 0, indexedRank)
	for ii := 0; ii < indexedRank; ii++ {
		insertedWindowsAxes = append(insertedWindowsAxes, ii)
	}
	scatterAxesToOperandAxes := make([]int, 0, 10)
	for ii := 0; ii < indexedRank; ii++ {
		scatterAxesToOperandAxes = append(scatterAxesToOperandAxes, ii)
	}
	return fn(operand, indices, updates, indicesRank-1, updateWindowsAxes, insertedWindowsAxes, scatterAxesToOperandAxes,
		sorted, unique)
}

// scatterSumVJP generates the adjoint gradient term for a ScatterSum node.
// Note: this may not work for the more general scatter form.
func scatterSumVJP(node, v *Node, _ shapes.Shape) []*Node {
	params := node.inputs.(*nodeInputsScatterSum)
	operand, scatterIndices, updates := params.operand, params.scatterIndices, params.updates
	_ = updates
	_ = scatterIndices
	_ = operand
	operandVJP := v // Since it's a sum of the initial values, the VJP is the identity of the gradient coming in.
	updatesVJP := Gather(v, scatterIndices, params.indicesAreSorted)
	return []*Node{ /*operand*/ operandVJP /*indices*/, nil /*initialValue*/, updatesVJP}
}

// scatterMaxOrMinVJP generates the adjoint gradient term for a ScatterMax or ScatterMin node.
// Note: this may not work for the more general scatter form.
func scatterMaxOrMinVJP(node, v *Node, _ shapes.Shape) []*Node {
	operand, scatterIndices, updates := node.inputNodes[0], node.inputNodes[1], node.inputNodes[2]
	var indicesAreSorted bool
	if node.Type() == NodeTypeScatterMax {
		indicesAreSorted = node.inputs.(*nodeInputsScatterMax).indicesAreSorted
	} else if node.Type() == NodeTypeScatterMin {
		indicesAreSorted = node.inputs.(*nodeInputsScatterMin).indicesAreSorted
	}

	// For the operand, we propagate v only if the operand value was chosen as max value.
	operandMask := Equal(node, operand)
	operandVJP := Where(operandMask, v, ZerosLike(v))

	// For the updates, we pick them only if they were chosen as max value.
	maxForUpdates := Gather(node, scatterIndices, indicesAreSorted)
	updatesMask := Equal(maxForUpdates, updates)
	updatesVJP := Gather(v, scatterIndices)
	updatesVJP = Where(updatesMask, updatesVJP, ZerosLike(updatesVJP))
	return []*Node{ /*operand*/ operandVJP /*indices*/, nil /*initialValue*/, updatesVJP}
}

// BackendScatterMax exposes the raw backend ScatterMax operator.
//
// This should be internal, and it is exposed only for testing and debugging purposes, please don't rely on it.
// If it turns out you need some functionality here that is not provided in ScatterMax,
// open an issue in GoMLX and we'll figure a betterAPI.
//
// Description in
// https://openxla.org/xla/operation_semantics#scatter
func BackendScatterMax(operand, indices, updates *Node, indexVectorAxis int, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int, indicesAreSorted, uniqueIndices bool) *Node {
	return backendScatterMax(operand, indices, updates, indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes,
		indicesAreSorted, uniqueIndices)
}

// BackendScatterMin exposes the raw backend ScatterMin operator.
//
// This should be internal, and it is exposed only for testing and debugging purposes, please don't rely on it.
// If it turns out you need some functionality here that is not provided in ScatterMin,
// open an issue in GoMLX and we'll figure a betterAPI.
//
// Description in
// https://openxla.org/xla/operation_semantics#scatter
func BackendScatterMin(operand, indices, updates *Node, indexVectorAxis int, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int, indicesAreSorted, uniqueIndices bool) *Node {
	return backendScatterMin(operand, indices, updates, indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes,
		indicesAreSorted, uniqueIndices)
}

// BackendScatterSum exposes the raw backend ScatterSum operator.
//
// This should be internal, and it is exposed only for testing and debugging purposes, please don't rely on it.
// If it turns out you need some functionality here that is not provided in ScatterSum,
// open an issue in GoMLX and we'll figure a betterAPI.
//
// Description in
// https://openxla.org/xla/operation_semantics#scatter
func BackendScatterSum(operand, indices, updates *Node, indexVectorAxis int, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int, indicesAreSorted, uniqueIndices bool) *Node {
	return backendScatterSum(operand, indices, updates, indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes,
		indicesAreSorted, uniqueIndices)
}
