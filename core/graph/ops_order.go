// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/internal/graphimpl"
	"github.com/gomlx/gomlx/support/exceptions"
)

// Sort sorts a single tensor along the specified axis.
//
// This is a simplified version of SortFunc that doesn't require a comparator closure.
// For sorting multiple tensors together or using a custom comparator, use SortFunc.
//
// Parameters:
//   - input: The tensor to sort.
//   - axis: The axis along which to sort (can be negative to count from the end).
//   - ascending: If true, sorts in ascending order; otherwise descending.
//
// Returns the sorted tensor.
//
// Example:
//
//	x := Const(g, []float32{5, 2, 8, 1, 9, 3})
//	sorted := Sort(x, 0, true)  // Returns [1, 2, 3, 5, 8, 9]
func Sort(input *Node, axis int, ascending bool) *Node {
	g := validateBuildingGraphFromInputs(input)
	dtype := input.Shape().DType

	// Create a simple comparator closure.
	comparator := NewClosure(g, func(g *Graph) []*Node {
		lhs := Parameter(g, "lhs", shapes.Make(dtype))
		rhs := Parameter(g, "rhs", shapes.Make(dtype))
		if ascending {
			return []*Node{LessThan(lhs, rhs)}
		}
		return []*Node{GreaterThan(lhs, rhs)}
	})

	results := SortFunc(comparator, axis, false, input)
	return results[0]
}

// SortFunc sorts one or more tensors along the specified axis using a comparator closure.
//
// The comparator closure should be created with NewClosure and must:
//   - Take 2*N scalar parameters (where N is the number of input tensors), in the order
//     (lhs_0, rhs_0, lhs_1, rhs_1, ..., lhs_N-1, rhs_N-1) where lhs_i and rhs_i are
//     scalar values from input tensor i being compared.
//   - Return a single boolean scalar: true if lhs should come before rhs.
//
// For a standard ascending sort on a single tensor with dtype Float32:
//
//	comparator := NewClosure(g, func(g *graph.Graph) []*Node {
//	    lhs := Parameter(g, "lhs", shapes.Make(dtypes.Float32))
//	    rhs := Parameter(g, "rhs", shapes.Make(dtypes.Float32))
//	    return []*Node{LessThan(lhs, rhs)}
//	})
//	sorted := SortFunc(comparator, 0, false, x)
//
// For sorting two tensors together (e.g., values and indices):
//
//	comparator := NewClosure(g, func(g *graph.Graph) []*Node {
//	    lhsVal := Parameter(g, "lhs_val", shapes.Make(dtypes.Float32))
//	    rhsVal := Parameter(g, "rhs_val", shapes.Make(dtypes.Float32))
//	    lhsIdx := Parameter(g, "lhs_idx", shapes.Make(dtypes.Int32))
//	    rhsIdx := Parameter(g, "rhs_idx", shapes.Make(dtypes.Int32))
//	    _ = lhsIdx // unused but required
//	    _ = rhsIdx // unused but required
//	    return []*Node{LessThan(lhsVal, rhsVal)} // Compare by values only
//	})
//	sortedResults := SortFunc(comparator, 0, false, values, indices)
//
// Parameters:
//   - comparator: A closure that compares two sets of scalar values.
//   - axis: The axis along which to sort (can be negative to count from the end).
//   - isStable: If true, maintains the relative order of equal elements.
//   - inputs: One or more tensors to sort. All must have the same dimensions (dtypes can differ).
//
// Returns the sorted tensors in the same order as inputs.
func SortFunc(comparator *Function, axis int, isStable bool, inputs ...*Node) []*Node {
	return wrapNodes(graphimpl.SortFunc(unwrapFunction(comparator), axis, isStable, unwrapNodes(inputs)...))
}

// TopK returns the top K largest elements and their indices along the specified axis.
//
// Parameters:
//   - x: The input tensor.
//   - k: The number of top elements to retrieve.
//   - axis: The axis along which to find top K (can be negative to count from the end).
//
// Returns:
//   - values: The top K largest values, sorted in descending order.
//   - indices: The indices of the top K values in the original tensor (dtype Int32).
//
// The output shapes have the same rank as input, with the specified axis having size K.
//
// Tie-breaking: When multiple elements have equal values, stable sort is used,
// meaning elements that appear earlier in the original order along the axis are selected first.
//
// Example:
//
//	x := Const(g, []float32{3, 1, 4, 1, 5, 9, 2, 6})
//	values, indices := TopK(x, 3, 0)
//	// values = [9, 6, 5]
//	// indices = [5, 7, 4]
func TopK(x *Node, k int, axis int) (values, indices *Node) {
	return topKImpl(x, k, axis, false)
}

// BottomK returns the K smallest elements and their indices along the specified axis.
//
// Parameters:
//   - x: The input tensor.
//   - k: The number of bottom elements to retrieve.
//   - axis: The axis along which to find bottom K (can be negative to count from the end).
//
// Returns:
//   - values: The K smallest values, sorted in ascending order.
//   - indices: The indices of the K smallest values in the original tensor (dtype Int32).
//
// The output shapes have the same rank as input, with the specified axis having size K.
//
// Tie-breaking: When multiple elements have equal values, stable sort is used,
// meaning elements that appear earlier in the original order along the axis are selected first.
//
// Example:
//
//	x := Const(g, []float32{3, 1, 4, 1, 5, 9, 2, 6})
//	values, indices := BottomK(x, 3, 0)
//	// values = [1, 1, 2]
//	// indices = [1, 3, 6]
func BottomK(x *Node, k int, axis int) (values, indices *Node) {
	return topKImpl(x, k, axis, true)
}

// topKImpl is the shared implementation for TopK and BottomK.
func topKImpl(x *Node, k int, axis int, ascending bool) (values, indices *Node) {
	g := x.Graph()
	g.impl.AssertBuilding()

	shape := x.Shape()
	rank := x.Rank()
	dtype := shape.DType
	axis = MustAdjustAxis(axis, x)

	// Validate k
	axisSize := shape.Dimensions[axis]
	if k <= 0 {
		exceptions.Panicf("TopK/BottomK: k must be positive, got %d", k)
	}
	if axisSize != shapes.DynamicDim && k > axisSize {
		exceptions.Panicf("TopK/BottomK: k=%d exceeds axis size %d", k, axisSize)
	}

	outputs, _ := graphimpl.FusedOpCallerMulti(
		func() []*graphimpl.NodeImpl {
			v, idx := graphimpl.BackendFusedTopK(unwrapNode(x), k, compute.FusedTopKConfig{
				Axis:     axis,
				Smallest: ascending,
			})
			return []*graphimpl.NodeImpl{v, idx}
		},
		func() []*graphimpl.NodeImpl {
			// Create indices tensor using IotaLike along the sort axis.
			// Indices will have Int32 dtype.
			indicesInput := ConvertDType(IotaLike(x, axis), dtypes.Int32)

			// Create comparator that compares based on values.
			// StableHLO sort expects parameters in order: (lhs_0, rhs_0, lhs_1, rhs_1, ...)
			// where input 0 is values and input 1 is indices.
			comparator := NewClosure(g, func(g *Graph) []*Node {
				lhsVal := Parameter(g, "lhs_val", shapes.Make(dtype))
				rhsVal := Parameter(g, "rhs_val", shapes.Make(dtype))
				_ = Parameter(g, "lhs_idx", shapes.Make(dtypes.Int32))
				_ = Parameter(g, "rhs_idx", shapes.Make(dtypes.Int32))
				if ascending {
					return []*Node{LessThan(lhsVal, rhsVal)}
				}
				return []*Node{GreaterThan(lhsVal, rhsVal)}
			})

			// Sort values and indices together.
			sortedResults := SortFunc(comparator, axis, true, x, indicesInput)
			sortedValues := sortedResults[0]
			sortedIndices := sortedResults[1]

			// Register VJP on the underlying Sort node so reverse-mode autodiff works
			// identically on backends that decompose TopK/BottomK via SortFunc (e.g., XLA).
			sortNode := sortedValues.impl.Inputs()[0]
			sortNode.SetCustomVJP(func(node *graphimpl.NodeImpl, vjps []*graphimpl.NodeImpl, _ shapes.Shape) []*graphimpl.NodeImpl {
				splitOutputs := wrapNodes(graphimpl.SplitNode(node))
				return []*graphimpl.NodeImpl{
					unwrapNode(scatterAlongAxis(wrapNode(node.Inputs()[0]), splitOutputs[1], wrapNode(vjps[0]), axis)),
					nil,
				}
			})

			// Slice the first K elements along the sort axis.
			sliceSpecs := make([]SliceAxisSpec, rank)
			for i := range rank {
				if i == axis {
					sliceSpecs[i] = AxisRange(0, k)
				} else {
					sliceSpecs[i] = AxisRange() // Full range
				}
			}

			return unwrapNodes([]*Node{
				Slice(sortedValues, sliceSpecs...),
				Slice(sortedIndices, sliceSpecs...),
			})
		},
		false, // useDecomposedAlternateVJP = false: FusedTopK has its own VJP, and decomposed SortFunc requires closures.
	)
	wrapped := wrapNodes(outputs)
	return wrapped[0], wrapped[1]
}

// scatterAlongAxis scatters updates into ZerosLike(operand) at positions given by indices along axis.
// Used by decomposed TopK/BottomK VJP, where indices along axis are unique.
func scatterAlongAxis(operand, indices, updates *Node, axis int) *Node {
	rank := operand.Rank()
	axis = MustAdjustAxis(axis, operand)

	coordParts := make([]*Node, rank)
	for i := range rank {
		if i == axis {
			coordParts[i] = indices
		} else {
			coordParts[i] = ConvertDType(IotaLike(indices, i), indices.DType())
		}
	}
	coords := Stack(coordParts, -1)
	return ScatterSum(ZerosLike(operand), coords, updates, false, true)
}

// TopKMask returns a boolean mask of the top-k elements along the specified axis.
// This is more efficient than TopK when you only need to know which elements are in the top-k.
//
// Parameters:
//   - x: Input tensor
//   - k: Number of top elements to mark
//   - axis: The axis along which to find top K (can be negative to count from the end)
//
// Returns:
//   - mask: Boolean mask with same shape as x, true for top-k elements
//
// Tie-breaking: When multiple elements have equal values, stable sort is used,
// meaning elements that appear earlier in the original order along the axis are selected first.
// This ensures exactly k elements are marked as true in the mask.
//
// Example:
//
//	x := Const(g, []float32{3, 1, 4, 1, 5, 9, 2, 6})
//	mask := TopKMask(x, 3, 0)
//	// mask = [false, false, false, false, true, true, false, true]
func TopKMask(x *Node, k int, axis int) *Node {
	g := x.Graph()
	g.impl.AssertBuilding()

	shape := x.Shape()
	axis = MustAdjustAxis(axis, x)
	axisSize := shape.Dimensions[axis]
	if k <= 0 {
		exceptions.Panicf("TopKMask: k must be positive, got %d", k)
	}
	if axisSize != shapes.DynamicDim && k > axisSize {
		exceptions.Panicf("TopKMask: k=%d exceeds axis size %d", k, axisSize)
	}

	// If k equals the axis size, all elements are in the top-k
	if axisSize != shapes.DynamicDim && k == axisSize {
		return OnesLike(ConvertDType(x, dtypes.Bool))
	}

	// Get the indices of the top-k elements
	// Using stable sort ensures deterministic tie-breaking
	_, topIndices := TopK(x, k, axis)

	// Create position indices for comparison
	positionIndices := ConvertDType(IotaLike(x, axis), dtypes.Int32)

	// Create a mask by checking if each position index is in topIndices
	// We'll expand both tensors to enable broadcasting comparison
	expandedPos := ExpandAxes(positionIndices, axis+1)
	expandedTop := ExpandAxes(topIndices, axis)
	matches := Equal(expandedPos, expandedTop)
	mask := ReduceLogicalOr(matches, axis+1)

	return mask
}
