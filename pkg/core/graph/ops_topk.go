package graph

import (
	. "github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gopjrt/dtypes"
)

// TopK returns the k largest elements and their indices along the last axis.
// This is an iterative implementation that finds the top-k elements one by one.
//
// Parameters:
//   - x: Input tensor of shape [..., n]
//   - k: Number of top elements to return
//
// Returns:
//   - values: Tensor of shape [..., k] with the top-k values sorted in descending order
//   - indices: Tensor of shape [..., k] with the indices of the top-k values in the original tensor
func TopK(x *Node, k int) (values, indices *Node) {
	g := validateBuildingGraphFromInputs(x)
	dtype := x.DType()
	lastDim := x.Shape().Dimensions[x.Rank()-1]

	if k <= 0 {
		Panicf("TopK: k must be positive, got %d", k)
	}
	if k > lastDim {
		Panicf("TopK: k=%d is larger than the last dimension size %d", k, lastDim)
	}

	working := x
	minValue := Infinity(g, dtype, -1)

	var topValues []*Node
	var topIndices []*Node
	for i := 0; i < k; i++ {
		idx := ArgMax(working, -1, dtypes.Int32)
		topIndices = append(topIndices, idx)
		val := ReduceMax(working, -1)
		topValues = append(topValues, val)

		oneHot := OneHot(idx, lastDim, dtype)
		mask := Mul(oneHot, minValue)
		notSelected := Sub(ScalarOne(g, dtype), oneHot)
		working = Add(Mul(working, notSelected), mask)
	}

	values = Stack(topValues, -1)
	indices = Stack(topIndices, -1)

	return values, indices
}

// TopKMask returns a boolean mask of the top-k elements along the last axis.
// This is more efficient than TopK when you only need to know which elements are in the top-k.
//
// Parameters:
//   - x: Input tensor of shape [..., n]
//   - k: Number of top elements to mark
//
// Returns:
//   - mask: Boolean mask, shape [..., n], true for top-k elements
func TopKMask(x *Node, k int) *Node {
	g := validateBuildingGraphFromInputs(x)
	dtype := x.DType()
	lastDim := x.Shape().Dimensions[x.Rank()-1]

	if k <= 0 {
		Panicf("TopKMask: k must be positive, got %d", k)
	}
	if k > lastDim {
		Panicf("TopKMask: k=%d is larger than the last dimension size %d", k, lastDim)
	}
	if k == lastDim {
		boolShape := x.Shape().Clone()
		boolShape.DType = dtypes.Bool
		return Ones(g, boolShape)
	}

	working := x
	minValue := Infinity(g, dtype, -1)
	for i := 0; i < k-1; i++ {
		idx := ArgMax(working, -1, dtypes.Int32)
		oneHot := OneHot(idx, lastDim, dtype)
		mask := Mul(oneHot, minValue)
		notSelected := Sub(ScalarOne(g, dtype), oneHot)
		working = Add(Mul(working, notSelected), mask)
	}

	threshold := ReduceMax(working, -1)
	threshold = ExpandDims(threshold, -1)

	return GreaterOrEqual(x, threshold)
}
