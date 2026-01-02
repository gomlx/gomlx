package graph

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// Sort sorts the input tensor along the specified axis in ascending order.
//
// Parameters:
//   - x: The tensor to sort
//   - axis: The axis along which to sort (negative values count from the end)
//
// Returns the sorted tensor with the same shape as the input.
//
// Example:
//
//	x := Const(g, []float32{3, 1, 4, 1, 5})
//	sorted := Sort(x, 0)  // Returns [1, 1, 3, 4, 5]
func Sort(x *Node, axis int) *Node {
	return sortWithOrder(x, axis, false, true)
}

// SortDescending sorts the input tensor along the specified axis in descending order.
//
// Parameters:
//   - x: The tensor to sort
//   - axis: The axis along which to sort (negative values count from the end)
//
// Returns the sorted tensor with the same shape as the input.
//
// Example:
//
//	x := Const(g, []float32{3, 1, 4, 1, 5})
//	sorted := SortDescending(x, 0)  // Returns [5, 4, 3, 1, 1]
func SortDescending(x *Node, axis int) *Node {
	return sortWithOrder(x, axis, true, true)
}

// SortWithIndices sorts the input tensor and returns both the sorted values and the indices.
//
// This is useful for implementing operations like argsort or top-k.
//
// Parameters:
//   - x: The tensor to sort
//   - axis: The axis along which to sort (negative values count from the end)
//   - descending: If true, sort in descending order; otherwise ascending
//
// Returns:
//   - sortedValues: The sorted tensor with the same shape as input
//   - indices: The indices that would sort the original tensor (same shape, dtype Int32)
//
// Example:
//
//	x := Const(g, []float32{3, 1, 4, 1, 5})
//	values, indices := SortWithIndices(x, 0, false)
//	// values = [1, 1, 3, 4, 5]
//	// indices = [1, 3, 0, 2, 4]
func SortWithIndices(x *Node, axis int, descending bool) (sortedValues, indices *Node) {
	g := x.Graph()
	g.AssertBuilding()

	// Normalize axis
	rank := x.Shape().Rank()
	if axis < 0 {
		axis = rank + axis
	}
	if axis < 0 || axis >= rank {
		exceptions.Panicf("SortWithIndices: axis %d out of range for tensor of rank %d", axis, rank)
	}

	// Get the sort closure creator
	closureCreator, ok := g.builder.(backends.SortClosureCreator)
	if !ok {
		exceptions.Panicf("SortWithIndices: backend %T does not support sort closure creation", g.builder)
	}

	// Create indices tensor with Iota (same shape as x but with Int32 dtype)
	indicesShape := shapes.Make(dtypes.Int32, x.Shape().Dimensions...)
	indicesNode := Iota(g, indicesShape, axis)

	// Create comparator for sorting values with indices (4 arguments)
	var comparator any
	var err error
	if descending {
		comparator, err = closureCreator.SortWithIndicesComparatorDescending(x.DType(), dtypes.Int32)
	} else {
		comparator, err = closureCreator.SortWithIndicesComparatorAscending(x.DType(), dtypes.Int32)
	}
	if err != nil {
		panic(err)
	}

	// Sort both values and indices together
	results := backendSort(comparator, axis, true, x, indicesNode)
	if len(results) != 2 {
		exceptions.Panicf("SortWithIndices: expected 2 outputs, got %d", len(results))
	}

	return results[0], results[1]
}

// sortWithOrder sorts a tensor with the specified order.
func sortWithOrder(x *Node, axis int, descending, isStable bool) *Node {
	g := x.Graph()
	g.AssertBuilding()

	// Normalize axis
	rank := x.Shape().Rank()
	if axis < 0 {
		axis = rank + axis
	}
	if axis < 0 || axis >= rank {
		exceptions.Panicf("Sort: axis %d out of range for tensor of rank %d", axis, rank)
	}

	// Get the sort closure creator
	closureCreator, ok := g.builder.(backends.SortClosureCreator)
	if !ok {
		exceptions.Panicf("Sort: backend %T does not support sort closure creation", g.builder)
	}

	// Create comparator
	var comparator any
	var err error
	if descending {
		comparator, err = closureCreator.SortComparatorDescending(x.DType())
	} else {
		comparator, err = closureCreator.SortComparatorAscending(x.DType())
	}
	if err != nil {
		panic(err)
	}

	// Call backend sort
	results := backendSort(comparator, axis, isStable, x)
	if len(results) != 1 {
		exceptions.Panicf("Sort: expected 1 output, got %d", len(results))
	}

	return results[0]
}

// ArgSort returns the indices that would sort the input tensor.
//
// This is equivalent to calling SortWithIndices and discarding the sorted values.
//
// Parameters:
//   - x: The tensor to sort
//   - axis: The axis along which to sort (negative values count from the end)
//   - descending: If true, sort in descending order; otherwise ascending
//
// Returns the indices (Int32) that would sort the original tensor.
//
// Example:
//
//	x := Const(g, []float32{3, 1, 4, 1, 5})
//	indices := ArgSort(x, 0, false)  // Returns [1, 3, 0, 2, 4]
func ArgSort(x *Node, axis int, descending bool) *Node {
	_, indices := SortWithIndices(x, axis, descending)
	return indices
}

// TopK returns the k largest elements and their indices along the specified axis.
//
// Parameters:
//   - x: The input tensor
//   - k: The number of top elements to return
//   - axis: The axis along which to find top-k (negative values count from the end)
//
// Returns:
//   - values: The top-k values (shape with axis dimension = k)
//   - indices: The indices of the top-k values (same shape as values, dtype Int32)
//
// Example:
//
//	x := Const(g, []float32{3, 1, 4, 1, 5, 9, 2, 6})
//	values, indices := TopK(x, 3, 0)
//	// values = [9, 6, 5]
//	// indices = [5, 7, 4]
func TopK(x *Node, k int, axis int) (values, indices *Node) {
	// Sort in descending order with indices
	sortedValues, sortedIndices := SortWithIndices(x, axis, true)

	// Normalize axis
	rank := x.Shape().Rank()
	if axis < 0 {
		axis = rank + axis
	}

	// Build slice specs: full range for all axes except the sort axis which is [0:k]
	sliceSpecs := make([]SliceAxisSpec, rank)
	for i := range rank {
		if i == axis {
			sliceSpecs[i] = AxisRange(0, k)
		} else {
			sliceSpecs[i] = AxisRange() // full range
		}
	}

	values = Slice(sortedValues, sliceSpecs...)
	indices = Slice(sortedIndices, sliceSpecs...)

	return values, indices
}

// BottomK returns the k smallest elements and their indices along the specified axis.
//
// Parameters:
//   - x: The input tensor
//   - k: The number of bottom elements to return
//   - axis: The axis along which to find bottom-k (negative values count from the end)
//
// Returns:
//   - values: The bottom-k values (shape with axis dimension = k)
//   - indices: The indices of the bottom-k values (same shape as values, dtype Int32)
func BottomK(x *Node, k int, axis int) (values, indices *Node) {
	// Sort in ascending order with indices
	sortedValues, sortedIndices := SortWithIndices(x, axis, false)

	// Normalize axis
	rank := x.Shape().Rank()
	if axis < 0 {
		axis = rank + axis
	}

	// Build slice specs: full range for all axes except the sort axis which is [0:k]
	sliceSpecs := make([]SliceAxisSpec, rank)
	for i := range rank {
		if i == axis {
			sliceSpecs[i] = AxisRange(0, k)
		} else {
			sliceSpecs[i] = AxisRange() // full range
		}
	}

	values = Slice(sortedValues, sliceSpecs...)
	indices = Slice(sortedIndices, sliceSpecs...)

	return values, indices
}
