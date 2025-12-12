package shapes

import (
	"slices"
	"testing"

	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/stretchr/testify/require"
)

func TestShape_Strides(t *testing.T) {
	// Test case 1: shape with dimensions [2, 3, 4]
	shape := Make(dtypes.F32, 2, 3, 4)
	strides := shape.Strides()
	require.Equal(t, []int{12, 4, 1}, strides)

	// Test case 2: shape with single dimension
	shape = Make(dtypes.F32, 5)
	strides = shape.Strides()
	require.Equal(t, []int{1}, strides)

	// Test case 3: shape with dimensions [3, 1, 2]
	shape = Make(dtypes.F32, 3, 1, 2)
	strides = shape.Strides()
	require.Equal(t, []int{2, 2, 1}, strides)
}

func TestShape_Iter(t *testing.T) {
	// Version 1: there is only one value to iterate:
	shape := Make(dtypes.F32, 1, 1, 1, 1)
	collect := make([][]int, 0, shape.Size())
	for flatIdx, indices := range shape.Iter() {
		collect = append(collect, slices.Clone(indices))
		require.Equal(t, 0, flatIdx) // There should only be one flatIdx, equal to 0.
	}
	require.Equal(t, [][]int{{0, 0, 0, 0}}, collect)

	// Version 2: all axes are "spatial" (dim > 1)
	shape = Make(dtypes.F64, 3, 2)
	collect = make([][]int, 0, shape.Size())
	var counter int
	for flatIdx, indices := range shape.Iter() {
		collect = append(collect, slices.Clone(indices))
		require.Equal(t, counter, flatIdx)
		counter++
	}
	want := [][]int{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
		{2, 0},
		{2, 1},
	}
	require.Equal(t, want, collect)

	// Version 3: with only 2 spatial axes.
	shape = Make(dtypes.BF16, 3, 1, 2, 1)
	collect = make([][]int, 0, shape.Size())
	counter = 0
	for flatIdx, indices := range shape.Iter() {
		collect = append(collect, slices.Clone(indices))
		require.Equal(t, counter, flatIdx)
		counter++
	}
	want = [][]int{
		{0, 0, 0, 0},
		{0, 0, 1, 0},
		{1, 0, 0, 0},
		{1, 0, 1, 0},
		{2, 0, 0, 0},
		{2, 0, 1, 0},
	}
	require.Equal(t, want, collect)
}

func TestShape_IterOnAxes(t *testing.T) {
	// Shape with dimensions [2, 3, 4]
	shape := Make(dtypes.F32, 2, 3, 4)

	// Test iteration on the first axis.
	var collect [][]int
	var flatIndices []int
	indices := make([]int, 3)
	indices[1] = 1               // Index 1 should be fixed to 1.
	axesToIterate := []int{0, 2} // We are only iterating on the axis 0 an 2.
	for flatIdx, indicesResult := range shape.IterOnAxes(axesToIterate, nil, indices) {
		collect = append(collect, slices.Clone(indicesResult))
		flatIndices = append(flatIndices, flatIdx)
	}
	require.Equal(t, [][]int{
		{0, 1, 0},
		{0, 1, 1},
		{0, 1, 2},
		{0, 1, 3},
		{1, 1, 0},
		{1, 1, 1},
		{1, 1, 2},
		{1, 1, 3},
	}, collect)
	require.Equal(t, []int{4, 5, 6, 7, 16, 17, 18, 19}, flatIndices)
}
