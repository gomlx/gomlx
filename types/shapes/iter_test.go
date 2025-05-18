package shapes

import (
	"slices"
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
)

func TestShape_Iter(t *testing.T) {
	// Version 1: there is only one value to iterate:
	shape := Make(dtypes.F32, 1, 1, 1, 1)
	collect := make([][]int, 0, shape.Size())
	for indices := range shape.Iter() {
		collect = append(collect, slices.Clone(indices))
	}
	require.Equal(t, [][]int{{0, 0, 0, 0}}, collect)

	// Version 2: all axes are "spatial" (dim > 1)
	shape = Make(dtypes.F64, 3, 2)
	collect = make([][]int, 0, shape.Size())
	for indices := range shape.Iter() {
		collect = append(collect, slices.Clone(indices))
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

	// Version 3: with only 2 spatial axis.
	shape = Make(dtypes.BF16, 3, 1, 2, 1)
	collect = make([][]int, 0, shape.Size())
	for indices := range shape.Iter() {
		collect = append(collect, slices.Clone(indices))
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
