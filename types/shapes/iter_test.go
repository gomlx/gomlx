package shapes

import (
	"slices"
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
)

func TestShape_Iter(t *testing.T) {
	shape := Make(dtypes.F32, 3, 1, 2, 1)
	collect := make([][]int, 0, shape.Size())
	for indices := range shape.Iter() {
		collect = append(collect, slices.Clone(indices))
	}
	want := [][]int{
		{0, 0, 0, 0},
		{0, 0, 1, 0},
		{1, 0, 0, 0},
		{1, 0, 1, 0},
		{2, 0, 0, 0},
		{2, 0, 1, 0},
	}
	require.Equal(t, want, collect)
}
