package ogbnmag

import (
	"fmt"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestSampler(t *testing.T) {
	s := NewSampler()
	s.AddNodeType("paper", 5)
	s.AddNodeType("author", 10)

	authorWritesPapers := tensor.FromValue([][]int32{
		{0, 2}, // Author 0 writes paper 2.
		{3, 2},
		{4, 2},
		{0, 3},
		{0, 4},
		{4, 4},
		{7, 4},
	})
	require.NoError(t, authorWritesPapers.Shape().Check(shapes.Int32, 7, 2))

	s.AddEdgeType("writes", "author", "paper", authorWritesPapers, false)
	fmt.Printf("writes:\n\tStarts: \t%#v\n\tTargets:\t%#v\n",
		s.EdgeTypes["writes"].Starts,
		s.EdgeTypes["writes"].EdgeTargets)
	assert.EqualValues(t, []int32{3, 3, 3, 4, 6, 6, 6, 7, 7, 7}, s.EdgeTypes["writes"].Starts)
	assert.EqualValues(t, []int32{2, 3, 4, 2, 2, 4, 4}, s.EdgeTypes["writes"].EdgeTargets)
	assert.EqualValues(t, []int32{2, 4}, s.EdgeTypes["writes"].EdgeTargetsForSourceIdx(4))
	assert.EqualValues(t, []int32{}, s.EdgeTypes["writes"].EdgeTargetsForSourceIdx(9))

	s.AddEdgeType("written_by", "author", "paper", authorWritesPapers, true)
	fmt.Printf("written_by:\n\tStarts: \t%#v\n\tTargets:\t%#v\n",
		s.EdgeTypes["written_by"].Starts,
		s.EdgeTypes["written_by"].EdgeTargets)
	assert.EqualValues(t, []int32{0, 0, 3, 4, 7}, s.EdgeTypes["written_by"].Starts)
	assert.EqualValues(t, []int32{0, 3, 4, 0, 0, 4, 7}, s.EdgeTypes["written_by"].EdgeTargets)
	assert.EqualValues(t, []int32{0, 4, 7}, s.EdgeTypes["written_by"].EdgeTargetsForSourceIdx(4))
	assert.EqualValues(t, []int32{}, s.EdgeTypes["written_by"].EdgeTargetsForSourceIdx(0))
}
