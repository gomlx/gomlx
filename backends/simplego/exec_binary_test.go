package simplego

import (
	"fmt"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestExecBinary_broadcastIterator(t *testing.T) {
	S := func(dims ...int) shapes.Shape {
		return shapes.Make(dtypes.Float32, dims...)
	}

	// Simple [2, 3] shape broadcast simultaneously by 2 different tensors.
	targetShape := S(2, 3)
	bi1 := newBroadcastIterator(S(2, 1), targetShape)
	bi2 := newBroadcastIterator(S(1, 3), targetShape)
	indices1 := make([]int, 0, targetShape.Size())
	indices2 := make([]int, 0, targetShape.Size())
	for _ = range targetShape.Size() {
		indices1 = append(indices1, bi1.Next())
		indices2 = append(indices2, bi2.Next())
	}
	fmt.Printf("\tindices1=%v\n\tindices2=%v\n", indices1, indices2)
	require.Equal(t, []int{0, 0, 0, 1, 1, 1}, indices1)
	require.Equal(t, []int{0, 1, 2, 0, 1, 2}, indices2)

	// Alternating broadcast axes.
	targetShape = S(3, 2, 4, 2)
	b3 := newBroadcastIterator(S(3, 1, 4, 1), targetShape)
	indices3 := make([]int, 0, targetShape.Size())
	for _ = range targetShape.Size() {
		indices3 = append(indices3, b3.Next())
	}
	fmt.Printf("\tindices3=%v\n", indices3)
	want3 := []int{
		0, 0, 1, 1, 2, 2, 3, 3,
		0, 0, 1, 1, 2, 2, 3, 3,
		4, 4, 5, 5, 6, 6, 7, 7,
		4, 4, 5, 5, 6, 6, 7, 7,
		8, 8, 9, 9, 10, 10, 11, 11,
		8, 8, 9, 9, 10, 10, 11, 11,
	}
	require.Equal(t, want3, indices3)
}

func TestExecBinary_Add(t *testing.T) {
	exec := graph.NewExec(backend, func(lhs, rhs *graph.Node) *graph.Node { return graph.Add(lhs, rhs) })

	// Test with scalar (or of size 1) values.
	y0 := exec.Call(bfloat16.FromFloat32(7), bfloat16.FromFloat32(11))[0]
	assert.Equal(t, bfloat16.FromFloat32(18), y0.Value())

	y1 := exec.Call([]int32{-1, 2}, []int32{1})[0]
	assert.Equal(t, []int32{0, 3}, y1.Value())

	y2 := exec.Call([][]int32{{-1}, {2}}, int32(-1))[0]
	assert.Equal(t, [][]int32{{-2}, {1}}, y2.Value())

	// Test with same sized shapes:
	y3 := exec.Call([][]uint64{{1, 2}, {3, 4}}, [][]uint64{{4, 3}, {2, 1}})[0]
	assert.Equal(t, [][]uint64{{5, 5}, {5, 5}}, y3.Value())

	// Test with broadcasting from both sides.
	y4 := exec.Call([][]int32{{-1}, {2}, {5}}, [][]int32{{10, 100}})[0]
	assert.Equal(t, [][]int32{{9, 99}, {12, 102}, {15, 105}}, y4.Value())
}

func TestExecBinary_Mul(t *testing.T) {
	exec := graph.NewExec(backend, func(lhs, rhs *graph.Node) *graph.Node { return graph.Mul(lhs, rhs) })

	// Test with scalar (or of size 1) values.
	y0 := exec.Call(bfloat16.FromFloat32(7), bfloat16.FromFloat32(11))[0]
	assert.Equal(t, bfloat16.FromFloat32(77), y0.Value())

	y1 := exec.Call([]int32{-1, 2}, []int32{2})[0]
	assert.Equal(t, []int32{-2, 4}, y1.Value())

	y2 := exec.Call([][]int32{{-1}, {2}}, int32(-1))[0]
	assert.Equal(t, [][]int32{{1}, {-2}}, y2.Value())

	// Test with same sized shapes:
	y3 := exec.Call([][]int32{{-1, 2}, {3, 4}}, [][]int32{{6, 3}, {2, 1}})[0]
	assert.Equal(t, [][]int32{{-6, 6}, {6, 4}}, y3.Value())

	// Test with broadcasting from both sides.
	y4 := exec.Call([][]int32{{-1}, {2}, {5}}, [][]int32{{10, 100}})[0]
	assert.Equal(t, [][]int32{{-10, -100}, {20, 200}, {50, 500}}, y4.Value())
}

func TestExecBinary_Sub(t *testing.T) {
	exec := graph.NewExec(backend, func(lhs, rhs *graph.Node) *graph.Node { return graph.Sub(lhs, rhs) })

	// Test with scalar (or of size 1) values.
	y0 := exec.Call(bfloat16.FromFloat32(7), bfloat16.FromFloat32(11))[0]
	assert.Equal(t, bfloat16.FromFloat32(-4), y0.Value())

	// Test scalar on right side
	y1 := exec.Call([]int32{-1, 2}, []int32{2})[0]
	assert.Equal(t, []int32{-3, 0}, y1.Value())

	// Test scalar on left side
	y2 := exec.Call(int32(5), []int32{1, 2})[0]
	assert.Equal(t, []int32{4, 3}, y2.Value())

	// Test with same sized shapes:
	y3 := exec.Call([][]int32{{-1, 2}, {3, 4}}, [][]int32{{6, 3}, {2, 1}})[0]
	assert.Equal(t, [][]int32{{-7, -1}, {1, 3}}, y3.Value())

	// Test with broadcasting from both sides.
	y4 := exec.Call([][]int32{{-1}, {2}, {5}}, [][]int32{{10, 100}})[0]
	assert.Equal(t, [][]int32{{-11, -101}, {-8, -98}, {-5, -95}}, y4.Value())
}

func TestExecBinary_Div(t *testing.T) {
	exec := graph.NewExec(backend, func(lhs, rhs *graph.Node) *graph.Node { return graph.Div(lhs, rhs) })

	// Test with scalar (or of size 1) values.
	y0 := exec.Call(bfloat16.FromFloat32(10), bfloat16.FromFloat32(2))[0]
	assert.Equal(t, bfloat16.FromFloat32(5), y0.Value())

	// Test scalar on right side
	y1 := exec.Call([]int32{-4, 8}, []int32{2})[0]
	assert.Equal(t, []int32{-2, 4}, y1.Value())

	// Test scalar on left side
	y2 := exec.Call(int32(6), []int32{2, 3})[0]
	assert.Equal(t, []int32{3, 2}, y2.Value())

	// Test with same sized shapes:
	y3 := exec.Call([][]int32{{-6, 9}, {12, 15}}, [][]int32{{2, 3}, {4, 5}})[0]
	assert.Equal(t, [][]int32{{-3, 3}, {3, 3}}, y3.Value())

	// Test with broadcasting from both sides.
	y4 := exec.Call([][]int32{{-10}, {20}, {50}}, [][]int32{{2, 10}})[0]
	assert.Equal(t, [][]int32{{-5, -1}, {10, 2}, {25, 5}}, y4.Value())
}

func TestExecBinary_Rem(t *testing.T) {
	exec := graph.NewExec(backend, func(lhs, rhs *graph.Node) *graph.Node { return graph.Rem(lhs, rhs) })

	// Test with scalar (or of size 1) values.
	y0 := exec.Call(bfloat16.FromFloat32(7), bfloat16.FromFloat32(4))[0]
	fmt.Printf("\ty0=%v\n", y0.GoStr())
	assert.Equal(t, bfloat16.FromFloat32(3), y0.Value())

	// Test scalar on right side
	y1 := exec.Call([]int32{7, 9}, []int32{4})[0]
	assert.Equal(t, []int32{3, 1}, y1.Value())

	// Test scalar on left side
	y2 := exec.Call(int32(7), []int32{4, 3})[0]
	assert.Equal(t, []int32{3, 1}, y2.Value())

	// Test with same sized shapes:
	y3 := exec.Call([][]int32{{7, 8}, {9, 10}}, [][]int32{{4, 3}, {2, 3}})[0]
	assert.Equal(t, [][]int32{{3, 2}, {1, 1}}, y3.Value())

	// Test with broadcasting from both sides.
	y4 := exec.Call([][]int32{{7}, {8}, {9}}, [][]int32{{4, 3}})[0]
	assert.Equal(t, [][]int32{{3, 1}, {0, 2}, {1, 0}}, y4.Value())
}
