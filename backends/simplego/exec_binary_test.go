package simplego

import (
	"fmt"
	"testing"

	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
	for range targetShape.Size() {
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
	for range targetShape.Size() {
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

func TestExecBinary_Pow(t *testing.T) {
	exec := graph.NewExec(backend, func(lhs, rhs *graph.Node) *graph.Node { return graph.Pow(lhs, rhs) })

	// Test with scalar (or of size 1) values.
	y0 := exec.Call(bfloat16.FromFloat32(16), bfloat16.FromFloat32(0.5))[0]
	assert.Equal(t, bfloat16.FromFloat32(4), y0.Value())

	// Test scalar on right side
	y1 := exec.Call([]int32{2, 3}, []int32{2})[0]
	assert.Equal(t, []int32{4, 9}, y1.Value())

	// Test scalar on left side
	y2 := exec.Call(int32(2), []int32{2, 3})[0]
	assert.Equal(t, []int32{4, 8}, y2.Value())

	// Test with same sized shapes:
	y3 := exec.Call([][]int32{{2, 3}, {4, 5}}, [][]int32{{2, 2}, {2, 2}})[0]
	assert.Equal(t, [][]int32{{4, 9}, {16, 25}}, y3.Value())

	// Test with broadcasting from both sides.
	y4 := exec.Call([][]int32{{2}, {3}, {4}}, [][]int32{{2, 3}})[0]
	assert.Equal(t, [][]int32{{4, 8}, {9, 27}, {16, 64}}, y4.Value())
}

func TestExecBinary_Max(t *testing.T) {
	exec := graph.NewExec(backend, func(lhs, rhs *graph.Node) *graph.Node { return graph.Max(lhs, rhs) })

	// Test with scalar (or of size 1) values.
	y0 := exec.Call(bfloat16.FromFloat32(7), bfloat16.FromFloat32(11))[0]
	assert.Equal(t, bfloat16.FromFloat32(11), y0.Value())

	// Test scalar on right side
	y1 := exec.Call([]int32{-1, 2}, []int32{0})[0]
	assert.Equal(t, []int32{0, 2}, y1.Value())

	// Test scalar on left side
	y2 := exec.Call(int32(5), []int32{1, 8})[0]
	assert.Equal(t, []int32{5, 8}, y2.Value())

	// Test with same sized shapes:
	y3 := exec.Call([][]int32{{-1, 2}, {3, 4}}, [][]int32{{6, 1}, {2, 5}})[0]
	assert.Equal(t, [][]int32{{6, 2}, {3, 5}}, y3.Value())

	// Test with broadcasting from both sides.
	y4 := exec.Call([][]int32{{-1}, {2}, {5}}, [][]int32{{0, 3}})[0]
	assert.Equal(t, [][]int32{{0, 3}, {2, 3}, {5, 5}}, y4.Value())
}

func TestExecBinary_Min(t *testing.T) {
	exec := graph.NewExec(backend, func(lhs, rhs *graph.Node) *graph.Node { return graph.Min(lhs, rhs) })

	// Test with scalar (or of size 1) values.
	y0 := exec.Call(bfloat16.FromFloat32(7), bfloat16.FromFloat32(11))[0]
	assert.Equal(t, bfloat16.FromFloat32(7), y0.Value())

	// Test scalar on right side
	y1 := exec.Call([]int32{-1, 2}, []int32{0})[0]
	assert.Equal(t, []int32{-1, 0}, y1.Value())

	// Test scalar on left side
	y2 := exec.Call(int32(5), []int32{1, 8})[0]
	assert.Equal(t, []int32{1, 5}, y2.Value())

	// Test with same sized shapes:
	y3 := exec.Call([][]int32{{-1, 2}, {3, 4}}, [][]int32{{6, 1}, {2, 5}})[0]
	assert.Equal(t, [][]int32{{-1, 1}, {2, 4}}, y3.Value())

	// Test with broadcasting from both sides.
	y4 := exec.Call([][]int32{{-1}, {2}, {5}}, [][]int32{{0, 3}})[0]
	assert.Equal(t, [][]int32{{-1, -1}, {0, 2}, {0, 3}}, y4.Value())
}

func TestExecBinary_BitwiseAnd(t *testing.T) {
	exec := graph.NewExec(backend, func(lhs, rhs *graph.Node) *graph.Node { return graph.BitwiseAnd(lhs, rhs) })

	// Test with scalar (or of size 1) values.
	y0 := exec.Call(uint8(0b11110000), uint8(0b10101010))[0]
	assert.Equal(t, uint8(0b10100000), y0.Value())

	// Test scalar on right side
	y1 := exec.Call([]int32{0b1100, 0b0011}, []int32{0b1010})[0]
	assert.Equal(t, []int32{0b1000, 0b0010}, y1.Value())

	// Test scalar on left side
	y2 := exec.Call(uint16(0b1111), []uint16{0b1010, 0b0101})[0]
	assert.Equal(t, []uint16{0b1010, 0b0101}, y2.Value())

	// Test with same sized shapes:
	y3 := exec.Call([][]int32{{0b1100, 0b0011}, {0b1111, 0b0000}}, [][]int32{{0b1010, 0b1010}, {0b0101, 0b0101}})[0]
	assert.Equal(t, [][]int32{{0b1000, 0b0010}, {0b0101, 0b0000}}, y3.Value())

	// Test with broadcasting from both sides.
	y4 := exec.Call([][]uint32{{0b1100}, {0b0011}, {0b1111}}, [][]uint32{{0b1010, 0b0101}})[0]
	assert.Equal(t, [][]uint32{{0b1000, 0b0100}, {0b0010, 0b0001}, {0b1010, 0b0101}}, y4.Value())
}

func TestExecBinary_BitwiseOr(t *testing.T) {
	exec := graph.NewExec(backend, func(lhs, rhs *graph.Node) *graph.Node { return graph.BitwiseOr(lhs, rhs) })

	// Test with scalar (or of size 1) values.
	y0 := exec.Call(uint8(0b11110000), uint8(0b10101010))[0]
	assert.Equal(t, uint8(0b11111010), y0.Value())

	// Test scalar on right side
	y1 := exec.Call([]int32{0b1100, 0b0011}, []int32{0b1010})[0]
	assert.Equal(t, []int32{0b1110, 0b1011}, y1.Value())

	// Test scalar on left side
	y2 := exec.Call(uint16(0b1111), []uint16{0b1010, 0b0101})[0]
	assert.Equal(t, []uint16{0b1111, 0b1111}, y2.Value())

	// Test with same sized shapes:
	y3 := exec.Call([][]int32{{0b1100, 0b0011}, {0b1111, 0b0000}}, [][]int32{{0b1010, 0b1010}, {0b0101, 0b0101}})[0]
	assert.Equal(t, [][]int32{{0b1110, 0b1011}, {0b1111, 0b0101}}, y3.Value())

	// Test with broadcasting from both sides.
	y4 := exec.Call([][]uint32{{0b1100}, {0b0011}, {0b1111}}, [][]uint32{{0b1010, 0b0101}})[0]
	assert.Equal(t, [][]uint32{{0b1110, 0b1101}, {0b1011, 0b0111}, {0b1111, 0b1111}}, y4.Value())
}

func TestExecBinary_BitwiseXor(t *testing.T) {
	exec := graph.NewExec(backend, func(lhs, rhs *graph.Node) *graph.Node { return graph.BitwiseXor(lhs, rhs) })

	// Test with scalar (or of size 1) values.
	y0 := exec.Call(uint8(0b11110000), uint8(0b10101010))[0]
	assert.Equal(t, uint8(0b01011010), y0.Value())

	// Test scalar on right side
	y1 := exec.Call([]int32{0b1100, 0b0011}, []int32{0b1010})[0]
	assert.Equal(t, []int32{0b0110, 0b1001}, y1.Value())

	// Test scalar on left side
	y2 := exec.Call(uint16(0b1111), []uint16{0b1010, 0b0101})[0]
	assert.Equal(t, []uint16{0b0101, 0b1010}, y2.Value())

	// Test with same sized shapes:
	y3 := exec.Call([][]int32{{0b1100, 0b0011}, {0b1111, 0b0000}}, [][]int32{{0b1010, 0b1010}, {0b0101, 0b0101}})[0]
	assert.Equal(t, [][]int32{{0b0110, 0b1001}, {0b1010, 0b0101}}, y3.Value())

	// Test with broadcasting from both sides.
	y4 := exec.Call([][]uint32{{0b1100}, {0b0011}, {0b1111}}, [][]uint32{{0b1010, 0b0101}})[0]
	assert.Equal(t, [][]uint32{{0b0110, 0b1001}, {0b1001, 0b0110}, {0b0101, 0b1010}}, y4.Value())
}

func TestExecBinary_LogicalAnd(t *testing.T) {
	exec := graph.NewExec(backend, func(lhs, rhs *graph.Node) *graph.Node { return graph.LogicalAnd(lhs, rhs) })

	// Test with scalar (or of size 1) values.
	y0 := exec.Call(true, false)[0]
	assert.Equal(t, false, y0.Value())

	// Test scalar on right side
	y1 := exec.Call([]bool{true, false}, []bool{true})[0]
	assert.Equal(t, []bool{true, false}, y1.Value())

	// Test scalar on left side
	y2 := exec.Call(true, []bool{true, false})[0]
	assert.Equal(t, []bool{true, false}, y2.Value())

	// Test with same sized shapes:
	y3 := exec.Call([][]bool{{true, false}, {true, true}}, [][]bool{{true, true}, {false, true}})[0]
	assert.Equal(t, [][]bool{{true, false}, {false, true}}, y3.Value())

	// Test with broadcasting from both sides.
	y4 := exec.Call([][]bool{{true}, {false}, {true}}, [][]bool{{true, false}})[0]
	assert.Equal(t, [][]bool{{true, false}, {false, false}, {true, false}}, y4.Value())
}

func TestExecBinary_LogicalOr(t *testing.T) {
	exec := graph.NewExec(backend, func(lhs, rhs *graph.Node) *graph.Node { return graph.LogicalOr(lhs, rhs) })

	// Test with scalar (or of size 1) values.
	y0 := exec.Call(true, false)[0]
	assert.Equal(t, true, y0.Value())

	// Test scalar on right side
	y1 := exec.Call([]bool{true, false}, []bool{true})[0]
	assert.Equal(t, []bool{true, true}, y1.Value())

	// Test scalar on left side
	y2 := exec.Call(true, []bool{true, false})[0]
	assert.Equal(t, []bool{true, true}, y2.Value())

	// Test with same sized shapes:
	y3 := exec.Call([][]bool{{true, false}, {true, true}}, [][]bool{{true, true}, {false, true}})[0]
	assert.Equal(t, [][]bool{{true, true}, {true, true}}, y3.Value())

	// Test with broadcasting from both sides.
	y4 := exec.Call([][]bool{{true}, {false}, {true}}, [][]bool{{true, false}})[0]
	assert.Equal(t, [][]bool{{true, true}, {true, false}, {true, true}}, y4.Value())
}

func TestExecBinary_LogicalXor(t *testing.T) {
	exec := graph.NewExec(backend, func(lhs, rhs *graph.Node) *graph.Node { return graph.LogicalXor(lhs, rhs) })

	// Test with scalar (or of size 1) values.
	y0 := exec.Call(true, false)[0]
	assert.Equal(t, true, y0.Value())

	// Test scalar on right side
	y1 := exec.Call([]bool{true, false}, []bool{true})[0]
	assert.Equal(t, []bool{false, true}, y1.Value())

	// Test scalar on left side
	y2 := exec.Call(true, []bool{true, false})[0]
	assert.Equal(t, []bool{false, true}, y2.Value())

	// Test with same sized shapes:
	y3 := exec.Call([][]bool{{true, false}, {true, true}}, [][]bool{{true, true}, {false, true}})[0]
	assert.Equal(t, [][]bool{{false, true}, {true, false}}, y3.Value())

	// Test with broadcasting from both sides.
	y4 := exec.Call([][]bool{{true}, {false}, {true}}, [][]bool{{true, false}})[0]
	assert.Equal(t, [][]bool{{false, true}, {true, false}, {false, true}}, y4.Value())
}

func TestExecBinary_Comparison(t *testing.T) {
	// Test Equal
	execEq := graph.NewExec(backend, func(lhs, rhs *graph.Node) *graph.Node { return graph.Equal(lhs, rhs) })
	y0 := execEq.Call(float32(1.5), float32(1.5))[0]
	assert.Equal(t, true, y0.Value())
	y1 := execEq.Call(bfloat16.FromFloat32(2.0), bfloat16.FromFloat32(2.0))[0]
	assert.Equal(t, true, y1.Value())
	y2 := execEq.Call([]uint16{1, 2, 3}, uint16(2))[0]
	assert.Equal(t, []bool{false, true, false}, y2.Value())
	y3 := execEq.Call([]int32{5}, []int32{5, 6})[0]
	assert.Equal(t, []bool{true, false}, y3.Value())

	// Test GreaterOrEqual
	execGe := graph.NewExec(backend, func(lhs, rhs *graph.Node) *graph.Node { return graph.GreaterOrEqual(lhs, rhs) })
	y4 := execGe.Call(float32(2.5), float32(1.5))[0]
	assert.Equal(t, true, y4.Value())
	y5 := execGe.Call(bfloat16.FromFloat32(1.0), bfloat16.FromFloat32(2.0))[0]
	assert.Equal(t, false, y5.Value())
	y6 := execGe.Call([]uint16{1, 2, 3}, uint16(2))[0]
	assert.Equal(t, []bool{false, true, true}, y6.Value())

	// Test GreaterThan
	execGt := graph.NewExec(backend, func(lhs, rhs *graph.Node) *graph.Node { return graph.GreaterThan(lhs, rhs) })
	y7 := execGt.Call(float32(2.5), float32(1.5))[0]
	assert.Equal(t, true, y7.Value())
	y8 := execGt.Call([]int32{1, 2, 3}, int32(2))[0]
	assert.Equal(t, []bool{false, false, true}, y8.Value())

	// Test LessOrEqual
	execLe := graph.NewExec(backend, func(lhs, rhs *graph.Node) *graph.Node { return graph.LessOrEqual(lhs, rhs) })
	y9 := execLe.Call(bfloat16.FromFloat32(1.0), bfloat16.FromFloat32(2.0))[0]
	assert.Equal(t, true, y9.Value())
	y10 := execLe.Call([]uint16{1, 2, 3}, uint16(2))[0]
	assert.Equal(t, []bool{true, true, false}, y10.Value())

	// Test LessThan
	execLt := graph.NewExec(backend, func(lhs, rhs *graph.Node) *graph.Node { return graph.LessThan(lhs, rhs) })
	y11 := execLt.Call(float32(1.5), float32(2.5))[0]
	assert.Equal(t, true, y11.Value())
	y12 := execLt.Call([]int32{1, 2, 3}, int32(2))[0]
	assert.Equal(t, []bool{true, false, false}, y12.Value())
}
