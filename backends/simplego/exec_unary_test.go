package simplego

import (
	"github.com/gomlx/gomlx/graph"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

var backend = New("go")

func TestBackendIsSimpleGo(t *testing.T) {
	assert.NotPanics(t, func() { _ = backend.(*Backend) })
}

func TestExecUnary_Neg(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.Neg(x) })
	y0 := exec.Call(float32(7))[0]
	assert.Equal(t, float32(-7), y0.Value())
	y1 := exec.Call([]int32{-1, 2})[0]
	assert.Equal(t, []int32{1, -2}, y1.Value())
	require.Panics(t, func() { _ = exec.Call([]uint32{1, 2, 3}) })
}

func TestExecUnary_Abs(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.Abs(x) })
	y0 := exec.Call(float32(-7))[0]
	assert.Equal(t, float32(7), y0.Value())
	y1 := exec.Call([]int32{-1, 2})[0]
	assert.Equal(t, []int32{1, 2}, y1.Value())
	y2 := exec.Call([]uint32{1, 2, 3})[0]
	assert.Equal(t, []uint32{1, 2, 3}, y2.Value())
}

func TestExecUnary_Sign(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.Sign(x) })
	y0 := exec.Call(float32(-7))[0]
	assert.Equal(t, float32(-1), y0.Value())
	y1 := exec.Call([]int32{-1, 0, 2})[0]
	assert.Equal(t, []int32{-1, 0, 1}, y1.Value())
	y2 := exec.Call([]uint32{1, 0, 3})[0]
	assert.Equal(t, []uint32{1, 0, 1}, y2.Value())
}

func TestExecUnary_LogicalNot(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.LogicalNot(x) })
	y0 := exec.Call(true)[0]
	assert.Equal(t, false, y0.Value())
	y1 := exec.Call([]bool{true, false, true})[0]
	assert.Equal(t, []bool{false, true, false}, y1.Value())
}

func TestExecUnary_BitwiseNot(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.BitwiseNot(x) })
	y0 := exec.Call(int32(7))[0]
	assert.Equal(t, int32(-8), y0.Value())
	y1 := exec.Call([]int32{-1, 2, 3})[0]
	assert.Equal(t, []int32{0, -3, -4}, y1.Value())
	y2 := exec.Call([]uint32{1, 2, 3})[0]
	assert.Equal(t, []uint32{^uint32(1), ^uint32(2), ^uint32(3)}, y2.Value())
}

func TestExecUnary_BitCount(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.BitCount(x) })
	y0 := exec.Call(int8(7))[0]
	assert.Equal(t, int8(3), y0.Value())
	y1 := exec.Call([]int8{-1, 2, 3})[0]
	assert.Equal(t, []int8{8, 1, 2}, y1.Value())

	y2 := exec.Call(uint16(15))[0]
	assert.Equal(t, uint16(4), y2.Value())
	y3 := exec.Call([]uint16{1, 2, 3})[0]
	assert.Equal(t, []uint16{1, 1, 2}, y3.Value())

	y4 := exec.Call(int32(31))[0]
	assert.Equal(t, int32(5), y4.Value())
	y5 := exec.Call([]int32{-1, 2, 3})[0]
	assert.Equal(t, []int32{32, 1, 2}, y5.Value())

	y6 := exec.Call(uint64(63))[0]
	assert.Equal(t, uint64(6), y6.Value())
	y7 := exec.Call([]uint64{1, 2, 3})[0]
	assert.Equal(t, []uint64{1, 1, 2}, y7.Value())
}
