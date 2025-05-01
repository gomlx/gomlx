package simplego

import (
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"math"
	"testing"
)

func TestExampleFunction1(t *testing.T) {
	println("Running TestExampleFunction1")
	// Your test logic here
}

func TestExampleFunction2(t *testing.T) {
	println("Running TestExampleFunction2")
	// Your test logic here
}
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

func TestExecUnary_Clz(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.Clz(x) })
	y0 := exec.Call(int8(7))[0]
	assert.Equal(t, int8(5), y0.Value())
	y1 := exec.Call([]int8{1, 2, 3})[0]
	assert.Equal(t, []int8{7, 6, 6}, y1.Value())

	y2 := exec.Call(uint16(15))[0]
	assert.Equal(t, uint16(12), y2.Value())
	y3 := exec.Call([]uint16{1, 2, 3})[0]
	assert.Equal(t, []uint16{15, 14, 14}, y3.Value())

	y4 := exec.Call(int32(31))[0]
	assert.Equal(t, int32(27), y4.Value())
	y5 := exec.Call([]int32{1, 2, 3})[0]
	assert.Equal(t, []int32{31, 30, 30}, y5.Value())

	y6 := exec.Call(uint64(63))[0]
	assert.Equal(t, uint64(58), y6.Value())
	y7 := exec.Call([]uint64{1, 2, 3})[0]
	assert.Equal(t, []uint64{63, 62, 62}, y7.Value())
}

func TestExecUnary_Exp(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.Exp(x) })
	y0 := exec.Call(float32(1.0))[0]
	assert.InDelta(t, float32(2.718281828459045), y0.Value(), 1e-6)
	y1 := exec.Call(float64(1.0))[0]
	assert.InDelta(t, 2.718281828459045, y1.Value(), 1e-15)
	y2 := exec.Call(bfloat16.FromFloat32(1.0))[0]
	want := bfloat16.FromFloat32(float32(math.E)).Float32()
	assert.InDelta(t, want, y2.Value().(bfloat16.BFloat16).Float32(), 1e-2)
}

func TestExecUnary_Expm1(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.Expm1(x) })
	y0 := exec.Call(float32(1.0))[0]
	assert.InDelta(t, float32(1.71828), y0.Value(), 1e-4)
	y1 := exec.Call(float64(1.0))[0]
	assert.InDelta(t, 1.71828, y1.Value(), 1e-4)
	y2 := exec.Call(bfloat16.FromFloat32(1.0))[0]
	want := bfloat16.FromFloat32(float32(math.E - 1.0)).Float32()
	assert.InDelta(t, want, y2.Value().(bfloat16.BFloat16).Float32(), 1e-2)
}

func TestExecUnary_Log(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.Log(x) })
	y0 := exec.Call(float32(2.718281828459045))[0]
	assert.InDelta(t, float32(1.0), y0.Value(), 1e-6)
	y1 := exec.Call(float64(2.718281828459045))[0]
	assert.InDelta(t, 1.0, y1.Value(), 1e-15)
	y2 := exec.Call(bfloat16.FromFloat32(2.718281828459045))[0]
	assert.InDelta(t, float32(1.0), y2.Value().(bfloat16.BFloat16).Float32(), 1e-2)
}

func TestExecUnary_Log1p(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.Log1p(x) })
	y0 := exec.Call(float32(1.718281828459045))[0]
	assert.InDelta(t, float32(1.0), y0.Value(), 1e-6)
	y1 := exec.Call(float64(1.718281828459045))[0]
	assert.InDelta(t, 1.0, y1.Value(), 1e-15)
	y2 := exec.Call(bfloat16.FromFloat32(1.718281828459045))[0]
	assert.InDelta(t, float32(1.0), y2.Value().(bfloat16.BFloat16).Float32(), 1e-2)
}

func TestExecUnary_Ceil(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.Ceil(x) })
	y0 := exec.Call(float32(1.6))[0]
	assert.Equal(t, float32(2.0), y0.Value())
	y1 := exec.Call(float64(1.6))[0]
	assert.Equal(t, 2.0, y1.Value())
	y2 := exec.Call(bfloat16.FromFloat32(1.6))[0]
	assert.Equal(t, bfloat16.FromFloat32(2.0), y2.Value())
}

func TestExecUnary_Floor(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.Floor(x) })
	y0 := exec.Call(float32(1.6))[0]
	assert.Equal(t, float32(1.0), y0.Value())
	y1 := exec.Call(float64(1.6))[0]
	assert.Equal(t, 1.0, y1.Value())
	y2 := exec.Call(bfloat16.FromFloat32(1.6))[0]
	assert.Equal(t, bfloat16.FromFloat32(1.0), y2.Value())
}

func TestExecUnary_Round(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.Round(x) })
	y0 := exec.Call(float32(1.6))[0]
	assert.Equal(t, float32(2.0), y0.Value())
	y1 := exec.Call(float64(1.6))[0]
	assert.Equal(t, 2.0, y1.Value())
	y2 := exec.Call(bfloat16.FromFloat32(1.6))[0]
	assert.Equal(t, bfloat16.FromFloat32(2.0), y2.Value())
}

func TestExecUnary_Rsqrt(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.Rsqrt(x) })
	y0 := exec.Call(float32(4.0))[0]
	assert.InDelta(t, float32(0.5), y0.Value(), 1e-6)
	y1 := exec.Call(float64(4.0))[0]
	assert.InDelta(t, 0.5, y1.Value(), 1e-15)
	y2 := exec.Call(bfloat16.FromFloat32(4.0))[0]
	assert.InDelta(t, float32(0.5), y2.Value().(bfloat16.BFloat16).Float32(), 1e-2)
}

func TestExecUnary_Sqrt(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.Sqrt(x) })
	y0 := exec.Call(float32(4.0))[0]
	assert.InDelta(t, float32(2.0), y0.Value(), 1e-6)
	y1 := exec.Call(float64(4.0))[0]
	assert.InDelta(t, 2.0, y1.Value(), 1e-15)
	y2 := exec.Call(bfloat16.FromFloat32(4.0))[0]
	assert.InDelta(t, float32(2.0), y2.Value().(bfloat16.BFloat16).Float32(), 1e-2)
}

func TestExecUnary_Cos(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.Cos(x) })
	y0 := exec.Call(float32(0.0))[0]
	assert.InDelta(t, float32(1.0), y0.Value(), 1e-6)
	y1 := exec.Call(float64(0.0))[0]
	assert.InDelta(t, 1.0, y1.Value(), 1e-15)
	y2 := exec.Call(bfloat16.FromFloat32(0.0))[0]
	assert.InDelta(t, float32(1.0), y2.Value().(bfloat16.BFloat16).Float32(), 1e-2)
}

func TestExecUnary_Sin(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.Sin(x) })
	y0 := exec.Call(float32(math.Pi / 2))[0]
	assert.InDelta(t, float32(1.0), y0.Value(), 1e-6)
	y1 := exec.Call(float64(math.Pi / 2))[0]
	assert.InDelta(t, 1.0, y1.Value(), 1e-15)
	y2 := exec.Call(bfloat16.FromFloat32(float32(math.Pi / 2)))[0]
	assert.InDelta(t, float32(1.0), y2.Value().(bfloat16.BFloat16).Float32(), 1e-2)
}

func TestExecUnary_Tanh(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.Tanh(x) })
	y0 := exec.Call(float32(0.0))[0]
	assert.InDelta(t, float32(0.0), y0.Value(), 1e-6)
	y1 := exec.Call(float64(0.0))[0]
	assert.InDelta(t, 0.0, y1.Value(), 1e-15)
	y2 := exec.Call(bfloat16.FromFloat32(0.0))[0]
	assert.InDelta(t, float32(0.0), y2.Value().(bfloat16.BFloat16).Float32(), 1e-2)
}

func TestExecUnary_Logistic(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.Logistic(x) })
	y0 := exec.Call(float32(0.0))[0]
	assert.InDelta(t, float32(0.5), y0.Value(), 1e-6)
	y1 := exec.Call(float64(2.0))[0]
	assert.InDelta(t, 0.8808, y1.Value(), 1e-4)
	y2 := exec.Call(bfloat16.FromFloat32(-2.0))[0]
	assert.InDelta(t, float32(0.1192), y2.Value().(bfloat16.BFloat16).Float32(), 1e-2)
}

func TestExecUnary_IsFinite(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.IsFinite(x) })

	// Test float32
	y0 := exec.Call(float32(1.0))[0]
	assert.Equal(t, true, y0.Value())
	y1 := exec.Call(float32(math.Inf(1)))[0]
	assert.Equal(t, false, y1.Value())

	// Test float64
	y2 := exec.Call(float64(1.0))[0]
	assert.Equal(t, true, y2.Value())
	y3 := exec.Call(math.Inf(-1))[0]
	assert.Equal(t, false, y3.Value())

	// Test bfloat16
	y4 := exec.Call(bfloat16.FromFloat32(float32(math.NaN())))[0]
	assert.Equal(t, false, y4.Value())
	y5 := exec.Call(bfloat16.FromFloat32(1.0))[0]
	assert.Equal(t, true, y5.Value())
}

func TestExecUnary_Erf(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.Erf(x) })
	y0 := exec.Call(float32(1.0))[0]
	assert.InDelta(t, float32(0.8427), y0.Value(), 1e-4)
	y1 := exec.Call(float64(1.0))[0]
	assert.InDelta(t, 0.8427, y1.Value(), 1e-4)
	y2 := exec.Call(bfloat16.FromFloat32(1.0))[0]
	assert.InDelta(t, float32(0.8427), y2.Value().(bfloat16.BFloat16).Float32(), 1e-2)
}
