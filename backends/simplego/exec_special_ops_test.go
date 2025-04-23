package simplego

import (
	"fmt"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestExecSpecialOps_Identity(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.Identity(x) })
	y0 := exec.Call(bfloat16.FromFloat32(7))[0]
	fmt.Printf("\ty0=%s\n", y0.GoStr())
	assert.Equal(t, bfloat16.FromFloat32(7), y0.Value())
}

func TestExecSpecialOps_Where(t *testing.T) {
	exec := graph.NewExec(backend, func(cond, onTrue, onFalse *graph.Node) *graph.Node { return graph.Where(cond, onTrue, onFalse) })

	// All scalars.
	y0 := exec.Call(true, bfloat16.FromFloat32(7), bfloat16.FromFloat32(11))[0]
	fmt.Printf("\ty0=%s\n", y0.GoStr())
	assert.Equal(t, bfloat16.FromFloat32(7), y0.Value())

	// Scalar cond, non-scalar values.
	y1 := exec.Call(false, []uint8{1, 2}, []uint8{11, 12})[0]
	fmt.Printf("\ty1=%s\n", y1.GoStr())
	assert.Equal(t, []uint8{11, 12}, y1.Value())

	// Non-scalar cond, scalar values.
	y2 := exec.Call([]bool{true, false}, int32(1), int32(0))[0]
	fmt.Printf("\ty2=%s\n", y2.GoStr())
	assert.Equal(t, []int32{1, 0}, y2.Value())

	// Non-scalar cond and values.
	y3 := exec.Call([]bool{false, true, true}, []float32{1, 2, 3}, []float32{101, 102, 103})[0]
	fmt.Printf("\ty3=%s\n", y3.GoStr())
	assert.Equal(t, []float32{101, 2, 3}, y3.Value())
}

func TestExecSpecialOps_Reshape(t *testing.T) {
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node { return graph.Reshape(x, 2, 2) })

	// Reshape scalar to array.
	y0 := exec.Call([]int32{42, 0, 1, 2})[0]
	fmt.Printf("\ty0=%s\n", y0.GoStr())
	assert.NoError(t, y0.Shape().Check(dtypes.Int32, 2, 2))
}
