package simplego

import (
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestExecSpecialOps_Where(t *testing.T) {
	exec := graph.NewExec(backend, func(cond, onTrue, onFalse *graph.Node) *graph.Node { return graph.Where(cond, onTrue, onFalse) })

	// All scalars.
	y0 := exec.Call(true, bfloat16.FromFloat32(7), bfloat16.FromFloat32(11))[0]
	assert.Equal(t, bfloat16.FromFloat32(7), y0.Value())

	// Scalar cond, non scalar values.
	y1 := exec.Call(false, []uint8{1, 2}, []uint8{11, 12})[0]
	assert.Equal(t, []uint8{11, 12}, y1.Value())
}
