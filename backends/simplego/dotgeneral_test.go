package simplego

import (
	"fmt"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestDotGeneral_transposeForDotGeneral(t *testing.T) {
	S := shapes.Make
	F32 := dtypes.Float32
	builder := backend.Builder("DotGeneral Test").(*Builder)
	operand := builder.Parameter("lhs", S(F32, 2, 3, 4, 5)).(*Node)
	transposed, batchDims, crossDims, contractingDims :=
		builder.transposeForDotGeneral(operand, "lhs", []int{2, 1}, []int{3, 0})
	fmt.Printf("\ttransposed.shape=%s\n", transposed.shape)

	assert.NoError(t, transposed.shape.CheckDims(10, 1, 12))
	assert.Equal(t, []int{5, 2}, batchDims)
	assert.Len(t, crossDims, 0)
	assert.Equal(t, []int{4, 3}, contractingDims)
}

func TestDotGeneral_Shape(t *testing.T) {
	S := shapes.Make
	F32 := dtypes.Float32
	builder := backend.Builder("DotGeneral Test").(*Builder)
	lhs := builder.Parameter("lhs", S(F32, 2, 3, 4, 5)).(*Node)
	rhs := builder.Parameter("lhs", S(F32, 5, 1, 2, 3)).(*Node)
	got := builder.DotGeneral(
		lhs, []int{1}, []int{3, 0},
		rhs, []int{3}, []int{0, 2},
	).(*Node)
	// Batch dims: 5 , 2
	// Contracting dims: 3
	// Cross dims: 4 (lhs) and 1 (rhs)
	fmt.Printf("\tdotgeneral.shape=%s\n", got.shape)
	assert.NoError(t, got.shape.Check(F32, 5, 2, 4, 1))
}

func TestDotGeneral_Exec(t *testing.T) {
	y0 := graph.ExecOnce(backend, func(lhs, rhs *graph.Node) *graph.Node {
		return graph.DotGeneral(lhs, []int{1}, []int{3, 0}, rhs, []int{1}, []int{0, 2})
	},
		tensors.FromFlatDataAndDimensions(xslices.Iota(float32(1), 2*3*1*5), 2, 3, 1, 5),
		tensors.FromFlatDataAndDimensions(xslices.Iota(float32(1), 5*3*2*4), 5, 3, 2, 4),
	)
	fmt.Printf("\ty0=%s\n", y0)
	want := [][][][]float32{
		{
			{{242, 260, 278, 296}},
			{{899, 962, 1025, 1088}},
		}, {
			{{773, 794, 815, 836}},
			{{2522, 2588, 2654, 2720}},
		}, {
			{{1448, 1472, 1496, 1520}},
			{{4289, 4358, 4427, 4496}},
		}, {
			{{2267, 2294, 2321, 2348}},
			{{6200, 6272, 6344, 6416}},
		}, {
			{{3230, 3260, 3290, 3320}},
			{{8255, 8330, 8405, 8480}},
		}}
	assert.Equal(t, want, y0.Value())
}
