package simplego

import (
	"fmt"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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

func TestExecSpecialOps_Reduce(t *testing.T) {
	y0 := graph.ExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ReduceMin(x, -1)
	}, [][]float32{{7, 0, 9}, {0, 3, 2}, {1001, 101, 11}})
	fmt.Printf("\ty0=%s\n", y0.GoStr())
	assert.Equal(t, []float32{0, 0, 11}, y0.Value())

	y1 := graph.ExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ReduceMax(x, -1)
	}, []float64{-1e8, -1e6, -1e16})
	fmt.Printf("\ty1=%s\n", y1.GoStr())
	assert.Equal(t, -1.0e6, y1.Value())

	input2 := tensors.FromFlatDataAndDimensions(xslices.Iota[uint32](0, 32), 2, 2, 2, 2, 2)
	fmt.Printf("\tinput2=%s\n", input2.GoStr())
	y2 := graph.ExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ReduceSum(x, 1, 3)
	}, input2)
	fmt.Printf("\ty2=%s\n", y2.GoStr())
	want2 := [][][]uint32{{{20, 24}, {36, 40}}, {{84, 88}, {100, 104}}}
	assert.Equal(t, want2, y2.Value())

	y3 := graph.ExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ReduceMultiply(x, 0)
	}, []float32{-1e-2, 1e5, -1e-3})
	fmt.Printf("\ty3=%s\n", y3.GoStr())
	assert.Equal(t, float32(1), y3.Value())

	bf16 := bfloat16.FromFloat32
	y4 := graph.ExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ReduceMin(x, 0)
	}, []bfloat16.BFloat16{bf16(-11), bf16(-17), bf16(-8)})
	fmt.Printf("\ty4=%s\n", y4.GoStr())
	assert.Equal(t, bf16(-17), y4.Value())

	// Test full reduction to scalar if no axes are given.
	y5 := graph.ExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ReduceSum(x)
	},
		[][]bfloat16.BFloat16{{bf16(-11), bf16(-17)}, {bf16(8), bf16(21)}})
	fmt.Printf("\ty5=%s\n", y5.GoStr())
	assert.Equal(t, bf16(1), y5.Value())
}

func TestExecSpecialOps_transposeIterator(t *testing.T) {
	operand := shapes.Make(dtypes.Int32, 2, 3, 4)
	permutations := []int{2, 0, 1}
	it := newTransposeIterator(operand, permutations)
	transposedFlatIndices := make([]int, 0, operand.Size())
	for _ = range operand.Size() {
		transposedFlatIndices = append(transposedFlatIndices, it.next())
	}
	fmt.Printf("\ttransposedFlatIndices=%#v\n", transposedFlatIndices)
	want := []int{
		// Operand axis 2 (the first being iterated) becomes output axis 0, in row-major order,
		// this is the largest one, with strides of 6:
		0, 6, 12, 18,
		1, 7, 13, 19,
		2, 8, 14, 20,

		3, 9, 15, 21,
		4, 10, 16, 22,
		5, 11, 17, 23}
	require.Equal(t, want, transposedFlatIndices)
}

func TestExecSpecialOps_Transpose(t *testing.T) {
	operand := tensors.FromFlatDataAndDimensions(xslices.Iota(float32(0), 24), 2, 3, 4)
	y0 := graph.ExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.TransposeAllDims(x, 2, 0, 1)
	}, operand)
	fmt.Printf("\ty0=%s\n", y0.GoStr())
	assert.NoError(t, y0.Shape().Check(dtypes.Float32, 4, 2, 3))
	want := [][][]float32{
		{{0, 4, 8}, {12, 16, 20}},
		{{1, 5, 9}, {13, 17, 21}},
		{{2, 6, 10}, {14, 18, 22}},
		{{3, 7, 11}, {15, 19, 23}}}
	require.Equal(t, want, y0.Value())
}

func TestExecSpecialOps_Iota(t *testing.T) {
	y0 := graph.ExecOnce(backend, func(g *graph.Graph) *graph.Node {
		return graph.Iota(g, shapes.Make(dtypes.Int8, 2, 3), 1)
	})
	fmt.Printf("\ty0=%s\n", y0.GoStr())
	assert.NoError(t, y0.Shape().Check(dtypes.Int8, 2, 3))
	require.Equal(t, [][]int8{{0, 1, 2}, {0, 1, 2}}, y0.Value())

	y1 := graph.ExecOnce(backend, func(g *graph.Graph) *graph.Node {
		return graph.Iota(g, shapes.Make(dtypes.BFloat16, 2, 3), 0)
	})
	fmt.Printf("\ty1=%s\n", y1.GoStr())
	assert.NoError(t, y1.Shape().Check(dtypes.BFloat16, 2, 3))
	bf16 := bfloat16.FromFloat32
	require.Equal(t, [][]bfloat16.BFloat16{{bf16(0), bf16(0), bf16(0)}, {bf16(1), bf16(1), bf16(1)}}, y1.Value())

}
