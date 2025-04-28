package simplego

import (
	"fmt"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/stretchr/testify/require"
	"testing"
)

var (
	// Shortcuts:

	// bf16 shortcut to create new BFloat16 numbers.
	bf16 = bfloat16.FromFloat32
)

// ... existing tests ...

func TestExecSpecialOps_Gather(t *testing.T) {
	y0 := graph.ExecOnce(backend, func(g *graph.Graph) *graph.Node {
		operand := graph.IotaFull(g, shapes.Make(dtypes.F32, 4, 3, 2, 2))
		startIndices := graph.Const(g, [][][]int{{{0, 1}, {0, 1}, {0, 1}}, {{0, 0}, {0, 0}, {1, 1}}, {{0, 0}, {1, 1}, {0, 0}}})
		startVectorAxis := 1
		fmt.Printf("\tstartIndices.shape=%s, startVectorAxis=%d\n", startIndices.Shape(), startVectorAxis)
		offsetOutputAxes := []int{1, 3}
		collapsedSliceAxes := []int{0, 2}
		startIndexMap := []int{0, 2, 3}
		sliceSizes := []int{1, 3, 1, 1}
		return graph.BackendGather(operand, startIndices, startVectorAxis, offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes, false)
	})
	fmt.Printf("\ty0=%s\n", y0.GoStr())
	want := [][][][]float32{
		{{{0}, {15}}, {{4}, {19}}, {{8}, {23}}},
		{{{1}, {1}}, {{5}, {5}}, {{9}, {9}}},
		{{{2}, {2}}, {{6}, {6}}, {{10}, {10}}}}
	require.Equal(t, want, y0.Value())
}

func TestExecSpecialOps_Concatenate(t *testing.T) {
	// Test Case 1: Concatenating vectors (rank 1) along axis 0
	y1 := graph.ExecOnce(backend, func(g *graph.Graph) *graph.Node {
		in1 := graph.Const(g, []float32{1, 2, 3})
		in2 := graph.Const(g, []float32{4, 5})
		return graph.Concatenate([]*graph.Node{in1, in2}, 0)
	})
	fmt.Printf("\ty1=%s\n", y1.GoStr())
	want1 := []float32{1, 2, 3, 4, 5}
	require.NoError(t, y1.Shape().Check(dtypes.Float32, 5))
	require.Equal(t, want1, y1.Value())

	// Test Case 2: Concatenating matrices (rank 2) along axis 0
	y2 := graph.ExecOnce(backend, func(g *graph.Graph) *graph.Node {
		in1 := graph.Const(g, [][]int8{{1, 2}, {3, 4}})
		in2 := graph.Const(g, [][]int8{{5, 6}})
		return graph.Concatenate([]*graph.Node{in1, in2}, 0)
	})
	fmt.Printf("\ty2=%s\n", y2.GoStr())
	want2 := [][]int8{{1, 2}, {3, 4}, {5, 6}}
	require.NoError(t, y2.Shape().Check(dtypes.Int8, 3, 2))
	require.Equal(t, want2, y2.Value())

	// Test Case 3: Concatenating matrices (rank 2) along axis 1
	y3 := graph.ExecOnce(backend, func(g *graph.Graph) *graph.Node {
		in1 := graph.Const(g, [][]bfloat16.BFloat16{{bf16(1)}, {bf16(2)}})
		in2 := graph.Const(g, [][]bfloat16.BFloat16{{bf16(3), bf16(4)}, {bf16(5), bf16(6)}})
		in3 := graph.Const(g, [][]bfloat16.BFloat16{{bf16(7)}, {bf16(8)}})
		return graph.Concatenate([]*graph.Node{in1, in2, in3}, 1)
	})
	fmt.Printf("\ty3=%s\n", y3.GoStr())
	want3 := [][]bfloat16.BFloat16{{bf16(1), bf16(3), bf16(4), bf16(7)}, {bf16(2), bf16(5), bf16(6), bf16(8)}}
	require.NoError(t, y3.Shape().Check(dtypes.BFloat16, 2, 4))
	require.Equal(t, want3, y3.Value())

	// Test Case 4: Concatenating rank 3 tensors along axis 1
	y4 := graph.ExecOnce(backend, func(g *graph.Graph) *graph.Node {
		in1 := graph.Const(g, [][][]int32{{{1, 2}}, {{3, 4}}})                    // Shape (2, 1, 2)
		in2 := graph.Const(g, [][][]int32{{{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}}) // Shape (2, 2, 2)
		return graph.Concatenate([]*graph.Node{in1, in2}, 1)
	})
	fmt.Printf("\ty4=%s\n", y4.GoStr())
	want4 := [][][]int32{{{1, 2}, {5, 6}, {7, 8}}, {{3, 4}, {9, 10}, {11, 12}}} // Shape (2, 3, 2)
	require.NoError(t, y4.Shape().Check(dtypes.Int32, 2, 3, 2))
	require.Equal(t, want4, y4.Value())

	// Test Case 5: Concatenating rank 3 tensors along axis 2
	y5 := graph.ExecOnce(backend, func(g *graph.Graph) *graph.Node {
		in1 := graph.Const(g, [][][]float64{{{1, 2}}, {{3, 4}}}) // Shape (2, 1, 2)
		in2 := graph.Const(g, [][][]float64{{{5}}, {{6}}})       // Shape (2, 1, 1)
		return graph.Concatenate([]*graph.Node{in1, in2}, 2)
	})
	fmt.Printf("\ty5=%s\n", y5.GoStr())
	want5 := [][][]float64{{{1, 2, 5}}, {{3, 4, 6}}} // Shape (2, 1, 3)
	require.NoError(t, y5.Shape().Check(dtypes.Float64, 2, 1, 3))
	require.Equal(t, want5, y5.Value())
}
