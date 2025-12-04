package simplego

import (
	"fmt"
	"math"
	"reflect"
	"slices"
	"sort"
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/shapeinference"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/xslices"
)

var (
	// Shortcuts:

	Bool = dtypes.Bool
	I8   = dtypes.Int8
	I32  = dtypes.Int32
	F32  = dtypes.Float32
	U64  = dtypes.Uint64
	MS   = shapes.Make

	// bf16 shortcut to create new BFloat16 numbers.
	bf16 = bfloat16.FromFloat32
)

func TestExecSpecialOps_Identity(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Identity)
	y0 := exec.MustExec(bfloat16.FromFloat32(7))[0]
	// fmt.Printf("\ty0=%s\n", y0.GoStr())
	assert.Equal(t, bfloat16.FromFloat32(7), y0.Value())
}

func TestExecSpecialOps_Where(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Where)

	// All scalars.
	y0 := exec.MustExec(true, bfloat16.FromFloat32(7), bfloat16.FromFloat32(11))[0]
	// fmt.Printf("\ty0=%s\n", y0.GoStr())
	assert.Equal(t, bfloat16.FromFloat32(7), y0.Value())

	// Scalar cond, non-scalar values.
	y1 := exec.MustExec(false, []uint8{1, 2}, []uint8{11, 12})[0]
	// fmt.Printf("\ty1=%s\n", y1.GoStr())
	assert.Equal(t, []uint8{11, 12}, y1.Value())

	// Non-scalar cond, scalar values.
	y2 := exec.MustExec([]bool{true, false}, int32(1), int32(0))[0]
	// fmt.Printf("\ty2=%s\n", y2.GoStr())
	assert.Equal(t, []int32{1, 0}, y2.Value())

	// Non-scalar cond and values.
	y3 := exec.MustExec([]bool{false, true, true}, []float32{1, 2, 3}, []float32{101, 102, 103})[0]
	// fmt.Printf("\ty3=%s\n", y3.GoStr())
	assert.Equal(t, []float32{101, 2, 3}, y3.Value())
}

func TestExecSpecialOps_Reshape(t *testing.T) {
	exec := graph.MustNewExec(backend, func(x *graph.Node) *graph.Node { return graph.Reshape(x, 2, 2) })

	// Reshape scalar to array.
	y0 := exec.MustExec([]int32{42, 0, 1, 2})[0]
	// fmt.Printf("\ty0=%s\n", y0.GoStr())
	assert.NoError(t, y0.Shape().Check(dtypes.Int32, 2, 2))
}

// =================================================================================================================
// Reduce* ---------------------------------------------------------------------------------------------------------
// =================================================================================================================

func TestExecSpecialOps_Reduce(t *testing.T) {
	y0 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ReduceMin(x, -1)
	}, [][]float32{{7, 0, 9}, {0, 3, 2}, {1001, 101, 11}})
	// fmt.Printf("\ty0=%s\n", y0.GoStr())
	assert.Equal(t, []float32{0, 0, 11}, y0.Value())

	y1 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ReduceMax(x, -1)
	}, []float64{-1e8, -1e6, -1e16})
	// fmt.Printf("\ty1=%s\n", y1.GoStr())
	assert.Equal(t, -1.0e6, y1.Value())

	input2 := tensors.FromFlatDataAndDimensions(xslices.Iota[uint32](0, 32), 2, 2, 2, 2, 2)
	// fmt.Printf("\tinput2=%s\n", input2.GoStr())
	y2 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ReduceSum(x, 1, 3)
	}, input2)
	// fmt.Printf("\ty2=%s\n", y2.GoStr())
	want2 := [][][]uint32{{{20, 24}, {36, 40}}, {{84, 88}, {100, 104}}}
	assert.Equal(t, want2, y2.Value())

	y3 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ReduceMultiply(x, 0)
	}, []float32{-1e-2, 1e5, -1e-3})
	// fmt.Printf("\ty3=%s\n", y3.GoStr())
	assert.Equal(t, float32(1), y3.Value())

	y4 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ReduceMin(x, 0)
	}, []bfloat16.BFloat16{bf16(-11), bf16(-17), bf16(-8)})
	// fmt.Printf("\ty4=%s\n", y4.GoStr())
	assert.Equal(t, bf16(-17), y4.Value())

	// Test full reduction to scalar if no axes are given.
	y5 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ReduceSum(x)
	},
		[][]bfloat16.BFloat16{{bf16(-11), bf16(-17)}, {bf16(8), bf16(21)}})
	// fmt.Printf("\ty5=%s\n", y5.GoStr())
	assert.Equal(t, bf16(1), y5.Value())
}

func TestExecSpecialOps_ReduceBitwise(t *testing.T) {
	y0 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ReduceBitwiseAnd(x, -1)
	}, []int32{7, 3, 2})
	// fmt.Printf("\tReduceBitwiseAnd: y0=%s\n", y0.GoStr())
	assert.Equal(t, int32(2), y0.Value())

	y1 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ReduceBitwiseOr(x)
	}, [][]uint8{{3}, {12}, {17}})
	// fmt.Printf("\tReduceBitwiseOr: y1=%s\n", y1.GoStr())
	assert.Equal(t, uint8(31), y1.Value())

	y2 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ReduceBitwiseXor(x, 0)
	}, [][]int64{{3}, {12}, {17}})
	fmt.Printf("\tReduceBitwiseXor: y2=%s\n", y2.GoStr())
	assert.Equal(t, []int64{30}, y2.Value())
}

func TestExecSpecialOps_ReduceLogical(t *testing.T) {
	y0 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ReduceLogicalAnd(x, -1)
	}, [][]bool{{true, false}, {true, true}})
	// fmt.Printf("\tReduceLogicalAnd: y0=%s\n", y0.GoStr())
	assert.Equal(t, []bool{false, true}, y0.Value())

	y1 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ReduceLogicalOr(x, 0)
	}, [][]bool{{true, false}, {false, false}})
	// fmt.Printf("\tReduceLogicalOr: y1=%s\n", y1.GoStr())
	assert.Equal(t, []bool{true, false}, y1.Value())

	y2 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ReduceLogicalXor(x, -1)
	}, [][]bool{{true, false}, {true, true}})
	// fmt.Printf("\tReduceLogicalXor: y2=%s\n", y2.GoStr())
	assert.Equal(t, []bool{true, false}, y2.Value())
}

func TestExecSpecialOps_transposeIterator(t *testing.T) {
	operand := shapes.Make(dtypes.Int32, 2, 3, 4)
	permutations := []int{2, 0, 1}
	it := newTransposeIterator(operand, permutations)
	transposedFlatIndices := make([]int, 0, operand.Size())
	for range operand.Size() {
		transposedFlatIndices = append(transposedFlatIndices, it.next())
	}
	// fmt.Printf("\ttransposedFlatIndices=%#v\n", transposedFlatIndices)
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
	y0 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.TransposeAllAxes(x, 2, 0, 1)
	}, operand)
	// fmt.Printf("\ty0=%s\n", y0.GoStr())
	assert.NoError(t, y0.Shape().Check(dtypes.Float32, 4, 2, 3))
	want := [][][]float32{
		{{0, 4, 8}, {12, 16, 20}},
		{{1, 5, 9}, {13, 17, 21}},
		{{2, 6, 10}, {14, 18, 22}},
		{{3, 7, 11}, {15, 19, 23}}}
	require.Equal(t, want, y0.Value())
}

func TestExecSpecialOps_Iota(t *testing.T) {
	y0 := graph.MustExecOnce(backend, func(g *graph.Graph) *graph.Node {
		return graph.Iota(g, shapes.Make(dtypes.Int8, 2, 3), 1)
	})
	// fmt.Printf("\ty0=%s\n", y0.GoStr())
	assert.NoError(t, y0.Shape().Check(dtypes.Int8, 2, 3))
	require.Equal(t, [][]int8{{0, 1, 2}, {0, 1, 2}}, y0.Value())

	y1 := graph.MustExecOnce(backend, func(g *graph.Graph) *graph.Node {
		return graph.Iota(g, shapes.Make(dtypes.BFloat16, 2, 3), 0)
	})
	// fmt.Printf("\ty1=%s\n", y1.GoStr())
	assert.NoError(t, y1.Shape().Check(dtypes.BFloat16, 2, 3))
	bf16 := bfloat16.FromFloat32
	require.Equal(t, [][]bfloat16.BFloat16{{bf16(0), bf16(0), bf16(0)}, {bf16(1), bf16(1), bf16(1)}}, y1.Value())

}

func TestExecSpecialOps_Broadcast(t *testing.T) {
	y0 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.BroadcastPrefix(x, 2, 3)
	}, []int8{1, 3})
	// fmt.Printf("\ty0=%s\n", y0.GoStr())
	assert.NoError(t, y0.Shape().Check(dtypes.Int8, 2, 3, 2))
	require.Equal(t, [][][]int8{{{1, 3}, {1, 3}, {1, 3}}, {{1, 3}, {1, 3}, {1, 3}}}, y0.Value())
}

func TestExecSpecialOps_BroadcastInDim(t *testing.T) {
	y0 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ExpandAndBroadcast(x, []int{2, 3, 2}, []int{0})
	}, [][]int8{{1, 3}})
	// fmt.Printf("\ty0=%s\n", y0.GoStr())
	assert.NoError(t, y0.Shape().Check(dtypes.Int8, 2, 3, 2))
	assert.Equal(t, [][][]int8{{{1, 3}, {1, 3}, {1, 3}}, {{1, 3}, {1, 3}, {1, 3}}}, y0.Value())

	y1 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ExpandAndBroadcast(x, []int{2}, []int{0})
	}, bf16(42))
	// fmt.Printf("\ty1=%s\n", y1.GoStr())
	assert.NoError(t, y1.Shape().Check(dtypes.BFloat16, 2))
	assert.Equal(t, []bfloat16.BFloat16{bf16(42), bf16(42)}, y1.Value())
}

func TestExecSpecialOps_gatherIterator(t *testing.T) {
	operandShape := shapes.Make(dtypes.F32, 4, 3, 2, 2)
	startIndicesShape := shapes.Make(dtypes.Int8, 3, 3, 2)
	startVectorAxis := 1
	offsetOutputAxes := []int{1, 3}
	collapsedSliceAxes := []int{0, 2}
	startIndexMap := []int{0, 2, 3}
	sliceSizes := []int{1, 3, 1, 1}
	outputShape, err := shapeinference.Gather(operandShape, startIndicesShape, startVectorAxis,
		offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes, false)
	require.NoError(t, err)
	// fmt.Printf("\toutputShape=%s\n", outputShape)
	require.NoError(t, outputShape.Check(dtypes.F32, 3, 3, 2, 1))
	it := newGatherIterator(startIndicesShape, startVectorAxis, outputShape, offsetOutputAxes)
	var gotStartIndices [][]int
	var gotOutputIndices []int
	indices := make([]int, 3)
	var outputBytesIdx int
	for it.Next(indices, &outputBytesIdx) {
		gotStartIndices = append(gotStartIndices, slices.Clone(indices))
		gotOutputIndices = append(gotOutputIndices, outputBytesIdx)
	}
	// fmt.Printf("\tgatherStartIndicesIterator got startIndices=%#v\n", gotStartIndices)
	// fmt.Printf("\tgatherStartIndicesIterator got outputBytesIndices=%#v\n", gotOutputIndices)
	wantStartIndirectIndices := [][]int{{0, 2, 4}, {1, 3, 5}, {6, 8, 10}, {7, 9, 11}, {12, 14, 16}, {13, 15, 17}}
	assert.Equal(t, wantStartIndirectIndices, gotStartIndices)
	dataSize := operandShape.DType.Size() // == 4 for Float32
	wantOutputFlatIndices := []int{0, 1, 6, 7, 12, 13}
	for ii := range wantOutputFlatIndices {
		wantOutputFlatIndices[ii] *= dataSize
	}
	assert.Equal(t, wantOutputFlatIndices, gotOutputIndices)
}

func TestExecSpecialOps_Gather(t *testing.T) {
	y0 := graph.MustExecOnce(backend, func(g *graph.Graph) *graph.Node {
		operand := graph.IotaFull(g, shapes.Make(dtypes.F32, 4, 3, 2, 2))
		startIndices := graph.Const(g, [][][]int{{{0, 1}, {0, 1}, {0, 1}}, {{0, 0}, {0, 0}, {1, 1}}, {{0, 0}, {1, 1}, {0, 0}}})
		startVectorAxis := 1
		// fmt.Printf("\tstartIndices.shape=%s, startVectorAxis=%d\n", startIndices.Shape(), startVectorAxis)
		offsetOutputAxes := []int{1, 3}
		collapsedSliceAxes := []int{0, 2}
		startIndexMap := []int{0, 2, 3}
		sliceSizes := []int{1, 3, 1, 1}
		return graph.BackendGather(operand, startIndices, startVectorAxis, offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes, false)
	})
	// fmt.Printf("\ty0=%s\n", y0.GoStr())
	want := [][][][]float32{
		{{{0}, {15}}, {{4}, {19}}, {{8}, {23}}},
		{{{1}, {1}}, {{5}, {5}}, {{9}, {9}}},
		{{{2}, {2}}, {{6}, {6}}, {{10}, {10}}}}
	require.Equal(t, want, y0.Value())
}

func TestExecSpecialOps_Concatenate(t *testing.T) {
	// Test Case 1: Concatenating vectors (rank 1) along axis 0
	y1 := graph.MustExecOnce(backend, func(g *graph.Graph) *graph.Node {
		in1 := graph.Const(g, []float32{1, 2, 3})
		in2 := graph.Const(g, []float32{4, 5})
		return graph.Concatenate([]*graph.Node{in1, in2}, 0)
	})
	// fmt.Printf("\ty1=%s\n", y1.GoStr())
	want1 := []float32{1, 2, 3, 4, 5}
	require.NoError(t, y1.Shape().Check(dtypes.Float32, 5))
	require.Equal(t, want1, y1.Value())

	// Test Case 2: Concatenating matrices (rank 2) along axis 0
	y2 := graph.MustExecOnce(backend, func(g *graph.Graph) *graph.Node {
		in1 := graph.Const(g, [][]int8{{1, 2}, {3, 4}})
		in2 := graph.Const(g, [][]int8{{5, 6}})
		return graph.Concatenate([]*graph.Node{in1, in2}, 0)
	})
	// fmt.Printf("\ty2=%s\n", y2.GoStr())
	want2 := [][]int8{{1, 2}, {3, 4}, {5, 6}}
	require.NoError(t, y2.Shape().Check(dtypes.Int8, 3, 2))
	require.Equal(t, want2, y2.Value())

	// Test Case 3: Concatenating matrices (rank 2) along axis 1
	y3 := graph.MustExecOnce(backend, func(g *graph.Graph) *graph.Node {
		in1 := graph.Const(g, [][]bfloat16.BFloat16{{bf16(1)}, {bf16(2)}})
		in2 := graph.Const(g, [][]bfloat16.BFloat16{{bf16(3), bf16(4)}, {bf16(5), bf16(6)}})
		in3 := graph.Const(g, [][]bfloat16.BFloat16{{bf16(7)}, {bf16(8)}})
		return graph.Concatenate([]*graph.Node{in1, in2, in3}, 1)
	})
	// fmt.Printf("\ty3=%s\n", y3.GoStr())
	want3 := [][]bfloat16.BFloat16{{bf16(1), bf16(3), bf16(4), bf16(7)}, {bf16(2), bf16(5), bf16(6), bf16(8)}}
	require.NoError(t, y3.Shape().Check(dtypes.BFloat16, 2, 4))
	require.Equal(t, want3, y3.Value())

	// Test Case 4: Concatenating rank 3 tensors along axis 1
	y4 := graph.MustExecOnce(backend, func(g *graph.Graph) *graph.Node {
		in1 := graph.Const(g, [][][]int32{{{1, 2}}, {{3, 4}}})                    // Shape (2, 1, 2)
		in2 := graph.Const(g, [][][]int32{{{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}}) // Shape (2, 2, 2)
		return graph.Concatenate([]*graph.Node{in1, in2}, 1)
	})
	// fmt.Printf("\ty4=%s\n", y4.GoStr())
	want4 := [][][]int32{{{1, 2}, {5, 6}, {7, 8}}, {{3, 4}, {9, 10}, {11, 12}}} // Shape (2, 3, 2)
	require.NoError(t, y4.Shape().Check(dtypes.Int32, 2, 3, 2))
	require.Equal(t, want4, y4.Value())

	// Test Case 5: Concatenating rank 3 tensors along axis 2
	y5 := graph.MustExecOnce(backend, func(g *graph.Graph) *graph.Node {
		in1 := graph.Const(g, [][][]float64{{{1, 2}}, {{3, 4}}}) // Shape (2, 1, 2)
		in2 := graph.Const(g, [][][]float64{{{5}}, {{6}}})       // Shape (2, 1, 1)
		return graph.Concatenate([]*graph.Node{in1, in2}, 2)
	})
	// fmt.Printf("\ty5=%s\n", y5.GoStr())
	want5 := [][][]float64{{{1, 2, 5}}, {{3, 4, 6}}} // Shape (2, 1, 3)
	require.NoError(t, y5.Shape().Check(dtypes.Float64, 2, 1, 3))
	require.Equal(t, want5, y5.Value())
}

func TestExecSpecialOps_ConvertDType(t *testing.T) {
	// Test int32 to float32
	y0 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ConvertDType(x, dtypes.Float32)
	}, int32(42))
	// fmt.Printf("\ty0=%s\n", y0.GoStr())
	assert.Equal(t, float32(42.0), y0.Value())

	// Test float32 to bfloat16
	y1 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ConvertDType(x, dtypes.BFloat16)
	}, float32(3.14))
	// fmt.Printf("\ty1=%s\n", y1.GoStr())
	assert.Equal(t, bf16(3.14), y1.Value())

	// Test bfloat16 to int32
	y2 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ConvertDType(x, dtypes.Int32)
	}, bf16(7.8))
	// fmt.Printf("\ty2=%s\n", y2.GoStr())
	assert.Equal(t, int32(7), y2.Value())

	// Test bool to int32
	y3 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ConvertDType(x, dtypes.Int32)
	}, true)
	// fmt.Printf("\ty3=%s\n", y3.GoStr())
	assert.Equal(t, int32(1), y3.Value())

	// Test float32 to bool
	y4 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ConvertDType(x, dtypes.Bool)
	}, float32(1.0))
	// fmt.Printf("\ty4=%s\n", y4.GoStr())
	assert.Equal(t, true, y4.Value())
}

func TestExecSpecialOps_Scatter(t *testing.T) {
	// Case 0: Typical scatter, except updates window is the first axis (usually it's the last)
	y0 := graph.MustExecOnce(backend, func(g *graph.Graph) *graph.Node {
		operand := graph.Zeros(g, MS(F32, 2, 2, 5))
		indices := graph.Const(g, [][]uint8{{0, 1}, {1, 0}})
		// updates: we use an unconventional update window in axis 0, and the batch axis 1.
		updates := graph.OnePlus(graph.IotaFull(g, MS(F32, 5, 2)))

		indexVectorAxis := 1
		updateWindowAxes := []int{0}
		insertedWindowAxes := []int{0, 1}
		scatterAxesToOperandAxes := []int{0, 1}
		return graph.BackendScatterMax(operand, indices, updates, indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes, true, true)
	})
	// fmt.Printf("\ty0=%s\n", y0.GoStr())
	want := [][][]float32{{{0, 0, 0, 0, 0}, {1, 3, 5, 7, 9}}, {{2, 4, 6, 8, 10}, {0, 0, 0, 0, 0}}}
	assert.Equal(t, want, y0.Value())

	// Case 1: operand axes shuffled; Operand initialized with ones instead.
	y1 := graph.MustExecOnce(backend, func(g *graph.Graph) *graph.Node {
		operand := graph.Ones(g, MS(F32, 2, 5, 2))
		indices := graph.Const(g, [][]uint8{{0, 1}, {1, 0}})
		// updates: we use an unconventional update window in axis 0, and the batch axis 1.
		updates := graph.OnePlus(graph.IotaFull(g, MS(F32, 5, 2)))
		indexVectorAxis := 1
		updateWindowAxes := []int{0}
		insertedWindowAxes := []int{0, 2}
		scatterAxesToOperandAxes := []int{0, 2}
		return graph.BackendScatterSum(operand, indices, updates, indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes, true, true)
	})
	// fmt.Printf("\ty1=%s\n", y1.GoStr())
	want = [][][]float32{{{1, 2}, {1, 4}, {1, 6}, {1, 8}, {1, 10}}, {{3, 1}, {5, 1}, {7, 1}, {9, 1}, {11, 1}}}
	assert.Equal(t, want, y1.Value())

	// Case 2: multi-dimension updates.
	y2 := graph.MustExecOnce(backend, func(g *graph.Graph) *graph.Node {
		operand := graph.Ones(g, MS(dtypes.BFloat16, 2, 3, 2))
		indices := graph.Const(g, [][]uint8{{0, 1}, {1, 0}})
		updates := graph.AddScalar(graph.IotaFull(g, MS(dtypes.BFloat16, 2, 2, 2)), -4)
		indexVectorAxis := 1
		updateWindowAxes := []int{1, 2}
		insertedWindowAxes := []int{0}
		scatterAxesToOperandAxes := []int{0, 1}
		return graph.BackendScatterMin(operand, indices, updates, indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes, true, true)
	})
	// fmt.Printf("\ty2=%s\n", y2.GoStr())
	want2 := [][][]bfloat16.BFloat16{{{bf16(1), bf16(1)}, {bf16(-4), bf16(-3)}, {bf16(-2), bf16(-1)}}, {{bf16(0), bf16(1)}, {bf16(1), bf16(1)}, {bf16(1), bf16(1)}}}
	assert.Equal(t, want2, y2.Value())
}

func rawSlice(operand *graph.Node, starts []int, limits []int, strides []int) *graph.Node {
	rank := operand.Shape().Rank()
	axisSpecs := make([]graph.SliceAxisSpec, rank)
	for axis := range rank {
		axisSpecs[axis] = graph.SliceAxisSpec{
			Start:       starts[axis],
			End:         limits[axis],
			StrideValue: strides[axis],
		}
	}
	return graph.Slice(operand, axisSpecs...)
}

func TestExecSpecialOps_Slice(t *testing.T) {
	// Test Case 1: Simple 1D slice
	y1 := graph.MustExecOnce(backend, func(g *graph.Graph) *graph.Node {
		operand := graph.Const(g, []int64{0, 1, 2, 3, 4}) // Shape [5]
		starts := []int{1}
		limits := []int{4} // Exclusive limit: indices 1, 2, 3
		strides := []int{1}
		// graph.Slice uses inclusive limits by default? Let's use SliceWithStride for clarity matching XLA Slice.
		// Assuming rawSlice maps to the backend op.
		// If graph.Slice takes end indices (inclusive) or sizes, adjust accordingly.
		return rawSlice(operand, starts, limits, strides)
	})
	// fmt.Printf("\ty1=%s\n", y1.GoStr())
	want1 := []int64{1, 2, 3}
	require.NoError(t, y1.Shape().Check(dtypes.Int64, 3)) // Default int is int64? Assuming so. Adjust if it's int32.
	require.Equal(t, want1, y1.Value())

	// Test Case 2: 2D slice with stride > 1
	y2 := graph.MustExecOnce(backend, func(g *graph.Graph) *graph.Node {
		operand := graph.Const(g, [][]int32{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}) // Shape [3, 3]
		starts := []int{0, 0}
		limits := []int{3, 3} // Exclusive limits for indices 0, 1, 2 in both axes
		strides := []int{2, 2}
		// Output shape: ceil((3-0)/2)=2, ceil((3-0)/2)=2 => [2, 2]
		// Values from indices: [0,0], [0,2], [2,0], [2,2]
		return rawSlice(operand, starts, limits, strides)
	})
	// fmt.Printf("\ty2=%s\n", y2.GoStr())
	want2 := [][]int32{{0, 2}, {6, 8}}
	require.NoError(t, y2.Shape().Check(dtypes.Int32, 2, 2))
	require.Equal(t, want2, y2.Value())

	// Test Case 3: Slice resulting in a rank-2 tensor with size 1x1
	y3 := graph.MustExecOnce(backend, func(g *graph.Graph) *graph.Node {
		operand := graph.Const(g, [][]int64{{0, 1}, {2, 3}}) // Shape [2, 2]
		starts := []int{1, 1}
		limits := []int{2, 2} // Exclusive limits for index 1 in both axes
		strides := []int{1, 1}
		// Output shape: ceil((2-1)/1)=1, ceil((2-1)/1)=1 => [1, 1]
		// Value from index: [1, 1]
		return rawSlice(operand, starts, limits, strides)
	})
	// fmt.Printf("\ty3=%s\n", y3.GoStr())
	want3 := [][]int64{{3}}                                  // Assuming int is int64
	require.NoError(t, y3.Shape().Check(dtypes.Int64, 1, 1)) // Adjust dtype if needed
	require.Equal(t, want3, y3.Value())

	// Test Case 4: Slice with bfloat16 and stride > 1
	y4 := graph.MustExecOnce(backend, func(g *graph.Graph) *graph.Node {
		operand := graph.Const(g, []bfloat16.BFloat16{bf16(0), bf16(1), bf16(2), bf16(3)}) // Shape [4]
		starts := []int{1}
		limits := []int{4} // Exclusive limit: indices 1, 2, 3
		strides := []int{2}
		// Output shape: ceil((4-1)/2)=ceil(1.5)=2 => [2]
		// Values from indices: 1, 3
		return rawSlice(operand, starts, limits, strides)
	})
	// fmt.Printf("\ty4=%s\n", y4.GoStr())
	want4 := []bfloat16.BFloat16{bf16(1), bf16(3)}
	require.NoError(t, y4.Shape().Check(dtypes.BFloat16, 2))
	require.Equal(t, want4, y4.Value())
}

func computeHistogram(values []float64, numBins int) []int {
	if len(values) == 0 {
		return nil
	}
	sort.Float64s(values)
	min, max := values[0], values[len(values)-1]
	binSize := (max - min) / float64(numBins)
	histogram := make([]int, numBins)
	for _, v := range values {
		bin := int((v - min) / binSize)
		if bin == numBins {
			bin--
		}
		histogram[bin]++
	}
	return histogram
}

func TestExecSpecialOps_RngBitsGenerator(t *testing.T) {
	const numSamples = 1000
	const numBins = 10
	const tolerance = 0.6 // Allow 60% deviation from the expected frequency

	testCases := []struct {
		dtype dtypes.DType
		name  string
	}{
		{dtypes.Float32, "float32"},
		{dtypes.Float64, "float64"},
		{dtypes.BFloat16, "bfloat16"},
	}

	state, err := graph.RNGState()
	require.NoError(t, err)
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			shape := shapes.Make(tc.dtype, numSamples)
			outputs := graph.MustExecOnceN(backend, func(state *graph.Node) []*graph.Node {
				var values *graph.Node
				state, values = graph.RandomUniform(state, shape)
				return []*graph.Node{state, values}
			}, state)
			// fmt.Printf("\toutput.shape=%s\n", shape)
			state = outputs[0]
			y := outputs[1]

			// Convert all values to float64 for histogram computation
			values := make([]float64, numSamples)
			switch tc.dtype {
			case dtypes.Float32:
				for i, v := range y.Value().([]float32) {
					values[i] = float64(v)
				}
			case dtypes.Float64:
				values = y.Value().([]float64)
			case dtypes.BFloat16:
				for i, v := range y.Value().([]bfloat16.BFloat16) {
					values[i] = float64(v.Float32())
				}
			}

			hist := computeHistogram(values, numBins)
			// fmt.Printf("\tshape=%s, hist=%v\n", shape, hist)
			expectedPerBin := numSamples / numBins
			maxDeviation := float64(expectedPerBin) * tolerance

			// Check each bin is within tolerance of expected frequency
			for bin, count := range hist {
				deviation := math.Abs(float64(count) - float64(expectedPerBin))
				if deviation > maxDeviation {
					t.Errorf("Bin %d count %d deviates too much from expected %d (deviation: %.2f > %.2f)",
						bin, count, expectedPerBin, deviation, maxDeviation)
				}
			}
		})
	}
}

func TestExecSpecialOps_ArgMinMaxOp(t *testing.T) {
	// Test Case 1: Simple 1D argmin
	y0 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ArgMin(x, 0)
	}, []float32{3, 1, 4, 1, 5})
	// fmt.Printf("\ty0=%s\n", y0.GoStr())
	require.Equal(t, int32(1), y0.Value())

	// Test Case 2: 2D argmax along axis 1 (columns)
	y1 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ArgMax(x, 1)
	}, [][]int32{{1, 2, 3}, {4, 1, 2}, {7, 8, 5}})
	// fmt.Printf("\ty1=%s\n", y1.GoStr())
	require.Equal(t, []int32{2, 0, 1}, y1.Value())

	// Test Case 3: 2D argmin along axis 0 (rows) with BFloat16
	y2 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ArgMin(x, 0)
	}, [][]bfloat16.BFloat16{
		{bf16(1), bf16(2)},
		{bf16(-1), bf16(3)},
		{bf16(4), bf16(-2)}})
	// fmt.Printf("\ty2=%s\n", y2.GoStr())
	require.Equal(t, []int32{1, 2}, y2.Value())

	// Test Case 4: 3D argmax with repeated values
	y3 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ArgMax(x, 1)
	}, [][][]float32{
		{{1, 2}, {1, 0}, {1, -1}},
		{{4, 3}, {4, 5}, {4, 2}}})
	// fmt.Printf("\ty3=%s\n", y3.GoStr())
	require.Equal(t, [][]int32{{0, 0}, {0, 1}}, y3.Value())
}

// =================================================================================================================
// ReduceWindow ----------------------------------------------------------------------------------------------------
// =================================================================================================================

func dtypeForSlice(slice any) dtypes.DType {
	t := reflect.TypeOf(slice)
	for t.Kind() == reflect.Slice {
		t = t.Elem()
	}
	return dtypes.FromGoType(t)
}

// Test case structure for ReduceWindow tests.
type reduceWindowGraphTestCase struct { // T is the Go type for data, e.g., float32, []float32
	name string
	// operandData will be the third argument to graph.MustExecOnce (inputs ...any)
	// graph.MustExecOnce infers shape and dtype from this.
	// If specific dtype/shape control is needed beyond inference, it's more complex.
	// For now, assume operandData's type and structure define the input tensor.
	operandData      any // e.g., []float32{1,2,3,4,5} or [][]int32{{1,2},{3,4}}
	reductionType    backends.ReduceOpType
	windowDimensions []int
	strides          []int    // Can be nil, graph.BackendReduceWindow should handle defaults.
	paddings         [][2]int // Can be nil.
	baseDilations    []int    // Can be nil.
	windowDilations  []int    // Can be nil.
	expectedOutput   any      // e.g., []float32{3,5,7,9}
	expectedShape    []int    // For verifying output shape explicitly
}

func TestExecSpecialOps_ReduceWindow(t *testing.T) { // Renamed for common Go test naming, or use user's preference
	// Helper to create BFloat16 slices for test cases
	bf16Values := func(vals ...float32) []bfloat16.BFloat16 {
		res := make([]bfloat16.BFloat16, len(vals))
		for i, v := range vals {
			res[i] = bfloat16.FromFloat32(v)
		}
		return res
	}

	// --- Test Cases for Float32 ---
	for _, tc := range []reduceWindowGraphTestCase{
		{
			name:             "F32_1D_Sum_Win2_Stride1_DefaultPadDil",
			operandData:      []float32{1, 2, 3, 4, 5},
			reductionType:    backends.ReduceOpSum,
			windowDimensions: []int{2},
			strides:          []int{1},
			// Nil for paddings, baseDilations, windowDilations will use graph.BackendReduceWindow defaults
			expectedOutput: []float32{3, 5, 7, 9},
			expectedShape:  []int{4},
		},
		{
			name:             "F32_1D_Product_Win2_Stride2_Pad1_1",
			operandData:      []float32{1, 2, 3, 4},
			reductionType:    backends.ReduceOpProduct,
			windowDimensions: []int{2},
			strides:          []int{2},
			paddings:         [][2]int{{1, 1}},
			// Calculation for expectedOutput:
			// Input: [1,2,3,4], Shape [4], DType F32
			// Window [2], Stride [2], Padding {{1,1}}
			// Shape inference: (InputDim + PadLow + PadHigh - WindowDim) / Stride + 1
			// (4 + 1 + 1 - 2) / 2 + 1 = (6 - 2) / 2 + 1 = 4 / 2 + 1 = 2 + 1 = 3. Output Shape [3]
			// Output[0]: input indices for window at output_idx 0: (0*stride - PadLow) to (0*stride - PadLow + WindowDim -1)
			// (0*2 - 1) = -1 to (0*2 - 1 + 2 -1) = 0. Indices: -1, 0. Valid: input[0]=1. Product=1 (init_val for padding/empty assumed 1 for product)
			// Output[1]: input indices for window at output_idx 1: (1*2 - 1) = 1 to (1*2 - 1 + 2 - 1) = 2. Indices: 1, 2. Valid: input[1]=2, input[2]=3. Prod=2*3=6.
			// Output[2]: input indices for window at output_idx 2: (2*2 - 1) = 3 to (2*2 - 1 + 2 - 1) = 4. Indices: 3, 4. Valid: input[3]=4. Prod=4.
			expectedOutput: []float32{1, 6, 4},
			expectedShape:  []int{3},
		},
		{
			name:             "F32_1D_Max_Win3_WindowDilation2",
			operandData:      []float32{1, 2, 3, 4, 5, 6, 7},
			reductionType:    backends.ReduceOpMax,
			windowDimensions: []int{3},
			strides:          []int{1},
			windowDilations:  []int{2}, // Effective window elements indices: k, k+2, k+4 related to input
			// Effective window span (DilatedWindowDim): (3-1)*2+1 = 5
			// Output shape: (7 - 5)/1 + 1 = 3.
			// Out[0]: input indices 0, 0+1*WinDil=2, 0+2*WinDil=4. Max(data[0], data[2], data[4]) = Max(1,3,5) = 5.
			// Out[1]: input indices 1, 1+1*WinDil=3, 1+2*WinDil=5. Max(data[1], data[3], data[5]) = Max(2,4,6) = 6.
			// Out[2]: input indices 2, 2+1*WinDil=4, 2+2*WinDil=6. Max(data[2], data[4], data[6]) = Max(3,5,7) = 7.
			expectedOutput: []float32{5, 6, 7},
			expectedShape:  []int{3},
		},
		{
			name:             "F32_2D_Sum_NoPadDilStride1",
			operandData:      [][]float32{{1, 2, 3}, {4, 5, 6}}, // Shape [2,3]
			reductionType:    backends.ReduceOpSum,
			windowDimensions: []int{2, 2},
			strides:          []int{1, 1},
			// Output shape: Dim0: (2-2)/1+1 = 1. Dim1: (3-2)/1+1 = 2. Shape [1,2]
			// Out[0,0]: sum of input[0:2, 0:2] = 1+2+4+5 = 12
			// Out[0,1]: sum of input[0:2, 1:3] = 2+3+5+6 = 16
			expectedOutput: [][]float32{{12, 16}},
			expectedShape:  []int{1, 2},
		},
		{
			name:             "I32_1D_Min_Win3_Stride2_BaseDil2",
			operandData:      []int32{10, 2, 5, 1, 8, 3, 9, 4}, // Shape [8]
			reductionType:    backends.ReduceOpMin,
			windowDimensions: []int{3},
			strides:          []int{2}, // Stride in the conceptually base-dilated input
			baseDilations:    []int{2}, // Conceptual input len (8-1)*2+1 = 15. Data: 10 H 2 H 5 H 1 H 8 H 3 H 9 H 4
			// Window takes 3 elements from conceptual input. EffWin=3.
			// Output shape on conceptual input (len 15): (15-3)/2+1 = 12/2+1=7.
			expectedOutput: []int32{2, 2, 1, 1, 3, 3, 4},
			expectedShape:  []int{7},
		},
		{
			name:             "I32_2D_Max",
			operandData:      [][]int32{{1, 5, 2}, {6, 3, 7}, {4, 9, 0}}, // Shape [3,3]
			reductionType:    backends.ReduceOpMax,
			windowDimensions: []int{2, 2},
			strides:          []int{1, 1},
			paddings:         [][2]int{{0, 1}, {1, 0}},
			expectedOutput:   [][]int32{{6, 6, 7}, {6, 9, 9}, {4, 9, 9}},
			expectedShape:    []int{3, 3},
		}, {
			name:             "I32_2D_Max_Win2x2_Stride1x1_NoPadDil",
			operandData:      [][]int32{{1, 2, 3}, {4, 5, 6}},
			reductionType:    backends.ReduceOpMax,
			windowDimensions: []int{2, 2},
			strides:          []int{1, 1},
			expectedOutput:   [][]int32{{5, 6}},
			expectedShape:    []int{1, 2},
		},
		{
			name:             "BF16_1D_Sum_Win2_NoParams",
			operandData:      bf16Values(1, 2, 3, 4), // Input as []bfloat16.Type
			reductionType:    backends.ReduceOpSum,
			windowDimensions: []int{2},
			strides:          []int{1},            // graph.ReduceWindow likely requires explicit strides
			expectedOutput:   bf16Values(3, 5, 7), // 1+2, 2+3, 3+4
			expectedShape:    []int{3},
		},
		{
			name:             "BF16_1D_Product_Win2_BaseDil2_Pad1",
			operandData:      bf16Values(2, 3, 4), // Shape [3]
			reductionType:    backends.ReduceOpProduct,
			windowDimensions: []int{2},
			strides:          []int{1},
			paddings:         [][2]int{{1, 0}}, // Pad low by 1
			baseDilations:    []int{2},         // Conceptual input: [2 H 3 H 4] (len 5). Padded: [PadVal 2 H 3 H 4]
			// Output shape on conceptual (len 5) with padding (1,0): (5+1+0 - 2)/1 + 1 = (6-2)/1+1 = 5
			// Assuming PadVal=1 for product identity if outside region
			// Out[0]: win over conceptual_padded indices [0,1] -> maps to input[0]=2 (via conceptual[1]). Product=2.
			// Out[1]: win over conceptual_padded indices [1,2] -> maps to input[0]=2 (via conceptual[1]), hole (via conceptual[2]). Product=2.
			// Out[2]: win over conceptual_padded indices [2,3] -> maps to input[1]=3 (via conceptual[3]), hole. Product=3.
			// Out[3]: win over conceptual_padded indices [3,4] -> maps to input[1]=3 (via conceptual[3]), hole. Product=3.
			// Out[4]: win over conceptual_padded indices [4,5] -> maps to input[2]=4 (via conceptual[5]), hole. Product=4.
			expectedOutput: bf16Values(2, 2, 3, 3, 4),
			expectedShape:  []int{5},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			y := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
				return graph.BackendReduceWindow(
					x, tc.reductionType,
					tc.windowDimensions, tc.strides, tc.baseDilations, tc.windowDilations,
					tc.paddings)
			}, tc.operandData)
			dtype := dtypeForSlice(tc.operandData)
			require.Equalf(t, dtype, y.DType(), "Unexpected dtype %s for test %q: wanted %s", y.DType(), tc.name, dtype)
			require.NoErrorf(t, y.Shape().CheckDims(tc.expectedShape...), "Got unexpected shape %s for %q: wanted %s", y.Shape(), tc.name, tc.expectedShape)
			require.Equal(t, tc.expectedOutput, y.Value(),
				"ReduceWindow: test %q: expected %v, got %v", tc.name, tc.expectedOutput, y.GoStr())
		})
	}
}
