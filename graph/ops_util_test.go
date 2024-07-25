package graph_test

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"math"
	"testing"
)

func TestEinsum(t *testing.T) {
	graphtest.RunTestGraphFn(t, "EinsumMatrixMul",
		func(g *Graph) (inputs, outputs []*Node) {
			lhs := IotaFull(g, shapes.Make(dtypes.Float32, 2, 4))
			lhs = OnePlus(lhs)
			rhs := MulScalar(Ones(g, shapes.Make(dtypes.Float32, 4, 3)), 0.1)
			inputs = []*Node{lhs, rhs}
			outputs = []*Node{Einsum("ij,jk->ik", lhs, rhs)}
			return
		}, []any{[][]float32{{1, 1, 1}, {2.6, 2.6, 2.6}}}, Epsilon)
	graphtest.RunTestGraphFn(t, "EinsumDotProduct",
		func(g *Graph) (inputs, outputs []*Node) {
			lhs := IotaFull(g, shapes.Make(dtypes.Float32, 4))
			lhs = OnePlus(lhs)
			rhs := MulScalar(Ones(g, shapes.Make(dtypes.Float32, 4)), 0.1)
			inputs = []*Node{lhs, rhs}
			outputs = []*Node{Einsum("i,i->", lhs, rhs)}
			return
		}, []any{float32(1)}, Epsilon)
	graphtest.RunTestGraphFn(t, "EinsumOuterProduct",
		func(g *Graph) (inputs, outputs []*Node) {
			lhs := OnePlus(IotaFull(g, shapes.Make(dtypes.Float32, 4)))
			rhs := OnePlus(IotaFull(g, shapes.Make(dtypes.Float32, 3)))
			inputs = []*Node{lhs, rhs}
			outputs = []*Node{Einsum("i,j->ij", lhs, rhs)}
			return
		}, []any{[][]float32{{1, 2, 3}, {2, 4, 6}, {3, 6, 9}, {4, 8, 12}}}, Epsilon)
	graphtest.RunTestGraphFn(t, "EinsumBatchMatrixMul",
		func(g *Graph) (inputs, outputs []*Node) {
			lhs := IotaFull(g, shapes.Make(dtypes.Float32, 5, 2, 4))
			lhs = OnePlus(lhs)
			rhs := MulScalar(Ones(g, shapes.Make(dtypes.Float32, 5, 4, 3)), 0.1)
			inputs = []*Node{lhs, rhs}
			outputs = []*Node{Einsum("bij,bjk->bik", lhs, rhs)}
			return
		}, []any{[][][]float32{
			{{1, 1, 1}, {2.6, 2.6, 2.6}},
			{{4.2, 4.2, 4.2}, {5.8, 5.8, 5.8}},
			{{7.4, 7.4, 7.4}, {9, 9, 9}},
			{{10.6, 10.6, 10.6}, {12.2, 12.2, 12.2}},
			{{13.8, 13.8, 13.8}, {15.4, 15.4, 15.4}}},
		}, Epsilon)
	graphtest.RunTestGraphFn(t, "EinsumBatchMatrixMulTransposed",
		func(g *Graph) (inputs, outputs []*Node) {
			lhs := IotaFull(g, shapes.Make(dtypes.Float32, 5, 2, 4))
			lhs = OnePlus(lhs)
			rhs := MulScalar(Ones(g, shapes.Make(dtypes.Float32, 5, 4, 3)), 0.1)
			inputs = []*Node{lhs, rhs}
			outputs = []*Node{Einsum("bij,bjk->ikb", lhs, rhs)}
			return
		}, []any{[][][]float32{
			{{1, 4.2, 7.4, 10.6, 13.8},
				{1, 4.2, 7.4, 10.6, 13.8},
				{1, 4.2, 7.4, 10.6, 13.8}},
			{{2.6, 5.8, 9, 12.2, 15.4},
				{2.6, 5.8, 9, 12.2, 15.4},
				{2.6, 5.8, 9, 12.2, 15.4}},
		}}, Epsilon)
}

func TestEinsumAxes(t *testing.T) {
	graphtest.RunTestGraphFn(t, "EinsumAxesMatrixMul",
		func(g *Graph) (inputs, outputs []*Node) {
			lhs := IotaFull(g, shapes.Make(dtypes.Float32, 2, 4))
			lhs = OnePlus(lhs)
			rhs := MulScalar(Ones(g, shapes.Make(dtypes.Float32, 4, 3)), 0.1)
			inputs = []*Node{lhs, rhs}
			outputs = []*Node{EinsumAxes(lhs, rhs, [][2]int{{1, 0}}, nil)}
			return
		}, []any{[][]float32{{1, 1, 1}, {2.6, 2.6, 2.6}}}, Epsilon)
	graphtest.RunTestGraphFn(t, "EinsumAxesDotProduct",
		func(g *Graph) (inputs, outputs []*Node) {
			lhs := IotaFull(g, shapes.Make(dtypes.Float32, 4))
			lhs = OnePlus(lhs)
			rhs := MulScalar(Ones(g, shapes.Make(dtypes.Float32, 4)), 0.1)
			inputs = []*Node{lhs, rhs}
			outputs = []*Node{EinsumAxes(lhs, rhs, [][2]int{{0, 0}}, nil)}
			return
		}, []any{float32(1)}, Epsilon)
	graphtest.RunTestGraphFn(t, "EinsumAxesOuterProduct",
		func(g *Graph) (inputs, outputs []*Node) {
			lhs := OnePlus(IotaFull(g, shapes.Make(dtypes.Float32, 4)))
			rhs := OnePlus(IotaFull(g, shapes.Make(dtypes.Float32, 3)))
			inputs = []*Node{lhs, rhs}
			outputs = []*Node{EinsumAxes(lhs, rhs, nil, nil)}
			return
		}, []any{[][]float32{{1, 2, 3}, {2, 4, 6}, {3, 6, 9}, {4, 8, 12}}}, Epsilon)
	graphtest.RunTestGraphFn(t, "EinsumAxesBatchMatrixMul",
		func(g *Graph) (inputs, outputs []*Node) {
			lhs := IotaFull(g, shapes.Make(dtypes.Float32, 5, 2, 4))
			lhs = OnePlus(lhs)
			rhs := MulScalar(Ones(g, shapes.Make(dtypes.Float32, 5, 4, 3)), 0.1)
			inputs = []*Node{lhs, rhs}
			outputs = []*Node{EinsumAxes(lhs, rhs, [][2]int{{2, 1}}, [][2]int{{0, 0}})}
			return
		}, []any{[][][]float32{
			{{1, 1, 1}, {2.6, 2.6, 2.6}},
			{{4.2, 4.2, 4.2}, {5.8, 5.8, 5.8}},
			{{7.4, 7.4, 7.4}, {9, 9, 9}},
			{{10.6, 10.6, 10.6}, {12.2, 12.2, 12.2}},
			{{13.8, 13.8, 13.8}, {15.4, 15.4, 15.4}}},
		}, Epsilon)
}

func TestLowerTriangular(t *testing.T) {
	graphtest.RunTestGraphFn(t, "LowerTriangular(3)",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{}
			outputs = []*Node{LowerTriangular(g, 3)}
			return
		}, []any{[][]bool{
			{true, false, false},
			{true, true, false},
			{true, true, true},
		}}, -1)
}

func TestUpperTriangular(t *testing.T) {
	graphtest.RunTestGraphFn(t, "LowerTriangular(3)",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{}
			outputs = []*Node{UpperTriangular(g, 3)}
			return
		}, []any{[][]bool{
			{true, true, true},
			{false, true, true},
			{false, false, true},
		}}, -1)
}

func TestDiagonalWithValue(t *testing.T) {
	graphtest.RunTestGraphFn(t, "DiagonalWithValue(5, 3)",
		func(g *Graph) (inputs, outputs []*Node) {
			value := Const(g, 5.0)
			inputs = []*Node{value}
			outputs = []*Node{DiagonalWithValue(value, 3)}
			return
		}, []any{[][]float64{
			{5, 0, 0},
			{0, 5, 0},
			{0, 0, 5},
		}}, -1)
}

func TestClip(t *testing.T) {
	testFuncOneInput(t, "Clip({0, 3, 6}, 2, 4)",
		func(g *Graph) (input, output *Node) {
			input = Const(g, [][][]float32{{{0}, {3}, {6}}})
			output = Clip(input, Const(g, float32(2)), Const(g, float32(4)))
			return
		}, [][][]float32{{{2}, {3}, {4}}})
}

func TestNorms(t *testing.T) {
	testFuncOneInput(t, "L2NormSquare",
		func(g *Graph) (input, output *Node) {
			input = Const(g, [][]float32{{1, 2}, {-3, 4}})
			output = L2NormSquare(input)
			return
		}, float32(5+25))

	testFuncOneInput(t, "L2Norm",
		func(g *Graph) (input, output *Node) {
			input = Const(g, [][]float32{{4, -3}, {-5, 12}})
			output = L2Norm(input, -1)
			return
		}, [][]float32{{5}, {13}})

	testFuncOneInput(t, "L1Norm",
		func(g *Graph) (input, output *Node) {
			input = Const(g, [][]float32{{-4, 3}, {5, -12}})
			output = L1Norm(input, -1)
			return
		}, [][]float32{{7}, {17}})

	invSqrt2 := float32(1.0 / math.Sqrt(2.0))
	testFuncOneInput(t, "L2Normalize",
		func(g *Graph) (input, output *Node) {
			input = Const(g, [][]float32{{5, 5}, {711, 711}})
			output = L2Normalize(input, -1)
			return
		}, [][]float32{{invSqrt2, invSqrt2}, {invSqrt2, invSqrt2}})

	testFuncOneInput(t, "L2NormalizeWithEpsilon",
		func(g *Graph) (input, output *Node) {
			input = Const(g, [][]float32{{0, 0}, {11, 11}})
			output = L2NormalizeWithEpsilon(input, 1e-9, -1)
			return
		}, [][]float32{{0, 0}, {invSqrt2, invSqrt2}}) // Epsilon shouldn't create a large enough difference to fail the test.
}

func TestMirroredLog1p(t *testing.T) {
	testFuncOneInput(t, "MirroredLog1p",
		func(g *Graph) (input, output *Node) {
			input = Const(g, []float64{0.0, 1.0, math.E - 1.0, -1.0, -(math.E - 1.0)})
			output = MirroredLog1p(input)
			return
		}, []float64{0.0, math.Log1p(1.0), 1.0, -math.Log1p(1.0), -1})

	testGradients(t, "MirroredLog1p gradient",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := Const(g, []float32{0.0, 2, -3})
			output = MirroredLog1p(input)
			return output, []*Node{input}
		}, []any{[]float32{0.0, 1.0 / (2 + 1), 1.0 / (3 + 1)}})
}

func TestShiftWithScalar(t *testing.T) {
	testFuncOneInput(t, "ShiftWithScalar(input, axis=1, left, n=1, fill=0.0)",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(dtypes.Float32, 3, 2, 2))
			output = ShiftWithScalar(input, 1, ShiftDirLeft, 1, 0.0)
			return
		}, [][][]float32{
			{{2, 3}, {0, 0}},
			{{6, 7}, {0, 0}},
			{{10, 11}, {0, 0}},
		})
	testFuncOneInput(t, "ShiftWithScalar(input, axis=-1, left, n=1, fill=100)",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(dtypes.Int32, 3, 2, 2))
			output = ShiftWithScalar(input, -1, ShiftDirLeft, 1, 100)
			return
		}, [][][]int32{
			{{1, 100}, {3, 100}},
			{{5, 100}, {7, 100}},
			{{9, 100}, {11, 100}},
		})
	testFuncOneInput(t, "ShiftWithScalar(input, axis=0, right, n=2, fill=1.0)",
		func(g *Graph) (input, output *Node) {
			input = Zeros(g, shapes.Make(dtypes.Bool, 3, 2, 2))
			output = ShiftWithScalar(input, 0, ShiftDirRight, 2, 1)
			return
		}, [][][]bool{ // Inserted `true` to the left (shift-right) of the tensor:
			{{true, true}, {true, true}},
			{{true, true}, {true, true}},
			{{false, false}, {false, false}},
		})
	testFuncOneInput(t, "ShiftWithScalar(input, axis=0, right, n=3, fill=1)",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(dtypes.Float64, 10.0))
			output = ShiftWithScalar(input, -1, ShiftDirRight, 3, 1)
			return
		}, []float64{1, 1, 1, 0, 1, 2, 3, 4, 5, 6})
}

func TestShiftWithValue(t *testing.T) {
	testFuncOneInput(t, "ShiftWithScalar(input, axis=0, left, n=3, value=1000)",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(dtypes.Float32, 10))
			output = ShiftWithValue(input, 0, ShiftDirLeft, 3, Scalar(g, F32, 1000))
			return
		}, []float32{3, 4, 5, 6, 7, 8, 9, 1000, 1000, 1000})
	testFuncOneInput(t, "ShiftWithScalar(input, axis=-1, right, n=3, value=[[100], [1000]])",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(dtypes.Int32, 2, 10))
			output = ShiftWithValue(input, -1, ShiftDirRight, 3, Const(g, [][]int32{{100}, {1000}}))
			return
		}, [][]int32{
			{100, 100, 100, 0, 1, 2, 3, 4, 5, 6},
			{1000, 1000, 1000, 10, 11, 12, 13, 14, 15, 16},
		})
}

func TestShift(t *testing.T) {
	testFuncOneInput(t, "Shift(input, axis=0, left, n=3)",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(dtypes.Float32, 10))
			output = Shift(input, 0, ShiftDirLeft, 3)
			return
		}, []float32{3, 4, 5, 6, 7, 8, 9, 9, 9, 9})
	testFuncOneInput(t, "Shift(input, axis=-1, right, n=3)",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(dtypes.Int32, 2, 10))
			output = Shift(input, -1, ShiftDirRight, 3)
			return
		}, [][]int32{
			{0, 0, 0, 0, 1, 2, 3, 4, 5, 6},
			{10, 10, 10, 10, 11, 12, 13, 14, 15, 16},
		})
}

func TestOneHot(t *testing.T) {
	testFuncOneInput(t, "OneHot 1 leading dimension",
		func(g *Graph) (input, output *Node) {
			input = Const(g, []int{1, 0, 3})
			output = OneHot(input, 4, dtypes.Float32)
			return
		}, [][]float32{{0, 1, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}})
	testFuncOneInput(t, "OneHot 2 leading dimensions",
		func(g *Graph) (input, output *Node) {
			input = Const(g, [][][]int{ // shape [2, 3, 2]
				{{1, 0}, {0, 2}, {3, 1}},
				{{2, 3}, {3, 1}, {0, 2}},
			})
			output = OneHot(input, 4, dtypes.Float32)
			return
		}, [][][][]float32{
			{{{0, 1, 0, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {0, 0, 1, 0}}, {{0, 0, 0, 1}, {0, 1, 0, 0}}},
			{{{0, 0, 1, 0}, {0, 0, 0, 1}}, {{0, 0, 0, 1}, {0, 1, 0, 0}}, {{1, 0, 0, 0}, {0, 0, 1, 0}}},
		})
}

func TestGrow(t *testing.T) {
	testFuncOneInput(t, "GrowLeft",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(dtypes.Int32, 3, 2))
			output = GrowLeft(input, 0, 2, 111)
			return
		}, [][]int32{
			{111, 111},
			{111, 111},
			{0, 1},
			{2, 3},
			{4, 5},
		})
	testFuncOneInput(t, "GrowRight",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(dtypes.Float32, 3, 2))
			output = GrowRight(input, 1, 1, 111)
			return
		}, [][]float32{
			{0, 1, 111},
			{2, 3, 111},
			{4, 5, 111},
		})
}
