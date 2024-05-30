package graph_test

import (
	"math"
	"testing"

	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
)

func TestEinsum(t *testing.T) {
	graphtest.RunTestGraphFn(t, "EinsumMatrixMul",
		func(g *Graph) (inputs, outputs []*Node) {
			lhs := IotaFull(g, shapes.Make(shapes.F32, 2, 4))
			lhs = OnePlus(lhs)
			rhs := MulScalar(Ones(g, shapes.Make(shapes.F32, 4, 3)), 0.1)
			inputs = []*Node{lhs, rhs}
			outputs = []*Node{Einsum("ij,jk->ik", lhs, rhs)}
			return
		}, []any{[][]float32{{1, 1, 1}, {2.6, 2.6, 2.6}}}, slices.Epsilon)
	graphtest.RunTestGraphFn(t, "EinsumDotProduct",
		func(g *Graph) (inputs, outputs []*Node) {
			lhs := IotaFull(g, shapes.Make(shapes.F32, 4))
			lhs = OnePlus(lhs)
			rhs := MulScalar(Ones(g, shapes.Make(shapes.F32, 4)), 0.1)
			inputs = []*Node{lhs, rhs}
			outputs = []*Node{Einsum("i,i->", lhs, rhs)}
			return
		}, []any{float32(1)}, slices.Epsilon)
	graphtest.RunTestGraphFn(t, "EinsumOuterProduct",
		func(g *Graph) (inputs, outputs []*Node) {
			lhs := OnePlus(IotaFull(g, shapes.Make(shapes.F32, 4)))
			rhs := OnePlus(IotaFull(g, shapes.Make(shapes.F32, 3)))
			inputs = []*Node{lhs, rhs}
			outputs = []*Node{Einsum("i,j->ij", lhs, rhs)}
			return
		}, []any{[][]float32{{1, 2, 3}, {2, 4, 6}, {3, 6, 9}, {4, 8, 12}}}, slices.Epsilon)
	graphtest.RunTestGraphFn(t, "EinsumBatchMatrixMul",
		func(g *Graph) (inputs, outputs []*Node) {
			lhs := IotaFull(g, shapes.Make(shapes.F32, 5, 2, 4))
			lhs = OnePlus(lhs)
			rhs := MulScalar(Ones(g, shapes.Make(shapes.F32, 5, 4, 3)), 0.1)
			inputs = []*Node{lhs, rhs}
			outputs = []*Node{Einsum("bij,bjk->bik", lhs, rhs)}
			return
		}, []any{[][][]float32{
			{{1, 1, 1}, {2.6, 2.6, 2.6}},
			{{4.2, 4.2, 4.2}, {5.8, 5.8, 5.8}},
			{{7.4, 7.4, 7.4}, {9, 9, 9}},
			{{10.6, 10.6, 10.6}, {12.2, 12.2, 12.2}},
			{{13.8, 13.8, 13.8}, {15.4, 15.4, 15.4}}},
		}, slices.Epsilon)
	graphtest.RunTestGraphFn(t, "EinsumBatchMatrixMulTransposed",
		func(g *Graph) (inputs, outputs []*Node) {
			lhs := IotaFull(g, shapes.Make(shapes.F32, 5, 2, 4))
			lhs = OnePlus(lhs)
			rhs := MulScalar(Ones(g, shapes.Make(shapes.F32, 5, 4, 3)), 0.1)
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
		}}, slices.Epsilon)
}

func TestEinsumAxes(t *testing.T) {
	graphtest.RunTestGraphFn(t, "EinsumAxesMatrixMul",
		func(g *Graph) (inputs, outputs []*Node) {
			lhs := IotaFull(g, shapes.Make(shapes.F32, 2, 4))
			lhs = OnePlus(lhs)
			rhs := MulScalar(Ones(g, shapes.Make(shapes.F32, 4, 3)), 0.1)
			inputs = []*Node{lhs, rhs}
			outputs = []*Node{EinsumAxes(lhs, rhs, [][2]int{{1, 0}}, nil)}
			return
		}, []any{[][]float32{{1, 1, 1}, {2.6, 2.6, 2.6}}}, slices.Epsilon)
	graphtest.RunTestGraphFn(t, "EinsumAxesDotProduct",
		func(g *Graph) (inputs, outputs []*Node) {
			lhs := IotaFull(g, shapes.Make(shapes.F32, 4))
			lhs = OnePlus(lhs)
			rhs := MulScalar(Ones(g, shapes.Make(shapes.F32, 4)), 0.1)
			inputs = []*Node{lhs, rhs}
			outputs = []*Node{EinsumAxes(lhs, rhs, [][2]int{{0, 0}}, nil)}
			return
		}, []any{float32(1)}, slices.Epsilon)
	graphtest.RunTestGraphFn(t, "EinsumAxesOuterProduct",
		func(g *Graph) (inputs, outputs []*Node) {
			lhs := OnePlus(IotaFull(g, shapes.Make(shapes.F32, 4)))
			rhs := OnePlus(IotaFull(g, shapes.Make(shapes.F32, 3)))
			inputs = []*Node{lhs, rhs}
			outputs = []*Node{EinsumAxes(lhs, rhs, nil, nil)}
			return
		}, []any{[][]float32{{1, 2, 3}, {2, 4, 6}, {3, 6, 9}, {4, 8, 12}}}, slices.Epsilon)
	graphtest.RunTestGraphFn(t, "EinsumAxesBatchMatrixMul",
		func(g *Graph) (inputs, outputs []*Node) {
			lhs := IotaFull(g, shapes.Make(shapes.F32, 5, 2, 4))
			lhs = OnePlus(lhs)
			rhs := MulScalar(Ones(g, shapes.Make(shapes.F32, 5, 4, 3)), 0.1)
			inputs = []*Node{lhs, rhs}
			outputs = []*Node{EinsumAxes(lhs, rhs, [][2]int{{2, 1}}, [][2]int{{0, 0}})}
			return
		}, []any{[][][]float32{
			{{1, 1, 1}, {2.6, 2.6, 2.6}},
			{{4.2, 4.2, 4.2}, {5.8, 5.8, 5.8}},
			{{7.4, 7.4, 7.4}, {9, 9, 9}},
			{{10.6, 10.6, 10.6}, {12.2, 12.2, 12.2}},
			{{13.8, 13.8, 13.8}, {15.4, 15.4, 15.4}}},
		}, slices.Epsilon)
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
