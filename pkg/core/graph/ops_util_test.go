package graph_test

import (
	"math"
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/gomlx/gopjrt/dtypes"
)

func TestScalar(t *testing.T) {
	graphtest.RunTestGraphFn(t, "EinsumMatrixMul",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{
				Scalar(g, dtypes.Float32, uint8(3)),
				Scalar(g, dtypes.Float64, int16(-2)),
				Scalar(g, dtypes.Int64, float32(7)),
				Scalar(g, dtypes.Uint8, float64(3)),
			}
			outputs = inputs
			return
		}, []any{
			float32(3),
			float64(-2),
			int64(7),
			uint8(3),
		}, -1)
}

func TestIsZero(t *testing.T) {
	graphtest.RunTestGraphFn(t, t.Name(),
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{
				Const(g, []uint8{3, 5, 0, 2}),
				Const(g, []float32{-2.2, 1e-10, 3.1, 0, -1e-10}),
				Const(g, []complex64{1e-10, 0, -1e-10i}),
			}
			outputs = xslices.Map(inputs, func(x *Node) *Node { return IsZero(x) })
			return
		}, []any{
			[]bool{false, false, true, false},
			[]bool{false, false, false, true, false},
			[]bool{false, true, false},
		}, -1)
}

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

func TestSoftmax(t *testing.T) {
	graphtest.RunTestGraphFn(t, "TestSoftmax()",
		func(g *Graph) (inputs, outputs []*Node) {
			logits := Const(g, [][]float64{{-1, 0, 1.}, {-1, 0, 0}})
			inputs = []*Node{logits}
			outputs = []*Node{Softmax(logits)}
			return
		}, []any{
			[][]float64{
				{0.09003057317038046, 0.24472847105479764, 0.6652409557748218},
				{0.15536240349696362, 0.4223187982515182, 0.4223187982515182}},
		}, xslices.Epsilon)
}

func TestMaskedSoftmax(t *testing.T) {
	// Values checked with Tensorflow's `tf.nn.softmax()` function.
	graphtest.RunTestGraphFn(t, "TestMaskedSoftmax()",
		func(g *Graph) (inputs, outputs []*Node) {
			logits := Const(g, [][]float64{{-1, 0, 1.}, {-1, 5, 10}})
			mask := Const(g, [][]bool{{true, true, true}, {true, false, false}})
			inputs = []*Node{logits, mask}
			outputs = []*Node{MaskedSoftmax(logits, mask, -1)}
			return
		}, []any{
			[][]float64{{0.09003057317038046, 0.24472847105479764, 0.6652409557748218}, {1, 0, 0}},
		}, xslices.Epsilon)
}

func TestLogSoftmax(t *testing.T) {
	graphtest.RunTestGraphFn(t, "TestSoftmax()",
		func(g *Graph) (inputs, outputs []*Node) {
			logits := Const(g, [][]float32{{-1, 0, 1.}, {-1, 0, 0}})
			inputs = []*Node{logits}
			outputs = []*Node{LogSoftmax(logits)}
			return
		}, []any{
			[][]float32{
				{-2.4076061, -1.407606, -0.40760604},
				{-1.8619947, -0.8619948, -0.8619948}},
		}, xslices.Epsilon)
}

func TestMaskedLogSoftmax(t *testing.T) {
	graphtest.RunTestGraphFn(t, "TestSoftmax()",
		func(g *Graph) (inputs, outputs []*Node) {
			logits := Const(g, [][]float64{{-1, 0, 1., 1000}, {-1, 0, 1000, 0}})
			mask := Const(g, [][]bool{{true, true, true, false}, {true, true, false, true}})
			inputs = []*Node{logits}
			outputs = []*Node{MaskedLogSoftmax(logits, mask)}
			return
		}, []any{
			[][]float64{
				{-2.4076061, -1.407606, -0.40760604, math.Inf(-1)},
				{-1.8619947, -0.8619948, math.Inf(-1), -0.8619948}},
		}, xslices.Epsilon)
}

func TestCumSum(t *testing.T) {
	graphtest.RunTestGraphFn(t, "TestCumSum()",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{
				OnePlus(IotaFull(g, shapes.Make(dtypes.Int32, 2, 3))),
			}
			outputs = []*Node{
				CumSum(inputs[0], -1),
				CumSum(inputs[0], 0),
			}
			return
		}, []any{
			[][]int32{
				{1, 3, 6},
				{4, 9, 15},
			},
			[][]int32{
				{1, 2, 3},
				{5, 7, 9},
			},
		}, xslices.Epsilon)
}

func TestConsecutiveDifference(t *testing.T) {
	graphtest.RunTestGraphFn(t, "TestConsecutiveDifference()",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{
				OnePlus(IotaFull(g, shapes.Make(dtypes.Int32, 2, 3))),
				AddScalar(Square(IotaFull(g, shapes.Make(dtypes.Float32, 5))), -2),
			}
			outputs = []*Node{
				ConsecutiveDifference(inputs[0], -1, true),
				ConsecutiveDifference(inputs[0], 0, true),
				ConsecutiveDifference(inputs[1], 0, true),
				ConsecutiveDifference(inputs[1], 0, false),
			}
			return
		}, []any{
			[][]int32{
				{1, 1, 1},
				{4, 1, 1},
			},
			[][]int32{
				{1, 2, 3},
				{3, 3, 3},
			},
			[]float32{-2, 1, 3, 5, 7},
			[]float32{1, 3, 5, 7},
		}, xslices.Epsilon)
}

func TestShapedLowerTriangular(t *testing.T) {
	graphtest.RunTestGraphFn(t, "ShapedLowerTriangular",
		func(g *Graph) (inputs, outputs []*Node) {
			outputs = []*Node{
				ShapedLowerTriangular(g, 3, 3, 0),
				ShapedLowerTriangular(g, 3, 3, -1),
				ShapedLowerTriangular(g, 2, 3, 1),
			}
			return
		}, []any{
			[][]bool{{true, false, false}, {true, true, false}, {true, true, true}},
			[][]bool{{false, false, false}, {true, false, false}, {true, true, false}},
			[][]bool{{true, true, false}, {true, true, true}},
		}, -1)
}

func TestTakeLowerTriangular(t *testing.T) {
	graphtest.RunTestGraphFn(t, "ShapedLowerTriangular",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{
				AddScalar(IotaFull(g, shapes.Make(dtypes.Float64, 2, 2)), 1),
				AddScalar(IotaFull(g, shapes.Make(dtypes.Float32, 1, 2, 3, 4)), 1),
			}
			outputs = []*Node{
				TakeLowerTriangular(inputs[0], 0),
				TakeLowerTriangular(inputs[1], 0),
				TakeLowerTriangular(inputs[1], -1),
				TakeLowerTriangular(inputs[1], 1),
			}
			return
		}, []any{
			[][]float64{{1, 0}, {3, 4}},
			[][][][]float32{{{{1, 0, 0, 0}, {5, 6, 0, 0}, {9, 10, 11, 0}}, {{13, 0, 0, 0}, {17, 18, 0, 0}, {21, 22, 23, 0}}}},
			[][][][]float32{{{{0, 0, 0, 0}, {5, 0, 0, 0}, {9, 10, 0, 0}}, {{0, 0, 0, 0}, {17, 0, 0, 0}, {21, 22, 0, 0}}}},
			[][][][]float32{{{{1, 2, 0, 0}, {5, 6, 7, 0}, {9, 10, 11, 12}}, {{13, 14, 0, 0}, {17, 18, 19, 0}, {21, 22, 23, 24}}}},
		}, -1)
}

func TestTakeUpperTriangular(t *testing.T) {
	graphtest.RunTestGraphFn(t, "ShapedLowerTriangular",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{
				OnePlus(IotaFull(g, shapes.Make(dtypes.Float64, 2, 2))),
				OnePlus(IotaFull(g, shapes.Make(dtypes.Float32, 1, 2, 3, 4))),
			}
			outputs = []*Node{
				TakeUpperTriangular(inputs[0], 0),
				TakeUpperTriangular(inputs[1], 0),
				TakeUpperTriangular(inputs[1], -1),
				TakeUpperTriangular(inputs[1], 1),
			}
			return
		}, []any{
			[][]float64{{1, 2}, {0, 4}},
			[][][][]float32{{{{1, 2, 3, 4}, {0, 6, 7, 8}, {0, 0, 11, 12}}, {{13, 14, 15, 16}, {0, 18, 19, 20}, {0, 0, 23, 24}}}},
			[][][][]float32{{{{1, 2, 3, 4}, {5, 6, 7, 8}, {0, 10, 11, 12}}, {{13, 14, 15, 16}, {17, 18, 19, 20}, {0, 22, 23, 24}}}},
			[][][][]float32{{{{0, 2, 3, 4}, {0, 0, 7, 8}, {0, 0, 0, 12}}, {{0, 14, 15, 16}, {0, 0, 19, 20}, {0, 0, 0, 24}}}},
		}, -1)
}

func TestReduceVariance(t *testing.T) {
	graphtest.RunTestGraphFn(t, "ReduceVariance",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{Const(g, []float32{3.0, 7.0})}
			outputs = []*Node{ReduceVariance(inputs[0])}
			return
		}, []any{
			float32(4.0),
		}, 0.001)

	graphtest.RunTestGraphFn(t, "ReduceVariance",
		func(g *Graph) (inputs, outputs []*Node) {
			rngState := Const(g, must1(RNGStateFromSeed(42)))
			_, values := RandomNormal(rngState, shapes.Make(dtypes.Float32, 4, 100_000))
			multiplier := OnePlus(Iota(g, shapes.Make(dtypes.Float32, 4, 1), 0))
			shift := AddScalar(Iota(g, shapes.Make(dtypes.Float32, 4, 1), 0), -2)
			values2 := Add(Mul(values, multiplier), shift)
			outputs = []*Node{
				ReduceVariance(values),
				ReduceVariance(values2, -1),
			}
			return
		}, []any{
			float32(1.0),
			[]float32{1.0, 4.0, 9.0, 16.0}, // Var(c*X) = c^2*Var(X).
		}, 0.1)
}

func TestReduceSkewness(t *testing.T) {
	graphtest.RunTestGraphFn(t, "ReduceSkewness",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{
				Const(g, []float32{1, 2, 3, 4, 5}),
				Const(g, []float32{2, 8, 0, 4, 1, 9, 9, 0}),
			}
			outputs = []*Node{
				ReduceSkewness(inputs[0]),
				ReduceSkewness(inputs[1]),
			}
			return
		}, []any{
			float32(0.0),
			float32(0.2650554122698573),
		}, 0.001)
}

func TestL2Normalize(t *testing.T) {
	graphtest.RunTestGraphFn(t, t.Name(),
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{
				Const(g, [][]float32{{3, 4}, {0, 0}}),
			}
			normalized := L2Normalize(inputs[0], -1)
			grad := Gradient(ReduceAllSum(normalized), inputs[0])[0]
			outputs = []*Node{
				normalized,
				grad,
			}
			return
		}, []any{
			[][]float32{{0.6, 0.8}, {0, 0}},
			[][]float32{{0.032, -0.024}, {1, 1}},
		}, 1e-3)
}

func TestCosineSimilarity(t *testing.T) {
	graphtest.RunTestGraphFn(t, t.Name(),
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, [][]float32{
				{0, 0, 0},
				{7, 0, 0},
				{0, 0, 13},
			})
			y := Const(g, [][]float32{
				{10, 20, 30},
				{2, 0, 0},
				{0, 0, -11},
			})
			inputs = []*Node{x, y}
			outputs = []*Node{
				CosineSimilarity(x, y, -1), // Rows: Axis -1
				CosineSimilarity(x, y, 0),  // Columns: Axis 0
			}
			return
		}, []any{
			[][]float32{{0}, {1}, {-1}},
			[][]float32{{0.19611612, 0, -0.34425467}},
		}, 0.0001)

	// Check the gradient of zero vectors won't yield NaNs.
	testGradients(t, "CosineSimilarity Gradient", func(g *Graph) (output *Node, nodesForGrad []*Node) {
		x := Const(g, [][]float32{{0, 0, 0}})
		y := Const(g, [][]float32{{10, 20, 30}})
		output = CosineSimilarity(x, y, -1)
		return output, []*Node{x}
	}, []any{
		[][]float32{{0, 0, 0}},
	})
}

func TestUtil(t *testing.T) {
	graphtest.RunTestGraphFn(t, "NegativeIndicator",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{Const(g, []float32{-2, -1, 0, 1, 2})}
			outputs = []*Node{NegativeIndicator(inputs[0])}
			return
		}, []any{[]float32{1, 1, 0, 0, 0}}, -1)

	graphtest.RunTestGraphFn(t, "PositiveIndicator",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{Const(g, []float32{-2, -1, 0, 1, 2})}
			outputs = []*Node{PositiveIndicator(inputs[0])}
			return
		}, []any{[]float32{0, 0, 0, 1, 1}}, -1)

	graphtest.RunTestGraphFn(t, "IsPositive",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{Const(g, []float32{-2, -1, 0, 1, 2})}
			outputs = []*Node{IsPositive(inputs[0])}
			return
		}, []any{[]bool{false, false, false, true, true}}, -1)

	graphtest.RunTestGraphFn(t, "IsNegative",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{Const(g, []float32{-2, -1, 0, 1, 2})}
			outputs = []*Node{IsNegative(inputs[0])}
			return
		}, []any{[]bool{true, true, false, false, false}}, -1)

	graphtest.RunTestGraphFn(t, "IsNonNegative",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{Const(g, []float32{-2, -1, 0, 1, 2})}
			outputs = []*Node{IsNonNegative(inputs[0])}
			return
		}, []any{[]bool{false, false, true, true, true}}, -1)

	graphtest.RunTestGraphFn(t, "IsNonPositive",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{Const(g, []float32{-2, -1, 0, 1, 2})}
			outputs = []*Node{IsNonPositive(inputs[0])}
			return
		}, []any{[]bool{true, true, true, false, false}}, -1)

	graphtest.RunTestGraphFn(t, "AddScalar",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{Const(g, []float32{1, 2, 3})}
			outputs = []*Node{AddScalar(inputs[0], float32(2))}
			return
		}, []any{[]float32{3, 4, 5}}, -1)

	graphtest.RunTestGraphFn(t, "SubScalar",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{Const(g, []float32{3, 4, 5})}
			outputs = []*Node{SubScalar(inputs[0], float32(1))}
			return
		}, []any{[]float32{2, 3, 4}}, -1)

	graphtest.RunTestGraphFn(t, "MulScalar",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{Const(g, []float32{1, 2, 3})}
			outputs = []*Node{MulScalar(inputs[0], float32(2))}
			return
		}, []any{[]float32{2, 4, 6}}, -1)

	graphtest.RunTestGraphFn(t, "DivScalar",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{Const(g, []float32{2, 4, 6})}
			outputs = []*Node{DivScalar(inputs[0], float32(2))}
			return
		}, []any{[]float32{1, 2, 3}}, -1)

	graphtest.RunTestGraphFn(t, "PowScalar",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{Const(g, []float32{1, 2, 3})}
			outputs = []*Node{PowScalar(inputs[0], float32(2))}
			return
		}, []any{[]float32{1, 4, 9}}, -1)

	graphtest.RunTestGraphFn(t, "ModScalar",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{Const(g, []float32{5, 7, 9})}
			outputs = []*Node{ModScalar(inputs[0], float32(2))}
			return
		}, []any{[]float32{1, 1, 1}}, -1)
}
