// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph_test

import (
	"math"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/graphtest"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/require"
)

func TestScalar(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		graphtest.RunTestGraphFnWithBackend(t, "EinsumMatrixMul", backend,
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

		graphtest.RunTestGraphFnWithBackend(t, "ScalarOne", backend,
			func(g *Graph) (inputs, outputs []*Node) {
				inputs = []*Node{
					ScalarOne(g, dtypes.Float32),
					ScalarOne(g, dtypes.Float64),
					ScalarOne(g, dtypes.Int64),
					ScalarOne(g, dtypes.Bool),
				}
				outputs = inputs
				return
			}, []any{
				float32(1),
				float64(1),
				int64(1),
				true,
			}, -1)

		graphtest.RunTestGraphFnWithBackend(t, "ScalarZero", backend,
			func(g *Graph) (inputs, outputs []*Node) {
				inputs = []*Node{
					ScalarZero(g, dtypes.Float32),
					ScalarZero(g, dtypes.Float64),
					ScalarZero(g, dtypes.Int64),
					ScalarZero(g, dtypes.Bool),
				}
				outputs = inputs
				return
			}, []any{
				float32(0),
				float64(0),
				int64(0),
				false,
			}, -1)

	})
}

func TestIsZero(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	graphtest.RunTestGraphFnWithBackend(t, "basic", backend,
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{
				Const(g, []uint8{3, 5, 0, 2}),
				Const(g, []float32{-2.2, 1e-10, 3.1, 0, -1e-10}),
			}
			outputs = xslices.Map(inputs, func(x *Node) *Node { return IsZero(x) })
			return
		}, []any{
			[]bool{false, false, true, false},
			[]bool{false, false, false, true, false},
		}, -1)

	t.Run("complex64", func(t *testing.T) {
		if !backend.Capabilities().DTypes[dtypes.Complex64] {
			t.Skipf("Backend %q does not support complex numbers", backend.Name())
		}
		graphtest.RunTestGraphFnWithBackend(t, "complex64", backend,
			func(g *Graph) (inputs, outputs []*Node) {
				inputs = []*Node{
					Const(g, []complex64{1e-10, 0, -1e-10i}),
				}
				outputs = xslices.Map(inputs, func(x *Node) *Node { return IsZero(x) })
				return
			}, []any{
				[]bool{false, true, false},
			}, -1)
	})
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
	testFuncOneInputDefaultBackend(t, "Clip({0, 3, 6}, 2, 4)",
		func(g *Graph) (input, output *Node) {
			input = Const(g, [][][]float32{{{0}, {3}, {6}}})
			output = Clip(input, Const(g, float32(2)), Const(g, float32(4)))
			return
		}, [][][]float32{{{2}, {3}, {4}}})
}

func TestNorms(t *testing.T) {
	testFuncOneInputDefaultBackend(t, "L2NormSquare",
		func(g *Graph) (input, output *Node) {
			input = Const(g, [][]float32{{1, 2}, {-3, 4}})
			output = L2NormSquare(input)
			return
		}, float32(5+25))

	testFuncOneInputDefaultBackend(t, "L2Norm",
		func(g *Graph) (input, output *Node) {
			input = Const(g, [][]float32{{4, -3}, {-5, 12}})
			output = L2Norm(input, -1)
			return
		}, [][]float32{{5}, {13}})

	testFuncOneInputDefaultBackend(t, "L1Norm",
		func(g *Graph) (input, output *Node) {
			input = Const(g, [][]float32{{-4, 3}, {5, -12}})
			output = L1Norm(input, -1)
			return
		}, [][]float32{{7}, {17}})

	invSqrt2 := float32(1.0 / math.Sqrt(2.0))
	testFuncOneInputDefaultBackend(t, "L2Normalize",
		func(g *Graph) (input, output *Node) {
			input = Const(g, [][]float32{{5, 5}, {711, 711}})
			output = L2Normalize(input, -1)
			return
		}, [][]float32{{invSqrt2, invSqrt2}, {invSqrt2, invSqrt2}})

	testFuncOneInputDefaultBackend(t, "L2NormalizeWithEpsilon",
		func(g *Graph) (input, output *Node) {
			input = Const(g, [][]float32{{0, 0}, {11, 11}})
			output = L2NormalizeWithEpsilon(input, 1e-9, -1)
			return
		}, [][]float32{{0, 0}, {invSqrt2, invSqrt2}}) // Epsilon shouldn't create a large enough difference to fail the test.
}

func TestMirroredLog1p(t *testing.T) {
	testFuncOneInputDefaultBackend(t, "MirroredLog1p",
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

func TestHyperbolicTrig(t *testing.T) {
	// Forward tests in Float64.
	xSinhCoshAsinh := []float64{-3.0, -0.5, -1e-4, 0.0, 1e-4, 0.5, 3.0}
	testFuncOneInputDefaultBackend(t, "Sinh",
		func(g *Graph) (input, output *Node) {
			input = Const(g, xSinhCoshAsinh)
			output = Sinh(input)
			return
		}, xslices.Map(xSinhCoshAsinh, math.Sinh))

	testFuncOneInputDefaultBackend(t, "Cosh",
		func(g *Graph) (input, output *Node) {
			input = Const(g, xSinhCoshAsinh)
			output = Cosh(input)
			return
		}, xslices.Map(xSinhCoshAsinh, math.Cosh))

	testFuncOneInputDefaultBackend(t, "Asinh",
		func(g *Graph) (input, output *Node) {
			input = Const(g, xSinhCoshAsinh)
			output = Asinh(input)
			return
		}, xslices.Map(xSinhCoshAsinh, math.Asinh))

	xAtanh := []float64{-0.9, -0.5, -1e-4, 0.0, 1e-4, 0.5, 0.9}
	testFuncOneInputDefaultBackend(t, "Atanh",
		func(g *Graph) (input, output *Node) {
			input = Const(g, xAtanh)
			output = Atanh(input)
			return
		}, xslices.Map(xAtanh, math.Atanh))

	xAcosh := []float64{1.0, 1.0001, 1.5, 3.0, 100.0}
	testFuncOneInputDefaultBackend(t, "Acosh",
		func(g *Graph) (input, output *Node) {
			input = Const(g, xAcosh)
			output = Acosh(input)
			return
		}, xslices.Map(xAcosh, math.Acosh))

	// Half-precision (Float16 and BFloat16) forward tests, including large values (> 256)
	// that would overflow x^2 in Float16 without Float32 promotion.
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		f16Vals := []float32{-500.0, -1.0, 0.0, 0.5, 500.0}
		f16Input := xslices.Map(f16Vals, float16.FromFloat32)
		graphtest.RunTestGraphFnWithBackend(t, "Asinh(Float16)", backend,
			func(g *Graph) (inputs, outputs []*Node) {
				x := Const(g, f16Input)
				return []*Node{x}, []*Node{Asinh(x)}
			}, []any{
				xslices.Map(f16Input, func(v float16.Float16) float16.Float16 {
					return float16.FromFloat32(float32(math.Asinh(float64(v.Float32()))))
				}),
			}, 1e-2)

		f16AcoshInput := xslices.Map([]float32{1.0, 2.0, 500.0}, float16.FromFloat32)
		graphtest.RunTestGraphFnWithBackend(t, "Acosh(Float16)", backend,
			func(g *Graph) (inputs, outputs []*Node) {
				x := Const(g, f16AcoshInput)
				return []*Node{x}, []*Node{Acosh(x)}
			}, []any{
				xslices.Map(f16AcoshInput, func(v float16.Float16) float16.Float16 {
					return float16.FromFloat32(float32(math.Acosh(float64(v.Float32()))))
				}),
			}, 1e-2)

		bf16AtanhInput := xslices.Map([]float32{-0.75, -0.25, 0.0, 0.25, 0.75}, bfloat16.FromFloat32)
		graphtest.RunTestGraphFnWithBackend(t, "Atanh(BFloat16)", backend,
			func(g *Graph) (inputs, outputs []*Node) {
				x := Const(g, bf16AtanhInput)
				return []*Node{x}, []*Node{Atanh(x)}
			}, []any{
				xslices.Map(bf16AtanhInput, func(v bfloat16.BFloat16) bfloat16.BFloat16 {
					return bfloat16.FromFloat32(float32(math.Atanh(float64(v.Float32()))))
				}),
			}, 2e-2)

		bf16SinhCoshInput := xslices.Map([]float32{-2.0, -0.25, 0.0, 0.25, 2.0}, bfloat16.FromFloat32)
		graphtest.RunTestGraphFnWithBackend(t, "Sinh_Cosh(BFloat16)", backend,
			func(g *Graph) (inputs, outputs []*Node) {
				x := Const(g, bf16SinhCoshInput)
				return []*Node{x}, []*Node{Sinh(x), Cosh(x)}
			}, []any{
				xslices.Map(bf16SinhCoshInput, func(v bfloat16.BFloat16) bfloat16.BFloat16 {
					return bfloat16.FromFloat32(float32(math.Sinh(float64(v.Float32()))))
				}),
				xslices.Map(bf16SinhCoshInput, func(v bfloat16.BFloat16) bfloat16.BFloat16 {
					return bfloat16.FromFloat32(float32(math.Cosh(float64(v.Float32()))))
				}),
			}, 2e-2)
	})

	// Gradient tests.
	gradX := []float64{-2.0, -0.5, 0.0, 0.5, 2.0}
	testGradients(t, "Sinh gradient",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := Const(g, gradX)
			return Sinh(input), []*Node{input}
		}, []any{xslices.Map(gradX, math.Cosh)})

	testGradients(t, "Cosh gradient",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := Const(g, gradX)
			return Cosh(input), []*Node{input}
		}, []any{xslices.Map(gradX, math.Sinh)})

	testGradients(t, "Asinh gradient",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := Const(g, gradX)
			return Asinh(input), []*Node{input}
		}, []any{xslices.Map(gradX, func(x float64) float64 {
			return 1.0 / math.Sqrt(x*x+1.0)
		})})

	gradXAtanh := []float64{-0.8, -0.3, 0.0, 0.3, 0.8}
	testGradients(t, "Atanh gradient",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := Const(g, gradXAtanh)
			return Atanh(input), []*Node{input}
		}, []any{xslices.Map(gradXAtanh, func(x float64) float64 {
			return 1.0 / (1.0 - x*x)
		})})

	gradXAcosh := []float64{1.2, 2.0, 5.0}
	testGradients(t, "Acosh gradient",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := Const(g, gradXAcosh)
			return Acosh(input), []*Node{input}
		}, []any{xslices.Map(gradXAcosh, func(x float64) float64 {
			return 1.0 / math.Sqrt(x*x-1.0)
		})})
}

func TestHyperbolicTrigNumericalStability(t *testing.T) {
	inf32, nan32 := float32(math.Inf(1)), float32(math.NaN())
	inf64 := math.Inf(1)
	asinh32 := func(x float32) float32 { return float32(math.Asinh(float64(x))) }
	acosh32 := func(x float32) float32 { return float32(math.Acosh(float64(x))) }
	ln2 := float32(math.Ln2)

	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		// Around the threshold for large values (20) both branches must agree with math.Sinh/math.Cosh.
		xAround20 := []float64{-20.5, -19.5, 19.5, 20.5}
		tiny := []float32{1e-6, -1e-10, 1e-30}
		graphtest.RunTestGraphFnWithBackend(t, "Sinh", backend,
			func(g *Graph) (inputs, outputs []*Node) {
				tinyX := Const(g, tiny)
				inputs = []*Node{tinyX}
				outputs = []*Node{
					// sinh(x)/x -> 1 for x -> 0: no cancellation of e^x - e^-x for small |x|.
					Div(Sinh(tinyX), tinyX),
					// sinh(±89) ~ ±2.2e38 fits in Float32 but e^89 doesn't: ln|sinh(±89)| = 89 - ln 2.
					Log(Abs(Sinh(Const(g, []float32{89, -89})))),
					Sinh(Const(g, xAround20)),
					Sinh(Const(g, []float32{inf32, -inf32})),
					IsNaN(Sinh(Const(g, []float32{nan32}))),
				}
				return
			}, []any{
				[]float32{1, 1, 1},
				[]float32{89 - ln2, 89 - ln2},
				xslices.Map(xAround20, math.Sinh),
				[]float32{inf32, -inf32},
				[]bool{true},
			}, 1e-4)

		graphtest.RunTestGraphFnWithBackend(t, "Cosh", backend,
			func(g *Graph) (inputs, outputs []*Node) {
				x := Const(g, []float32{89, -89})
				inputs = []*Node{x}
				outputs = []*Node{
					// cosh(±89) ~ 2.2e38 fits in Float32 but e^89 doesn't: ln(cosh(±89)) = 89 - ln 2.
					Log(Cosh(x)),
					Cosh(Const(g, xAround20)),
					Cosh(Const(g, []float32{inf32, -inf32})),
					IsNaN(Cosh(Const(g, []float32{nan32}))),
				}
				return
			}, []any{
				[]float32{89 - ln2, 89 - ln2},
				xslices.Map(xAround20, math.Cosh),
				[]float32{inf32, inf32},
				[]bool{true},
			}, 1e-4)

		// Float32 threshold for large values is 4096; x² overflows Float32 above ~1.8e19.
		xAsinh32 := []float32{-1e30, -100, -0.5, 0, 0.5, 4095, 4097, 1e20}
		xAsinh64 := []float64{-1e300, 1e-200, 0.5, 6e7, 7e7, 1e154}
		graphtest.RunTestGraphFnWithBackend(t, "Asinh", backend,
			func(g *Graph) (inputs, outputs []*Node) {
				x := Const(g, xAsinh32)
				tinyX := Const(g, tiny)
				inputs = []*Node{x}
				outputs = []*Node{
					Asinh(x),
					// asinh(x)/x -> 1 for x -> 0: no precision loss for small |x|.
					Div(Asinh(tinyX), tinyX),
					Asinh(Const(g, xAsinh64)),
					Asinh(Const(g, []float32{inf32, -inf32})),
					IsNaN(Asinh(Const(g, []float32{nan32}))),
				}
				return
			}, []any{
				xslices.Map(xAsinh32, asinh32),
				[]float32{1, 1, 1},
				xslices.Map(xAsinh64, math.Asinh),
				[]float32{inf32, -inf32},
				[]bool{true},
			}, 1e-4)

		// x + sqrt(x-1)sqrt(x+1) overflows Float32 for x > ~1.7e38.
		xAcosh32 := []float32{1.5, 4095, 4097, 1e20, 3e38}
		xAcosh64 := []float64{1 + 1e-10, 6e7, 7e7, 1e300}
		graphtest.RunTestGraphFnWithBackend(t, "Acosh", backend,
			func(g *Graph) (inputs, outputs []*Node) {
				x := Const(g, xAcosh32)
				inputs = []*Node{x}
				outputs = []*Node{
					Acosh(x),
					Acosh(Const(g, xAcosh64)),
					Acosh(Const(g, []float64{inf64})),
					// Outside the domain.
					IsNaN(Acosh(Const(g, []float32{0.5, nan32}))),
				}
				return
			}, []any{
				xslices.Map(xAcosh32, acosh32),
				xslices.Map(xAcosh64, math.Acosh),
				[]float64{inf64},
				[]bool{true, true},
			}, 1e-4)
	})

	// Gradients in the branch for large values, and of Expm1 (used by Sinh for small values).
	gradX := []float64{-21, 21}
	testGradients(t, "Sinh gradient for large |x|",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := Const(g, gradX)
			return Sinh(input), []*Node{input}
		}, []any{xslices.Map(gradX, math.Cosh)})

	testGradients(t, "Cosh gradient for large |x|",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := Const(g, gradX)
			return Cosh(input), []*Node{input}
		}, []any{xslices.Map(gradX, math.Sinh)})

	gradXInverse := []float64{-1e8, 1e8}
	testGradients(t, "Asinh gradient for large |x|",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := Const(g, gradXInverse)
			return Asinh(input), []*Node{input}
		}, []any{xslices.Map(gradXInverse, func(x float64) float64 {
			return 1.0 / math.Sqrt(x*x+1.0)
		})})

	gradXExpm1 := []float64{-2, 0, 1e-8, 2}
	testGradients(t, "Expm1 gradient",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := Const(g, gradXExpm1)
			return Expm1(input), []*Node{input}
		}, []any{xslices.Map(gradXExpm1, math.Exp)})
}

func TestShiftWithScalar(t *testing.T) {
	testFuncOneInputDefaultBackend(t, "ShiftWithScalar(input, axis=1, left, n=1, fill=0.0)",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(dtypes.Float32, 3, 2, 2))
			output = ShiftWithScalar(input, 1, ShiftDirLeft, 1, 0.0)
			return
		}, [][][]float32{
			{{2, 3}, {0, 0}},
			{{6, 7}, {0, 0}},
			{{10, 11}, {0, 0}},
		})
	testFuncOneInputDefaultBackend(t, "ShiftWithScalar(input, axis=-1, left, n=1, fill=100)",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(dtypes.Int32, 3, 2, 2))
			output = ShiftWithScalar(input, -1, ShiftDirLeft, 1, 100)
			return
		}, [][][]int32{
			{{1, 100}, {3, 100}},
			{{5, 100}, {7, 100}},
			{{9, 100}, {11, 100}},
		})
	testFuncOneInputDefaultBackend(t, "ShiftWithScalar(input, axis=0, right, n=2, fill=1.0)",
		func(g *Graph) (input, output *Node) {
			input = Zeros(g, shapes.Make(dtypes.Bool, 3, 2, 2))
			output = ShiftWithScalar(input, 0, ShiftDirRight, 2, 1)
			return
		}, [][][]bool{ // Inserted `true` to the left (shift-right) of the tensor:
			{{true, true}, {true, true}},
			{{true, true}, {true, true}},
			{{false, false}, {false, false}},
		})
	testFuncOneInputDefaultBackend(t, "ShiftWithScalar(input, axis=0, right, n=3, fill=1)",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(dtypes.Float64, 10.0))
			output = ShiftWithScalar(input, -1, ShiftDirRight, 3, 1)
			return
		}, []float64{1, 1, 1, 0, 1, 2, 3, 4, 5, 6})
}

func TestShiftWithValue(t *testing.T) {
	testFuncOneInputDefaultBackend(t, "ShiftWithScalar(input, axis=0, left, n=3, value=1000)",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(dtypes.Float32, 10))
			output = ShiftWithValue(input, 0, ShiftDirLeft, 3, Scalar(g, F32, 1000))
			return
		}, []float32{3, 4, 5, 6, 7, 8, 9, 1000, 1000, 1000})
	testFuncOneInputDefaultBackend(t, "ShiftWithScalar(input, axis=-1, right, n=3, value=[[100], [1000]])",
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
	testFuncOneInputDefaultBackend(t, "Shift(input, axis=0, left, n=3)",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(dtypes.Float32, 10))
			output = Shift(input, 0, ShiftDirLeft, 3)
			return
		}, []float32{3, 4, 5, 6, 7, 8, 9, 9, 9, 9})
	testFuncOneInputDefaultBackend(t, "Shift(input, axis=-1, right, n=3)",
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
	testFuncOneInputDefaultBackend(t, "OneHot 1 leading dimension",
		func(g *Graph) (input, output *Node) {
			input = Const(g, []int{1, 0, 3})
			output = OneHot(input, 4, dtypes.Float32)
			return
		}, [][]float32{{0, 1, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}})
	testFuncOneInputDefaultBackend(t, "OneHot 2 leading dimensions",
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
	testFuncOneInputDefaultBackend(t, "GrowLeft",
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
	testFuncOneInputDefaultBackend(t, "GrowRight",
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

func TestCumMax(t *testing.T) {
	graphtest.RunTestGraphFn(t, "TestCumMax()",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{
				Const(g, [][]int32{{3, 1, 4}, {6, 5, 2}}),
			}
			outputs = []*Node{
				CumMax(inputs[0], -1),
				CumMax(inputs[0], 0),
				CumMax(inputs[0], -1, &CumSumOptions{Exclusive: true}),
				CumMax(inputs[0], 0, &CumSumOptions{Exclusive: true}),
				CumMax(inputs[0], -1, &CumSumOptions{Reverse: true}),
				CumMax(inputs[0], 0, &CumSumOptions{Reverse: true}),
				CumMax(inputs[0], -1, &CumSumOptions{Exclusive: true, Reverse: true}),
				CumMax(inputs[0], 0, &CumSumOptions{Exclusive: true, Reverse: true}),
				// Also verify decomposed fallback implementation matches directly:
				CumMaxDecomposedForTest(inputs[0], 1, CumSumOptions{}),
				CumMaxDecomposedForTest(inputs[0], 1, CumSumOptions{Exclusive: true}),
				CumMaxDecomposedForTest(inputs[0], 1, CumSumOptions{Reverse: true}),
				CumMaxDecomposedForTest(inputs[0], 1, CumSumOptions{Exclusive: true, Reverse: true}),
			}
			return
		}, []any{
			// -1, default
			[][]int32{
				{3, 3, 4},
				{6, 6, 6},
			},
			// 0, default
			[][]int32{
				{3, 1, 4},
				{6, 5, 4},
			},
			// -1, exclusive
			[][]int32{
				{math.MinInt32, 3, 3},
				{math.MinInt32, 6, 6},
			},
			// 0, exclusive
			[][]int32{
				{math.MinInt32, math.MinInt32, math.MinInt32},
				{3, 1, 4},
			},
			// -1, reverse
			[][]int32{
				{4, 4, 4},
				{6, 5, 2},
			},
			// 0, reverse
			[][]int32{
				{6, 5, 4},
				{6, 5, 2},
			},
			// -1, exclusive+reverse
			[][]int32{
				{4, 4, math.MinInt32},
				{5, 2, math.MinInt32},
			},
			// 0, exclusive+reverse
			[][]int32{
				{6, 5, 2},
				{math.MinInt32, math.MinInt32, math.MinInt32},
			},
			// Decomposed: axis 1, default
			[][]int32{
				{3, 3, 4},
				{6, 6, 6},
			},
			// Decomposed: axis 1, exclusive
			[][]int32{
				{math.MinInt32, 3, 3},
				{math.MinInt32, 6, 6},
			},
			// Decomposed: axis 1, reverse
			[][]int32{
				{4, 4, 4},
				{6, 5, 2},
			},
			// Decomposed: axis 1, exclusive+reverse
			[][]int32{
				{4, 4, math.MinInt32},
				{5, 2, math.MinInt32},
			},
		}, xslices.Epsilon)

	// Test gradients (autodiff)
	graphtest.RunTestGraphFn(t, "TestCumMaxGradients",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, []float64{3, 1, 4, 2})
			inputs = []*Node{x}

			// Forward
			yDefault := CumMax(x, 0)
			yExclusive := CumMax(x, 0, &CumSumOptions{Exclusive: true})
			yReverse := CumMax(x, 0, &CumSumOptions{Reverse: true})
			yExclRev := CumMax(x, 0, &CumSumOptions{Exclusive: true, Reverse: true})

			gradDefault := Gradient(ReduceAllSum(yDefault), x)[0]
			gradExclusive := Gradient(ReduceAllSum(yExclusive), x)[0]
			gradReverse := Gradient(ReduceAllSum(yReverse), x)[0]
			gradExclRev := Gradient(ReduceAllSum(yExclRev), x)[0]

			outputs = []*Node{gradDefault, gradExclusive, gradReverse, gradExclRev}
			return
		}, []any{
			[]float64{2, 0, 2, 0},
			[]float64{2, 0, 1, 0},
			[]float64{0, 0, 3, 1},
			[]float64{0, 0, 2, 1},
		}, xslices.Epsilon)

	// Test panic when more than one option is passed.
	require.Panics(t, func() {
		g := NewGraph(compute.MustNew(), "test_cummax_multiple_options")
		x := IotaFull(g, shapes.Make(dtypes.Float32, 3))
		CumMax(x, 0, &CumSumOptions{}, &CumSumOptions{})
	})
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
				CumSum(inputs[0], -1, &CumSumOptions{Exclusive: true}),
				CumSum(inputs[0], 0, &CumSumOptions{Exclusive: true}),
				CumSum(inputs[0], -1, &CumSumOptions{Reverse: true}),
				CumSum(inputs[0], 0, &CumSumOptions{Reverse: true}),
				CumSum(inputs[0], -1, &CumSumOptions{Exclusive: true, Reverse: true}),
				CumSum(inputs[0], 0, &CumSumOptions{Exclusive: true, Reverse: true}),
			}
			return
		}, []any{
			// -1, default
			[][]int32{
				{1, 3, 6},
				{4, 9, 15},
			},
			// 0, default
			[][]int32{
				{1, 2, 3},
				{5, 7, 9},
			},
			// -1, exclusive
			[][]int32{
				{0, 1, 3},
				{0, 4, 9},
			},
			// 0, exclusive
			[][]int32{
				{0, 0, 0},
				{1, 2, 3},
			},
			// -1, reverse
			[][]int32{
				{6, 5, 3},
				{15, 11, 6},
			},
			// 0, reverse
			[][]int32{
				{5, 7, 9},
				{4, 5, 6},
			},
			// -1, exclusive+reverse
			[][]int32{
				{5, 3, 0},
				{11, 6, 0},
			},
			// 0, exclusive+reverse
			[][]int32{
				{4, 5, 6},
				{0, 0, 0},
			},
		}, xslices.Epsilon)

	// Test gradients (autodiff)
	graphtest.RunTestGraphFn(t, "TestCumSumGradients",
		func(g *Graph) (inputs, outputs []*Node) {
			x := IotaFull(g, shapes.Make(dtypes.Float64, 3))
			inputs = []*Node{x}

			// Forward
			yDefault := CumSum(x, 0)
			yExclusive := CumSum(x, 0, &CumSumOptions{Exclusive: true})
			yReverse := CumSum(x, 0, &CumSumOptions{Reverse: true})
			yExclRev := CumSum(x, 0, &CumSumOptions{Exclusive: true, Reverse: true})

			gradDefault := Gradient(ReduceAllSum(yDefault), x)[0]
			gradExclusive := Gradient(ReduceAllSum(yExclusive), x)[0]
			gradReverse := Gradient(ReduceAllSum(yReverse), x)[0]
			gradExclRev := Gradient(ReduceAllSum(yExclRev), x)[0]

			outputs = []*Node{gradDefault, gradExclusive, gradReverse, gradExclRev}
			return
		}, []any{
			[]float64{3, 2, 1},
			[]float64{2, 1, 0},
			[]float64{1, 2, 3},
			[]float64{0, 1, 2},
		}, xslices.Epsilon)

	// Test panic when more than one option is passed.
	require.Panics(t, func() {
		g := NewGraph(compute.MustNew(), "test_multiple_options")
		x := IotaFull(g, shapes.Make(dtypes.Float32, 3))
		CumSum(x, 0, &CumSumOptions{}, &CumSumOptions{})
	})
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
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		graphtest.RunTestGraphFnWithBackend(t, t.Name(), backend,
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
		testGradientsWithBackend(t, "CosineSimilarity Gradient", backend, func(g *Graph) (output *Node, nodesForGrad []*Node) {
			x := Const(g, [][]float32{{0, 0, 0}})
			y := Const(g, [][]float32{{10, 20, 30}})
			output = CosineSimilarity(x, y, -1)
			return output, []*Node{x}
		}, []any{
			[][]float32{{0, 0, 0}},
		})
	})
}

func TestCrossCosineSimilarity(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		graphtest.RunTestGraphFnWithBackend(t, t.Name(), backend,
			func(g *Graph) (inputs, outputs []*Node) {
				x := Const(g, [][]float32{
					{0, 0, 0},
					{7, 0, 0},
					{0, 0, 13},
				})
				y := Const(g, [][]float32{
					{10, 20, 30},
					{0, 0, 0},
					{0, 0, -11},
				})
				inputs = []*Node{x, y}
				outputs = []*Node{
					CrossCosineSimilarity(x, y, -1, 0),
				}
				return
			}, []any{
				[][]float32{
					{0, 0, 0},
					{0.26726124, 0, 0},
					{0.8017837, 0, -1},
				},
			}, 0.0001)

		// Check the gradient of zero vectors won't yield NaNs.
		testGradientsWithBackend(t, "CrossCosineSimilarity Gradient", backend, func(g *Graph) (output *Node, nodesForGrad []*Node) {
			x := Const(g, [][]float32{{0, 0, 0}})
			y := Const(g, [][]float32{{10, 20, 30}})
			output = CrossCosineSimilarity(x, y, -1, 0)
			return output, []*Node{x}
		}, []any{
			[][]float32{{0, 0, 0}},
		})
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
