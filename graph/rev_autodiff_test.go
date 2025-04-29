/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package graph_test

import (
	"fmt"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
	"math"
	"testing"
)

func TestGradientAdd(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	g := NewGraph(backend, "TestDense")
	{
		c1 := Const(g, []float32{1, 2})
		c2 := Const(g, []float32{10})
		output := ReduceAllSum(Add(c1, c2))
		gradients := Gradient(output, c1, c2)
		outputs := []*Node{output}
		outputs = append(outputs, gradients...)
		g.Compile(outputs...)
	}
	outputs := g.Run()

	{
		got := outputs[0]
		fmt.Printf("output=%v\n", got)
		want := float32(23)
		require.Equalf(t, want, got.Value(), "%s: wanted %v, got %v", t.Name(), want, got)
	}

	{
		got := outputs[1]
		fmt.Printf("\tgrad output/A=%v\n", got)
		want := []float32{1, 1}
		require.Equalf(t, want, got.Value(), "%s: wanted %v, got %v", t.Name(), want, got)
	}

	{
		got := outputs[2]
		fmt.Printf("\tgrad output/B=%v\n", got)
		want := []float32{2}
		require.Equalf(t, want, got.Value(), "%s: wanted %v, got %v", t.Name(), want, got)
	}
}

func TestGradientDot(t *testing.T) {
	graphtest.RunTestGraphFn(t, "GradientDot: dot(vector, vector)", func(g *Graph) (inputs, outputs []*Node) {
		v1 := Mul(Ones(g, MakeShape(F32, 4)), Const(g, float32(2)))
		v2 := Mul(Ones(g, MakeShape(F32, 4)), Const(g, float32(3)))
		output := Dot(v1, v2)
		gradients := Gradient(output, v1, v2)
		inputs = []*Node{v1, v2}
		outputs = append([]*Node{output}, gradients...)
		return
	}, []any{
		float32(24),           // dot product output
		[]float32{3, 3, 3, 3}, // gradient with respect to v1
		[]float32{2, 2, 2, 2}, // gradient with respect to v2
	}, Epsilon)

	graphtest.RunTestGraphFn(t, "GradientDot: dot(matrix, vector)", func(g *Graph) (inputs, outputs []*Node) {
		v1 := Add(Iota(g, MakeShape(F32, 2, 4), 0), Const(g, float32(2)))
		v2 := Mul(Ones(g, MakeShape(F32, 4)), Const(g, float32(3)))
		output := Dot(v1, v2)
		sum := ReduceAllSum(output)
		gradients := Gradient(sum, v1, v2)
		inputs = []*Node{v1, v2}
		outputs = append([]*Node{output}, gradients...)
		return
	}, []any{
		[]float32{24, 36},                       // dot product output
		[][]float32{{3, 3, 3, 3}, {3, 3, 3, 3}}, // gradient with respect to v1
		[]float32{5, 5, 5, 5},                   // gradient with respect to v2
	}, Epsilon)

	graphtest.RunTestGraphFn(t, "GradientDot: dot(matrix, matrix)", func(g *Graph) (inputs, outputs []*Node) {
		v1 := Add(Iota(g, MakeShape(F32, 2, 4), 0), Const(g, float32(2)))
		v2 := Add(Iota(g, MakeShape(F32, 4, 1), 0), Const(g, float32(1)))
		output := Dot(v1, v2)
		require.NoError(t, output.Shape().CheckDims(2, 1))
		sum := ReduceAllSum(output)
		gradients := Gradient(sum, v1, v2)
		inputs = []*Node{v1, v2}
		outputs = append([]*Node{output}, gradients...)
		return
	}, []any{
		[][]float32{{20}, {30}},                 // dot product output
		[][]float32{{1, 2, 3, 4}, {1, 2, 3, 4}}, // gradient with respect to v1
		[][]float32{{5}, {5}, {5}, {5}},         // gradient with respect to v2
	}, Epsilon)
}

func TestGradientSlice(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Slice Gradient Tests",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, [][]int64{{1, 2, 3}, {4, 5, 6}})
			o1 := Slice(x)
			o2 := Slice(x, AxisRange(), AxisRange(1))
			o3 := Slice(x, AxisRange(0, -1))
			o4 := Slice(x, AxisRange(-1), AxisRange().Stride(2))
			inputs = []*Node{x, o1, o2, o3, o4}
			outputs = []*Node{
				Gradient(ReduceAllSum(o1), x)[0],
				Gradient(ReduceAllSum(o2), x)[0],
				Gradient(ReduceAllSum(o3), x)[0],
				Gradient(ReduceAllSum(o4), x)[0],
			}
			return
		}, []any{
			[][]int64{{1, 1, 1}, {1, 1, 1}},
			[][]int64{{0, 1, 1}, {0, 1, 1}},
			[][]int64{{1, 1, 1}, {0, 0, 0}},
			[][]int64{{0, 0, 0}, {1, 0, 1}},
		}, Epsilon)
}

func TestGradientGather(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Gradient Gather",
		func(g *Graph) (inputs, outputs []*Node) {
			numbers := IotaFull(g, MakeShape(F64, 5, 3))
			gather := Gather(numbers, ScalarOne(g, dtypes.Int64), true)
			gradients := Gradient(ReduceAllSum(gather), numbers)
			inputs = []*Node{numbers}
			outputs = []*Node{gather, gradients[0]}
			return
		}, []any{
			[]float64{3, 4, 5},
			[][]float64{{0, 0, 0}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
		}, Epsilon)

	graphtest.RunTestGraphFn(t, "Gradient Gather: simple leading indices dimension",
		func(g *Graph) (inputs, outputs []*Node) {
			numbers := IotaFull(g, MakeShape(F64, 5, 3))
			numbersMul := Mul(numbers, Add(IotaFull(g, MakeShape(F64, 5, 1)), Const(g, 1.0)))
			indices := Const(g, [][]int{{2}, {0}})
			gather := Gather(numbersMul, indices, false)
			gradients := Gradient(ReduceAllSum(gather), numbers)
			inputs = []*Node{numbers}
			outputs = []*Node{gather, gradients[0]}
			return
		}, []any{
			[][]float64{{6 * 3, 7 * 3, 8 * 3}, {0, 1, 2}},
			// Indices are {{2}, {0}}, where the {2} is multiplied by 3, and {0} is multiplied by 1:
			[][]float64{{1, 1, 1}, {0, 0, 0}, {3, 3, 3}, {0, 0, 0}, {0, 0, 0}},
		}, Epsilon)

	graphtest.RunTestGraphFn(t, "Gradient Gather: with 2D leading indices dimension.",
		func(g *Graph) (inputs, outputs []*Node) {
			numbers := IotaFull(g, MakeShape(F64, 5, 3))
			indices := Const(g, [][][]int{{{2}, {0}}, {{2}, {1}}})
			gather := Gather(numbers, indices, false)
			gradients := Gradient(ReduceAllSum(gather), numbers)
			inputs = []*Node{numbers}
			outputs = []*Node{gather, gradients[0]}
			return
		}, []any{
			[][][]float64{{{6, 7, 8}, {0, 1, 2}}, {{6, 7, 8}, {3, 4, 5}}},
			// Indices gathered from {0}, {1} and 2x{2}:
			[][]float64{{1, 1, 1}, {1, 1, 1}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0}},
		}, Epsilon)
}

// gradTestFunc takes a graph and returns the output being tested, along the nodes that
// we want the gradients for.
type gradTestFunc func(g *Graph) (output *Node, nodesForGrad []*Node)

// testGradients run testFn to build a graph, calculates the gradients of the ReduceAllSum(output) with respect
// to the nodesForGrad, and check that it gets close to the corresponding values in wantForGrad.
//
// It will print out the inputNodes and outputs to help debugging.
func testGradients(t *testing.T, name string, testFn gradTestFunc, wantForGrad []any) {
	testGradientsInDelta(t, name, testFn, wantForGrad, Epsilon)
}

// testGradientsInDelta run testFn to build a graph, calculates the gradients of the ReduceAllSum(output) with respect
// to the nodesForGrad, and check that it gets the corresponding values in wantForGrad, withing a delta-margin (at every element).
//
// It will print out the inputNodes and outputs to help debugging.
func testGradientsInDelta(t *testing.T, name string, testFn gradTestFunc, wantForGrad []any, delta float64) {
	backend := graphtest.BuildTestBackend()
	fmt.Printf("%s:\n", name)
	// Create a function that can be used by computation.Exec.
	fn := func(g *Graph) []*Node {
		output, nodesForGrad := testFn(g)
		grads := Gradient(ReduceAllSum(output), nodesForGrad...)
		return append([]*Node{output}, grads...)
	}
	exec := NewExec(backend, fn)
	results := exec.Call()
	fmt.Printf("\toutput=%v\n", results[0].GoStr())
	gradients := results[1:]
	for ii, output := range gradients {
		fmt.Printf("\tGradient #%d: %s\n", ii, output.GoStr())
	}
	require.Equalf(t, len(wantForGrad), len(gradients), "%s: number of wanted results different from number of gradients", name)
	for ii, output := range gradients {
		require.Truef(t, tensors.FromAnyValue(wantForGrad[ii]).InDelta(output, delta),
			"%s: gradient #%d doesn't match wanted value (withing %g delta/margin)\n\t%v", name, ii, delta, wantForGrad[ii])
	}
}

// testGradientsExact run testFn to build a graph, calculates the gradients of the
// ReduceAllSum(output) with respect to the nodesForGrad, and check that it gets the
// exact corresponding values in wantForGrad.
//
// It will print out the inputNodes and outputs to help debugging.
func testGradientsExact(t *testing.T, name string, testFn gradTestFunc, wantForGrad []any) {
	backend := graphtest.BuildTestBackend()
	fmt.Printf("%s:\n", name)
	// Create a function that can be used by computation.Exec.
	fn := func(g *Graph) []*Node {
		output, nodesForGrad := testFn(g)
		grads := Gradient(ReduceAllSum(output), nodesForGrad...)
		return append([]*Node{output}, grads...)
	}
	exec := NewExec(backend, fn)
	results := exec.Call()
	fmt.Printf("\tgradient=%v\n", results[0].GoStr())
	gradients := results[1:]
	for ii, gradient := range gradients {
		fmt.Printf("\tGradient #%d: %s\n", ii, gradient.GoStr())
	}
	require.Equalf(t, len(wantForGrad), len(gradients), "%s: number of wanted results different from number of gradients", name)
	for ii, gradient := range gradients {
		require.Equalf(t, gradient.Value(), wantForGrad[ii], "%s: gradient #%d doesn't match wanted value %#v",
			name, ii, wantForGrad[ii])
	}
}

func TestGradientConvertDType(t *testing.T) {
	testGradients(t, "gradient_of_ConvertType",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			inputs := Const(g, []float32{1e6, 1e-6, 0, -1e-8, -1e6})
			values := ConvertDType(inputs, dtypes.Float64)
			output = Mul(Const(g, []float64{2, 1, 3, -4, 5}), values)
			return output, []*Node{inputs}
		}, []any{[]float32{2, 1, 3, -4, 5}},
	)
	testGradients(t, "gradient_of_ConvertType",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			inputs := Const(g, []float32{1e6, 1e-6, 0, -1e-8, -1e6})
			values := ConvertDType(inputs, dtypes.Complex64)
			scaled := Mul(Const(g, []complex64{2, 1, 3, -4, 5}), values)
			output = ReduceAllSum(Add(Real(scaled), Imag(scaled)))
			return output, []*Node{values, inputs}
		}, []any{
			[]complex64{2 + 2i, 1 + 1i, 3 + 3i, -4 - 4i, 5 + 5i},
			[]float32{2, 1, 3, -4, 5},
		},
	)
}

func TestGradientAbs(t *testing.T) {
	testGradients(t, "TestGradientAbs",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			inputs := Const(g, []float64{1e6, 1e-6, 0, -1e-8, -1e6})
			output = Mul(Const(g, []float64{2, 1, 3, 4, 5}), Abs(inputs))
			return output, []*Node{inputs}
		}, []any{[]float64{2, 1, 3, -4, -5}},
	)

	testGradients(t, "TestGradientAbs-Complex",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			in0 := Const(g, []complex64{1 + 1i, 3 - 4i, -4 + 3i, -1 - 1i})
			in1 := Const(g, []complex128{1 + 1i, 3 - 4i, -4 + 3i, -1 - 1i})
			out0 := ReduceAllSum(Abs(in0))
			out1 := ReduceAllSum(Abs(in1))
			output = Add(ConvertDType(out0, dtypes.Float64), out1)
			return output, []*Node{in0, in1}
		}, []any{
			[]complex64{0.70710677 + 0.70710677i, 0.6 - 0.8i, -0.8 + 0.6i, -0.70710677 - 0.70710677i},
			[]complex128{0.7071067811865475 + 0.7071067811865475i, 0.6 - 0.8i, -0.8 + 0.6i, -0.7071067811865475 - 0.7071067811865475i},
		},
	)

}

func TestGradientMinMax(t *testing.T) {
	testGradients(t, "gradient_of_max",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			inputs := Const(g, []float64{1e6, 1e-6, 0, -1e-8, -1e6})
			zeros := Zeros(g, inputs.Shape())
			output = Mul(Const(g, []float64{2, 1, 3, 4, 5}), Max(inputs, zeros))
			return output, []*Node{inputs, zeros}
		}, []any{
			[]float64{2, 1, 3, 0, 0},
			[]float64{0, 0, 0, 4, 5}})
	testGradients(t, "gradient_of_min",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			inputs := Const(g, []float64{1e6, 1e-6, 0, -1e-8, -1e6})
			zeros := ScalarZero(g, inputs.DType())
			output = Mul(Const(g, []float64{2, 1, 3, 4, 5}), Min(inputs, zeros))
			return output, []*Node{inputs, zeros}
		}, []any{
			[]float64{0, 0, 0, 4, 5},
			6.0})
}

func TestGradientExp(t *testing.T) {
	testGradients(t, "gradient_of_exp",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			inputs := Const(g, []float64{6, 1, 0, -2, -3})
			output = Mul(Const(g, []float64{2, 1, 3, 4, 5}), Exp(inputs))
			return output, []*Node{inputs}
		}, []any{[]float64{2 * math.Exp(6), math.Exp(1), 3, 4 * math.Exp(-2), 5 * math.Exp(-3)}},
	)
}

func TestGradientPow(t *testing.T) {
	a := []float64{3, 1, 0.5}
	b := []float64{3, 1, -2}
	testGradientsInDelta(t, "gradient_of_pow",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			aNode, bNode := Const(g, a), Const(g, b)
			aNode = BroadcastToDims(Reshape(aNode, 1, 3), 3, 3)
			bNode = BroadcastToDims(Reshape(bNode, 3, 1), 3, 3)
			output = Pow(aNode, bNode)
			return output, []*Node{aNode, bNode}
		}, []any{
			[][]float64{{27, 3, 0.75}, {1, 1, 1}, {-0.0741, -2, -16}},
			[][]float64{{29.663, 0, -0.0866}, {3.296, 0, -0.347}, {0.122, 0, -2.77}},
		},
		0.01,
	)
}

func TestGradientLog1p(t *testing.T) {
	testGradients(t, "gradient_of_log1p",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			inputs := Const(g, []float64{0, 2, 10, 100})
			output = Mul(Const(g, []float64{2, 1, 3, 4}), Log1P(inputs))
			return output, []*Node{inputs}
		}, []any{[]float64{2, 1.0 / (2.0 + 1.0), 3.0 / (10.0 + 1.0), 4.0 / (100.0 + 1.0)}},
	)
}

func TestGradientReduceMax(t *testing.T) {
	testGradients(t, "gradient_of_reduce_max",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			inputs := Const(g, [][]float64{{1e6, 1e-6, 0, -1e-8, -1e6}, {0, 0, 0, 0, 0}})
			// ReduceMax at dimension 0: result should be {1e6, 1e-6, 0, 0, 0}.
			output = Mul(Const(g, []float64{2, 1, 3, 4, 5}), ReduceMax(inputs, 0))
			return output, []*Node{inputs}
		}, []any{
			[][]float64{{2, 1, 3, 0, 0}, {0, 0, 3, 4, 5}},
		})
}

func TestGradientBatchNorm(t *testing.T) {
	testGradients(t, "BatchNorm - offset dependent",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := Iota(g, MakeShape(dtypes.Float32, 11, 3), 0)
			input = Div(input, Const(g, float32(10))) // Values from 0.0 to 1.0 (step 0.1) on batch axis.
			scale := Const(g, []float32{1.0, 2.0, 3.0})
			offset := Const(g, []float32{1.0, 10.0, 100.0})
			var mean, variance *Node
			output, mean, variance = InternalBatchNormForTraining(input, scale, offset, 1e-7, -1)
			mean.SetLogged("mean")
			variance.SetLogged("variance")
			nodesForGrad = []*Node{input, scale, offset}
			return
		}, []any{
			// Notice the gradient of the output with respect to the inputs is ~0 because we sum everything, and if
			// one input change a bit, all outputs are changed to preserve the mean, and in the end the sum remains
			// the same.
			[][]float32{
				{0, 0, 0},
				{0, 0, 0},
				{0, 0, 0},
				{0, 0, 0},
				{0, 0, 0},
				{0, 0, 0},
				{0, 0, 0},
				{0, 0, 0},
				{0, 0, 0},
				{0, 0, 0},
				{0, 0, 0},
			},
			[]float32{0, 0, 0},
			[]float32{11, 11, 11},
		})

	graphtest.RunTestGraphFn(t, "BatchNorm gradient - operand dependent", func(g *Graph) (inputs, outputs []*Node) {
		input := Iota(g, MakeShape(dtypes.Float32, 11, 3), 0)
		input = Div(input, Const(g, float32(10))) // Values from 0.0 to 1.0 (step 0.1) on batch axis.
		inputs = []*Node{input}
		scale := Const(g, []float32{1.0, 2.0, 3.0})
		offset := Const(g, []float32{1.0, 10.0, 100.0})
		output, mean, variance := InternalBatchNormForTraining(input, scale, offset, 1e-7, -1)
		output.SetLogged("\tbatch normalized output")
		mean.SetLogged("\tmean")
		variance.SetLogged("\tvariance")
		scaleContributions := Pow(Scalar(g, input.DType(), 10.0), Iota(g, output.Shape(), -1))
		scaleContributions.SetLogged("\tcontributions")
		loss := ReduceSum(Mul(output, scaleContributions))
		loss.SetLogged("\tloss")
		outputs = Gradient(loss, input, scale, offset)
		return
	}, []any{
		[][]float32{{0, 1.3708086e-06, 1.6449703e-05}, {0, 1.0966469e-06, 1.3159763e-05}, {0, 8.2248516e-07, 9.869822e-06}, {0, 5.483234e-07, 6.579881e-06}, {0, 2.741617e-07, 3.2899404e-06}, {0, 0, 0}, {0, -2.7416178e-07, -3.2899416e-06}, {0, -5.483234e-07, -6.579881e-06}, {0, -8.2248516e-07, -9.869822e-06}, {0, -1.0966469e-06, -1.31597635e-05}, {0, -1.3708086e-06, -1.6449703e-05}},
		[]float32{0, 1.5078908e-06, 1.2063127e-05},
		[]float32{11, 110, 1100},
	}, 1e-05)
}

func TestStopGradient(t *testing.T) {
	testGradients(t, "output=\\sum{3*x_0 + 1+StopGradient(x_1)} == 15, x_0 = x_1 = [0,1,2]",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input1 := IotaFull(g, MakeShape(F64, 3))
			input2 := IotaFull(g, MakeShape(F64, 3))
			output = Add(MulScalar(input1, 3), OnePlus(StopGradient(input2)))
			output = ReduceAllSum(output) // \sum{3*[0,1,2] + [1,2,3]} == 15
			return output, []*Node{input1, input2}
		}, []any{
			[]float64{3, 3, 3},
			[]float64{0, 0, 0},
		})
}

func TestGradientGatherSlices(t *testing.T) {
	testGradients(t, "gradient_gather_slices",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := IotaFull(g, shapes.Make(dtypes.Float32, 3, 2, 4))
			start := Const(g, [][]int32{{0, 1}, {1, 2}})
			sizes := []int{1, 2} // Take a sub-matrix
			output = GatherSlices(input, []int{1, 2}, start, sizes, true)
			return output, []*Node{input}
		}, []any{
			[][][]float32{
				{{0, 1, 1, 0}, {0, 0, 1, 1}},
				{{0, 1, 1, 0}, {0, 0, 1, 1}},
				{{0, 1, 1, 0}, {0, 0, 1, 1}}},
		})
}

// TestGradientBroadcastInDim test the underlying XLA's broadcastInDim operator, since
// it powers BroadcastToShape, BroadcastToDims and ExpandAndBroadcast operators.
func TestGradientBroadcastInDim(t *testing.T) {
	testGradients(t, "broadcastInDim: scalar to outputShapes",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := Const(g, float32(1))
			output = BroadcastToDims(input, 2, 2)
			output = Mul(output, OnePlus(IotaFull(g, output.Shape())))
			return output, []*Node{input}
		}, []any{float32(10)})

	testGradients(t, "broadcastInDim: with expansion (a)",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := Const(g, []float32{10, 20})
			output = ExpandAndBroadcast(input, []int{2, 2}, []int{0})
			output = Mul(output, OnePlus(IotaFull(g, output.Shape())))
			return output, []*Node{input}
		}, []any{[]float32{4, 6}})

	testGradients(t, "broadcastInDim: with expansion (b)",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := Const(g, []float32{10, 20})
			output = ExpandAndBroadcast(input, []int{2, 2}, []int{1})
			output = Mul(output, OnePlus(IotaFull(g, output.Shape())))
			return output, []*Node{input}
		}, []any{[]float32{3, 7}})
}

// TestGradientTranspose makes sure it gets the reverse transpose correct.
func TestGradientTranspose(t *testing.T) {
	testGradients(t, "Transpose",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := Ones(g, MakeShape(F64, 2, 3, 1))
			output = TransposeAllDims(input, 1, 2, 0) // Rotate axes left.
			scale := OnePlus(IotaFull(g, MakeShape(F64, 3, 1, 2)))
			output = ReduceAllSum(Mul(output, scale))
			return output, []*Node{input}
		}, []any{
			[][][]float64{{{1}, {3}, {5}}, {{2}, {4}, {6}}},
		})
}

func TestGradientRealImagAndConj(t *testing.T) {
	testGradientsExact(t, "gradient_of_Real",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			inputs := Const(g, []complex128{1e6, 1e-6, 0, -1e-8, -1e6})
			values := Real(inputs)
			output = Mul(Const(g, []float64{2, 1, 3, 4, 5}), values)
			return output, []*Node{inputs}
		}, []any{[]complex128{2, 1, 3, 4, 5}},
	)
	testGradientsExact(t, "gradient_of_Imag",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			inputs := Const(g, []complex128{1e6i, 1e-6i, 0, -1e-8i, -1e6i})
			values := Imag(inputs)
			output = Mul(Const(g, []float64{2, 1, 3, 4, 5}), values)
			return output, []*Node{inputs}
		}, []any{[]complex128{2i, 1i, 3i, 4i, 5i}},
	)
	testGradientsExact(t, "gradient_of_Conj",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			inputs := Const(g, []complex128{1e6i, 1e-6i, 0, -1e-8i, -1e6i})
			values := Imag(Conj(inputs))
			output = Mul(Const(g, []float64{2, 1, 3, 4, 5}), values)
			return output, []*Node{inputs}
		}, []any{[]complex128{-2i, -1i, -3i, -4i, -5i}},
	)
}

func TestGradientComplex(t *testing.T) {
	testGradients(t, "gradient_of_Real",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			realPart := Const(g, []float32{1.0, 3.0})
			imagPart := Const(g, []float32{-1.0, 4.0})
			values := Complex(realPart, imagPart)
			output = Abs(Mul(Const(g, []complex64{complex64(math.Sqrt2), 5}), values))
			return output, []*Node{realPart, imagPart}
		}, []any{
			[]float32{1.0, 3},
			[]float32{-1.0, 4},
		},
	)
}

func TestIdentityWithCustomGradient(t *testing.T) {
	testGradients(t, "IdentityWithCustomGradient",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := IotaFull(g, shapes.Make(dtypes.Float32, 5))
			output = IdentityWithCustomGradient(input, func(x, v *Node) *Node {
				factor := AddScalar(Neg(x), 5)
				fmt.Printf("> custom gradient: x.outputShapes=%s, v.outputShapes=%s\n", x.Shape(), v.Shape())
				return Mul(v, factor)
			})
			output = MulScalar(output, 2)
			return output, []*Node{input}
		}, []any{
			[]float32{10, 8, 6, 4, 2},
		},
	)
}

func TestDynamicSliceGradient(t *testing.T) {
	testGradients(t, "DynamicSlice Gradient",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := IotaFull(g, shapes.Make(dtypes.Float32, 3, 2))
			startIndices := []*Node{Const(g, int32(0)), Const(g, int32(1))}
			output = DynamicSlice(input, startIndices, []int{2, 1})
			output = MulScalar(output, 7)
			return output, []*Node{input}
		}, []any{
			[][]float32{
				{0, 7},
				{0, 7},
				{0, 0},
			},
		})
}

func TestDynamicUpdateSliceGradient(t *testing.T) {
	testGradients(t, "DynamicUpdateSlice Gradient",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := IotaFull(g, shapes.Make(dtypes.Float32, 3, 2))
			update := MulScalar(Ones(g, shapes.Make(dtypes.Float32, 2, 1)), 100)
			startIndices := []*Node{Const(g, int32(0)), Const(g, int32(1))}
			output = DynamicUpdateSlice(input, update, startIndices)
			output = MulScalar(output, 11)
			return output, []*Node{input, update}
		}, []any{
			[][]float32{
				{11, 0},
				{11, 0},
				{11, 11},
			},
			[][]float32{{11}, {11}},
		})
}

func TestGradientErf(t *testing.T) {
	testGradients(t, "TestGradientErf",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			inputs := Const(g, []float64{0, -1, 2, -3, 4, -5, 6})
			output = Erf(inputs)
			return output, []*Node{inputs}
		}, []any{
			[]float64{1.1283792, 0.41510755, 0.020666987, 1.3925305e-04,
				1.2698236e-07, 1.5670867e-11, 2.6173016e-16},
		},
	)
}

func TestGradientWhere(t *testing.T) {
	testGradients(t, "Where gradient: no broadcast",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			cond := Const(g, []bool{true, false, true})
			ifTrue := Const(g, []float32{1, 1, 1})
			ifFalse := Const(g, []float32{0, 0, 0})
			output = Where(cond, ifTrue, ifFalse)
			output = Dot(OnePlus(IotaFull(g, output.Shape())), output)
			return output, []*Node{ifTrue, ifFalse}
		}, []any{
			[]float32{1, 0, 3},
			[]float32{0, 2, 0},
		},
	)

	testGradients(t, "Where gradient: broadcast from scalar #1",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			cond := Const(g, []bool{true, false, true})
			ifTrue := Const(g, float32(1))
			ifFalse := Const(g, []float32{0, 0, 0})
			output = Where(cond, ifTrue, ifFalse)
			output = Dot(OnePlus(IotaFull(g, output.Shape())), output)
			return output, []*Node{ifTrue, ifFalse}
		}, []any{
			float32(1 + 3),
			[]float32{0, 2, 0},
		},
	)

	testGradients(t, "Where gradient: broadcast from scalar #2",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			cond := Const(g, []bool{true, false, true})
			ifTrue := Const(g, float32(1))
			ifFalse := Const(g, float32(0))
			output = Where(cond, ifTrue, ifFalse)
			output = Dot(OnePlus(IotaFull(g, output.Shape())), output)
			return output, []*Node{ifTrue, ifFalse}
		}, []any{
			float32(1 + 3),
			float32(2),
		},
	)

	testGradients(t, "Where gradient: broadcast from prefix condition",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			cond := Const(g, []bool{true, false, true})
			ifTrue := IotaFull(g, shapes.Make(dtypes.Float32, 3, 2))
			ifFalse := AddScalar(IotaFull(g, shapes.Make(dtypes.Float32, 3, 2)), 100)
			output = Where(cond, ifTrue, ifFalse)
			output = ReduceAllSum(Mul(OnePlus(IotaFull(g, output.Shape())), output))
			return output, []*Node{ifTrue, ifFalse}
		}, []any{
			[][]float32{
				{1, 2},
				{0, 0},
				{5, 6},
			},
			[][]float32{
				{0, 0},
				{3, 4},
				{0, 0},
			},
		},
	)

	testGradients(t, "Where gradient: broadcast from prefix and scalar",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			cond := Const(g, []bool{true, false, true})
			ifTrue := IotaFull(g, shapes.Make(dtypes.Float32, 3, 2))
			ifFalse := Const(g, float32(100))
			output = Where(cond, ifTrue, ifFalse)
			output = ReduceAllSum(Mul(OnePlus(IotaFull(g, output.Shape())), output))
			return output, []*Node{ifTrue, ifFalse}
		}, []any{
			[][]float32{
				{1, 2},
				{0, 0},
				{5, 6},
			},
			float32(7),
		},
	)
}

func TestGradientMaskedReducedMax(t *testing.T) {
	testGradients(t, "MaskedReduceMax",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			values := Const(g, []float64{3.0, 7.0, 5.0}) // 7.0 should be masked.
			mask := Const(g, []bool{true, false, true})
			output = MaskedReduceMax(values, mask)
			return output, []*Node{values}
		}, []any{
			[]float64{0, 0, 1},
		},
	)

	// Check it is robust to NaNs.
	testGradients(t, "MaskedReduceMax with NaNs",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			values := Const(g, []float64{3.0, math.Inf(1), 5.0, math.NaN()}) // 7.0 should be masked.
			mask := Const(g, []bool{true, false, true, false})
			output = MaskedReduceMax(values, mask)
			return output, []*Node{values}
		}, []any{
			[]float64{0, 0, 1, 0},
		},
	)
}

func TestGradientMaskedReducedSum(t *testing.T) {
	testGradients(t, "MaskedReduceSum",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			values := Const(g, []float64{3.0, 7.0, 5.0}) // 7.0 should be masked.
			mask := Const(g, []bool{true, false, true})
			output = MaskedReduceSum(values, mask)
			return output, []*Node{values}
		}, []any{
			[]float64{1, 0, 1},
		},
	)

	// Check it is robust to NaNs.
	testGradients(t, "MaskedReduceSum with NaNs",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			values := Const(g, []float64{3.0, math.Inf(1), 5.0, math.NaN()}) // 7.0 should be masked.
			mask := Const(g, []bool{true, false, true, false})
			output = MaskedReduceSum(values, mask)
			return output, []*Node{values}
		}, []any{
			[]float64{1, 0, 1, 0},
		},
	)
}
