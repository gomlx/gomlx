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
	"github.com/stretchr/testify/require"
	"math"
	"testing"
)

func TestGradientAdd(t *testing.T) {
	manager := buildTestManager()
	g := manager.NewGraph().WithName("TestDense")
	c1 := Const(g, []float32{1, 2})
	c2 := Const(g, []float32{10})
	output := ReduceAllSum(Add(c1, c2))
	gradients := Gradient(output, c1, c2)
	g.Compile(output, Tuple(gradients...))
	results := g.Run(nil)
	resultsSplit := results.SplitTuple()

	{
		got := resultsSplit[0].Local().Value()
		fmt.Printf("output=%v\n", got)
		want := float32(23)
		if !xslices.DeepSliceCmp(got, want, xslices.Equal[float32]) {
			t.Fatalf("Want %v, Got %v", want, got)
		}
	}

	gradientsSplit := resultsSplit[1].SplitTuple()

	{
		got := gradientsSplit[0].Local().Value()
		fmt.Printf("\tgrad output/A=%v\n", got)
		want := []float32{1, 1}
		if !xslices.DeepSliceCmp(got, want, xslices.Equal[float32]) {
			t.Fatalf("Want %v, Got %v", want, got)
		}
	}

	{
		got := gradientsSplit[1].Local().Value()
		fmt.Printf("\tgrad output/B=%v\n", got)
		want := []float32{2}
		if !xslices.DeepSliceCmp(got, want, xslices.Equal[float32]) {
			t.Fatalf("Want %v, Got %v", want, got)
		}
	}
}

func TestGradientDot(t *testing.T) {
	manager := buildTestManager()

	// vector x vector case: simple dot product.
	{
		g := manager.NewGraph().WithName("TestDense")
		v1 := Mul(Ones(g, MakeShape(F32, 4)), Const(g, float32(2)))
		v2 := Mul(Ones(g, MakeShape(F32, 4)), Const(g, float32(3)))
		fmt.Printf("\tv1=2s, %s\n", v1)
		fmt.Printf("\tv2=3s, %s\n", v2)
		output := Dot(v1, v2)
		gradients := Gradient(output, v1, v2)
		g.Compile(output, Tuple(gradients...))
		results := g.Run(nil)
		resultsSplit := results.SplitTuple()
		{
			got := resultsSplit[0].Local()
			fmt.Printf("\toutput=%s\n", got)
			want := float32(24)
			if !xslices.DeepSliceCmp(got.Value(), want, xslices.Equal[float32]) {
				t.Fatalf("Want %v, Got %v", want, got.Value())
			}
		}
		gradientsSplit := resultsSplit[1].SplitTuple()
		{
			got := gradientsSplit[0].Local()
			fmt.Printf("\tgrad output/v1=%v\n", got)
			want := []float32{3, 3, 3, 3}
			if !xslices.DeepSliceCmp(got.Value(), want, xslices.Equal[float32]) {
				t.Fatalf("Want %v, Got %v", want, got)
			}
		}

		{
			got := gradientsSplit[1].Local()
			fmt.Printf("\tgrad output/v2=%v\n", got)
			want := []float32{2, 2, 2, 2}
			if !xslices.DeepSliceCmp(got.Value(), want, xslices.Equal[float32]) {
				t.Fatalf("Want %v, Got %v", want, got)
			}
		}
	}

	// matrix x vector case: simple dot product.
	{
		g := manager.NewGraph().WithName("TestDense")
		v1 := Add(Iota(g, MakeShape(F32, 2, 4), 0), Const(g, float32(2)))
		v2 := Mul(Ones(g, MakeShape(F32, 4)), Const(g, float32(3)))
		output := Dot(v1, v2)
		sum := ReduceAllSum(output)
		gradients := Gradient(sum, v1, v2)
		g.Compile(sum, output, v1, v2, Tuple(gradients...))
		results := g.Run(nil).SplitTuple()
		fmt.Println()
		fmt.Printf("\tv1=%v\n", results[2].Local())     // {{2, 2, 2, 2}, {3, 3, 3, 3}}
		fmt.Printf("\tv2=%v\n", results[3].Local())     // {3, 3, 3, 3}
		fmt.Printf("\toutput=%v\n", results[1].Local()) // {24, 36}
		fmt.Printf("\tsum=%v\n", results[0].Local())    // 60
		{
			got := results[0].Local()
			want := float32(60)
			if !xslices.DeepSliceCmp(got.Value(), want, xslices.Equal[float32]) {
				t.Errorf("Want %v, Got %v", want, got.Value())
			}
		}
		gradientsSplit := results[4].SplitTuple()
		{
			got := gradientsSplit[0].Local()
			fmt.Printf("\tgrad output/v1=%v\n", got)
			want := [][]float32{{3, 3, 3, 3}, {3, 3, 3, 3}}
			if !xslices.DeepSliceCmp(got.Value(), want, xslices.Equal[float32]) {
				t.Errorf("Want %v, Got %v", want, got)
			}
		}

		{
			got := gradientsSplit[1].Local()
			fmt.Printf("\tgrad output/v2=%v\n", got)
			want := []float32{5, 5, 5, 5}
			if !xslices.DeepSliceCmp(got.Value(), want, xslices.Equal[float32]) {
				t.Errorf("Want %v, Got %v", want, got)
			}
		}
	}

	// matrix x matrix case: simple dot product.
	{
		g := manager.NewGraph().WithName("TestDense")
		v1 := Add(Iota(g, MakeShape(F32, 2, 4), 0), Const(g, float32(2)))
		v2 := Add(Iota(g, MakeShape(F32, 4, 1), 0), Const(g, float32(1)))
		output := Dot(v1, v2)
		sum := ReduceAllSum(output)
		gradients := Gradient(sum, v1, v2)
		g.Compile(sum, output, v1, v2, Tuple(gradients...))
		results := g.Run(nil).SplitTuple()
		fmt.Println()
		fmt.Printf("\tv1=%v\n", results[2].Local())     // {{2, 2, 2, 2}, {3, 3, 3, 3}}
		fmt.Printf("\tv2=%v\n", results[3].Local())     // {{1}, {2}, {3}, {4}}
		fmt.Printf("\toutput=%v\n", results[1].Local()) // {20, 30}
		fmt.Printf("\tsum=%v\n", results[0].Local())    // 50
		{
			got := results[0].Local()
			want := float32(50)
			if !xslices.DeepSliceCmp(got.Value(), want, xslices.Equal[float32]) {
				t.Errorf("Want %v, Got %v", want, got.Value())
			}
		}
		gradientsSplit := results[4].SplitTuple()
		{
			got := gradientsSplit[0].Local()
			fmt.Printf("\tgrad output/v1=%v\n", got)
			want := [][]float32{{1, 2, 3, 4}, {1, 2, 3, 4}}
			if !xslices.DeepSliceCmp(got.Value(), want, xslices.Equal[float32]) {
				t.Errorf("Want %v, Got %v", want, got)
			}
		}

		{
			got := gradientsSplit[1].Local()
			fmt.Printf("\tgrad output/v2=%v\n", got)
			want := [][]float32{{5}, {5}, {5}, {5}}
			if !xslices.DeepSliceCmp(got.Value(), want, xslices.Equal[float32]) {
				t.Errorf("Want %v, Got %v", want, got)
			}
		}
	}

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
		}, xslices.Epsilon)
}

func TestGradientGather(t *testing.T) {
	manager := buildTestManager()
	{ // Trivial scalar gather.
		fmt.Println("\tGather(): trivial scalar gather.")
		g := manager.NewGraph()
		// numbers=(Float64)[5 3]: [[0 1 2] [3 4 5] [6 7 8] [9 10 11] [12 13 14]]
		numbers := IotaFull(g, MakeShape(F64, 5, 3))
		gather := Gather(numbers, ScalarOne(g, dtypes.Int64))
		gradients := Gradient(ReduceAllSum(gather), numbers)
		g.Compile(gather, gradients[0])
		results := g.Run(nil).SplitTuple()
		{
			got := results[0].Local()
			fmt.Printf("\t\tGather=%s\n", got.GoStr())
			want := []float64{3, 4, 5}
			if !xslices.DeepSliceCmp(got.Value(), want, xslices.Equal[float64]) {
				t.Errorf("Gather: want %v, got %v", want, got)
			}
		}
		{
			got := results[1].Local()
			fmt.Printf("\t\tGradient=%v\n", got.GoStr())
			want := [][]float64{{0, 0, 0}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}
			if !xslices.DeepSliceCmp(got.Value(), want, xslices.Equal[float64]) {
				t.Errorf("Gather: want %v, got %v", want, got)
			}
		}
	}

	{ // Simple leading indices dimension.
		fmt.Println("\tGather(): simple leading indices dimension.")
		g := manager.NewGraph()
		// numbers=(Float64)[5 3]: [[0 1 2] [3 4 5] [6 7 8] [9 10 11] [12 13 14]]
		numbers := IotaFull(g, MakeShape(F64, 5, 3))
		// Multiply numbers, so we can see that adjoint gradients are properly passed.
		numbersMul := Mul(numbers, Add(IotaFull(g, MakeShape(F64, 5, 1)), Const(g, 1.0)))
		indices := Const(g, [][]int{{2}, {0}})
		gather := Gather(numbersMul, indices)
		gradients := Gradient(ReduceAllSum(gather), numbers)
		g.Compile(gather, gradients[0])
		results := g.Run(nil).SplitTuple()
		{
			got := results[0].Local()
			fmt.Printf("\t\tGather=%v\n", got.GoStr())
			want := [][]float64{{6 * 3, 7 * 3, 8 * 3}, {0, 1, 2}}
			if !xslices.DeepSliceCmp(got.Value(), want, xslices.Equal[float64]) {
				t.Errorf("Gather: want %v, got %v", want, got)
			}
		}
		{
			got := results[1].Local()
			fmt.Printf("\t\tGradient=%v\n", got.GoStr())
			// Indices are {{2}, {0}}, where the {2} is multiplied by 3, and {0} is multiplied by 1:
			want := [][]float64{{1, 1, 1}, {0, 0, 0}, {3, 3, 3}, {0, 0, 0}, {0, 0, 0}}
			if !xslices.DeepSliceCmp(got.Value(), want, xslices.Equal[float64]) {
				t.Errorf("Gather: want %v, got %v", want, got)
			}
		}
	}

	{ // With 2D leading indices dimension.
		fmt.Println("\tGather(): with 2D leading indices dimension.")
		g := manager.NewGraph()
		// numbers=(Float64)[5 3]: [[0 1 2] [3 4 5] [6 7 8] [9 10 11] [12 13 14]]
		numbers := IotaFull(g, MakeShape(F64, 5, 3))
		indices := Const(g, [][][]int{{{2}, {0}}, {{2}, {1}}})
		gather := Gather(numbers, indices)
		gradients := Gradient(ReduceAllSum(gather), numbers)
		g.Compile(gather, gradients[0])
		results := g.Run(nil).SplitTuple()
		{
			got := results[0].Local()
			fmt.Printf("\t\tGather=%v\n", got.GoStr())
			want := [][][]float64{{{6, 7, 8}, {0, 1, 2}}, {{6, 7, 8}, {3, 4, 5}}}
			if !xslices.DeepSliceCmp(got.Value(), want, xslices.Equal[float64]) {
				t.Errorf("Gather: want %v, got %v", want, got)
			}
		}
		{
			got := results[1].Local()
			fmt.Printf("\t\tGradient=%v\n", got.GoStr())
			// Indices gathered from {0}, {1} and 2x{2}:
			want := [][]float64{{1, 1, 1}, {1, 1, 1}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0}}
			if !xslices.DeepSliceCmp(got.Value(), want, xslices.Equal[float64]) {
				t.Errorf("Gather: want %v, got %v", want, got)
			}
		}
	}
}

// gradTestFunc takes a graph and returns the output being tested, along the nodes that
// we want the gradients for.
type gradTestFunc func(g *Graph) (output *Node, nodesForGrad []*Node)

// testGradients run testFn to build a graph, calculates the gradients of the ReduceAllSum(output) with respect
// to the nodesForGrad, and check that it gets close to the corresponding values in wantForGrad.
//
// It will print out the inputs and outputs to help debugging.
func testGradients(t *testing.T, name string, testFn gradTestFunc, wantForGrad []any) {
	manager := buildTestManager()
	fmt.Printf("%s:\n", name)
	// Create a function that can be used by computation.Exec.
	fn := func(g *Graph) []*Node {
		output, nodesForGrad := testFn(g)
		grads := Gradient(ReduceAllSum(output), nodesForGrad...)
		all := make([]*Node, len(grads)+1)
		all[0] = output
		copy(all[1:], grads)
		return all
	}
	exec := NewExec(manager, fn)
	results := exec.Call()
	fmt.Printf("\toutput=%v\n", results[0].Local().GoStr())
	gradients := results[1:]
	for ii, output := range gradients {
		fmt.Printf("\tGradient #%d: %s\n", ii, output.Local().GoStr())
	}
	require.Equalf(t, len(wantForGrad), len(gradients), "%s: number of wanted results different from number of gradients", name)
	const delta = 1e-4
	for ii, output := range gradients {
		require.Truef(t, xslices.SlicesInDelta(output.Value(), wantForGrad[ii], delta), "%s: gradient #%d doesn't match wanted value %#v",
			name, ii, wantForGrad[ii])
	}
}

// testGradientsExact run testFn to build a graph, calculates the gradients of the
// ReduceAllSum(output) with respect to the nodesForGrad, and check that it gets the
// exact corresponding values in wantForGrad.
//
// It will print out the inputs and outputs to help debugging.
func testGradientsExact(t *testing.T, name string, testFn gradTestFunc, wantForGrad []any) {
	manager := buildTestManager()
	fmt.Printf("%s:\n", name)
	// Create a function that can be used by computation.Exec.
	fn := func(g *Graph) []*Node {
		output, nodesForGrad := testFn(g)
		grads := Gradient(ReduceAllSum(output), nodesForGrad...)
		all := make([]*Node, len(grads)+1)
		all[0] = output
		copy(all[1:], grads)
		return all
	}
	exec := NewExec(manager, fn)
	results := exec.Call()
	fmt.Printf("\toutput=%v\n", results[0].Local().GoStr())
	gradients := results[1:]
	for ii, output := range gradients {
		fmt.Printf("\tGradient #%d: %s\n", ii, output.Local().GoStr())
	}
	require.Equalf(t, len(wantForGrad), len(gradients), "%s: number of wanted results different from number of gradients", name)
	for ii, output := range gradients {
		require.Equalf(t, output.Value(), wantForGrad[ii], "%s: gradient #%d doesn't match wanted value %#v",
			name, ii, wantForGrad[ii])
	}
}

func TestGradientConvertType(t *testing.T) {
	testGradients(t, "gradient_of_ConvertType",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			inputs := Const(g, []float32{1e6, 1e-6, 0, -1e-8, -1e6})
			values := ConvertType(inputs, dtypes.Float64)
			output = Mul(Const(g, []float64{2, 1, 3, -4, 5}), values)
			return output, []*Node{inputs}
		}, []any{[]float32{2, 1, 3, -4, 5}},
	)
	testGradients(t, "gradient_of_ConvertType",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			inputs := Const(g, []float32{1e6, 1e-6, 0, -1e-8, -1e6})
			values := ConvertType(inputs, shapes.Complex64)
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
			output = Add(ConvertType(out0, dtypes.Float64), out1)
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
	testGradients(t, "BatchNorm",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := Iota(g, MakeShape(dtypes.Float32, 11, 3), 0) // Values from 0.0 to 6.0 on batch axis.
			input = Div(input, Const(g, float32(10)))
			scale := Const(g, []float32{1.0, 2.0, 3.0})
			offset := Const(g, []float32{1.0, 10.0, 100.0})
			var mean, variance *Node
			output, mean, variance = BatchNormTrainingXLA(input, scale, offset, 1e-7, -1)
			mean.SetLogged("mean")
			variance.SetLogged("variance")
			nodesForGrad = []*Node{input, scale, offset}
			return
		}, []any{
			[][]float32{
				{-1.7135108e-07, -3.4270215e-07, -5.140532e-07},
				{-1.3708086e-07, -2.7416172e-07, -4.1124258e-07},
				{-1.02810645e-07, -2.0562129e-07, -3.0843194e-07},
				{-6.854043e-08, -1.3708086e-07, -2.0562129e-07},
				{-3.4270215e-08, -6.854043e-08, -1.0281065e-07},
				{-5.10666e-15, -1.021332e-14, -1.531998e-14},
				{3.4270215e-08, 6.854043e-08, 1.0281065e-07},
				{6.8540416e-08, 1.3708083e-07, 2.0562126e-07},
				{1.02810645e-07, 2.0562129e-07, 3.0843194e-07},
				{1.3708086e-07, 2.7416172e-07, 4.112426e-07},
				{1.7135108e-07, 3.4270215e-07, 5.140532e-07}},
			[]float32{-3.769727e-07, -3.769727e-07, -3.769727e-07},
			[]float32{11, 11, 11},
		})
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
			output = GatherSlices(input, []int{1, 2}, start, sizes)
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
	testGradients(t, "broadcastInDim: scalar to shape",
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
			real := Const(g, []float32{1.0, 3.0})
			imag := Const(g, []float32{-1.0, 4.0})
			values := Complex(real, imag)
			output = Abs(Mul(Const(g, []complex64{complex64(math.Sqrt2), 5}), values))
			return output, []*Node{real, imag}
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
				fmt.Printf("> custom gradient: x.shape=%s, v.shape=%s\n", x.Shape(), v.Shape())
				return Mul(v, factor)
			})
			output = MulScalar(output, 2)
			return output, []*Node{input}
		}, []any{
			[]float32{10, 8, 6, 4, 2},
		},
	)
}
