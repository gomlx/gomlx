/*
 *	Copyright 2024 Jan Pfeifer
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
	"math"
	"reflect"
	"testing"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var (
	// Aliases:

	MakeShape = shapes.Make
	F32       = dtypes.Float32
	F64       = dtypes.Float64

	Epsilon = 1e-4
)

type graphFnOneInputToTest func(g *Graph) (input, output *Node)

// testFuncOneInput makes it easy to test a function with one input and one output. It
// compiles and executes the given graph building function graphFn and checks that the
// result is as expected.
func testFuncOneInput(t *testing.T, testName string, graphFn graphFnOneInputToTest, want any) {
	fmt.Printf("%s\n", testName)
	wantTensor := tensors.FromAnyValue(want)
	backend := graphtest.BuildTestBackend()
	g := NewGraph(backend, testName)
	inputNode, outputNode := graphFn(g)
	g.Compile(inputNode, outputNode)
	results := g.Run()
	input, got := results[0], results[1]
	if !wantTensor.InDelta(got, Epsilon) {
		t.Errorf("%s(%#v):\n\twant=%v, got=%s", testName, input, wantTensor.GoStr(), got.GoStr())
	}
}

func TestConstant(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	{
		g := NewGraph(backend, "")
		n := Const(g, 5)
		shape := n.Shape()
		if shape.DType != dtypes.Int64 || shape.Rank() != 0 {
			t.Errorf("ConstTensor has invalid outputShapes: %s", shape)
		}
	}
	{
		g := NewGraph(backend, "")
		n := Const(g, [][]float32{{1.2, 1.3}, {2.4, 2.5}, {2.6, 2.7}})
		shape := n.Shape()
		if shape.DType != dtypes.Float32 || !reflect.DeepEqual(shape.Dimensions, []int{3, 2}) {
			fmt.Printf("\tTestConstant: node %s\n", n)
			t.Errorf("ConstTensor has invalid outputShapes: %s", shape)
		}
	}
	{
		g := NewGraph(backend, "")
		t1 := tensors.FromValue([][]float32{{1.2, 1.3}, {2.4, 2.5}, {2.6, 2.7}})
		n1 := ConstCachedTensor(g, t1)
		n2 := ConstCachedTensor(g, t1)
		require.Equal(t, n1, n2)
	}
}

func compileRunAndTakeFirst(t *testing.T, g *Graph) *tensors.Tensor {
	var output *tensors.Tensor
	require.NotPanics(t, func() {
		g.Compile(g.LastNode())
		output = g.Run()[0]
	})
	return output
}

func TestAdd(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	{
		// Test scalars.
		g := NewGraph(backend, "scalar graph")
		x := Const(g, 5)
		y := Const(g, 7)
		n := Add(x, y)
		wantShape := shapes.Shape{DType: dtypes.Int64}
		require.Truef(t, n.Shape().Equal(wantShape), "Add invalid outputShapes %s, wanted %s", n.Shape(), wantShape)
		gotTensor := compileRunAndTakeFirst(t, g)
		got := tensors.ToScalar[int64](gotTensor)
		if got != 12 {
			fmt.Printf("%s\n", g)
			fmt.Printf("\tResult: %d %s\n", got, gotTensor.Shape())
			t.Errorf("Wanted 5 + 7 = 12, got %d", got)
		}
	}
	{
		// Test multi-dimension arrays.
		g := NewGraph(backend, "[2, 2] Graph")
		x := Const(g, [][]float32{{1.1, 1.2}, {1.3, 1.4}})
		y := Const(g, [][]float32{{10, 10}, {20, 20}})
		n := Add(x, y)
		wantShape := shapes.Make(dtypes.Float32, 2, 2)
		if !n.Shape().Equal(wantShape) {
			t.Fatalf("Add invalid outputShapes %s, wanted %s", n.Shape(), wantShape)
		}
		got := compileRunAndTakeFirst(t, g)
		want := tensors.FromValue([][]float32{{11.1, 11.2}, {21.3, 21.4}})
		if !want.Equal(got) {
			fmt.Printf("%s\n", g)
			fmt.Printf("\tResult: %v %s\n", got, got.Shape())
			t.Errorf("Wanted %v, got %v", want, got)
		}
	}
	{
		// Test multi-dimension arrays of same rank with broadcast.
		g := NewGraph(backend, "[2, 2] Graph")
		x := Const(g, [][]float32{{1.1, 1.2}, {1.3, 1.4}})
		y := Const(g, [][]float32{{1}, {10}})
		n := Add(x, y)
		wantShape := shapes.Make(dtypes.Float32, 2, 2)
		if !n.Shape().Equal(wantShape) {
			t.Fatalf("Add invalid outputShapes %s, wanted %s", n.Shape(), wantShape)
		}
		local := compileRunAndTakeFirst(t, g)
		got := local.Value().([][]float32)
		want := [][]float32{{2.1, 2.2}, {11.3, 11.4}}
		if !reflect.DeepEqual(got, want) {
			fmt.Printf("%s\n", g)
			fmt.Printf("\tResult: %v %s\n", got, local.Shape())
			t.Errorf("Wanted %v, got %v", want, got)
		}
	}
	{
		// Test add multi-dimension array with a scalar (different ranks).
		g := NewGraph(backend, "[2, 2] Graph")
		x := Const(g, [][]float32{{1.1, 1.2}, {1.3, 1.4}})
		y := Const(g, float32(1))
		n := Add(x, y)
		wantShape := shapes.Make(dtypes.Float32, 2, 2)
		if !n.Shape().Equal(wantShape) {
			t.Fatalf("Add invalid outputShapes %s, wanted %s", n.Shape(), wantShape)
		}
		local := compileRunAndTakeFirst(t, g)
		got := local.Value().([][]float32)
		want := [][]float32{{2.1, 2.2}, {2.3, 2.4}}
		if !reflect.DeepEqual(got, want) {
			fmt.Printf("%s\n", g)
			fmt.Printf("\tResult: %v %s\n", got, local.Shape())
			t.Errorf("Wanted %v, got %v", want, got)
		}
	}
}

func TestParameter(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	// Test passing of values.
	{
		// Test scalars.
		g := NewGraph(backend, "")
		x := Parameter(g, "x", shapes.Scalar[float32]())
		y := Parameter(g, "y", shapes.Scalar[float32]())
		Add(x, y)
		if x.GetParameterHandle() == InvalidParameterHandle || y.GetParameterHandle() == InvalidParameterHandle || x.GetParameterHandle() == y.GetParameterHandle() {
			t.Fatalf("Invalid parameter handles: x=%d, y=%d", x.GetParameterHandle(), y.GetParameterHandle())
		}
		g.Compile(g.LastNode())

		// Tests for various parameters.
		for xV := float32(0); xV < 3; xV += 1 {
			for yV := float32(0); yV < 3; yV += 1 {
				gotTensor := g.RunWithMap(ParamsMap{x: xV, y: yV})[0]
				got := tensors.ToScalar[float32](gotTensor)
				if got != xV+yV {
					fmt.Printf("%s\n", g)
					t.Errorf("%f + %f : got %s, wanted %f", xV, yV, gotTensor, xV+yV)
				}
			}
		}
	}
}

func TestConvertDType(t *testing.T) {
	// Test that number can be converted to complex types.
	wantF32 := []float32{3.0, -5.0}
	wantF64 := []float64{-7.0, 11.0}
	graphtest.RunTestGraphFn(t, "ConvertToComplex", func(g *Graph) (inputs, outputs []*Node) {
		inputs = []*Node{
			Const(g, wantF32),
			Const(g, wantF64),
		}
		c64 := ConvertDType(inputs[0], dtypes.Complex64)
		assert.Equal(t, dtypes.Complex64, c64.DType())
		c128 := ConvertDType(inputs[1], dtypes.Complex128)
		assert.Equal(t, dtypes.Complex128, c128.DType())
		outputs = []*Node{Real(c64), Real(c128)}
		return
	}, []any{wantF32, wantF64}, -1)
}

type TwoArgsTestCase[T dtypes.Supported] struct {
	fnGraph  func(x, y *Node) *Node
	fnScalar func(x, y T) T
	name     string
}

func TestTwoArgsOps(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	{
		casesFloat32 := []TwoArgsTestCase[float32]{
			{Mul, func(x, y float32) float32 { return x * y }, "Mul"},
			{Sub, func(x, y float32) float32 { return x - y }, "Sub"},
			{Div, func(x, y float32) float32 { return x / y }, "Div"},
			{Mod, func(x, y float32) float32 { return float32(math.Mod(float64(x), float64(y))) }, "Mod"},
			{Min, func(x, y float32) float32 { return min(x, y) }, "Min"},
			{Max, func(x, y float32) float32 { return max(x, y) }, "Max"},
			{Pow, func(x, y float32) float32 {
				return float32(math.Pow(float64(x), float64(y)))
			}, "Pow"},
		}
		xSlices := [][]float32{{11, 12}, {13, 14}}
		yValue := float32(3)
		for _, test := range casesFloat32 {
			g := NewGraph(backend, "TwoArgsOps [2, 2] Graph")
			x := Const(g, xSlices)
			y := Const(g, yValue)
			n := test.fnGraph(x, y)
			wantShape := shapes.Make(dtypes.Float32, 2, 2)
			if !n.Shape().Equal(wantShape) {
				t.Fatalf("Add invalid outputShapes %s, wanted %s", n.Shape(), wantShape)
			}
			local := compileRunAndTakeFirst(t, g)
			got := local.Value().([][]float32)
			want := [][]float32{{11, 12}, {13, 14}}
			for _, s1 := range want {
				for ii := range s1 {
					s1[ii] = test.fnScalar(s1[ii], yValue)
				}
			}
			if !reflect.DeepEqual(got, want) {
				fmt.Printf("%s\n", g)
				fmt.Printf("\tResult: %v %s\n", got, local.Shape())
				t.Errorf("Wanted %v, got %v", want, got)
			}
		}
	}

	{
		casesInt := []TwoArgsTestCase[int64]{
			{BitwiseAnd, func(x, y int64) int64 { return x & y }, "BitwiseAnd"},
			{BitwiseOr, func(x, y int64) int64 { return x | y }, "BitwiseOr"},
			{BitwiseXor, func(x, y int64) int64 { return x ^ y }, "BitwiseXor"},
		}
		xSlices := [][]int64{{11, 12}, {13, 14}}
		yValue := int64(3)
		for _, test := range casesInt {
			g := NewGraph(backend, "TwoArgs: bitwise functions [2, 2] Graph")
			x := Const(g, xSlices)
			y := Const(g, yValue)
			n := test.fnGraph(x, y)
			wantShape := shapes.Make(dtypes.Int64, 2, 2)
			if !n.Shape().Equal(wantShape) {
				t.Fatalf("Add invalid outputShapes %s, wanted %s", n.Shape(), wantShape)
			}
			local := compileRunAndTakeFirst(t, g)
			got := local.Value().([][]int64)
			want := [][]int64{{11, 12}, {13, 14}}
			for _, s1 := range want {
				for ii := range s1 {
					s1[ii] = test.fnScalar(s1[ii], yValue)
				}
			}
			if !reflect.DeepEqual(got, want) {
				fmt.Printf("%s\n", g)
				fmt.Printf("\tResult: %v %s\n", got, local.Shape())
				t.Errorf("Wanted %v, got %v", want, got)
			}
		}
	}

	{
		casesBool := []TwoArgsTestCase[bool]{
			{LogicalAnd, func(x, y bool) bool { return x && y }, "LogicalAnd"},
			{LogicalOr, func(x, y bool) bool { return x || y }, "LogicalOr"},
			{LogicalXor, func(x, y bool) bool { return x != y }, "LogicalXor"},
		}
		for _, boolCase := range casesBool {
			exec := NewExec(backend, boolCase.fnGraph)
			for _, x := range []bool{true, false} {
				for _, y := range []bool{true, false} {
					got := tensors.ToScalar[bool](exec.Call(x, y)[0])
					want := boolCase.fnScalar(x, y)
					require.Equal(t, want, got, "%s(%v,%v)=%v, wanted %v", boolCase.name, x, y, got, want)
				}
			}
		}
	}
}

type OneArgTestCase[T dtypes.Supported] struct {
	fnGraph    func(x *Node) *Node
	goFnScalar func(x T) T
}

func TestOneArgOps(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	casesFloat64 := []OneArgTestCase[float64]{
		{Abs, func(x float64) float64 { return math.Abs(x) }},
		{Neg, func(x float64) float64 { return -x }},
		{Exp, func(x float64) float64 { return math.Exp(x) }},
		{Expm1, func(x float64) float64 { return math.Expm1(x) }},
		{Floor, func(x float64) float64 { return math.Floor(x) }},
		{Ceil, func(x float64) float64 { return math.Ceil(x) }},
		{Round, func(x float64) float64 { return math.Round(x) }},
		{Log, func(x float64) float64 { return math.Log(x) }},
		{Log1P, func(x float64) float64 { return math.Log1p(x) }},
		{Sign, func(x float64) float64 {
			if math.Signbit(x) {
				return -1
			} else {
				return 1
			}
		}},
		{Logistic, func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }},
		{Sigmoid, func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }},
		{Cos, func(x float64) float64 { return math.Cos(x) }},
		{Sin, func(x float64) float64 { return math.Sin(x) }},
		{Tanh, func(x float64) float64 { return math.Tanh(x) }},
		{Sqrt, func(x float64) float64 { return math.Sqrt(x) }},
		{Rsqrt, func(x float64) float64 { return 1.0 / math.Sqrt(x) }},
	}
	xSlices := [][]float64{{11.1, 12.8}, {-13.2, -14.9}}
	for _, test := range casesFloat64 {
		g := NewGraph(backend, "[2, 2] graph for one-arg operation")
		x := Const(g, xSlices)
		n := test.fnGraph(x)
		wantShape := shapes.Make(dtypes.Float64, 2, 2)
		if !n.Shape().Equal(wantShape) {
			t.Fatalf("Add invalid outputShapes %s, wanted %s", n.Shape(), wantShape)
		}
		local := compileRunAndTakeFirst(t, g)
		got := local.Value().([][]float64)
		want := [][]float64{{0, 0}, {0, 0}}
		for i0, x0Slice := range xSlices {
			for i1, value := range x0Slice {
				want[i0][i1] = test.goFnScalar(value)
			}
		}
		if !xslices.DeepSliceCmp(got, want, xslices.Close[float64]) {
			fmt.Printf("%s\n", g)
			fmt.Printf("\tResult: %v %s\n", got, local.Shape())
			t.Errorf("Wanted %v, got %v", want, got)
		}
	}

	// Test imag/real for complex numbers.
	graphtest.RunTestGraphFn(t, "RealImagConj()", func(g *Graph) (inputs, outputs []*Node) {
		inputs = []*Node{Const(g, []complex64{1.0, 0.0 - 1.0i, -2.0 + 2.0i})}
		outputs = []*Node{Real(inputs[0]), Imag(inputs[0]), Conj(inputs[0])}
		return
	}, []any{
		[]float32{1.0, 0.0, -2.0},
		[]float32{0.0, -1.0, 2.0},
		[]complex64{1.0, 0.0 + 1.0i, -2.0 - 2.0i},
	}, -1)

	// Test Not ops
	graphtest.RunTestGraphFn(t, "LogicalNot", func(g *Graph) (inputs, outputs []*Node) {
		inputs = []*Node{Const(g, []bool{false, true})}
		outputs = []*Node{LogicalNot(inputs[0])}
		return
	}, []any{
		[]bool{true, false},
	}, -1)
	graphtest.RunTestGraphFn(t, "BitwiseNot", func(g *Graph) (inputs, outputs []*Node) {
		inputs = []*Node{Const(g, []uint8{12, 255, 243, 0})}
		outputs = []*Node{BitwiseNot(inputs[0])}
		return
	}, []any{
		[]uint8{243, 0, 12, 255},
	}, -1)
}

func TestClzOp(t *testing.T) {
	testFuncOneInput(t, "Clz()",
		func(g *Graph) (input, output *Node) {
			input = Const(g, []int64{16, 14})
			output = Clz(input)
			return
		}, []int64{64 - 5, 64 - 4})
}

func TestDot(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	g := NewGraph(backend, "").WithName("Dot")

	// Shape: [batch=4, dims=3]
	inputs := Const(g, [][]float32{{1.1, 2.2, 3.3}, {11, 22, 33}, {111, 222, 333}, {1111, 2222, 3333}})
	// Layer 0: outputShapes [3, 2], that is the inputNodes have dim=3, and should output dims=2
	w0 := Const(g, [][]float32{{1, 0}, {1, -1}, {-1, 1}})
	// Dot(inputNodes, w0) -> outputShapes [batch=4, dims=2]
	Dot(inputs, w0) // Last node created in the graph is taken as output by default.
	got := compileRunAndTakeFirst(t, g)
	want := tensors.FromValue([][]float32{{0, 1.1}, {0, 11}, {0, 111}, {0, 1111}})
	if !want.InDelta(got, Epsilon) {
		fmt.Printf("%s\n", g)
		fmt.Printf("\tResult=%v\n", got)
		t.Errorf("Wanted %v, got %v", want, got)
	}
}

func TestBroadcast(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	{
		g := NewGraph(backend, "BroadcastToDims")
		input := Const(g, 7)
		BroadcastToDims(input, 2, 3) // Last node created in the graph is taken as output by default.
		got := compileRunAndTakeFirst(t, g)
		want := [][]int64{{7, 7, 7}, {7, 7, 7}}
		assert.Equal(t, want, got.Value())
	}

	{
		g := NewGraph(backend, "BroadcastPrefix")
		input := Const(g, []float32{1.1, 1.2})
		BroadcastPrefix(input, 2, 1) // The last node created in the graph is taken as output by default.
		got := compileRunAndTakeFirst(t, g)
		want := [][][]float32{{{1.1, 1.2}}, {{1.1, 1.2}}} // Shape [2, 1, 2].
		assert.Equal(t, want, got.Value())
	}

	// Using now the new testFuncOneInput testing tool:
	testFuncOneInput(t, "ExpandAndBroadcast()",
		func(g *Graph) (input, output *Node) {
			input = Const(g, []int32{10, 20})
			output = ExpandAndBroadcast(input, []int{2, 2}, []int{0})
			return
		}, [][]int32{{10, 20}, {10, 20}})
	testFuncOneInput(t, "ExpandAndBroadcast()",
		func(g *Graph) (input, output *Node) {
			input = Const(g, []int32{10, 20})
			output = ExpandAndBroadcast(input, []int{2, 2}, []int{1})
			return
		}, [][]int32{{10, 10}, {20, 20}})

	// Test broadcasting new axes as well as existing axes with dimension 1.
	graphtest.RunTestGraphFn(t, "ExpandAndBroadcast() 2", func(g *Graph) (inputs, outputs []*Node) {
		inputs = []*Node{Const(g, [][]float32{{1, 2}})}
		outputs = []*Node{ExpandAndBroadcast(inputs[0], []int{3, 2, 2}, []int{0})}
		return
	}, []any{
		[][][]float32{
			{{1, 2}, {1, 2}},
			{{1, 2}, {1, 2}},
			{{1, 2}, {1, 2}},
		},
	}, -1)

}

// TestBroadcastToDims and BroadcastToShape.
func TestBroadcastToDims(t *testing.T) {
	graphtest.RunTestGraphFn(t, "BroadcastToDims: scalar", func(g *Graph) (inputs, outputs []*Node) {
		inputs = []*Node{Scalar(g, dtypes.Float32, 3.0)}
		outputs = []*Node{BroadcastToDims(inputs[0], 1, 3)}
		return
	}, []any{
		[][]float32{{3, 3, 3}},
	}, -1)

	graphtest.RunTestGraphFn(t, "BroadcastToDims: expansion", func(g *Graph) (inputs, outputs []*Node) {
		// DType of shape should be ignored, and an extra dimension should be added to the input.
		inputs = []*Node{Const(g, []float32{3})}
		outputs = []*Node{BroadcastToShape(inputs[0], shapes.Make(dtypes.Bool, 1, 3))}
		return
	}, []any{
		[][]float32{{3, 3, 3}},
	}, -1)

	graphtest.RunTestGraphFn(t, "BroadcastToDims: normal", func(g *Graph) (inputs, outputs []*Node) {
		// DType of shape should be ignored, and an extra dimension should be added to the input.
		inputs = []*Node{Const(g, [][]float32{{3}})}
		outputs = []*Node{BroadcastToDims(inputs[0], 3, 1)}
		return
	}, []any{
		[][]float32{{3}, {3}, {3}},
	}, -1)

}

func TestFill(t *testing.T) {
	testFuncOneInput(t, "FillScalar", func(g *Graph) (input, output *Node) {
		input = FillScalar(g, shapes.Make(dtypes.Int64, 3, 1), 4.0)
		output = input
		return
	}, [][]int64{{4}, {4}, {4}})

	testFuncOneInput(t, "Ones", func(g *Graph) (input, output *Node) {
		input = Ones(g, shapes.Make(dtypes.Float32, 3, 1))
		output = input
		return
	}, [][]float32{{1}, {1}, {1}})

	testFuncOneInput(t, "Zeros", func(g *Graph) (input, output *Node) {
		input = Zeros(g, shapes.Make(dtypes.Float64, 3, 1))
		output = input
		return
	}, [][]float64{{0}, {0}, {0}})
}

func reduceSumGraph(t *testing.T, backend backends.Backend, reduceDims []int) *Graph {
	var g *Graph
	require.NotPanics(t, func() {
		g = NewGraph(backend, "main")
		n0 := Const(g, [][]float64{{5.0, 1.0}})
		n1 := Ones(g, shapes.Make(dtypes.Float64, 2, 1))
		n2 := Add(n1, n0)
		o0 := ReduceSum(n2, reduceDims...)
		g.Compile(o0)
	})
	return g
}

func TestReduceSum(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	cases := []struct {
		dims []int
		want any
	}{
		{want: 16.0},
		{dims: []int{0}, want: []float64{12, 4}},
		{dims: []int{1}, want: []float64{8, 8}},
	}
	for _, testCase := range cases {
		g := reduceSumGraph(t, backend, testCase.dims)
		got := g.Run()[0]
		wantT := tensors.FromAnyValue(testCase.want)
		if !wantT.InDelta(got, Epsilon) {
			t.Errorf("Wanted %v, got %v", testCase.want, got)
		}
	}
}

func TestReduceMean(t *testing.T) {
	testFuncOneInput(t, "ReduceMean(dims=1, 2)",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, MakeShape(dtypes.Float32, 3, 2, 4))
			output = ReduceMean(input, 1, 2)
			return
		}, []float32{3.5, 11.5, 19.5})
	testFuncOneInput(t, "ReduceAllMean()",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, MakeShape(dtypes.Float32, 3, 2, 4))
			output = ReduceAllMean(input)
			return
		}, float32(11.5))
	graphtest.RunTestGraphFn(t, "ReduceMean",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{
				IotaFull(g, shapes.Make(dtypes.Float32, 3, 5)),
				Const(g, [][]bool{
					{true, true, true, true, true},
					{true, true, false, false, false},
					{false, false, false, false, false}}),
			}
			outputs = []*Node{
				MaskedReduceMean(inputs[0], inputs[1], -1),
			}
			return
		}, []any{
			[]float32{2.0, 5.5, 0.0},
		}, -1)
}

func TestReduceMax(t *testing.T) {
	graphtest.RunTestGraphFn(t, "ReduceMax()",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, []float64{1, 5, math.Inf(-1)})
			output := ReduceMax(x)
			inputs = []*Node{x}
			outputs = []*Node{output}
			return
		}, []any{5.0}, xslices.Epsilon)

	// float64 NaN
	backend := graphtest.BuildTestBackend()

	// float32 NaN:
	// It works if input is passed as a constant:
	{
		gotT := ExecOnce(backend, func(g *Graph) *Node {
			return ReduceMax(Const(g, []float64{math.NaN(), 1}))
		})
		got := tensors.ToScalar[float64](gotT)
		require.Truef(t, math.IsNaN(float64(got)), "ReduceMax({NaN, 1}) of NaN values should be NaN, got %v", got)
	}

	// But it doesn't if tensor to reduce is passed as a parameter.
	{
		gotT := ExecOnce(backend, func(x *Node) *Node {
			return ReduceMax(x)
		}, []float64{math.NaN(), 1})
		got := tensors.ToScalar[float64](gotT)
		// TODO: PJRT for CPU is bugged here, and it fails in this example (it works for CUDA). So commented out for now.
		// See https://github.com/openxla/xla/issues/21461
		//	require.Truef(t, math.IsNaN(got), "ReduceMax of NaN values should be NaN.")
		if !math.IsNaN(got) {
			fmt.Printf("ReduceMax({NaN, 1}): should be NaN, got %v -- see issue in https://github.com/openxla/xla/issues/21461\n", got)
		}
	}
}

func TestMaskedReduceMax(t *testing.T) {
	graphtest.RunTestGraphFn(t, "MaskedReduceMax()",
		func(g *Graph) (inputs, outputs []*Node) {
			x := IotaFull(g, MakeShape(dtypes.Float32, 4, 3))
			mask := Const(g, [][]bool{
				{true, false, false},
				{true, true, false},
				{true, false, true},
				{true, true, true}})
			output := MaskedReduceMax(x, mask, 1)
			inputs = []*Node{x, mask}
			outputs = []*Node{output}
			return
		}, []any{[]float32{0, 4, 8, 11}}, xslices.Epsilon)
}

func TestLogicalAllAndAny(t *testing.T) {
	graphtest.RunTestGraphFn(t, "TestLogicalAllAndAny", func(g *Graph) (inputs, outputs []*Node) {
		inputs = []*Node{
			Const(g, [][]bool{
				{false, false, false},
				{false, true, false},
				{true, false, true},
				{true, true, true},
			}),
		}
		outputs = []*Node{
			LogicalAll(inputs[0], -1),
			LogicalAny(inputs[0], -1),
		}
		return
	}, []any{
		[]bool{false, false, false, true},
		[]bool{false, true, true, true},
	}, 0)
}

func TestReshape(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	{
		g := NewGraph(backend, "")
		input := Const(g, [][][]float32{{{1.1, 1.2}}}) // Shape [1, 1, 2]
		ReshapeWithShape(input, shapes.Make(input.DType(), 2, 1))
		got := compileRunAndTakeFirst(t, g)
		want := tensors.FromValue([][]float32{{1.1}, {1.2}})
		if !want.Equal(got) {
			fmt.Printf("%s\n", g)
			fmt.Printf("\tResult=%v\n", got)
			t.Errorf("Wanted %v, got %v", want, got)
		}
	}
}

func TestIota(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	{
		g := NewGraph(backend, "").WithName("iota0")
		Iota(g, MakeShape(F64, 2, 2), 0)
		g.Compile(g.LastNode())
		got := g.Run()[0]
		want := tensors.FromValue([][]float64{{0, 0}, {1, 1}})
		if !want.Equal(got) {
			t.Fatalf("Iota: want %v, got %v", want, got)
		}
	}
	{
		g := NewGraph(backend, "").WithName("iota0")
		Iota(g, MakeShape(F64, 2, 2), 1)
		g.Compile(g.LastNode())
		got := g.Run()[0]
		want := tensors.FromValue([][]float64{{0, 1}, {0, 1}})
		if !want.Equal(got) {
			t.Fatalf("Iota: want %v, got %v", want, got)
		}
	}
}

func TestSlice(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Slice Tests with Rank 1",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, []int64{1, 2, 3, 4})
			inputs = []*Node{x}
			outputs = []*Node{
				Slice(x),
				Slice(x, AxisRange()),
				Slice(x, AxisRange(2)),
				Slice(x, AxisRange(1, -1)),
				Slice(x, AxisRange().Stride(2)),
				Slice(x, AxisElem(2)),
			}
			return
		}, []any{
			[]int64{1, 2, 3, 4},
			[]int64{1, 2, 3, 4},
			[]int64{3, 4},
			[]int64{2, 3},
			[]int64{1, 3},
			[]int64{3},
		}, xslices.Epsilon)

	graphtest.RunTestGraphFn(t, "Slice Tests with Rank 1",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, [][]int32{{1, 2, 3}, {4, 5, 6}})
			inputs = []*Node{x}
			outputs = []*Node{
				Slice(x, AxisRange(), AxisElem(0)),
				Slice(x, AxisRange(1, 2)),
				Slice(x, AxisRange().Stride(2), AxisElem(-1)),
				SliceAxis(x, -1, AxisElem(2)),
			}
			return
		}, []any{
			[][]int32{{1}, {4}},
			[][]int32{{4, 5, 6}},
			[][]int32{{3}},
			[][]int32{{3}, {6}},
		}, xslices.Epsilon)

	graphtest.RunTestGraphFn(t, "Slice Tests with Rank 1",
		func(g *Graph) (inputs, outputs []*Node) {
			x := IotaFull(g, shapes.Make(dtypes.Int64, 2, 2, 2, 2))
			inputs = []*Node{x}
			outputs = []*Node{
				Slice(x, AxisRange(), AxisElem(0).Spacer(), AxisElem(-1)),

				// Check that a spacer matches 0 elements also.
				Slice(x, AxisElem(0), AxisElem(0), AxisRange().Spacer(),
					AxisElem(0), AxisElem(0)),
			}
			return
		}, []any{
			[][][][]int64{{{{1}}}, {{{9}}}},
			[][][][]int64{{{{0}}}},
		}, xslices.Epsilon)

}

func TestPad(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Slice Tests with Rank 1",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, [][]int64{{1, 2}, {3, 4}})
			zero := ScalarZero(g, x.DType())
			inputs = []*Node{x, zero}
			outputs = []*Node{
				Pad(x, zero),
				Pad(x, zero, PadAxis{}, PadAxis{Start: 1, Interior: 1}),
			}
			return
		}, []any{
			[][]int64{{1, 2}, {3, 4}},
			[][]int64{{0, 1, 0, 2}, {0, 3, 0, 4}},
		}, Epsilon)
}

func TestConcatenate(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	{
		fmt.Println("\tConcatenate(): 1D concatenation.")
		g := NewGraph(backend, "")
		// numbers=(Float64)[3]: [2 3 4]
		x1 := IotaFull(g, MakeShape(F64, 3))
		x2 := Add(IotaFull(g, MakeShape(F64, 5)), Const(g, float64(3)))
		concat := Concatenate([]*Node{x1, x2}, 0)
		g.Compile(concat)
		got := g.Run()[0]
		fmt.Printf("\t\tresult=%s\n", got.GoStr())
		want := tensors.FromValue([]float64{0, 1, 2, 3, 4, 5, 6, 7})
		if !want.Equal(got) {
			t.Errorf("scatter: want %v, got %v", want, got)
		}
	}
	{
		fmt.Println("\tConcatenate(): 3D concatenation at middle dimension.")
		g := NewGraph(backend, "")
		// numbers=(Float64)[3]: [2 3 4]
		x1 := IotaFull(g, MakeShape(F64, 2, 2, 2))
		x2 := Add(IotaFull(g, MakeShape(F64, 2, 1, 2)), Const(g, float64(8)))
		concat := Concatenate([]*Node{x1, x2}, 1)
		g.Compile(concat)
		got := g.Run()[0]
		fmt.Printf("\t\tresult=%s\n", got.GoStr())
		want := tensors.FromValue([][][]float64{{{0, 1}, {2, 3}, {8, 9}}, {{4, 5}, {6, 7}, {10, 11}}})
		if !want.Equal(got) {
			t.Errorf("scatter: want %v, got %v", want, got)
		}
	}
}

func TestStack(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Stack",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{
				Zeros(g, shapes.Make(dtypes.Int32, 3)),
				Ones(g, shapes.Make(dtypes.Int32, 3)),
			}
			outputs = []*Node{
				Stack(inputs, 0),
				Stack(inputs, 1),
				Stack(inputs, -1),
			}
			return
		}, []any{
			[][]int32{{0, 0, 0}, {1, 1, 1}},
			[][]int32{{0, 1}, {0, 1}, {0, 1}},
			[][]int32{{0, 1}, {0, 1}, {0, 1}},
		}, -1)
}

func TestNonNegativeIndicator(t *testing.T) {
	testFuncOneInput(t, "NonNegativeIndicator",
		func(g *Graph) (input, output *Node) {
			input = Const(g, []float64{1.0, 0.0001, 0, -0.2, -3.0})
			output = NonNegativeIndicator(input)
			return
		}, []float64{1, 1, 1, 0, 0})
}

func TestPositiveIndicator(t *testing.T) {
	testFuncOneInput(t, "PositiveIndicator",
		func(g *Graph) (input, output *Node) {
			input = Const(g, []float64{1.0, 0.0001, 0, -0.2, -3.0})
			output = PositiveIndicator(input)
			return
		}, []float64{1, 1, 0, 0, 0})
}

func TestNonPositiveIndicator(t *testing.T) {
	testFuncOneInput(t, "NonPositiveIndicator",
		func(g *Graph) (input, output *Node) {
			input = Const(g, []float64{1.0, 0.0001, 0, -0.2, -3.0})
			output = NonPositiveIndicator(input)
			return
		}, []float64{0, 0, 1, 1, 1})
}

func TestNegativeIndicator(t *testing.T) {
	testFuncOneInput(t, "NegativeIndicator",
		func(g *Graph) (input, output *Node) {
			input = Const(g, []float64{1.0, 0.0001, 0, -0.2, -3.0})
			output = NegativeIndicator(input)
			return
		}, []float64{0, 0, 0, 1, 1})
}

func TestReduceAndKeep(t *testing.T) {
	testFuncOneInput(t, "TestReduceAndKeep last dimension",
		func(g *Graph) (input, output *Node) {
			input = Const(g, [][]float64{{1, 0, 3}, {2, -1, 1}})
			output = ReduceAndKeep(input, ReduceSum, -1)
			return
		}, [][]float64{{4}, {2}})
	testFuncOneInput(t, "TestReduceAndKeep middle dimension",
		func(g *Graph) (input, output *Node) {
			input = Const(g, [][][]float32{{{1, 0, 3}, {2, -1, 1}}})
			output = ReduceAndKeep(input, ReduceMax, -2)
			return
		}, [][][]float32{{{2, 0, 3}}})
	testFuncOneInput(t, "TestReduceAndKeep first dimension",
		func(g *Graph) (input, output *Node) {
			input = Const(g, [][][]float32{{{1, 0, 3}, {2, -1, 1}}})
			output = ReduceAndKeep(input, ReduceMax, 0)
			return
		}, [][][]float32{{{1, 0, 3}, {2, -1, 1}}}) // Nothing happened here, since dimensions[0] == 1, nothing to reduce.
}

func TestReverse(t *testing.T) {
	testFuncOneInput(t, "Reverse(dimensions={1, 2})",
		func(g *Graph) (input, output *Node) {
			input = Iota(g, MakeShape(dtypes.Float32, 9), 0)
			input = Reshape(input, 1, 3, 3, 1)
			output = Reverse(input, 1, 2)
			return
		}, [][][][]float32{{{{8}, {7}, {6}}, {{5}, {4}, {3}}, {{2}, {1}, {0}}}})
}

func TestTranspose(t *testing.T) {
	testFuncOneInput(t, "Transpose(dims=1, 2)",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, MakeShape(dtypes.Float32, 3, 2, 4))
			output = Transpose(input, 1, 2)
			return
		}, [][][]float32{{{0, 4}, {1, 5}, {2, 6}, {3, 7}}, {{8, 12}, {9, 13}, {10, 14}, {11, 15}}, {{16, 20}, {17, 21}, {18, 22}, {19, 23}}})
}

func TestInternalBatchNormForInference(t *testing.T) {
	testFuncOneInput(t, "BatchNormInference()",
		func(g *Graph) (input, output *Node) {
			input = Iota(g, MakeShape(dtypes.Float32, 7, 3), 0) // Values from 0.0 to 6.0 on batch axis.
			scale := Const(g, []float32{1.0, 2.0, 3.0})
			offset := Const(g, []float32{10.0, 100.0, 1000.0})
			mean := Const(g, []float32{0.5, 0.5, 1.0})
			variance := Const(g, []float32{1.0, 1.0, 10.0})
			output = InternalBatchNormForInference(input, scale, offset, mean, variance, 1e-7, -1)
			return
		}, [][]float32{
			{9.5, 99, 999.05133},
			{10.5, 101, 1000},
			{11.5, 103, 1000.94867},
			{12.5, 105, 1001.8974},
			{13.5, 107, 1002.84607},
			{14.5, 109, 1003.79474},
			{15.5, 111, 1004.7434},
		})
}

func TestInternalBatchNormForTraining(t *testing.T) {
	graphtest.RunTestGraphFn(t, "BatchNormInference()",
		func(g *Graph) (inputs, outputs []*Node) {
			input := Iota(g, MakeShape(dtypes.Float32, 7, 3), 0) // Values from 0.0 to 6.0 on batch axis.
			inputs = []*Node{input}
			scale := Const(g, []float32{1.0, 2.0, 3.0})
			offset := Const(g, []float32{10.0, 100.0, 1000.0})
			normalized, batchMean, batchVariance := InternalBatchNormForTraining(input, scale, offset, 1e-7, -1)
			normalized2 := InternalBatchNormForInference(input, scale, offset, batchMean, batchVariance, 1e-7, -1)
			outputs = []*Node{normalized, normalized2, batchMean, batchVariance}

			return
		}, []any{
			[][]float32{
				{8.5, 97, 995.5},
				{9, 98, 997},
				{9.5, 99, 998.5},
				{10, 100, 1000},
				{10.5, 101, 1001.5},
				{11, 102, 1003},
				{11.5, 103, 1004.5},
			},
			[][]float32{
				{8.5, 97, 995.5},
				{9, 98, 997},
				{9.5, 99, 998.5},
				{10, 100, 1000},
				{10.5, 101, 1001.5},
				{11, 102, 1003},
				{11.5, 103, 1004.5},
			},
			[]float32{3, 3, 3}, // Mean = (0+1+2+3+4+5+6) / 7 = 3
			[]float32{4, 4, 4}, // ReduceVariance = (9+4+1+0+1+4+9) / 7 = 4
		}, 1e-4)
}

func TestExpandAxes(t *testing.T) {
	graphtest.RunTestGraphFn(t, "ExpandAxes and InsertAxes",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{
				Zeros(g, shapes.Make(dtypes.Int64, 2)),
			}
			outputs = []*Node{
				ExpandAxes(inputs[0], 0, 1, 3),  // -> shape [1, 1, 2, 1]
				ExpandAxes(inputs[0], 0, 1, -1), // -> same, shape [1, 1, 2, 1]
				InsertAxes(inputs[0], 0, 0, -1), // -> same, shape [1, 1, 2, 1]
			}
			return
		}, []any{
			[][][][]int64{{{{0}, {0}}}},
			[][][][]int64{{{{0}, {0}}}},
			[][][][]int64{{{{0}, {0}}}},
		}, -1)
}

func TestSqueeze(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Squeeze()",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{
				Zeros(g, shapes.Make(dtypes.Int64, 1, 2, 1, 2)),
				Ones(g, shapes.Make(dtypes.Int64, 1, 1, 1)),
			}
			outputs = []*Node{
				Squeeze(inputs[0], 0, -2),
				Squeeze(inputs[0]),
				Squeeze(inputs[1], 0, 1, -1),
				Squeeze(inputs[1]),
			}
			return
		}, []any{
			[][]int64{
				{0, 0},
				{0, 0},
			},
			[][]int64{
				{0, 0},
				{0, 0},
			},
			int64(1),
			int64(1),
		}, -1)
}

func TestArgMax(t *testing.T) {
	for _, dtype := range []dtypes.DType{dtypes.Float64, dtypes.Float32, dtypes.Int64, dtypes.Int32} {
		graphtest.RunTestGraphFn(t, fmt.Sprintf("ArgMax()/ArgMin() for dtype %q", dtype),
			func(g *Graph) (inputs, outputs []*Node) {
				inputs = []*Node{
					IotaFull(g, shapes.Make(dtype, 3, 5)),
				}
				outputs = []*Node{
					ArgMax(inputs[0], -1),
					ArgMax(inputs[0], 0),
					ArgMin(inputs[0], 1, dtypes.Uint8),
				}
				return
			}, []any{
				[]int32{4, 4, 4},
				[]int32{2, 2, 2, 2, 2},
				[]uint8{0, 0, 0},
			}, -1)
	}
}

func TestComplex(t *testing.T) {
	re := []float32{1.0, -3.0}
	im := []float32{-5.0, 7.0}
	re64 := []float64{11, 17}
	graphtest.RunTestGraphFn(t, "Complex", func(g *Graph) (inputs, outputs []*Node) {
		inputs = []*Node{
			Const(g, re),
			Const(g, im),
			Const(g, re64),
		}
		outputs = []*Node{
			Complex(inputs[0], inputs[1]),
			Complex(inputs[2], ScalarOne(g, inputs[2].DType())), // Test broadcast of scalar.
		}
		return
	}, []any{
		[]complex64{complex(re[0], im[0]), complex(re[1], im[1])},
		[]complex128{complex(re64[0], 1.0), complex(re64[1], 1.0)},
	}, -1)
}

func TestWhere(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Where: no broadcast", func(g *Graph) (inputs, outputs []*Node) {
		inputs = []*Node{
			Const(g, []bool{true, false, true}),
			Const(g, []float32{1, 1, 1}),
			Const(g, []float32{0, 0, 0}),
		}
		outputs = []*Node{Where(inputs[0], inputs[1], inputs[2])}
		return
	}, []any{
		[]float32{1, 0, 1},
	}, -1)

	graphtest.RunTestGraphFn(t, "Where: broadcast from scalar #1", func(g *Graph) (inputs, outputs []*Node) {
		inputs = []*Node{
			Const(g, []bool{true, false, true}),
			Const(g, float32(1)),
			Const(g, []float32{0, 0, 0}),
		}
		outputs = []*Node{Where(inputs[0], inputs[1], inputs[2])}
		return
	}, []any{
		[]float32{1, 0, 1},
	}, -1)

	graphtest.RunTestGraphFn(t, "Where: broadcast from scalar #2", func(g *Graph) (inputs, outputs []*Node) {
		inputs = []*Node{
			Const(g, []bool{true, false, true}),
			Const(g, float32(1)),
			Const(g, float32(0)),
		}
		outputs = []*Node{Where(inputs[0], inputs[1], inputs[2])}
		return
	}, []any{
		[]float32{1, 0, 1},
	}, -1)

	graphtest.RunTestGraphFn(t, "Where: broadcast from prefix condition", func(g *Graph) (inputs, outputs []*Node) {
		inputs = []*Node{
			Const(g, []bool{true, false, true}),
			IotaFull(g, shapes.Make(dtypes.Float32, 3, 2)),
			AddScalar(IotaFull(g, shapes.Make(dtypes.Float32, 3, 2)), 100),
		}
		outputs = []*Node{Where(inputs[0], inputs[1], inputs[2])}
		return
	}, []any{
		[][]float32{
			{0, 1},
			{102, 103},
			{4, 5},
		},
	}, -1)

	graphtest.RunTestGraphFn(t, "Where: broadcast from prefix condition and scalar", func(g *Graph) (inputs, outputs []*Node) {
		inputs = []*Node{
			Const(g, []bool{true, false, true}),
			IotaFull(g, shapes.Make(dtypes.Float32, 3, 2)),
			Const(g, float32(100)),
		}
		outputs = []*Node{Where(inputs[0], inputs[1], inputs[2])}
		return
	}, []any{
		[][]float32{
			{0, 1},
			{100, 100},
			{4, 5},
		},
	}, -1)
}

func TestDynamicSlice(t *testing.T) {
	// All indices for the slice given in one tensor.
	graphtest.RunTestGraphFn(t, "DynamicSlice: all indices in one tensor", func(g *Graph) (inputs, outputs []*Node) {
		inputs = []*Node{
			IotaFull(g, shapes.Make(dtypes.Float32, 3, 2)),
			Const(g, []int32{0, 1}),
		}
		outputs = []*Node{
			DynamicSlice(inputs[0], inputs[1:], []int{2, 1}),
		}
		return
	}, []any{
		[][]float32{{1}, {3}},
	}, -1)

	// Each index to startIndices given on a separate tensor.
	graphtest.RunTestGraphFn(t, "DynamicSlice: indices in separate tensors", func(g *Graph) (inputs, outputs []*Node) {
		inputs = []*Node{
			IotaFull(g, shapes.Make(dtypes.Float32, 3, 2)),
			Const(g, int32(0)),
			Const(g, int32(1)),
		}
		outputs = []*Node{
			DynamicSlice(inputs[0], inputs[1:], []int{2, 1}),
		}
		return
	}, []any{
		[][]float32{{1}, {3}},
	}, -1)
}

func TestDynamicUpdateSlice(t *testing.T) {
	// This injects [][]float32{{100}, {100}} at position {0, 1} of the input.
	graphtest.RunTestGraphFn(t, "DynamicUpdateSlice: all indices in one tensor", func(g *Graph) (inputs, outputs []*Node) {
		operand := IotaFull(g, shapes.Make(dtypes.Float32, 3, 2))
		updates := MulScalar(Ones(g, shapes.Make(dtypes.Float32, 2, 1)), 100)
		startIndices := Const(g, []int32{0, 1})
		inputs = []*Node{operand, updates, startIndices}
		outputs = []*Node{
			DynamicUpdateSlice(operand, updates, []*Node{startIndices}),
		}
		return
	}, []any{
		[][]float32{{0, 100}, {2, 100}, {4, 5}},
	}, -1)

	graphtest.RunTestGraphFn(t, "DynamicUpdateSlice: all indices in one tensor", func(g *Graph) (inputs, outputs []*Node) {
		operand := IotaFull(g, shapes.Make(dtypes.Float32, 3, 2))
		updates := MulScalar(Ones(g, shapes.Make(dtypes.Float32, 2, 1)), 100)
		startIndices := []*Node{Const(g, int32(0)), Const(g, int32(1))}
		inputs = []*Node{operand, updates}
		inputs = append(inputs, startIndices...)
		outputs = []*Node{
			DynamicUpdateSlice(operand, updates, startIndices),
		}
		return
	}, []any{
		[][]float32{{0, 100}, {2, 100}, {4, 5}},
	}, -1)
}

func TestBitCount(t *testing.T) {
	graphtest.RunTestGraphFn(t, "BitCount", func(g *Graph) (inputs, outputs []*Node) {
		operand := IotaFull(g, shapes.Make(dtypes.Int32, 8))
		inputs = []*Node{operand}
		outputs = []*Node{
			BitCount(operand),
		}
		return
	}, []any{
		[]int32{0, 1, 1, 2, 1, 2, 2, 3}, // Number of bits set.
	}, 0)
}

func TestIsFinite(t *testing.T) {
	graphtest.RunTestGraphFn(t, "IsFinite", func(g *Graph) (inputs, outputs []*Node) {
		inputs = []*Node{
			Const(g, []float64{0.0, math.NaN(), -1, math.Inf(-1), 1, math.Inf(1)}),
		}
		outputs = []*Node{IsFinite(inputs[0])}
		return
	}, []any{
		[]bool{true, false, true, false, true, false}, // Number of bits set.
	}, 0)
}

func TestMatMul(t *testing.T) {
	graphtest.RunTestGraphFn(t, "MatMul 1:", func(g *Graph) (inputs, outputs []*Node) {
		inputs = []*Node{
			IotaFull(g, shapes.Make(dtypes.Float32, 3, 2, 5)),
			Transpose(
				Const(g, [][]float32{{100, 10, 1, 0.1, 0.01}, {1000, 100, 10, 1, 0.1}}),
				0, 1),
		}
		outputs = []*Node{MatMul(inputs[0], inputs[1])}
		return
	}, []any{
		// Shape: [3, 2, 2]
		[][][]float32{
			{{12.34, 123.4}, {567.89, 5678.9}},
			{{1123.4401, 11234.4}, {1678.99, 16789.9}},
			{{2234.54, 22345.4}, {2790.09, 27900.9}},
		},
	}, 0)

	graphtest.RunTestGraphFn(t, "MatMul 2:", func(g *Graph) (inputs, outputs []*Node) {
		inputs = []*Node{
			IotaFull(g, shapes.Make(dtypes.Float32, 3, 2, 5)),
			Const(g, []float32{100, 10, 1, 0.1, 0.01}),
		}
		outputs = []*Node{
			MatMul(inputs[0], inputs[1]),
			MatMul(inputs[0], InsertAxes(inputs[1], -1)),
		}
		return
	}, []any{
		// Shape: [3, 2]
		[][]float32{{12.34, 567.89}, {1123.4401, 1678.99}, {2234.54, 2790.09}},
		// Shape: [3, 2, 1]
		[][][]float32{{{12.34}, {567.89}}, {{1123.4401}, {1678.99}}, {{2234.54}, {2790.09}}},
	}, 0)

	// Test shapes of edge cases:
	backend := graphtest.BuildTestBackend()
	g := NewGraph(backend, "matmul_test")
	testShapes := [][3][]int{
		// Format: {<lhs shape>, <rhs shape>, <expected output shape>} :
		{{3}, {3, 5}, {5}},
		{{3, 5}, {5}, {3}},
		{{3, 5}, {5, 4}, {3, 4}},

		// Notice that the complex ordering of the axes in numpy's MatMul
		{{3, 5}, {7, 5, 4}, {7, 3, 4}},
		{{7, 3, 5}, {4, 7, 5, 4}, {4, 7, 3, 4}},

		// Broadcasting:
		{{1, 3, 5}, {7, 5, 4}, {7, 3, 4}},
		{{8, 3, 5}, {1, 5, 4}, {8, 3, 4}},
		{{1, 2, 6, 5}, {1, 3, 1, 5, 4}, {1, 3, 2, 6, 4}},
	}
	fmt.Println("Shape checking:")
	for _, dims := range testShapes {
		fmt.Printf("\tMatMul(%v, %v) -> %v\n", dims[0], dims[1], dims[2])
		lhs := Ones(g, shapes.Make(dtypes.F32, dims[0]...))
		rhs := Ones(g, shapes.Make(dtypes.F32, dims[1]...))
		got := MatMul(lhs, rhs)
		require.NoError(t, got.Shape().CheckDims(dims[2]...))
	}
}
