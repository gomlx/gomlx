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
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"math"
	"reflect"
	"testing"
)

var (
	// Aliases:

	MakeShape = shapes.Make
	F32       = shapes.Float32
	F64       = shapes.Float64
)

// buildTestManager using "Host" by default -- can be overwritten by GOMLX_PLATFORM environment variable.
func buildTestManager() *Manager {
	return graphtest.BuildTestManager()
}

type graphFnOneInputToTest func(g *Graph) (input, output *Node)

// testFuncOneInput makes it easy to test a function with one input and one output. It
// compiles and executes the given graph building function graphFn and checks that the
// result is as expected.
func testFuncOneInput(t *testing.T, testName string, graphFn graphFnOneInputToTest, want any) {
	fmt.Printf("%s\n", testName)
	manager := buildTestManager()
	g := manager.NewGraph(testName)
	inputNode, outputNode := graphFn(g)
	g.Compile(inputNode, outputNode)
	results := g.Run(nil).SplitTuple()
	input, got := results[0].Value(), results[1].Value()
	if !slices.SlicesInDelta(want, got, slices.Epsilon) {
		t.Errorf("%s(%#v): want=%v, got=%v", testName, input, want, got)
	}
}

func TestConstant(t *testing.T) {
	platforms, _ := GetPlatforms()
	fmt.Printf("Platforms: %v\n", platforms)
	manager := buildTestManager()
	{
		g := manager.NewGraph("")
		n := Const(g, 5)
		shape := n.Shape()
		if shape.DType != shapes.Int64 || shape.Rank() != 0 {
			t.Errorf("ConstLocal has invalid shape: %s", shape)
		}
	}
	{
		g := manager.NewGraph("")
		n := Const(g, [][]float32{{1.2, 1.3}, {2.4, 2.5}, {2.6, 2.7}})
		shape := n.Shape()
		if shape.DType != shapes.Float32 || !reflect.DeepEqual(shape.Dimensions, []int{3, 2}) {
			fmt.Printf("\tTestConstant: node %s\n", n)
			t.Errorf("ConstLocal has invalid shape: %s", shape)
		}
	}
}

func compileRunTransfer(t *testing.T, g *Graph, msg string) *tensor.Local {
	g.Compile()
	device := g.Run(nil)
	local := device.Local()
	return local
}

func TestAdd(t *testing.T) {
	manager := buildTestManager()
	{
		// Test scalars.
		g := manager.NewGraph("")
		x := Const(g, 5)
		y := Const(g, 7)
		n := Add(x, y)
		wantShape := shapes.Shape{DType: shapes.Int64}
		require.Truef(t, n.Shape().Eq(wantShape), "Add invalid shape %s, wanted %s", n.Shape(), wantShape)
		local := compileRunTransfer(t, g, "scalar Graph")
		got := local.Value().(int64)
		if got != 12 {
			fmt.Printf("%s\n", g)
			fmt.Printf("\tResult: %d %s\n", got, local.Shape())
			t.Errorf("Wanted 5 + 7 = 12, got %d", got)
		}
	}
	{
		// Test multi-dimension arrays.
		g := manager.NewGraph("")
		x := Const(g, [][]float32{{1.1, 1.2}, {1.3, 1.4}})
		y := Const(g, [][]float32{{10, 10}, {20, 20}})
		n := Add(x, y)
		wantShape := shapes.Make(shapes.Float32, 2, 2)
		if !n.Shape().Eq(wantShape) {
			t.Fatalf("Add invalid shape %s, wanted %s", n.Shape(), wantShape)
		}
		local := compileRunTransfer(t, g, "[2, 2] Graph")
		got := local.Value().([][]float32)
		want := [][]float32{{11.1, 11.2}, {21.3, 21.4}}
		if !reflect.DeepEqual(got, want) {
			fmt.Printf("%s\n", g)
			fmt.Printf("\tResult: %v %s\n", got, local.Shape())
			t.Errorf("Wanted %v, got %v", want, got)
		}
	}
	{
		// Test multi-dimension arrays of same rank with broadcast.
		g := manager.NewGraph("")
		x := Const(g, [][]float32{{1.1, 1.2}, {1.3, 1.4}})
		y := Const(g, [][]float32{{1}, {10}})
		n := Add(x, y)
		wantShape := shapes.Make(shapes.Float32, 2, 2)
		if !n.Shape().Eq(wantShape) {
			t.Fatalf("Add invalid shape %s, wanted %s", n.Shape(), wantShape)
		}
		local := compileRunTransfer(t, g, "[2, 2] Graph")
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
		g := manager.NewGraph("")
		x := Const(g, [][]float32{{1.1, 1.2}, {1.3, 1.4}})
		y := Const(g, float32(1))
		n := Add(x, y)
		wantShape := shapes.Make(shapes.Float32, 2, 2)
		if !n.Shape().Eq(wantShape) {
			t.Fatalf("Add invalid shape %s, wanted %s", n.Shape(), wantShape)
		}
		local := compileRunTransfer(t, g, "[2, 2] Graph")
		got := local.Value().([][]float32)
		want := [][]float32{{2.1, 2.2}, {2.3, 2.4}}
		if !reflect.DeepEqual(got, want) {
			fmt.Printf("%s\n", g)
			fmt.Printf("\tResult: %v %s\n", got, local.Shape())
			t.Errorf("Wanted %v, got %v", want, got)
		}
	}
}

func testTupleParameter(t *testing.T, manager *Manager) {
	g := manager.NewGraph("")
	xyS := shapes.MakeTuple([]shapes.Shape{shapes.Scalar[float64](), shapes.Scalar[float64]()})
	xy := g.Parameter("xy", xyS)
	if !xy.Shape().Eq(xyS) {
		fmt.Printf("\t(before) xy.shape=%s\n", xyS)
		fmt.Printf("\t(after) xy.shape=%s\n", xy.Shape())
		t.Fatalf("Tuple shape changed after creating parameter.")
	}
	x := GetTupleElement(xy, 0)
	y := GetTupleElement(xy, 1)
	// x^2 + 2*y
	Add(Mul(x, x), Mul(Const(g, 2.0), y))
	if xy.ParameterHandle() == InvalidParameterHandle {
		t.Fatalf("Invalid parameter xlaHandle for tuple")
	}
	g.Compile()

	// Tests for various parameters.
	for xV := float64(0); xV < 20; xV += 1 {
		for yV := float64(0); yV < 20; yV += 1 {
			xyV := tensor.MakeLocalTupleAny(xV, yV)
			device := g.Run(ParamsMap{xy: xyV})
			local := device.Local()
			got := local.Value().(float64)
			want := xV*xV + 2*yV
			if got != want {
				fmt.Printf("%s\n", g)
				t.Errorf("%f + %f : got %s, wanted %f", xV, yV, local, want)
			}
		}
	}
}

func TestParameter(t *testing.T) {
	manager := buildTestManager()

	// Test passing of values.
	{
		// Test scalars.
		g := manager.NewGraph("")
		x := g.Parameter("x", shapes.Scalar[float32]())
		y := g.Parameter("y", shapes.Scalar[float32]())
		Add(x, y)
		if x.ParameterHandle() == InvalidParameterHandle || y.ParameterHandle() == InvalidParameterHandle || x.ParameterHandle() == y.ParameterHandle() {
			t.Fatalf("Invalid parameter handles: x=%d, y=%d", x.ParameterHandle(), y.ParameterHandle())
		}
		g.Compile()

		// Tests for various parameters.
		for xV := float32(0); xV < 3; xV += 1 {
			for yV := float32(0); yV < 3; yV += 1 {
				device := g.Run(ParamsMap{x: xV, y: yV})
				local := device.Local()
				got := local.Value().(float32)
				if got != xV+yV {
					fmt.Printf("%s\n", g)
					t.Errorf("%f + %f : got %s, wanted %f", xV, yV, local, xV+yV)
				}
			}
		}
	}

	// Test tuple parameters.
	testTupleParameter(t, manager)
}

func TestConvertType(t *testing.T) {
	// Test that number can be converted to complex types.
	wantF32 := []float32{3.0, -5.0}
	wantF64 := []float64{-7.0, 11.0}
	graphtest.RunTestGraphFn(t, "ConvertToComplex", func(g *Graph) (inputs, outputs []*Node) {
		inputs = []*Node{
			Const(g, wantF32),
			Const(g, wantF64),
		}
		c64 := ConvertType(inputs[0], shapes.Complex64)
		assert.Equal(t, shapes.Complex64, c64.DType())
		c128 := ConvertType(inputs[1], shapes.Complex128)
		assert.Equal(t, shapes.Complex128, c128.DType())
		outputs = []*Node{Real(c64), Real(c128)}
		return
	}, []any{wantF32, wantF64}, -1)
}

type TwoArgsTestCase[T shapes.Number] struct {
	fnGraph  func(x, y *Node) *Node
	fnScalar func(x, y T) T
}

func TestTwoArgsOps(t *testing.T) {
	manager := buildTestManager()

	{
		casesFloat32 := []TwoArgsTestCase[float32]{
			{Mul, func(x, y float32) float32 { return x * y }},
			{Sub, func(x, y float32) float32 { return x - y }},
			{Div, func(x, y float32) float32 { return x / y }},
			{Mod, func(x, y float32) float32 { return float32(math.Mod(float64(x), float64(y))) }},
			{Min, func(x, y float32) float32 {
				if x < y {
					return x
				} else {
					return y
				}
			}},
			{Max, func(x, y float32) float32 {
				if x > y {
					return x
				} else {
					return y
				}
			}},
			{Pow, func(x, y float32) float32 {
				return float32(math.Pow(float64(x), float64(y)))
			}},
		}
		xSlices := [][]float32{{11, 12}, {13, 14}}
		yValue := float32(3)
		for _, test := range casesFloat32 {
			g := manager.NewGraph("")
			x := Const(g, xSlices)
			y := Const(g, yValue)
			n := test.fnGraph(x, y)
			wantShape := shapes.Make(shapes.Float32, 2, 2)
			if !n.Shape().Eq(wantShape) {
				t.Fatalf("Add invalid shape %s, wanted %s", n.Shape(), wantShape)
			}
			local := compileRunTransfer(t, g, "[2, 2] Graph")
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
			{And, func(x, y int64) int64 { return x & y }},
			{Or, func(x, y int64) int64 { return x | y }},
			{Xor, func(x, y int64) int64 { return x ^ y }},
		}
		xSlices := [][]int64{{11, 12}, {13, 14}}
		yValue := int64(3)
		for _, test := range casesInt {
			g := manager.NewGraph("")
			x := Const(g, xSlices)
			y := Const(g, yValue)
			n := test.fnGraph(x, y)
			wantShape := shapes.Make(shapes.Int64, 2, 2)
			if !n.Shape().Eq(wantShape) {
				t.Fatalf("Add invalid shape %s, wanted %s", n.Shape(), wantShape)
			}
			local := compileRunTransfer(t, g, "[2, 2] Graph")
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
}

type OneArgTestCase[T shapes.Number] struct {
	fnGraph    func(x *Node) *Node
	goFnScalar func(x T) T
}

func TestOneArgOps(t *testing.T) {
	manager := buildTestManager()

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
		{RSqrt, func(x float64) float64 { return 1.0 / math.Sqrt(x) }},
	}
	xSlices := [][]float64{{11.1, 12.8}, {-13.2, -14.9}}
	for _, test := range casesFloat64 {
		g := manager.NewGraph("")
		x := Const(g, xSlices)
		n := test.fnGraph(x)
		wantShape := shapes.Make(shapes.Float64, 2, 2)
		if !n.Shape().Eq(wantShape) {
			t.Fatalf("Add invalid shape %s, wanted %s", n.Shape(), wantShape)
		}
		local := compileRunTransfer(t, g, "[2, 2] graph for one-arg operation")
		got := local.Value().([][]float64)
		want := [][]float64{{0, 0}, {0, 0}}
		for i0, x0Slice := range xSlices {
			for i1, value := range x0Slice {
				want[i0][i1] = test.goFnScalar(value)
			}
		}
		if !slices.DeepSliceCmp(got, want, slices.Close[float64]) {
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
}

func TestClzOp(t *testing.T) {
	testFuncOneInput(t, "Clz()",
		func(g *Graph) (input, output *Node) {
			input = Const(g, []int64{16, 14})
			output = Clz(input)
			return
		}, []int64{64 - 5, 64 - 4})
}

func TestLogicalOps(t *testing.T) {
	//fmt.Printf("Node type %s: #%d\n", xla.LogicalNotNode, xla.LogicalNotNode)
	//testFuncOneInput(t, "Not()",
	//	func(g *Graph) (input, output *Node) {
	//		input = Const(g, []bool{true, false, true, true})
	//		output = Not(input)
	//		return
	//	}, []bool{false, true, false, false})
}

// compileAndRun compiles, runs and returns the value on the tensor. Doesn't work for tuples though.
func compileAndRun(g *Graph) any {
	g.Compile()
	device := g.Run(nil)
	got := device.Local().Value()
	return got
}

func TestDot(t *testing.T) {
	manager := buildTestManager()
	g := manager.NewGraph("Dot")

	// Shape: [batch=4, dims=3]
	inputs := Const(g, [][]float32{{1.1, 2.2, 3.3}, {11, 22, 33}, {111, 222, 333}, {1111, 2222, 3333}})
	// Layer 0: shape [3, 2], that is the inputs have dim=3, and should output dims=2
	w0 := Const(g, [][]float32{{1, 0}, {1, -1}, {-1, 1}})
	// Dot(inputs, w0) -> shape [batch=4, dims=2]
	Dot(inputs, w0) // Last node created in the graph is taken as output by default.
	got := compileAndRun(g)
	want := [][]float32{{0, 1.1}, {0, 11}, {0, 111}, {0, 1111}}
	if !slices.DeepSliceCmp(got, want, slices.Close[float32]) {
		fmt.Printf("%s\n", g)
		fmt.Printf("\tResult=%v\n", got)
		t.Errorf("Wanted %v, got %v", want, got)
	}
}

func TestBroadcast(t *testing.T) {
	manager := buildTestManager()
	{
		g := manager.NewGraph("")
		input := Const(g, 7)
		BroadcastToDims(input, 2, 3) // Last node created in the graph is taken as output by default.
		got := compileAndRun(g)
		want := [][]int64{{7, 7, 7}, {7, 7, 7}}
		assert.Equal(t, want, got)
	}

	{
		g := manager.NewGraph("")
		input := Const(g, []float32{1.1, 1.2})
		BroadcastPrefix(input, []int{2, 1}) // The last node created in the graph is taken as output by default.
		got := compileAndRun(g)
		want := [][][]float32{{{1.1, 1.2}}, {{1.1, 1.2}}} // Shape [2, 1, 2].
		assert.Equal(t, want, got)
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

}

func TestFill(t *testing.T) {
	manager := buildTestManager()
	{
		g := manager.NewGraph("FillScalar")
		FillScalar(g, shapes.Make(shapes.Int64, 3, 1), 4.0)
		got := compileAndRun(g)
		want := [][]int64{{4}, {4}, {4}}
		if !slices.DeepSliceCmp(got, want, slices.Equal[int64]) {
			t.Errorf("Wanted %#v, got %#v", want, got)
		}
	}
	{
		g := manager.NewGraph("Ones")
		Ones(g, shapes.Make(shapes.Float32, 3, 1))
		got := compileAndRun(g)
		want := [][]float32{{1}, {1}, {1}}
		if !slices.DeepSliceCmp(got, want, slices.Equal[float32]) {
			t.Errorf("Wanted %#v, got %#v", want, got)
		}
	}
	{
		g := manager.NewGraph("Zeros")
		Zeros(g, shapes.Make(shapes.Float64, 3, 1))
		got := compileAndRun(g)
		want := [][]float64{{0}, {0}, {0}}
		if !slices.DeepSliceCmp(got, want, slices.Equal[float64]) {
			t.Errorf("Wanted %#v, got %#v", want, got)
		}
	}
}

func reduceSumGraph(t *testing.T, m *Manager, reduceDims []int) *Graph {
	g := m.NewGraph("main")
	n0 := Const(g, [][]float64{{5.0, 1.0}})
	n1 := Ones(g, shapes.Make(shapes.Float64, 2, 1))
	n2 := Add(n1, n0)
	o0 := ReduceSum(n2, reduceDims...)
	g.Compile(o0)
	return g
}

func TestReduceSum(t *testing.T) {
	manager := buildTestManager()
	cases := []struct {
		dims []int
		want any
	}{
		{want: 16.0},
		{dims: []int{0}, want: []float64{12, 4}},
		{dims: []int{1}, want: []float64{8, 8}},
	}
	for _, testCase := range cases {
		g := reduceSumGraph(t, manager, testCase.dims)
		gotT := g.Run(nil)
		got := gotT.Local().Value()
		if !slices.DeepSliceCmp(got, testCase.want, slices.Close[float64]) {
			t.Errorf("Wanted %v, got %v", testCase.want, got)
		}
	}
}

func TestReduceMean(t *testing.T) {
	testFuncOneInput(t, "ReduceMean(dims=1, 2)",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, MakeShape(shapes.Float32, 3, 2, 4))
			output = ReduceMean(input, 1, 2)
			return
		}, []float32{3.5, 11.5, 19.5})
	testFuncOneInput(t, "ReduceAllMean()",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, MakeShape(shapes.Float32, 3, 2, 4))
			output = ReduceAllMean(input)
			return
		}, float32(11.5))
	graphtest.RunTestGraphFn(t, "ReduceMean",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{
				IotaFull(g, shapes.Make(shapes.F32, 3, 5)),
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

func TestReduceMaskedMax(t *testing.T) {
	graphtest.RunTestGraphFn(t, "MaskedReduceMax()",
		func(g *Graph) (inputs, outputs []*Node) {
			x := IotaFull(g, MakeShape(shapes.Float32, 4, 3))
			mask := Const(g, [][]bool{
				{true, false, false},
				{true, true, false},
				{true, false, true},
				{true, true, true}})
			output := MaskedReduceMax(x, mask, 1)
			inputs = []*Node{x, mask}
			outputs = []*Node{output}
			return
		}, []any{[]float32{0, 4, 8, 11}}, slices.Epsilon)
}

func TestReshape(t *testing.T) {
	manager := buildTestManager()
	{
		g := manager.NewGraph("")
		input := Const(g, [][][]float32{{{1.1, 1.2}}}) // Shape [1, 1, 2]
		ReshapeWithShape(input, shapes.Make(input.DType(), 2, 1))
		got := compileAndRun(g)
		want := [][]float32{{1.1}, {1.2}}
		if !slices.DeepSliceCmp(got, want, slices.Equal[float32]) {
			fmt.Printf("%s\n", g)
			fmt.Printf("\tResult=%v\n", got)
			t.Errorf("Wanted %v, got %v", want, got)
		}
	}
}

func TestTuple(t *testing.T) {
	manager := buildTestManager()
	{
		g := manager.NewGraph("")
		a := Const(g, []float32{1.1, 1.2})
		b := Const(g, 5)
		tuple := Tuple(a, b)
		if !tuple.Shape().IsTuple() {
			t.Errorf("Expected shape to be tuple, got %s instead", tuple.Shape())
		}
		GetTupleElement(tuple, 0)
		got := compileAndRun(g)
		want := []float32{1.1, 1.2}
		if !slices.DeepSliceCmp(got, want, slices.Equal[float32]) {
			fmt.Printf("%s\n", g)
			fmt.Printf("\tResult=%v\n", got)
			t.Errorf("Wanted %v, got %v", want, got)
		}
	}

	{
		g := manager.NewGraph("")
		a := Const(g, []float32{1.1, 1.2})
		b := Const(g, 5)
		tupleN := Tuple(a, b)
		if !tupleN.Shape().IsTuple() {
			t.Errorf("Expected shape to be tuple, got %s instead", tupleN.Shape())
		}
		g.Compile()
		tupleT := g.Run(nil)
		if !tupleT.IsTuple() {
			t.Errorf("Expected tensor shape to be tuple, got %s instead", tupleN.Shape())
		}
		/*
			splits := tupleT.SplitTupleError()
			if splits == nil {
				t.Errorf("Failed to split Device tuple: %v", tupleT.error)
			}
			want := []any{[]float32{1.1, 1.2}, 5}
			if !types.DeepSliceCmp(splits[0].Local().Value(), want[0], types.Equal[float32]) || splits[1].Local().Value().(int) != 5 {
				fmt.Printf("%s\n", g)
				fmt.Printf("\tResult=(%v, %v)\n", splits[0].Local().Value(), splits[1].Local().Value())
				t.Fatalf("Wanted %v", want)
			}

			// Split a second time, to check that works.
			splits = tupleT.SplitTupleError()
			if splits == nil {
				t.Errorf("Failed to split result tuple a second time: %v", tupleT.error)
			}
			if !types.DeepSliceCmp(splits[0].Local().Value(), want[0], types.Equal[float32]) || splits[1].Local().Value().(int) != 5 {
				fmt.Printf("\tResult=(%v, %v)\n", splits[0].Local().Value(), splits[1].Local().Value())
				t.Errorf("Failed at 2nd split of tuple: wanted %v", want)
			}
		*/
	}
}

func TestIota(t *testing.T) {
	manager := buildTestManager()
	{
		g := manager.NewGraph("iota0")
		Iota(g, MakeShape(F64, 2, 2), 0)
		g.Compile()
		got := g.Run(nil).Local().Value()
		want := [][]float64{{0, 0}, {1, 1}}
		if !slices.DeepSliceCmp(got, want, slices.Equal[float64]) {
			t.Fatalf("Iota: want %v, got %v", want, got)
		}
	}
	{
		g := manager.NewGraph("iota0")
		Iota(g, MakeShape(F64, 2, 2), 1)
		g.Compile()
		got := g.Run(nil).Local().Value()
		want := [][]float64{{0, 1}, {0, 1}}
		if !slices.DeepSliceCmp(got, want, slices.Equal[float64]) {
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
		}, slices.Epsilon)

	graphtest.RunTestGraphFn(t, "Slice Tests with Rank 1",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, [][]int32{{1, 2, 3}, {4, 5, 6}})
			inputs = []*Node{x}
			outputs = []*Node{
				Slice(x, AxisRange(), AxisElem(0)),
				Slice(x, AxisRange(1, 2)),
				Slice(x, AxisRange().Stride(2), AxisElem(-1)),
			}
			return
		}, []any{
			[][]int32{{1}, {4}},
			[][]int32{{4, 5, 6}},
			[][]int32{{3}},
		}, slices.Epsilon)

	graphtest.RunTestGraphFn(t, "Slice Tests with Rank 1",
		func(g *Graph) (inputs, outputs []*Node) {
			x := IotaFull(g, shapes.Make(shapes.I64, 2, 2, 2, 2))
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
		}, slices.Epsilon)

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
		}, slices.Epsilon)
}

func TestGather(t *testing.T) {
	manager := buildTestManager()
	{ // Trivial scalar gather.
		fmt.Println("\tGather(): trivial scalar gather.")
		g := manager.NewGraph("")
		// numbers=(Float64)[5 3]: [[0 1 2] [3 4 5] [6 7 8] [9 10 11] [12 13 14]]
		numbers := IotaFull(g, MakeShape(F64, 5, 3))
		indices := Const(g, 1)
		gather := Gather(numbers, indices)
		g.Compile(gather)
		got := g.Run(nil).Local()
		fmt.Printf("\t\tGather=%v\n", got)
		want := []float64{3, 4, 5}
		if !slices.DeepSliceCmp(got.Value(), want, slices.Equal[float64]) {
			t.Errorf("Gather: want %v, got %v", want, got)
		}
	}

	{ // Simple leading indices dimension.
		fmt.Println("\tGather(): simple leading indices dimension.")
		g := manager.NewGraph("")
		// numbers=(Float64)[5 3]: [[0 1 2] [3 4 5] [6 7 8] [9 10 11] [12 13 14]]
		numbers := IotaFull(g, MakeShape(F64, 5, 3))
		indices := Const(g, [][]int{{2}, {0}})
		gather := Gather(numbers, indices)
		g.Compile(gather)
		got := g.Run(nil).Local()
		fmt.Printf("\t\tGather=%v\n", got)
		want := [][]float64{{6, 7, 8}, {0, 1, 2}}
		if !slices.DeepSliceCmp(got.Value(), want, slices.Equal[float64]) {
			t.Errorf("Gather: want %v, got %v", want, got)
		}
	}

	{ // With 2D leading indices dimension.
		fmt.Println("\tGather(): with 2D leading indices dimension.")
		g := manager.NewGraph("")
		// numbers=(Float64)[5 3]: [[0 1 2] [3 4 5] [6 7 8] [9 10 11] [12 13 14]]
		numbers := IotaFull(g, MakeShape(F64, 5, 3))
		indices := Const(g, [][][]int{{{2}, {0}}, {{2}, {1}}})
		gather := Gather(numbers, indices)
		g.Compile(gather)
		got := g.Run(nil).Local()
		fmt.Printf("\t\tGather=%v\n", got)
		want := [][][]float64{{{6, 7, 8}, {0, 1, 2}}, {{6, 7, 8}, {3, 4, 5}}}
		if !slices.DeepSliceCmp(got.Value(), want, slices.Equal[float64]) {
			t.Errorf("Gather: want %v, got %v", want, got)
		}
	}

	{ // With leading indices dimension, and 3D params tailing dimensions.
		fmt.Println("\tGather(): With leading indices dimension, and 2D params tailing dimensions.")
		g := manager.NewGraph("")
		// numbers=(Float64)[5 3]: [[0 1 2] [3 4 5] [6 7 8] [9 10 11] [12 13 14]]
		numbers := IotaFull(g, MakeShape(F64, 5, 2, 2))
		indices := Const(g, [][]int{{2}, {0}, {1}, {3}})
		gather := Gather(numbers, indices)
		g.Compile(gather)
		got := g.Run(nil).Local()
		fmt.Printf("\t\tGather=%v\n", got)
		want := [][][]float64{{{8, 9}, {10, 11}}, {{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}, {{12, 13}, {14, 15}}}
		if !slices.DeepSliceCmp(got.Value(), want, slices.Equal[float64]) {
			t.Errorf("Gather: want %v, got %v", want, got.GoStr())
		}
	}

}

func TestGatherSlices(t *testing.T) {
	testFuncOneInput(t, "GatherSlices(input, slicedAxes={1}, start={{0}, {1}, {0}}, sizes={1})",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(shapes.F32, 4, 5))
			start := Const(g, [][]int32{{0}, {1}, {0}}) // Slice from rows 0, 2 and 0 of each example in the batch.
			sizes := []int{1}                           // Take only one row per start.
			output = GatherSlices(input, []int{0}, start, sizes)
			return
		}, [][][]float32{{{0, 1, 2, 3, 4}}, {{5, 6, 7, 8, 9}}, {{0, 1, 2, 3, 4}}})

	testFuncOneInput(t, "GatherSlices(input, slicedAxes={0}, start={{0}, {1}}, sizes={2})",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(shapes.F32, 4, 3))
			start := Const(g, [][]int32{{0}, {1}}) // Slice from rows 0 and 1.
			sizes := []int{2}                      // Take two rows per start.
			output = GatherSlices(input, []int{0}, start, sizes)
			return
		}, [][][]float32{{{0, 1, 2}, {3, 4, 5}}, {{3, 4, 5}, {6, 7, 8}}})

	testFuncOneInput(t, "GatherSlices(input, slicedAxes={0,1}, start={1, 1}, sizes={2, 3})",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(shapes.F32, 4, 10))
			start := Const(g, []int32{1, 1}) // Slice in middle of matrix.
			sizes := []int{2, 3}             // Take a sub-matrix
			output = GatherSlices(input, []int{0, 1}, start, sizes)
			return
		}, [][]float32{{11, 12, 13}, {21, 22, 23}})
}

func TestIndicesForShape(t *testing.T) {
	manager := buildTestManager()
	g := manager.NewGraph("")
	shape := MakeShape(F64, 2, 3, 4)
	numbers := IndicesForShape(g, shape)
	g.Compile(numbers)
	got := g.Run(nil).Local()
	fmt.Printf("\tIndicesForShape(%s)=%v\n", shape, got)
	want := [][]int64{{0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {0, 0, 3}, {0, 1, 0}, {0, 1, 1}, {0, 1, 2}, {0, 1, 3}, {0, 2, 0}, {0, 2, 1}, {0, 2, 2}, {0, 2, 3}, {1, 0, 0}, {1, 0, 1}, {1, 0, 2}, {1, 0, 3}, {1, 1, 0}, {1, 1, 1}, {1, 1, 2}, {1, 1, 3}, {1, 2, 0}, {1, 2, 1}, {1, 2, 2}, {1, 2, 3}}
	if !slices.DeepSliceCmp(got.Value(), want, slices.Equal[int64]) {
		t.Errorf("IndicesForShape(%s): want %v, got %v", shape, want, got)
	}
}

func TestScatter(t *testing.T) {
	manager := buildTestManager()
	{ // Trivial scalar scatter.
		fmt.Println("\tScatter(): trivial scalar scatter.")
		g := manager.NewGraph("")
		// numbers=(Float64)[3]: [2 3 4]
		numbers := Add(IotaFull(g, MakeShape(F64, 3)), Const(g, float64(2)))
		indices := Const(g, 1)
		scatter := Scatter(indices, numbers, MakeShape(F64, 2, 3))
		g.Compile(scatter)
		got := g.Run(nil).Local()
		fmt.Printf("\t\tscatter=%v\n", got)
		want := [][]float64{{0, 0, 0}, {2, 3, 4}}
		if !slices.DeepSliceCmp(got.Value(), want, slices.Equal[float64]) {
			t.Errorf("scatter: want %v, got %v", want, got)
		}
	}

	{ // Simple leading indices dimension.
		fmt.Println("\tScatterAdd(): leading indices dimension, and deeper slice dimension.")
		g := manager.NewGraph("")
		// numbers=(Float64)[5 3, 1]: [[[0] [1] [2]] [[3] [4] [5]]]
		numbers := IotaFull(g, MakeShape(F64, 2, 3, 1))
		indices := Const(g, [][]int{{2}, {0}})
		operand := Ones(g, MakeShape(F64, 3, 3, 1))
		scatter := ScatterAdd(operand, indices, numbers)
		g.Compile(scatter)
		got := g.Run(nil).Local()
		fmt.Printf("\t\tscatter=%v\n", got)
		want := [][][]float64{{{4}, {5}, {6}}, {{1}, {1}, {1}}, {{1}, {2}, {3}}}
		if !slices.DeepSliceCmp(got.Value(), want, slices.Equal[float64]) {
			t.Errorf("scatter: want %v, got %v", want, got)
		}
	}
}

func TestConcatenate(t *testing.T) {
	manager := buildTestManager()
	{
		fmt.Println("\tConcatenate(): 1D concatenation.")
		g := manager.NewGraph("")
		// numbers=(Float64)[3]: [2 3 4]
		x1 := IotaFull(g, MakeShape(F64, 3))
		x2 := Add(IotaFull(g, MakeShape(F64, 5)), Const(g, float64(3)))
		concat := Concatenate([]*Node{x1, x2}, 0)
		g.Compile(concat)
		got := g.Run(nil).Local()
		fmt.Printf("\t\tresult=%s\n", got.GoStr())
		want := []float64{0, 1, 2, 3, 4, 5, 6, 7}
		if !slices.DeepSliceCmp(got.Value(), want, slices.Equal[float64]) {
			t.Errorf("scatter: want %v, got %v", want, got)
		}
	}
	{
		fmt.Println("\tConcatenate(): 3D concatenation at middle dimension.")
		g := manager.NewGraph("")
		// numbers=(Float64)[3]: [2 3 4]
		x1 := IotaFull(g, MakeShape(F64, 2, 2, 2))
		x2 := Add(IotaFull(g, MakeShape(F64, 2, 1, 2)), Const(g, float64(8)))
		concat := Concatenate([]*Node{x1, x2}, 1)
		g.Compile(concat)
		got := g.Run(nil).Local()
		fmt.Printf("\t\tresult=%s\n", got.GoStr())
		want := [][][]float64{{{0, 1}, {2, 3}, {8, 9}}, {{4, 5}, {6, 7}, {10, 11}}}
		if !slices.DeepSliceCmp(got.Value(), want, slices.Equal[float64]) {
			t.Errorf("scatter: want %v, got %v", want, got)
		}
	}
}

func TestPositiveIndicator(t *testing.T) {
	testFuncOneInput(t, "PositiveIndicator",
		func(g *Graph) (input, output *Node) {
			input = Const(g, []float64{1.0, 0.0001, 0, -0.2, -3.0})
			output = PositiveIndicator(input)
			return
		}, []float64{1, 1, 1, 0, 0})
}

func TestStrictlyPositiveIndicator(t *testing.T) {
	testFuncOneInput(t, "StrictlyPositiveIndicator",
		func(g *Graph) (input, output *Node) {
			input = Const(g, []float64{1.0, 0.0001, 0, -0.2, -3.0})
			output = StrictlyPositiveIndicator(input)
			return
		}, []float64{1, 1, 0, 0, 0})
}

func TestOneHot(t *testing.T) {
	testFuncOneInput(t, "OneHot 1 leading dimension",
		func(g *Graph) (input, output *Node) {
			input = Const(g, []int{1, 0, 3})
			output = OneHot(input, 4, shapes.Float32)
			return
		}, [][]float32{{0, 1, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}})
	testFuncOneInput(t, "OneHot 2 leading dimensions",
		func(g *Graph) (input, output *Node) {
			input = Const(g, [][]int{{1, 0}, {0, 2}, {3, 1}})
			output = OneHot(input, 4, shapes.Float32)
			return
		}, [][][]float32{{{0, 1, 0, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {0, 0, 1, 0}}, {{0, 0, 0, 1}, {0, 1, 0, 0}}})
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
		}, slices.Epsilon)
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
		}, slices.Epsilon)
}

func TestReverse(t *testing.T) {
	testFuncOneInput(t, "Reverse(dimensions={1, 2})",
		func(g *Graph) (input, output *Node) {
			input = Iota(g, MakeShape(shapes.Float32, 9), 0)
			input = Reshape(input, 1, 3, 3, 1)
			output = Reverse(input, 1, 2)
			return
		}, [][][][]float32{{{{8}, {7}, {6}}, {{5}, {4}, {3}}, {{2}, {1}, {0}}}})
}

func TestTranspose(t *testing.T) {
	testFuncOneInput(t, "Transpose(dims=1, 2)",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, MakeShape(shapes.Float32, 3, 2, 4))
			output = Transpose(input, 1, 2)
			return
		}, [][][]float32{{{0, 4}, {1, 5}, {2, 6}, {3, 7}}, {{8, 12}, {9, 13}, {10, 14}, {11, 15}}, {{16, 20}, {17, 21}, {18, 22}, {19, 23}}})
}

func TestBatchNormInferenceXLA(t *testing.T) {
	testFuncOneInput(t, "BatchNormInference()",
		func(g *Graph) (input, output *Node) {
			input = Iota(g, MakeShape(shapes.Float32, 7, 3), 0) // Values from 0.0 to 6.0 on batch axis.
			scale := Const(g, []float32{1.0, 2.0, 3.0})
			offset := Const(g, []float32{10.0, 100.0, 1000.0})
			mean := Const(g, []float32{0.5, 0.5, 1.0})
			variance := Const(g, []float32{1.0, 1.0, 10.0})
			output = BatchNormInferenceXLA(input, scale, offset, mean, variance, 1e-7, -1)
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

func TestSqueeze(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Squeeze()",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{
				Zeros(g, shapes.Make(shapes.Int64, 1, 2, 1, 2)),
				Ones(g, shapes.Make(shapes.Int64, 1, 1, 1)),
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
	for _, dtype := range []shapes.DType{shapes.F64, shapes.F32, shapes.I64, shapes.I32} {
		graphtest.RunTestGraphFn(t, fmt.Sprintf("ArgMax()/ArgMin() for dtype %q", dtype),
			func(g *Graph) (inputs, outputs []*Node) {
				inputs = []*Node{
					IotaFull(g, shapes.Make(dtype, 3, 5)),
				}
				outputs = []*Node{
					ArgMax(inputs[0], -1),
					ArgMax(inputs[0], 0),
					ArgMin(inputs[0], 1, shapes.Uint8),
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
