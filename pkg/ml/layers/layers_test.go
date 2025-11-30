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

package layers

import (
	"flag"
	"fmt"
	"strings"
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/initializers"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"

	_ "github.com/gomlx/gomlx/backends/default"
)

var (
	flagPlot = flag.Bool("plot", false, "output plot of layer tests.")
)

type Shape = shapes.Shape

var (
	S   = shapes.Make
	F32 = dtypes.Float32
)

func IotaP1Initializer(g *Graph, shape Shape) *Node {
	return AddScalar(Iota(g, shape, 0), 1.0)
}

func testSimpleFunc(t *testing.T, name string, input any,
	fn func(ctx *context.Context, input *Node) *Node, want any) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New().WithInitializer(IotaP1Initializer)
	exec := context.MustNewExec(backend, ctx, fn)
	var outputs []*tensors.Tensor
	require.NotPanicsf(t, func() { outputs = exec.MustExec(input) }, "%s: failed to exec graph", name)
	fmt.Printf("\t%s(%v) = %s\n", name, input, outputs[0].GoStr())
	require.Truef(t, xslices.SlicesInDelta(outputs[0].Value(), want, xslices.Epsilon),
		"%s(%v): want=%v, got=%v", name, input, want, outputs[0].GoStr())
}

func testSimpleFuncMany(t *testing.T, name string, inputs []any,
	fn func(ctx *context.Context, inputs []*Node) *Node, want any) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New().WithInitializer(IotaP1Initializer)
	exec := context.MustNewExec(backend, ctx, fn)
	var outputs []*tensors.Tensor
	require.NotPanicsf(t, func() { outputs = exec.MustExec(inputs...) }, "%s: failed to exec graph", name)
	parts := make([]string, len(inputs))
	for ii, input := range inputs {
		parts[ii] = fmt.Sprintf("%v", input)
	}
	inputsStr := strings.Join(parts, ", ")
	fmt.Printf("\t%s(%s) = %s\n", name, inputsStr, outputs[0].GoStr())
	require.Truef(t, xslices.SlicesInDelta(outputs[0].Value(), want, xslices.Epsilon),
		"%s(%s): want=%v, got=%v", name, inputsStr, want, outputs[0].GoStr())
}

func TestDense(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New().WithInitializer(IotaP1Initializer)
	g := NewGraph(backend, "TestDense")
	input := tensors.FromValue([][]float32{{1, 2}, {10, 20}, {100, 200}})
	fmt.Printf("\tinput=%v\n", input)

	inputNode := Parameter(g, "input", input.Shape())
	output := DenseWithBias(ctx, inputNode, 3)
	sum := ReduceAllSum(output)

	// Generate the gradients with respect to everything (inputs and variables).
	gradients := Gradient(sum,
		inputNode, ctx.GetVariableByScopeAndName("/dense", "weights").ValueGraph(g),
		ctx.GetVariableByScopeAndName("/dense", "biases").ValueGraph(g))
	outputs := append([]*Node{sum, output}, gradients...)
	g.Compile(outputs...)

	// Before running the graph initialize the variables.
	ctx.InitializeVariables(backend, nil)
	ctx.EnumerateVariables(func(v *context.Variable) {
		fmt.Printf("\t%s=%v\n", v.ParameterName(), v.MustValue())
	})

	params := make(ParamsMap)
	ctx.ExecPopulateGraphParamsMap(g, params)
	params[inputNode] = input
	results := g.RunWithMap(params)
	fmt.Printf("\tsum=%v\n", results[0])
	fmt.Printf("\toutput=%v\n", results[1])
	got := results[1].Value()
	want := [][]float32{{6, 7, 8}, {51, 52, 53}, {501, 502, 503}}
	require.Equalf(t, want, got, "Got %v, Want %v", got, want)

	{
		got := results[2]
		fmt.Printf("\t\tgrad sum/input=%v\n", got)
		want := [][]float32{{3, 6}, {3, 6}, {3, 6}}
		require.Equalf(t, want, got.Value(), "grad sum/input: got %v, want %v", got, want)
	}
	{
		got := results[3]
		fmt.Printf("\t\tgrad sum/weights=%v\n", got)
		want := [][]float32{{111, 111, 111}, {222, 222, 222}}
		require.Equalf(t, want, got.Value(), "grad sum/weights: got %v, want %v", got, want)
	}
	{
		got := results[4]
		fmt.Printf("\t\tgrad sum/biases=%v\n", got)
		want := []float32{3, 3, 3}
		require.Equalf(t, want, got.Value(), "grad sum/biases: got %v, want %v", got, want)
	}
}

func TestDense2(t *testing.T) {
	testSimpleFunc(t, "Dense([4,1,2], true, 3, 1)",
		[][][]float32{{{1, 2}}, {{10, 20}}, {{100, 200}}, {{100, 200}}},
		func(ctx *context.Context, input *Node) *Node {
			return Dense(ctx, input, true, 3, 1)
		},
		[][][][]float32{{{{6}, {7}, {8}}}, {{{51}, {52}, {53}}}, {{{501}, {502}, {503}}}, {{{501}, {502}, {503}}}},
	)

	testSimpleFunc(t, "DenseWithBias([100, 3072], 4) == 0",
		float32(0),
		func(ctx *context.Context, input *Node) *Node {
			g := input.Graph()
			input = Ones(g, shapes.Make(dtypes.Float32, 100, 3072))
			output := DenseWithBias(ctx.WithInitializer(initializers.Zero), input, 4)
			fmt.Printf("\toutput.shape=%s\n", output.Shape())
			return ReduceAllSum(output)
		},
		float32(0),
	)
}

func plotComputation(title string, start, end float64, fns ...func(x float64) float64) {
	p := plot.New()
	p.Title.Text = title

	p.X.Label.Text = "x"
	p.X.Min = start
	p.X.Max = end

	p.Y.Label.Text = "f(x)"
	p.Y.Min = 0
	p.Y.Max = 1.1

	var plotters []plot.Plotter
	for _, fn := range fns {
		fnPlot := plotter.NewFunction(fn)
		fnPlot.Samples = 1000
		plotters = append(plotters, fnPlot)
	}
	p.Add(plotters...)
	if err := p.Save(12*vg.Inch, 6*vg.Inch, title+".png"); err != nil {
		panic(err)
	}
}

func TestPieceWiseLinearCalibration(t *testing.T) {
	manager := graphtest.BuildTestBackend()
	{
		ctx := context.New()
		g := NewGraph(manager, "test")
		const numKeypoints = 5
		const maxInput = 100

		var input *Node
		if *flagPlot {
			// For plotting, the input is a parameter of the computation.
			input = Parameter(g, "x", S(F32))
		} else {
			// For normal testing we use fixed input values.
			input = Const(g, []float32{-1, 5, 25, 50, 110})
		}
		keypoints := Div(IotaFull(g, S(F32, numKeypoints)), Const(g, float32(numKeypoints-1)))
		keypoints = Mul(keypoints, keypoints)
		keypoints = Mul(keypoints, Const(g, float32(maxInput)))
		calibrated := PieceWiseLinearCalibrationCascaded(ctx, input, keypoints, true)
		g.Compile(keypoints, calibrated)
		params := make(ParamsMap)
		ctx.ExecSetVariablesInParams(params, g)

		if *flagPlot {
			// Plot calibration of a point x.
			calibration := func(x float64) float64 {
				params[input] = tensors.FromValue(float32(x))
				result := g.RunWithMap(params)[1]
				return float64(tensors.ToScalar[float32](result))
			}

			plotComputation("weights", -10, 110, calibration)
			return
		}

		// Continue with normal test, with fixed input values.
		results := g.RunWithMap(params)
		fmt.Printf("\tinput=%v\n", input)
		fmt.Printf("\tkeypoints=%s\n", results[0])
		got := results[1]
		fmt.Printf("\tpwl=%s\n", got)
		want := []float32{0, 0.2, 0.5, 0.7, 1}
		require.InDeltaSlicef(t, want, got.Value(), 1e-4,
			"Expected a log-like output, trimmed at the edges: got=%s, want=%v", got, want)
	}
}

func TestLayerNormalization(t *testing.T) {
	testSimpleFunc(t, "LayerNormalization()",
		[][]float32{{0, 10}, {20, 30}, {40, 50}},
		func(ctx *context.Context, input *Node) *Node {
			return LayerNormalization(ctx, input, -1).LearnedOffset(false).LearnedGain(false).Epsilon(0).Done()
		},
		[][]float32{{-1, 1}, {-1, 1}, {-1, 1}},
	)

	testSimpleFunc(t, "LayerNormalization()",
		[][]float32{{0, 10}, {20, 30}, {40, 50}},
		func(ctx *context.Context, input *Node) *Node {
			return LayerNormalization(ctx, input, -1).LearnedOffset(false).LearnedGain(false).Epsilon(0).ScaleNormalization(false).Done()
		},
		[][]float32{{-5, 5}, {-5, 5}, {-5, 5}},
	)
	testSimpleFuncMany(t, "LayerNormalization()",
		[]any{
			[][]float32{{0, 10, 5}, {20, 30, 0}, {0, 30, 50}, {0, 0, 0}},
			[][]bool{{true, true, true}, {true, true, false}, {true, false, true}, {false, false, false}},
		},
		func(ctx *context.Context, inputs []*Node) *Node {
			return LayerNormalization(ctx, inputs[0], -1).Mask(inputs[1]).
				LearnedOffset(false).LearnedGain(false).Epsilon(0).
				ScaleNormalization(false).Done()
		},
		[][]float32{{-5, 5, 0}, {-5, 5, 0}, {-25, 0, 25}, {0, 0, 0}},
	)
}
