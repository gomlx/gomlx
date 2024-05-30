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
	"testing"

	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	"google3/third_party/golang/plot/plotter/plotter"
	"google3/third_party/golang/plot/vg/vg"
	"github.com/stretchr/testify/require"

	"google3/third_party/golang/plot/plot"
)

var (
	flagPlot = flag.Bool("plot", false, "output plot of layer tests.")
)

type Shape = shapes.Shape

var (
	S   = shapes.Make
	F32 = shapes.Float32
)

func IotaP1Initializer(g *Graph, shape Shape) *Node {
	return Add(Iota(g, shape, 0), Const(g, float32(1.0)))
}

func TestDense(t *testing.T) {
	manager := graphtest.BuildTestManager()
	ctx := context.NewContext(manager).WithInitializer(IotaP1Initializer)
	g := manager.NewGraph("TestDense")
	input := tensor.FromValue([][]float32{{1, 2}, {10, 20}, {100, 200}})
	fmt.Printf("\tinput=%v\n", input)

	inputNode := g.Parameter("input", input.Shape())
	output := DenseWithBias(ctx, inputNode, 3)
	sum := ReduceAllSum(output)

	// Generate the gradients with respect to everything (inputs and variables).
	gradients := Gradient(sum,
		inputNode, ctx.InspectVariable("/dense", "weights").ValueGraph(g),
		ctx.InspectVariable("/dense", "biases").ValueGraph(g))
	g.Compile(sum, output, Tuple(gradients...))

	// Before running the graph initialize the variables.
	ctx.InitializeVariables()
	ctx.EnumerateVariables(func(v *context.Variable) {
		fmt.Printf("\t%s=%v\n", v.ParameterName(), v.Value().Local())
	})

	params := make(ParamsMap)
	ctx.ExecPopulateGraphParamsMap(g, params)
	params[inputNode] = input
	results := g.Run(params).SplitTuple()
	fmt.Printf("\tsum=%v\n", results[0].Local())
	fmt.Printf("\toutput=%v\n", results[1].Local())
	got := results[1].Local().Value()
	want := [][]float32{{6, 7, 8}, {51, 52, 53}, {501, 502, 503}}
	if !slices.DeepSliceCmp(want, got, slices.Equal[float32]) {
		t.Errorf("Got %v, Want %v", got, want)
	}

	{
		gradients := results[2].SplitTuple()

		got := gradients[0].Local()
		fmt.Printf("\t\tgrad sum/input=%v\n", got)
		if want := [][]float32{{3, 6}, {3, 6}, {3, 6}}; !slices.DeepSliceCmp(want, got.Value(), slices.Equal[float32]) {
			t.Errorf("grad sum/input: got=%v, want=%v", got, want)
		}

		got = gradients[1].Local()
		fmt.Printf("\t\tgrad sum/weights=%v\n", got)
		if want := [][]float32{{111, 111, 111}, {222, 222, 222}}; !slices.DeepSliceCmp(want, got.Value(), slices.Equal[float32]) {
			t.Errorf("grad sum/weights: got=%v, want=%v", got, want)
		}

		got = gradients[2].Local()
		fmt.Printf("\t\tgrad sum/biases=%v\n", got)
		if want := []float32{3, 3, 3}; !slices.DeepSliceCmp(want, got.Value(), slices.Equal[float32]) {
			t.Errorf("grad sum/weights: got=%v, want=%v", got, want)
		}
	}
}

func testSimpleFunc(t *testing.T, name string, input any,
	fn func(ctx *context.Context, input *Node) *Node, want any) {
	manager := graphtest.BuildTestManager()
	ctx := context.NewContext(manager).WithInitializer(IotaP1Initializer)
	exec := context.NewExec(manager, ctx, fn)
	var outputs []tensor.Tensor
	require.NotPanicsf(t, func() { outputs = exec.Call(input) }, "%s: failed to exec graph", name)
	fmt.Printf("\t%s(%v) = %s\n", name, input, outputs[0].Local().GoStr())
	require.Truef(t, slices.SlicesInDelta(outputs[0].Local().Value(), want, slices.Epsilon),
		"%s(%v): want=%v, got=%v", name, input, want, outputs[0].Local().GoStr())
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
			input = Ones(g, shapes.Make(shapes.Float32, 100, 3072))
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
	manager := graphtest.BuildTestManager()
	{
		ctx := context.NewContext(manager)
		g := manager.NewGraph("test")
		const numKeypoints = 5
		const maxInput = 100

		var input *Node
		if *flagPlot {
			// For plotting, the input is a parameter of the computation.
			input = g.Parameter("x", S(F32))
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
				params[input] = tensor.FromValue(float32(x))
				tuple := g.Run(params)
				result := tuple.SplitTuple()[1].Local()
				return float64(result.Value().(float32))
			}

			plotComputation("weights", -10, 110, calibration)
			return
		}

		// Continue with normal test, with fixed input values.
		tuple := g.Run(params)
		results := tuple.SplitTuple()
		fmt.Printf("\tinput=%v\n", input)
		fmt.Printf("\tkeypoints=%s\n", results[0].Local().GoStr())
		got := results[1].Local()
		fmt.Printf("\tpwl=%s\n", got.GoStr())
		want := []float32{0, 0.2, 0.5, 0.7, 1}
		if !slices.DeepSliceCmp(want, got.Value(), slices.Close[float32]) {
			t.Errorf("Expected a log-like output, trimmed at the edges: got=%s, want=%v", got.GoStr(), want)
		}
	}
}

func TestLayerNormalization(t *testing.T) {
	testSimpleFunc(t, "LayerNormalization()",
		[][]float32{{0, 10}, {20, 30}, {40, 50}},
		func(ctx *context.Context, input *Node) *Node {
			return LayerNormalization(ctx, input, -1).LearnedOffset(false).LearnedScale(false).Epsilon(0).Done()
		},
		[][]float32{{-1, 1}, {-1, 1}, {-1, 1}},
	)
	testSimpleFunc(t, "LayerNormalization()",
		[][]float32{{0, 10}, {20, 30}, {40, 50}},
		func(ctx *context.Context, input *Node) *Node {
			return LayerNormalization(ctx, input, -1).LearnedOffset(false).LearnedScale(false).Epsilon(0).ScaleNormalization(false).Done()
		},
		[][]float32{{-5, 5}, {-5, 5}, {-5, 5}},
	)
}
