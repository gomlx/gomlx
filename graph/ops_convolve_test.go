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
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/tensors/images"
	"github.com/gomlx/gopjrt/dtypes"
	"testing"
)

func TestConvolve(t *testing.T) {
	testFuncOneInput(t, "Convolve(...).ChannelsAxis(images.ChannelsFirst).NoPadding()",
		func(g *Graph) (input, output *Node) {
			channelA := Iota(g, MakeShape(dtypes.Float32, 1, 1, 3, 3), 2)
			channelB := Mul(channelA, Const(g, float32(0.1)))
			input = Concatenate([]*Node{channelA, channelB}, 1)
			kernel := Ones(g, MakeShape(dtypes.Float32, 2, 3, 3, 1))
			output = Convolve(input, kernel).ChannelsAxis(images.ChannelsFirst).NoPadding().Done()
			return
		}, [][][][]float32{{{{9.9}}}})

	testFuncOneInput(t, "Convolve(...).ChannelsAxis(images.ChannelsLast).NoPadding()",
		func(g *Graph) (input, output *Node) {
			channelA := Iota(g, MakeShape(dtypes.Float64, 1, 3, 3, 1), 2)
			channelB := Mul(channelA, Const(g, 0.1))
			input = Concatenate([]*Node{channelA, channelB}, -1)
			kernel := Ones(g, MakeShape(dtypes.Float64, 3, 3, 2, 1))
			output = Convolve(input, kernel).ChannelsAxis(images.ChannelsLast).NoPadding().Done()
			return
		}, [][][][]float64{{{{9.9}}}})

	testFuncOneInput(t, "Convolve(...).ChannelsAxis(images.ChannelsFirst).PadSame()",
		func(g *Graph) (input, output *Node) {
			channelA := Iota(g, MakeShape(dtypes.Float32, 1, 1, 3, 3), 2)
			channelB := Mul(channelA, Const(g, float32(0.1)))
			input = Concatenate([]*Node{channelA, channelB}, 1)
			kernel := Ones(g, MakeShape(dtypes.Float32, 2, 3, 3, 1))
			output = Convolve(input, kernel).ChannelsAxis(images.ChannelsFirst).PadSame().Done()
			return
		}, [][][][]float32{{{{2.2, 3.3, 2.2}, {6.6, 9.9, 6.6}, {6.6, 9.9, 6.6}}}})

	testFuncOneInput(t, "Convolve(...kernel=[2,2])",
		func(g *Graph) (input, output *Node) {
			channelA := Iota(g, MakeShape(dtypes.Float64, 1, 3, 3, 1), 1)
			channelB := Mul(channelA, Const(g, 0.1))
			input = Concatenate([]*Node{channelA, channelB}, -1)
			kernel := Ones(g, MakeShape(dtypes.Float64, 2, 2, 2, 1))
			output = Convolve(input, kernel).PadSame().Done()
			return
		}, [][][][]float64{{{{2.2}, {2.2}, {1.1}}, {{6.6}, {6.6}, {3.3}}, {{4.4}, {4.4}, {2.2}}}})

	testFuncOneInput(t, "Convolve(...).Strides(2)",
		func(g *Graph) (input, output *Node) {
			channelA := Iota(g, MakeShape(dtypes.Float64, 1, 3, 3, 1), 2)
			channelB := Mul(channelA, Const(g, 0.1))
			input = Concatenate([]*Node{channelA, channelB}, -1)
			kernel := Ones(g, MakeShape(dtypes.Float64, 3, 3, 2, 1))
			output = Convolve(input, kernel).Strides(2).PadSame().Done()
			return
		}, [][][][]float64{{{{2.2}, {6.6}}, {{2.2}, {6.6}}}})
	testFuncOneInput(t, "Convolve().NoPadding().Dilations(2)",
		func(g *Graph) (input, output *Node) {
			channelA := Iota(g, MakeShape(dtypes.Float64, 1, 5, 5, 1), 1)
			channelB := Mul(channelA, Const(g, 0.01))
			input = Concatenate([]*Node{channelA, channelB}, -1)
			kernel := Ones(g, MakeShape(dtypes.Float64, 3, 3, 2, 1))
			output = Convolve(input, kernel).NoPadding().Dilations(2).Done()
			return
		}, [][][][]float64{{{{18.18}}}})

	testFuncOneInput(t, "Convolve().Dilations(2)",
		func(g *Graph) (input, output *Node) {
			channelA := Iota(g, MakeShape(dtypes.Float64, 1, 5, 5, 1), 1)
			channelB := Mul(channelA, Const(g, 0.01))
			input = Concatenate([]*Node{channelA, channelB}, -1)
			kernel := Ones(g, MakeShape(dtypes.Float64, 3, 3, 2, 1))
			output = Convolve(input, kernel).Dilations(2).PadSame().Done()
			return
		}, [][][][]float64{
			{{{4.04}, {4.04}, {6.06}, {4.04}, {4.04}},
				{{8.08}, {8.08}, {12.12}, {8.08}, {8.08}},
				{{12.12}, {12.12}, {18.18}, {12.12}, {12.12}},
				{{8.08}, {8.08}, {12.12}, {8.08}, {8.08}},
				{{12.12}, {12.12}, {18.18}, {12.12}, {12.12}}}})

	testFuncOneInput(t, "Convolve().InputDilations(2)",
		func(g *Graph) (input, output *Node) {
			input = Add(IotaFull(g, MakeShape(dtypes.Float64, 1, 2, 1)), Const(g, 1.0))
			kernel := Add(IotaFull(g, MakeShape(dtypes.Float64, 3, 1, 1)), Const(g, 1.0))
			output = Convolve(input, kernel).PaddingPerDim([][2]int{{2, 2}}).InputDilationPerDim(2).Done()
			return
		}, [][][]float64{{{3}, {2}, {7}, {4}, {2}}})
}

func TestGradientConvolve(t *testing.T) {
	testGradients(t, "Gradient 1D Convolve().NoPadding() -- scaled output",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := Add(IotaFull(g, MakeShape(dtypes.Float64, 1, 5, 1)), Const(g, 1.0))
			kernel := Add(IotaFull(g, MakeShape(dtypes.Float64, 3, 1, 1)), Const(g, 1.0))
			output = Convolve(input, kernel).NoPadding().Done()
			scale := Add(IotaFull(g, output.Shape()), Const(g, 1.0))
			output.SetLogged("Output")
			output = Mul(output, scale)
			return output, []*Node{input, kernel}
		}, []any{
			[][][]float64{{{1}, {4}, {10}, {12}, {9}}},
			[][][]float64{{{14}}, {{20}}, {{26}}},
		})

	testGradients(t, "Gradient 2D Convolve().NoPadding()",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			channelA := IotaFull(g, MakeShape(dtypes.Float64, 1, 5, 5, 1))
			channelB := Mul(channelA, Const(g, 0.001))
			input := Concatenate([]*Node{channelA, channelB}, -1)
			kernel := IotaFull(g, MakeShape(dtypes.Float64, 3, 3, 1))
			kernel = Concatenate([]*Node{kernel, kernel}, -1)
			kernel = Reshape(kernel, 3, 3, 2, 1)
			output = Convolve(input, kernel).NoPadding().Done()
			return output, []*Node{input, kernel}
		}, []any{
			[][][][]float64{{
				{{0, 0}, {1, 1}, {3, 3}, {3, 3}, {2, 2}},
				{{3, 3}, {8, 8}, {15, 15}, {12, 12}, {7, 7}},
				{{9, 9}, {21, 21}, {36, 36}, {27, 27}, {15, 15}},
				{{9, 9}, {20, 20}, {33, 33}, {24, 24}, {13, 13}},
				{{6, 6}, {13, 13}, {21, 21}, {15, 15}, {8, 8}},
			}},
			[][][][]float64{
				{{{54}, {0.054}}, {{63}, {0.063}}, {{72}, {0.072}}},
				{{{99}, {0.099}}, {{108}, {0.108}}, {{117}, {0.117}}},
				{{{144}, {0.144}}, {{153}, {0.153}}, {{162}, {0.162}}},
			},
		})

	testGradients(t, "Gradient Convolve().PadSame()",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			channelA := IotaFull(g, MakeShape(dtypes.Float64, 1, 3, 3, 1))
			channelB := Mul(channelA, Const(g, 0.1))
			input := Concatenate([]*Node{channelA, channelB}, -1)
			kernel := IotaFull(g, MakeShape(dtypes.Float64, 3, 3, 1, 2))
			kernel = Concatenate([]*Node{kernel, Mul(kernel, Const(g, 0.1))}, 2)
			output = Convolve(input, kernel).PadSame().Done()
			return output, []*Node{input, kernel}
		}, []any{
			[][][][]float64{{
				{{36, 3.6}, {66, 6.6}, {52, 5.2}},
				{{90, 9}, {153, 15.3}, {114, 11.4}},
				{{84, 8.4}, {138, 13.8}, {100, 10}},
			}},
			[][][][]float64{
				{{{8, 8}, {0.8, 0.8}}, {{15, 15}, {1.5, 1.5}}, {{12, 12}, {1.2, 1.2}}},
				{{{21, 21}, {2.1, 2.1}}, {{36, 36}, {3.6, 3.6}}, {{27, 27}, {2.7, 2.7}}},
				{{{20, 20}, {2, 2}}, {{33, 33}, {3.3, 3.3}}, {{24, 24}, {2.4, 2.4}}},
			},
		})

	testGradients(t, "Gradient Convolve().NoPadding().Dilations(2)",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			channelA := IotaFull(g, MakeShape(dtypes.Float64, 1, 5, 5, 1))
			channelB := Mul(channelA, Const(g, 0.001))
			input := Concatenate([]*Node{channelA, channelB}, -1)
			kernel := IotaFull(g, MakeShape(dtypes.Float64, 3, 3, 1, 2))
			kernel = Concatenate([]*Node{kernel, Mul(kernel, Const(g, 0.1))}, 2)
			output = Convolve(input, kernel).NoPadding().Dilations(2).Done()
			return output, []*Node{input, kernel}
		}, []any{
			[][][][]float64{{
				{{1, 0.1}, {0, 0}, {5, 0.5}, {0, 0}, {9, 0.9}},
				{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
				{{13, 1.3}, {0, 0}, {17, 1.7}, {0, 0}, {21, 2.1}},
				{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
				{{25, 2.5}, {0, 0}, {29, 2.9}, {0, 0}, {33, 3.3}},
			}},
			[][][][]float64{
				{{{0, 0}, {0, 0}}, {{2, 2}, {0.002, 0.002}}, {{4, 4}, {0.004, 0.004}}},
				{{{10, 10}, {0.01, 0.01}}, {{12, 12}, {0.012, 0.012}}, {{14, 14}, {0.014, 0.014}}},
				{{{20, 20}, {0.02, 0.02}}, {{22, 22}, {0.022, 0.022}}, {{24, 24}, {0.024, 0.024}}},
			},
		})

	testGradients(t, "Gradient 2D Convolve().NoPadding().Strides(2)",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			channelA := IotaFull(g, MakeShape(dtypes.Float64, 1, 5, 5, 1))
			channelB := Mul(channelA, Const(g, 0.001))
			input := Concatenate([]*Node{channelA, channelB}, -1)
			kernel := IotaFull(g, MakeShape(dtypes.Float64, 3, 3, 1, 2))
			kernel = Concatenate([]*Node{kernel, Mul(kernel, Const(g, 0.1))}, 2)
			output = Convolve(input, kernel).NoPadding().Strides(2).Done()
			return output, []*Node{input, kernel}
		}, []any{
			[][][][]float64{{
				{{1, 0.1}, {5, 0.5}, {10, 1}, {5, 0.5}, {9, 0.9}},
				{{13, 1.3}, {17, 1.7}, {34, 3.4}, {17, 1.7}, {21, 2.1}},
				{{26, 2.6}, {34, 3.4}, {68, 6.8}, {34, 3.4}, {42, 4.2}},
				{{13, 1.3}, {17, 1.7}, {34, 3.4}, {17, 1.7}, {21, 2.1}},
				{{25, 2.5}, {29, 2.9}, {58, 5.8}, {29, 2.9}, {33, 3.3}},
			}},
			[][][][]float64{
				{{{24, 24}, {0.024, 0.024}}, {{28, 28}, {0.028, 0.028}}, {{32, 32}, {0.032, 0.032}}},
				{{{44, 44}, {0.044, 0.044}}, {{48, 48}, {0.048, 0.048}}, {{52, 52}, {0.052, 0.052}}},
				{{{64, 64}, {0.064, 0.064}}, {{68, 68}, {0.068, 0.068}}, {{72, 72}, {0.072, 0.072}}},
			},
		})

	testGradients(t, "Gradient 2D Convolve().NoPadding().Strides(2): shape check",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := Zeros(g, MakeShape(dtypes.Float64, 2, 8, 7, 3))
			kernel := Zeros(g, MakeShape(dtypes.Float64, 1, 1, 3, 6))
			output = Convolve(input, kernel).NoPadding().Strides(2).Done()
			return output, []*Node{input, kernel}
		}, []any{
			tensors.FromScalarAndDimensions(0.0, 2, 8, 7, 3).Value(),
			tensors.FromScalarAndDimensions(0.0, 1, 1, 3, 6).Value(),
		})

	testGradients(t, "Gradient 2D Convolve().NoPadding().Dilations(2): shape check",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := Zeros(g, MakeShape(dtypes.Float64, 2, 5, 5, 3))
			kernel := Zeros(g, MakeShape(dtypes.Float64, 2, 2, 3, 6))
			output = Convolve(input, kernel).NoPadding().Dilations(2).Done()
			return output, []*Node{input, kernel}
		}, []any{
			tensors.FromScalarAndDimensions(0.0, 2, 5, 5, 3).Value(),
			tensors.FromScalarAndDimensions(0.0, 2, 2, 3, 6).Value(),
		})
}

func TestConvolveWithGroupCount(t *testing.T) {
	testFuncOneInput(t, "SUCCESS: Convolution with FeatureGroupCount=2",
		func(g *Graph) (input, output *Node) {
			// Input with 2 channels (matches FeatureGroupCount)
			input = IotaFull(g, MakeShape(dtypes.Float32, 1, 2, 3, 3))

			// Create kernel for grouped convolution (1 input channel per group)
			kernel := Const(g, [][][][]float32{{{{0, 0}, {0, 0}, {0, 0}}, {{0, 0}, {1, 2}, {0, 0}}, {{0, 0}, {0, 0}, {0, 0}}}})

			output = Convolve(input, kernel).
				ChannelsAxis(images.ChannelsFirst).
				NoPadding().
				FeatureGroupCount(2).
				Done()
			return
		},
		[][][][]float32{{{{4.0}}, {{26.0}}}})

	testFuncOneInput(t, "SUCCESS: Convolution with BatchGroupCount=2",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, MakeShape(dtypes.Float32, 2, 1, 3, 3))

			// Create kernel for grouped convolution (1 input channel per group)
			kernel := Const(g, [][][][]float32{{{{0, 0}, {0, 0}, {0, 0}}, {{0, 0}, {1, 2}, {0, 0}}, {{0, 0}, {0, 0}, {0, 0}}}})

			output = Convolve(input, kernel).
				ChannelsAxis(images.ChannelsFirst).
				NoPadding().
				BatchGroupCount(2).
				Done()
			return
		},
		[][][][]float32{{{{4.0}}, {{26.0}}}})
}
