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
	"github.com/gomlx/gomlx/types/shapes"
	"testing"
)

func TestMaxPool(t *testing.T) {
	testFuncOneInput(t, "MaxPool(...).ChannelsFirst().NoPadding()",
		func(g *Graph) (input, output *Node) {
			channelA := Iota(g, MakeShape(shapes.Float32, 1, 1, 3, 3), 2)
			channelB := Mul(channelA, Const(g, float32(0.1)))
			input = Concatenate([]*Node{channelA, channelB}, 1)
			output = MaxPool(input).ChannelsFirst().Window(3).NoPadding().Done()
			return
		}, [][][][]float32{{{{2}}, {{0.2}}}})

	testFuncOneInput(t, "MaxPool(...).ChannelsAfter().NoPadding()",
		func(g *Graph) (input, output *Node) {
			channelA := Iota(g, MakeShape(shapes.Float64, 1, 3, 3, 1), 2)
			channelB := Mul(channelA, Const(g, 0.1))
			input = Concatenate([]*Node{channelA, channelB}, -1)
			output = MaxPool(input).Window(3).ChannelsAfter().NoPadding().Done()
			return
		}, [][][][]float64{{{{2, 0.2}}}})

	testFuncOneInput(t, "MaxPool(...).Window(3).ChannelsFirst().PadSame().Strides(1)",
		func(g *Graph) (input, output *Node) {
			channelA := Iota(g, MakeShape(shapes.Float32, 1, 1, 3, 3), 2)
			channelB := Mul(channelA, Const(g, float32(0.1)))
			input = Concatenate([]*Node{channelA, channelB}, 1)
			output = MaxPool(input).Window(3).ChannelsFirst().PadSame().Strides(1).Done()
			return
		}, [][][][]float32{{{{1, 1, 1}, {2, 2, 2}, {2, 2, 2}}, {{0.1, 0.1, 0.1}, {0.2, 0.2, 0.2}, {0.2, 0.2, 0.2}}}})

	testFuncOneInput(t, "MaxPool(...).Window(2).Strides(1)",
		func(g *Graph) (input, output *Node) {
			channelA := IotaFull(g, MakeShape(shapes.Float64, 1, 5, 1))
			channelB := Mul(channelA, Const(g, 0.1))
			input = Concatenate([]*Node{channelA, channelB}, -1)
			output = MaxPool(input).Window(2).Strides(1).Done()
			return
		}, [][][]float64{{{1, 0.1}, {2, 0.2}, {3, 0.3}, {4, 0.4}}})
	testFuncOneInput(t, "MaxPool(...).Window(2)",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, MakeShape(shapes.Float64, 1, 5, 5, 1))
			output = MaxPool(input).Window(2).Done()
			return
		}, [][][][]float64{{{{6}, {8}}, {{16}, {18}}}})
}

func TestGradientMaxPool(t *testing.T) {
	testGradients[float32](t, "Gradient 1D MaxPool().NoPadding() -- scaled output",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := Add(IotaFull(g, MakeShape(shapes.Float32, 1, 6, 1)), Const(g, float32(1.0)))
			output = MaxPool(input).NoPadding().Window(3).Strides(1).Done()
			output.SetLogged("output")
			scale := Add(IotaFull(g, output.Shape()), Const(g, float32(1.0)))
			output = Mul(output, scale)
			return output, []*Node{input}
		}, []any{
			[][][]float32{{{0}, {0}, {1}, {2}, {3}, {4}}},
		})

	testGradients[float64](t, "Gradient 1D MaxPool(...).Window(2).Strides(1)",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			channelA := IotaFull(g, MakeShape(shapes.Float64, 1, 5, 1))
			channelB := Mul(channelA, Const(g, 0.1))
			input := Concatenate([]*Node{channelA, channelB}, -1)
			output = MaxPool(input).Window(2).Strides(1).Done()
			return output, []*Node{input}
		}, []any{
			[][][]float64{{{0, 0}, {1, 1}, {1, 1}, {1, 1}, {1, 1}}},
		})

	testGradients[float64](t, "Gradient 2D MaxPool(...).Window(2)",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := IotaFull(g, MakeShape(shapes.Float64, 1, 5, 5, 1))
			output = MaxPool(input).Window(2).Done()
			return output, []*Node{input}
		}, []any{
			[][][][]float64{{
				{{0}, {0}, {0}, {0}, {0}},
				{{0}, {1}, {0}, {1}, {0}},
				{{0}, {0}, {0}, {0}, {0}},
				{{0}, {1}, {0}, {1}, {0}},
				{{0}, {0}, {0}, {0}, {0}},
			}},
		})
}
