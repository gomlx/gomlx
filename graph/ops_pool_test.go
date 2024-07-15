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
	timage "github.com/gomlx/gomlx/types/tensor/image"
	"testing"
)

func TestMaxPool(t *testing.T) {
	testFuncOneInput(t, "MaxPool(...).ChannelsFirst().NoPadding()",
		func(g *Graph) (input, output *Node) {
			channelA := Iota(g, MakeShape(dtypes.Float32, 1, 1, 3, 3), 2)
			channelB := Mul(channelA, Const(g, float32(0.1)))
			input = Concatenate([]*Node{channelA, channelB}, 1)
			output = MaxPool(input).ChannelsAxis(timage.ChannelsFirst).Window(3).NoPadding().Done()
			return
		}, [][][][]float32{{{{2}}, {{0.2}}}})

	testFuncOneInput(t, "MaxPool(...).ChannelsAfter().NoPadding()",
		func(g *Graph) (input, output *Node) {
			channelA := Iota(g, MakeShape(dtypes.Float64, 1, 3, 3, 1), 2)
			channelB := Mul(channelA, Const(g, 0.1))
			input = Concatenate([]*Node{channelA, channelB}, -1)
			output = MaxPool(input).Window(3).ChannelsAxis(timage.ChannelsLast).NoPadding().Done()
			return
		}, [][][][]float64{{{{2, 0.2}}}})

	testFuncOneInput(t, "MaxPool(...).Window(3).ChannelsFirst().PadSame().Strides(1)",
		func(g *Graph) (input, output *Node) {
			channelA := Iota(g, MakeShape(dtypes.Float32, 1, 1, 3, 3), 2)
			channelB := Mul(channelA, Const(g, float32(0.1)))
			input = Concatenate([]*Node{channelA, channelB}, 1)
			output = MaxPool(input).Window(3).ChannelsAxis(timage.ChannelsFirst).PadSame().Strides(1).Done()
			return
		}, [][][][]float32{{{{1, 1, 1}, {2, 2, 2}, {2, 2, 2}}, {{0.1, 0.1, 0.1}, {0.2, 0.2, 0.2}, {0.2, 0.2, 0.2}}}})

	testFuncOneInput(t, "MaxPool(...).Window(2).Strides(1)",
		func(g *Graph) (input, output *Node) {
			channelA := IotaFull(g, MakeShape(dtypes.Float64, 1, 5, 1))
			channelB := Mul(channelA, Const(g, 0.1))
			input = Concatenate([]*Node{channelA, channelB}, -1)
			output = MaxPool(input).Window(2).Strides(1).Done()
			return
		}, [][][]float64{{{1, 0.1}, {2, 0.2}, {3, 0.3}, {4, 0.4}}})
	testFuncOneInput(t, "MaxPool(...).Window(2)",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, MakeShape(dtypes.Float64, 1, 5, 5, 1))
			output = MaxPool(input).Window(2).Done()
			return
		}, [][][][]float64{{{{6}, {8}}, {{16}, {18}}}})
}

func TestGradientMaxPool(t *testing.T) {
	testGradients(t, "Gradient 1D MaxPool().NoPadding() -- scaled output",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := Add(IotaFull(g, MakeShape(dtypes.Float32, 1, 6, 1)), Const(g, float32(1.0)))
			output = MaxPool(input).NoPadding().Window(3).Strides(1).Done()
			scale := Add(IotaFull(g, output.Shape()), Const(g, float32(1.0)))
			output = Mul(output, scale)
			return output, []*Node{input}
		}, []any{
			[][][]float32{{{0}, {0}, {1}, {2}, {3}, {4}}},
		})

	testGradients(t, "Gradient 1D MaxPool(...).Window(2).Strides(1)",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			channelA := IotaFull(g, MakeShape(dtypes.Float64, 1, 5, 1))
			channelB := Mul(channelA, Const(g, 0.1))
			input := Concatenate([]*Node{channelA, channelB}, -1)
			output = MaxPool(input).Window(2).Strides(1).Done()
			return output, []*Node{input}
		}, []any{
			[][][]float64{{{0, 0}, {1, 1}, {1, 1}, {1, 1}, {1, 1}}},
		})

	testGradients(t, "Gradient 2D MaxPool(...).Window(2)",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := IotaFull(g, MakeShape(dtypes.Float64, 1, 5, 5, 1))
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

// TestSumPool only tests that the correct reduction is applied. Windows and strides are already
// tested with TestMaxPool.
func TestSumPool(t *testing.T) {
	testFuncOneInput(t, "SumPool(...)",
		func(g *Graph) (input, output *Node) {
			channelA := IotaFull(g, MakeShape(dtypes.Float64, 1, 3, 3, 1))
			channelB := Mul(channelA, Const(g, 0.1))
			input = Concatenate([]*Node{channelA, channelB}, -1)
			output = SumPool(input).Window(3).ChannelsAxis(timage.ChannelsLast).NoPadding().Done()
			return
		}, [][][][]float64{{{{36, 3.6}}}})
}

// TestGradientSumPool only tests that the correct gradient for the reduction is applied. More fine-grained testing
// is done in TestGradientMeanPool, which we can verify the result with Tensorflow.
func TestGradientSumPool(t *testing.T) {
	testGradients(t, "Gradient 2D SumPool(...).Window(3).PadSame().Strides(1)",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := IotaFull(g, MakeShape(dtypes.Float32, 1, 3, 1))
			output = SumPool(input).Window(3).PadSame().Strides(1).Done()
			return output, []*Node{input}
		}, []any{
			[][][]float32{{{2}, {3}, {2}}},
		})

	testGradients(t, "Gradient 2D Even Spatial Dimensions SumPool(...).Window(2)",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := IotaFull(g, MakeShape(dtypes.Float64, 1, 5, 1))
			output = SumPool(input).Window(2).Strides(3).
				PaddingPerDim([][2]int{{0, 1}}).Done()
			output = Mul(output, OnePlus(IotaFull(g, output.Shape())))
			return output, []*Node{input}
		}, []any{
			[][][]float64{{{1}, {1}, {0}, {2}, {2}}},
		})

	testGradients(t, "Gradient 2D Even Spatial Dimensions SumPool(...).Window(2)",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := IotaFull(g, MakeShape(dtypes.Float64, 1, 4, 4, 1))
			output = SumPool(input).Window(2).Done()
			output = Mul(output, OnePlus(IotaFull(g, output.Shape())))
			return output, []*Node{input}
		}, []any{
			[][][][]float64{{
				{{1}, {1}, {2}, {2}},
				{{1}, {1}, {2}, {2}},
				{{3}, {3}, {4}, {4}},
				{{3}, {3}, {4}, {4}},
			}},
		})

	testGradients(t, "Gradient 1D SumPool() -- scaled output",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := Add(IotaFull(g, MakeShape(dtypes.Float32, 1, 6, 1)), Const(g, float32(1.0)))
			output = SumPool(input).NoPadding().Window(3).Strides(1).Done()
			scale := OnePlus(IotaFull(g, output.Shape()))
			output = Mul(output, scale)
			return output, []*Node{input}
		}, []any{
			[][][]float32{{{1}, {3}, {6}, {9}, {7}, {4}}},
		})
}

// TestMeanPool only tests that the correct reduction and normalization are applied. Windows and strides are already
// tested with TestMaxPool.
func TestMeanPool(t *testing.T) {
	testFuncOneInput(t, "MeanPool(...).ChannelsAfter().NoPadding()",
		func(g *Graph) (input, output *Node) {
			channelA := IotaFull(g, MakeShape(dtypes.Float64, 1, 3, 3, 1))
			channelB := Mul(channelA, Const(g, 0.1))
			input = Concatenate([]*Node{channelA, channelB}, -1)
			output = MeanPool(input).Window(3).ChannelsAxis(timage.ChannelsLast).NoPadding().Done()
			return
		}, [][][][]float64{{{{4, 0.4}}}})

	fmt.Println()
	testFuncOneInput(t, "MeanPool(...).ChannelsAfter().PadSame().",
		func(g *Graph) (input, output *Node) {
			channelA := IotaFull(g, MakeShape(dtypes.Float64, 1, 3, 3, 1))
			channelB := Mul(channelA, Const(g, 0.1))
			input = Concatenate([]*Node{channelA, channelB}, -1)
			input = Concatenate([]*Node{input, input}, 0) // Batch of 2.
			output = MeanPool(input).Window(3).ChannelsAxis(timage.ChannelsLast).PadSame().Strides(1).Done()
			return
		}, [][][][]float64{{
			{{2.0, 0.20}, {2.5, 0.25}, {3.0, 0.30}},
			{{3.5, 0.35}, {4.0, 0.40}, {4.5, 0.45}},
			{{5.0, 0.50}, {5.5, 0.55}, {6.0, 0.60}},
		}, {
			{{2.0, 0.20}, {2.5, 0.25}, {3.0, 0.30}},
			{{3.5, 0.35}, {4.0, 0.40}, {4.5, 0.45}},
			{{5.0, 0.50}, {5.5, 0.55}, {6.0, 0.60}},
		}})
}

// TestGradientMeanPool can leverage `tensorflow.nn.pool` with "AVG" type. Results should be identical.
func TestGradientMeanPool(t *testing.T) {
	// The results below should reflect that the last column and row are not used in the output.
	testGradients(t, "Gradient 2D MeanPool(...).Window(2)",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := IotaFull(g, MakeShape(dtypes.Float64, 1, 5, 5, 1))
			output = MeanPool(input).Window(2).Done()
			return output, []*Node{input}
		}, []any{
			[][][][]float64{{
				{{0.25}, {0.25}, {0.25}, {0.25}, {0}},
				{{0.25}, {0.25}, {0.25}, {0.25}, {0}},
				{{0.25}, {0.25}, {0.25}, {0.25}, {0}},
				{{0.25}, {0.25}, {0.25}, {0.25}, {0}},
				{{0}, {0}, {0}, {0}, {0}},
			}},
		})

	testGradients(t, "Gradient Unbalanced 1D MeanPool(...).Window(2)",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := IotaFull(g, MakeShape(dtypes.Float64, 1, 3, 1))
			output = MeanPool(input).Window(2).PaddingPerDim([][2]int{{2, 0}}).Done()
			return output, []*Node{input}
		}, []any{
			[][][]float64{{{0.5}, {0.5}, {0.0}}},
		})

	testGradients(t, "Gradient 1D MeanPool() -- scaled output",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := Add(IotaFull(g, MakeShape(dtypes.Float32, 1, 6, 1)), Const(g, float32(1.0)))
			output = MeanPool(input).NoPadding().Window(3).Strides(1).Done()
			scale := Add(IotaFull(g, output.Shape()), ScalarOne(g, dtypes.Float32))
			output = Mul(output, scale)
			return output, []*Node{input}
		}, []any{
			[][][]float32{{{1.0 / 3.0}, {1.0}, {2.0}, {3.0}, {2.0 + 1.0/3.0}, {4.0 / 3.0}}},
		})
	fmt.Println()

	testGradients(t, "Gradient 2D MeanPool(...).Window(3).ChannelsFirst().PadSame().Strides(1)",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			channelA := IotaFull(g, MakeShape(dtypes.Float32, 1, 1, 3, 3))
			channelB := Mul(channelA, Const(g, float32(0.1)))
			input := Concatenate([]*Node{channelA, channelB}, 1)
			output = MeanPool(input).ChannelsAxis(timage.ChannelsFirst).Window(3).PadSame().Strides(1).Done()
			return output, []*Node{input}
		}, []any{
			[][][][]float32{{{
				{0.6944445, 1.1111112, 0.6944445},
				{1.1111112, 1.7777778, 1.1111112},
				{0.6944445, 1.1111112, 0.6944445},
			}, {
				{0.6944445, 1.1111112, 0.6944445},
				{1.1111112, 1.7777778, 1.1111112},
				{0.6944445, 1.1111112, 0.6944445},
			}}},
		})
	fmt.Println()

	testGradients(t, "Gradient 2D even spatial dims MeanPool(...).Window(2)",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			input := IotaFull(g, MakeShape(dtypes.Float64, 1, 2, 2, 1))
			output = MeanPool(input).Window(2).Done()
			return output, []*Node{input}
		}, []any{
			[][][][]float64{{{{0.25}, {0.25}}, {{0.25}, {0.25}}}},
		})

}
