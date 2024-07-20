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

package losses

import (
	"fmt"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/stretchr/testify/require"
	"testing"

	_ "github.com/gomlx/gomlx/backends/xla"
)

// gradTestFunc takes a graph and returns the output being tested, along the nodes that
// we want the gradients for.
type gradTestFunc func(g *Graph) (output *Node, nodesForGrad []*Node)

const deltaForTests = 1e-3

func testGradients[T interface{ float32 | float64 }](t *testing.T, name string, testFn gradTestFunc, wantForGrad [][]T) {
	manager := graphtest.BuildTestBackend()
	g := NewGraph(manager, name)
	fmt.Printf("%s:\n", name)
	output, nodesForGrad := testFn(g)
	grads := Gradient(ReduceAllSum(output), nodesForGrad...)
	all := make([]*Node, len(grads)+1)
	all[0] = output
	copy(all[1:], grads)
	g.Compile(all...)
	results := g.Run()
	fmt.Printf("\toutput=%v\n", results[0])
	for ii, want := range wantForGrad {
		got := results[ii+1]
		fmt.Printf("\tgrad(f)/grad(x_%d): got=%v\n", ii, got)
		require.InDeltaSlicef(t, want, got.Value(), deltaForTests, "grad f(x)/x_%d: want %v, got %v", ii, want, got)
	}
}

type fnToTest func(g *Graph) (input, output *Node)

func testSomeFunc[T interface{ float32 | float64 }](t *testing.T, name string, fn fnToTest, want any, inDelta bool) {
	fmt.Printf("%s\n", name)
	manager := graphtest.BuildTestBackend()
	g := NewGraph(manager, name)
	input, output := fn(g)
	g.Compile(input, output)
	results := g.Run()
	fmt.Printf("\t%s(%s) = %s\n", name, results[0], results[1])
	if inDelta {
		if results[1].IsScalar() {
			require.InDeltaf(t, want, results[1].Value(), deltaForTests, "%s(%v): want=%v, got=%v", name, results[0], want, results[1])
		} else {
			require.InDeltaSlicef(t, want, results[1].Value(), deltaForTests, "%s(%v): want=%v, got=%v", name, results[0], want, results[1])
		}
	} else {
		require.Equal(t, want, results[1].Value(), deltaForTests, "%s(%v): want=%v, got=%v", name, results[0], want, results[1])
	}
}

func TestMeanSquaredError(t *testing.T) {
	testSomeFunc[float32](t, "MeanSquaredErrorWithWeightsAndMask",
		func(g *Graph) (input, output *Node) {
			labels := Const(g, []float32{1.0, 2.0, 7.0})
			mask := Const(g, []bool{true, true, false})
			weights := Const(g, []float32{5.0, 1.0, 3.2})
			predictions := Const(g, []float32{2.0, 4.0, 0})
			output = MeanSquaredError([]*Node{labels, mask, weights}, []*Node{predictions})
			return predictions, output
		}, float32(5.0*1.0+1.0*4.0)/3, true)
}

func TestMeanAbsoluteError(t *testing.T) {
	testSomeFunc[float32](t, "MeanAbsoluteErrorWithWeightsAndMask",
		func(g *Graph) (input, output *Node) {
			labels := Const(g, []float32{1.0, 2.0, 7.0})
			mask := Const(g, []bool{true, true, false})
			weights := Const(g, []float32{5.0, 1.0, 3.2})
			predictions := Const(g, []float32{0.0, 4.0, 0})
			output = MeanAbsoluteError([]*Node{labels, mask, weights}, []*Node{predictions})
			return predictions, output
		}, float32(5.0*1.0+1.0*2.0)/3, true)
}

func TestGradientBinaryCrossentropy(t *testing.T) {
	testGradients[float64](t, "Gradient BinaryCrossentropy",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			logits := Const(g, []float64{5, 1e-6, 0, 0, -1e-6, -5})
			predictions := Sigmoid(logits)
			labels := Const(g, []float64{1, 0, 1, 0, 0, 1})
			output = ReduceAllSum(BinaryCrossentropy([]*Node{labels}, []*Node{predictions}))
			return output, []*Node{logits}
		}, [][]float64{{-0.00669, 0.5, -0.5, 0.5, 0.5, -0.9933}})
}

func TestGradientBinaryCrossentropyLogits(t *testing.T) {
	testGradients[float64](t, "Gradient BinaryCrossentropyLogits",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			logits := Const(g, []float64{5, 1e-6, 0, 0, -1e-6, -5})
			labels := Const(g, []float64{1, 0, 1, 0, 0, 1})
			output = ReduceAllSum(BinaryCrossentropyLogits([]*Node{labels}, []*Node{logits}))
			return output, []*Node{logits}
		}, [][]float64{{-0.00669285, 0.50000025, -0.5, 0.5, 0.49999975, -0.99330715}})
}

func TestCategoricalCrossEntropy(t *testing.T) {
	testSomeFunc[float32](t, "CategoricalCrossEntropy",
		func(g *Graph) (input, output *Node) {
			labels := Const(g, [][]float32{{0, 1, 0}, {0, 0, 1}})
			predictions := Const(g, [][]float32{{0.05, 0.95, 0}, {0.1, 0.8, 0.1}})
			output = CategoricalCrossEntropy([]*Node{labels}, []*Node{predictions})
			return predictions, output
		}, []float32{0.05129, 2.3026}, true)

	testSomeFunc[float32](t, "CategoricalCrossEntropyWithMask",
		func(g *Graph) (input, output *Node) {
			labels := Const(g, [][]float32{{0, 1, 0}, {0, 0, 1}, {0, 0, 0}})
			mask := Const(g, []bool{true, true, false})
			predictions := Const(g, [][]float32{{0.05, 0.95, 0}, {0.1, 0.8, 0.1}, {0, 0, 0}})
			output = CategoricalCrossEntropy([]*Node{labels, mask}, []*Node{predictions})
			return predictions, output
		}, []float32{0.05129, 2.3026, 0}, true)

	testSomeFunc[float32](t, "CategoricalCrossEntropyWithMaskAndWeights",
		func(g *Graph) (input, output *Node) {
			labels := Const(g, [][]float32{{0, 1, 0}, {0, 0, 1}, {0, 0, 0}})
			mask := Const(g, []bool{true, true, false})
			weights := Const(g, []float32{1.0, 2.0, 0.0})
			predictions := Const(g, [][]float32{{0.05, 0.95, 0}, {0.1, 0.8, 0.1}, {0, 0, 0}})
			output = CategoricalCrossEntropy([]*Node{labels, weights, mask}, []*Node{predictions})
			return predictions, output
		}, []float32{0.05129, 4.6052, 0}, true)

	testSomeFunc[float32](t, "SparseCategoricalCrossEntropyLogits",
		func(g *Graph) (input, output *Node) {
			labels := Const(g, [][]int32{{1}, {2}, {0}})
			mask := Const(g, []bool{true, true, false})
			predictions := Const(g, [][]float32{{0, 10.0, 0}, {10.0, 0, 0}, {0, 0, 0}})
			output = SparseCategoricalCrossEntropyLogits([]*Node{labels, mask}, []*Node{predictions})
			return predictions, output
		}, []float32{0, 10.0, 0}, true)
}
