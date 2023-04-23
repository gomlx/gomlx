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
	"github.com/gomlx/gomlx/types/slices"
	"testing"
)

// gradTestFunc takes a graph and returns the output being tested, along the nodes that
// we want the gradients for.
type gradTestFunc func(g *Graph) (output *Node, nodesForGrad []*Node)

func testGradients[T interface{ float32 | float64 }](t *testing.T, name string, testFn gradTestFunc, wantForGrad [][]T) {
	manager := BuildManager().MustDone()
	g := manager.NewGraph(name)
	fmt.Printf("%s:\n", name)
	output, nodesForGrad := testFn(g)
	grads := Gradient(ReduceAllSum(output), nodesForGrad...)
	all := make([]*Node, len(grads)+1)
	all[0] = output
	copy(all[1:], grads)
	g.MustOk()
	g.Compile(all...)
	g.MustOk()
	tuple := g.Run(nil)
	if !tuple.Ok() {
		t.Fatalf("Failed to run graph: %+v", tuple.Error())
	}
	results := tuple.SplitTuple()
	fmt.Printf("\toutput=%v\n", results[0].Local().GoStr())
	for ii, want := range wantForGrad {
		got := results[ii+1].Local()
		fmt.Printf("\tgrad(f)/grad(x_%d): got=%v\n", ii, got.GoStr())
		if !slices.DeepSliceCmp(got.Value(), want, slices.Close[T]) {
			t.Errorf("grad f(x)/x_%d: want %v, got %v", ii, want, got.GoStr())
		}
	}
}

func TestGradientBinaryCrossentropy(t *testing.T) {
	testGradients[float64](t, "Gradient BinaryCrossentropy",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			logits := Const(g, []float64{5, 1e-6, 0, 0, -1e-6, -5})
			predictions := Sigmoid(logits)
			labels := Const(g, []float64{1, 0, 1, 0, 0, 1})
			output = ReduceAllSum(BinaryCrossentropy([]*Node{labels}, []*Node{predictions}))
			return output, []*Node{logits}
		}, [][]float64{{-0.00669285, 0.50000025, -0.5, 0.5, 0.49999975, -0.99330715}})
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

type fnToTest func(g *Graph) (input, output *Node)

func testSomeFunc[T interface{ float32 | float64 }](t *testing.T, name string, fn fnToTest, want any, close bool) {
	fmt.Printf("%s\n", name)
	manager := BuildManager().MustDone()
	g := manager.NewGraph(name)
	input, output := fn(g)
	g.Compile(input, output)
	g.MustOk()
	tuple := g.Run(nil)
	if !tuple.Ok() {
		t.Fatalf("Failed to run graph: %+v", tuple.Error())
	}
	results := tuple.SplitTuple()
	fmt.Printf("\t%s(%s) = %s\n", name, results[0].Local().GoStr(), results[1].Local().GoStr())
	if close {
		// Check close.
		if !slices.DeepSliceCmp(results[1].Local().Value(), want, slices.Close[T]) {
			t.Errorf("%s(%v): want=%v, got=%v", name, results[0].Local(), want, results[1].Local().GoStr())
		}
	} else {
		// Check equality.
		if !slices.DeepSliceCmp(results[1].Local().Value(), want, slices.Equal[T]) {
			t.Errorf("%s(%v): want=%v, got=%v", name, results[0].Local(), want, results[1].Local().GoStr())
		}
	}
}

func TestCategoricalCrossEntropy(t *testing.T) {
	testSomeFunc[float32](t, "CategoricalCrossEntropy",
		func(g *Graph) (input, output *Node) {
			labels := Const(g, [][]float32{{0, 1, 0}, {0, 0, 1}})
			predictions := Const(g, [][]float32{{0.05, 0.95, 0}, {0.1, 0.8, 0.1}})
			output = CategoricalCrossEntropy([]*Node{labels}, []*Node{predictions})
			return predictions, output
		}, []float32{0.05129, 2.3026}, true)
}
