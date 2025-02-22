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
	"testing"

	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/xla"
)

// gradTestFunc takes a graph and returns the output being tested, along the nodes that
// we want the gradients for.
type gradTestFunc func(g *Graph) (output *Node, nodesForGrad []*Node)

const deltaForTests = 1e-3

func testGradients[T interface{ float32 | float64 }](t *testing.T, name string, testFn gradTestFunc, wantForGrad [][]T) {
	testGradientsInDelta(t, name, testFn, wantForGrad, deltaForTests)
}

func testGradientsInDelta[T interface{ float32 | float64 }](t *testing.T, name string, testFn gradTestFunc, wantForGrad [][]T, delta float64) {
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
		require.InDeltaSlicef(t, want, got.Value(), delta, "grad f(x)/x_%d: want %v, got %v", ii, want, got)
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

func TestCheckExtraLabelsForWeightsAndMask(t *testing.T) {
	graphtest.RunTestGraphFn(t, t.Name(), func(g *Graph) (inputs, outputs []*Node) {
		inputs = []*Node{
			Const(g, []float32{6.0, 12.0, 3.0, 0.0}),  // Weights
			Const(g, []bool{true, false, true, true}), // Mask
		}
		weights, mask := CheckExtraLabelsForWeightsAndMask(inputs[0].Shape(), inputs)
		// Check that order doesn't matter.
		weights2, mask2 := CheckExtraLabelsForWeightsAndMask(inputs[0].Shape(), []*Node{inputs[1], inputs[0]})
		weights3, mask3 := CheckExtraLabelsForWeightsAndMask(inputs[0].Shape(), []*Node{inputs[0]})
		if mask3 != nil {
			fmt.Printf("%s: mask not passed so it should have returned nil", t.Name())
			t.Fail()
		}
		outputs = []*Node{
			weights, mask,
			weights2, mask2,
			weights3,
		}

		return
	}, []any{
		[]float32{6, 0, 3, 0},
		[]bool{true, false, true, true},
		[]float32{6, 0, 3, 0},
		[]bool{true, false, true, true},
		[]float32{6, 12, 3, 0},
	}, -1)
}

func TestMeanSquaredError(t *testing.T) {
	graphtest.RunTestGraphFn(t, t.Name(),
		func(g *Graph) (inputs, outputs []*Node) {
			labels := Const(g, []float32{1.0, 2.0, 7.0})
			mask := Const(g, []bool{true, true, false})
			weights := Const(g, []float32{5.0, 1.0, 3.2})
			predictions := Const(g, []float32{2.0, 4.0, 0})
			inputs = []*Node{predictions, labels, weights, mask}
			outputs = []*Node{MeanSquaredError([]*Node{labels, mask, weights}, []*Node{predictions})}
			return
		}, []any{
			float32(5*1+1*4.0) / 2,
		}, -1)
}

func TestMeanAbsoluteError(t *testing.T) {
	graphtest.RunTestGraphFn(t, t.Name(),
		func(g *Graph) (inputs, outputs []*Node) {
			labels := Const(g, []float32{1.0, 2.0, 7.0})
			mask := Const(g, []bool{true, true, false})
			weights := Const(g, []float32{5.0, 1.0, 3.2})
			predictions := Const(g, []float32{2.0, 4.0, 0})
			inputs = []*Node{predictions, labels, weights, mask}
			outputs = []*Node{MeanAbsoluteError([]*Node{labels, mask, weights}, []*Node{predictions})}
			return
		}, []any{
			float32(5*1+1*2.0) / 2,
		}, 1e-4)
}

func TestGradientBinaryCrossentropy(t *testing.T) {
	const count = 6
	testGradients[float64](t, "Gradient BinaryCrossentropy",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			logits := Const(g, []float64{5, 1e-6, 0, 0, -1e-6, -5})
			predictions := Sigmoid(logits)
			labels := Const(g, []float64{1, 0, 1, 0, 0, 1})
			output = BinaryCrossentropy([]*Node{labels}, []*Node{predictions})
			return output, []*Node{logits}
		}, [][]float64{{-0.00669 / count, 0.5 / count, -0.5 / count, 0.5 / count, 0.5 / count, -0.9933 / count}})

	// Now we add one extra example at the end, that is masked out.
	// The gradient should remain the same (since the masked out element shouldn't count), and
	// the last element doesn't have any gradient (0).
	testGradients[float64](t, "Gradient BinaryCrossentropy with Mask",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			logits := Const(g, []float64{5, 1e-6, 0, 0, -1e-6, -5, 0})
			mask := Const(g, []bool{true, true, true, true, true, true, false})
			predictions := Sigmoid(logits)
			labels := Const(g, []float64{1, 0, 1, 0, 0, 1, 0})
			output = BinaryCrossentropy([]*Node{labels, mask}, []*Node{predictions})
			return output, []*Node{logits}
		}, [][]float64{{-0.00669 / count, 0.5 / count, -0.5 / count, 0.5 / count, 0.5 / count, -0.9933 / count, 0}})

}

func TestGradientBinaryCrossentropyLogits(t *testing.T) {
	const count = 6
	testGradients[float64](t, "Gradient BinaryCrossentropyLogits",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			logits := Const(g, []float64{5, 1e-6, 0, 0, -1e-6, -5})
			labels := Const(g, []float64{1, 0, 1, 0, 0, 1})
			output = ReduceAllSum(BinaryCrossentropyLogits([]*Node{labels}, []*Node{logits}))
			return output, []*Node{logits}
		}, [][]float64{{-0.00669285 / count, 0.50000025 / count, -0.5 / count, 0.5 / count, 0.5 / count, -0.99330715 / count}})

	// Now we add one extra example at the end, that is masked out.
	// The gradient should remain the same (since the masked out element shouldn't count), and
	// the last element doesn't have any gradient (0).
	testGradients[float64](t, "Gradient BinaryCrossentropyLogits With Mask",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			logits := Const(g, []float64{5, 1e-6, 0, 0, -1e-6, -5, 13})
			mask := Const(g, []bool{true, true, true, true, true, true, false})
			labels := Const(g, []float64{1, 0, 1, 0, 0, 1, 1})
			output = ReduceAllSum(BinaryCrossentropyLogits([]*Node{labels, mask}, []*Node{logits}))
			return output, []*Node{logits}
		}, [][]float64{{-0.00669285 / count, 0.50000025 / count, -0.5 / count, 0.5 / count, 0.5 / count, -0.99330715 / count, 0}})

}

func TestCategoricalCrossEntropy(t *testing.T) {
	testSomeFunc[float32](t, "CategoricalCrossEntropy",
		func(g *Graph) (input, output *Node) {
			labels := Const(g, [][]float32{{0, 1, 0}, {0, 0, 1}})
			predictions := Const(g, [][]float32{{0.05, 0.95, 0}, {0.1, 0.8, 0.1}})
			output = CategoricalCrossEntropy([]*Node{labels}, []*Node{predictions})
			return predictions, output
		}, float32((0.05129+2.3026)/2), true)

	testSomeFunc[float32](t, "CategoricalCrossEntropyWithMask",
		func(g *Graph) (input, output *Node) {
			labels := Const(g, [][]float32{{0, 1, 0}, {0, 0, 1}, {0, 0, 0}})
			mask := Const(g, []bool{true, true, false})
			predictions := Const(g, [][]float32{{0.05, 0.95, 0}, {0.1, 0.8, 0.1}, {0, 0, 0}})
			output = CategoricalCrossEntropy([]*Node{labels, mask}, []*Node{predictions})
			return predictions, output
		}, float32((0.05129+2.3026)/2), true)

	testSomeFunc[float32](t, "CategoricalCrossEntropyWithMaskAndWeights",
		func(g *Graph) (input, output *Node) {
			labels := Const(g, [][]float32{{0, 1, 0}, {0, 0, 1}, {0, 0, 0}})
			mask := Const(g, []bool{true, true, false})
			weights := Const(g, []float32{1.0, 3.0, 0.0})
			predictions := Const(g, [][]float32{{0.05, 0.95, 0}, {0.1, 0.8, 0.1}, {0, 0, 0}})
			output = CategoricalCrossEntropy([]*Node{labels, weights, mask}, []*Node{predictions})
			return predictions, output
		}, (float32(0.05129)*1+float32(2.30259)*3)/2, true)

	testSomeFunc[float32](t, "SparseCategoricalCrossEntropyLogits",
		func(g *Graph) (input, output *Node) {
			labels := Const(g, [][]int32{{1}, {2}, {0}})
			mask := Const(g, []bool{true, true, false})
			predictions := Const(g, [][]float32{{0, 10.0, 0}, {10.0, 0, 0}, {0, 0, 0}})
			output = SparseCategoricalCrossEntropyLogits([]*Node{labels, mask}, []*Node{predictions})
			return predictions, output
		}, float32(0+10)/2, true)
}

func TestHuberLoss(t *testing.T) {
	graphtest.RunTestGraphFn(t, "MakeHuberLoss", func(g *Graph) (inputs, outputs []*Node) {
		inputs = []*Node{
			Const(g, []float32{1.1, 0.9, 3.0, -1.0}), // Predictions
			Const(g, []float32{1, 1, 1, 1}),          // Labels
		}
		lossFn := MakeHuberLoss(1.0)
		predictions := []*Node{inputs[0]}
		labels := []*Node{inputs[1]}
		outputs = []*Node{lossFn(labels, predictions)}
		return
	}, []any{
		float32(0.005+0.005+1.5+1.5) / 4,
	}, 1e-4)

	testGradients[float64](t, "MakeHuberLoss: Gradient",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			predictions := Const(g, []float64{1.1, 0.9, 3.0, -1.0})
			labels := Const(g, []float64{1, 1, 1, 1})
			lossFn := MakeHuberLoss(1.0)
			output = lossFn([]*Node{labels}, []*Node{predictions})
			return output, []*Node{predictions}
		}, [][]float64{{
			0.1 / 4, -0.1 / 4, // L2 region: gradient is the absolute error +/- 0.1
			float64(1) / 4, float64(-1) / 4, // L1 region: gradient is constant +/- 1 (while absolute error is +/- 2).
		}})
}

func TestAdaptivePowerLoss(t *testing.T) {
	const numElements = 5
	graphtest.RunTestGraphFn(t, "MakeAdaptivePowerLoss", func(g *Graph) (inputs, outputs []*Node) {
		predictions := Const(g, []float32{0.0, 0.1, -0.1, 10.0, -1000.0})
		predictions = OnePlus(predictions) // Shifted from 0.
		labels := OnesLike(predictions)
		inputs = []*Node{predictions}
		lossFn := MakeAdaptivePowerLoss(3.0, 1, 10.0, 1.0)
		outputs = []*Node{lossFn([]*Node{labels}, []*Node{predictions})}
		return
	}, []any{
		float32(
			0+ // Zero when predictions==labels
				(0.1*0.1*0.1)+(0.1*0.1*0.1)+ // "Near": use powerNear == 3. Also, checks it is symmetric.
				10*10+ // Half-way, it should be ~10^((powerNear+powerFar)/2), so 10^2
				1001.38275) / numElements, // Far, it should be ~1000^1; Then take the mean (divide by numElements).
	}, 1e-3)

	testGradientsInDelta[float64](t, "MakeAdaptiveLoss: Gradient",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			predictions := Const(g, []float32{0.0, 0.1, -0.1, 10.0, -1000.0})
			predictions = OnePlus(predictions) // Shifted from 0.
			labels := OnesLike(predictions)
			lossFn := MakeAdaptivePowerLoss(3.0, 1, 10.0, 1.0)
			output = lossFn([]*Node{labels}, []*Node{predictions})
			return output, []*Node{predictions}
		}, [][]float64{{
			0.0 / numElements,                                 // Exactly 0, gradient is zero.
			3.0 * 0.01 / numElements, 3 * -0.01 / numElements, // L3 region: d(x^3)/dx = 3x^2 ->
			2.0 * 10 / numElements, // L2 region: gradient
			-1.0 / numElements,     // L1 region: gradient is constant +/- 1 (while absolute error is +/- 2).
		}}, 1e-2)
}
