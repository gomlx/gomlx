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
		[]float32{0.005, 0.005, 1.5, 1.5},
	}, 1e-4)

	testGradients[float64](t, "MakeHuberLoss: Gradient",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			predictions := Const(g, []float64{1.1, 0.9, 3.0, -1.0})
			labels := Const(g, []float64{1, 1, 1, 1})
			lossFn := MakeHuberLoss(1.0)
			output = ReduceAllSum(lossFn([]*Node{labels}, []*Node{predictions}))
			return output, []*Node{predictions}
		}, [][]float64{{
			0.1, -0.1, // L2 region: gradient is the absolute error +/- 0.1
			1, -1, // L1 region: gradient is constant +/- 1 (while absolute error is +/- 2).
		}})
}

func TestAdaptivePowerLoss(t *testing.T) {
	graphtest.RunTestGraphFn(t, "MakeAdaptivePowerLoss", func(g *Graph) (inputs, outputs []*Node) {
		predictions := Const(g, []float32{0.0, 0.1, -0.1, 10.0, -1000.0})
		predictions = OnePlus(predictions) // Shifted from 0.
		labels := OnesLike(predictions)
		inputs = []*Node{predictions}
		lossFn := MakeAdaptivePowerLoss(3.0, 1, 10.0, 1.0)
		outputs = []*Node{lossFn([]*Node{labels}, []*Node{predictions})}
		return
	}, []any{
		[]float32{
			0,                                // Zero when predictions==labels
			0.1 * 0.1 * 0.1, 0.1 * 0.1 * 0.1, // "Near": use powerNear == 3. Also, checks it is symmetric.
			10 * 10,    // Half-way, it should be ~10^((powerNear+powerFar)/2), so 10^2
			1001.38275, // Far, it should be ~1000^1
		},
	}, 1e-3)

	testGradientsInDelta[float64](t, "MakeAdaptiveLoss: Gradient",
		func(g *Graph) (output *Node, nodesForGrad []*Node) {
			predictions := Const(g, []float32{0.0, 0.1, -0.1, 10.0, -1000.0})
			predictions = OnePlus(predictions) // Shifted from 0.
			labels := OnesLike(predictions)
			lossFn := MakeAdaptivePowerLoss(3.0, 1, 10.0, 1.0)
			output = ReduceAllSum(lossFn([]*Node{labels}, []*Node{predictions}))
			return output, []*Node{predictions}
		}, [][]float64{{
			0.0,                 // Exactly 0, gradient is zero.
			3 * 0.01, 3 * -0.01, // L3 region: d(x^3)/dx = 3x^2 ->
			2 * 10, // L2 region: gradient
			-1,     // L1 region: gradient is constant +/- 1 (while absolute error is +/- 2).
		}}, 1e-2)
}

func TestTripletLoss(t *testing.T) {
	graphtest.RunTestGraphFn(t, "TripletLoss",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{
				Const(g, [][]float32{{0}, {1}, {0}, {1}}),
				Const(g, [][]float32{
					{0.23, 0.75},
					{0.89, 0.41},
					{0.37, 0.62},
					{0.78, 0.24},
				}), // embeddings
				Const(g, [][]float32{{1}, {0}, {0}, {0}, {3}, {2}, {3}, {2}, {1}, {2}}), // labels
				Const(g, [][]float32{
					{0.08208963, 0.11788353, 0.46360782, 0.3360519, 0.2702437, 0.6951965},
					{0.598121, 0.14609586, 0.07872304, 0.949776, 0.41479972, 0.36961815},
					{0.11646613, 0.8878409, 0.4034519, 0.9632401, 0.6313564, 0.0198459},
					{0.03582959, 0.3428808, 0.843301, 0.6335877, 0.8623248, 0.16186231},
					{0.09054314, 0.746887, 0.56099737, 0.7181275, 0.60642695, 0.02207313},
					{0.2735666, 0.08748698, 0.13752021, 0.4570993, 0.8813543, 0.98528206},
					{0.5412437, 0.2382705, 0.6263132, 0.29713312, 0.9241606, 0.734765},
					{0.22289598, 0.84535605, 0.4398808, 0.5816502, 0.31203038, 0.5436755},
					{0.5512105, 0.6922551, 0.11149547, 0.6343566, 0.20425326, 0.3884894},
					{0.51529086, 0.35541356, 0.77092594, 0.3715265, 0.40550032, 0.7369012},
				}), // embeddings
			}
			outputs = []*Node{
				TripletLoss([]*Node{inputs[0]}, []*Node{inputs[1]}, TripletMiningStrategyAll, 1.0, PairwiseDistanceMetricL2),
				TripletLoss([]*Node{inputs[0]}, []*Node{inputs[1]}, TripletMiningStrategySemiHard, 1.0, PairwiseDistanceMetricL2),
				TripletLoss([]*Node{inputs[0]}, []*Node{inputs[1]}, TripletMiningStrategyHard, 1.0, PairwiseDistanceMetricL2),
				TripletLoss([]*Node{inputs[0]}, []*Node{inputs[1]}, TripletMiningStrategySemiHard, 1.0, PairwiseDistanceMetricCosine),

				TripletLoss([]*Node{inputs[2]}, []*Node{inputs[3]}, TripletMiningStrategyAll, 1.0, PairwiseDistanceMetricL2),
				TripletLoss([]*Node{inputs[2]}, []*Node{inputs[3]}, TripletMiningStrategySemiHard, 1.0, PairwiseDistanceMetricL2),
				TripletLoss([]*Node{inputs[2]}, []*Node{inputs[3]}, TripletMiningStrategyHard, 1.0, PairwiseDistanceMetricL2),
				TripletLoss([]*Node{inputs[2]}, []*Node{inputs[3]}, TripletMiningStrategyHard, -1.0, PairwiseDistanceMetricL2),
				TripletLoss([]*Node{inputs[2]}, []*Node{inputs[3]}, TripletMiningStrategyHard, 1.0, PairwiseDistanceMetricCosine),
				// TripletLoss([]*Node{inputs[2]}, []*Node{inputs[3]}, TripletMiningStrategySemiHard, 1.0, PairwiseDistanceMetricCosine),
			}
			return
		}, []any{
			float32(0.54368829), // 1 - all 	margin	+1	L2
			float32(0.5914507),  // 1 - semi-hard 	margin	+1	L2
			float32(0.5914507),  // 1 - hard 		margin	+1	L2
			float32(0.783348),   // 1 - hard 		margin	+1	Cosine

			float32(1.0418946),  // 2 - all			margin	+1	L2
			float32(0.9323689),  // 2 - semi-hard	margin	+1	L2
			float32(1.5172637),  // 2 - hard		margin 	+1	L2
			float32(0.98718655), // 2 - hard		margin	-1	L2
			float32(1.2390234),  // 2 - hard		margin 	+1	Cosine
			// float32(0.97456443), // 2 - semi-hard	margin 	+1	Cosine // 0.98323786

			// distances: [][]float32{
			// {0,			0.34174055, 0.4378963,	0.26221365, 0.3858019, 	0.1401844,	0.1377036, 	0.20757705, 0.3607911, 	0.07693392},
			// {0.34174055, 0, 			0.25997412,	0.35250682, 0.31524062, 0.2544706, 	0.286155,	0.28480804, 0.14722657, 0.28982067},
			// {0.4378963, 	0.25997412, 0,			0.15627801, 0.017833054,0.4559992, 	0.37475657, 0.12468529, 0.17564374, 0.35807443},
			// {0.26221365, 0.35250682, 0.15627801, 0, 			0.08797246,	0.3274415, 	0.17841232, 0.24157673, 0.40529507, 0.21925491},
			// {0.3858019, 	0.31524062, 0.017833054,0.08797246, 0, 			0.44810057,	0.3093226, 	0.121177316,0.22279435, 0.2916208},
			// {0.1377036, 	0.286155, 	0.37475657, 0.17841232, 0.3093226, 	0.09821546, 0, 			0.24765933, 0.32003987, 0.07203978},
			// {0.20757705, 0.28480804, 0.12468529, 0.24157673, 0.121177316,0.31774896, 0.24765933, 0, 			0.085817754,0.14851892},
			// {0.3607911, 	0.14722657, 0.17564374, 0.40529507, 0.22279435, 0.361889, 	0.32003987, 0.085817754,0, 			0.2350465},
			// {0.07693392, 0.28982067, 0.35807443, 0.21925491, 0.2916208, 	0.21125132, 0.07203978, 0.14851892, 0.2350465, 	0}}

			// tensorflow semi-hard negatives
			//[[0.        , 0.38580203, 0.43789637, 0.3417406 , 0.43789637, 0.20757711, 0.1401844 , 0.2622137 , 0.        , 0.13770366],
			// [0.3417406 , 0.        , 0.        , 0.        , 0.3417406 , 0.28480804, 0.28982067, 0.28615505, 0.25447053, 0.31524068],
			// [0.4559992 , 0.        , 0.        , 0.        , 0.12468535, 0.4559992 , 0.43789637, 0.17564374, 0.35807443, 0.3747567 ],
			// [0.3274415 , 0.        , 0.        , 0.        , 0.1784125 , 0.40529507, 0.21925491, 0.2622137 , 0.40529507, 0.24157673],
			// [0.44810057, 0.38580203, 0.08797252, 0.12117732, 0.        , 0.44810057, 0.        , 0.22279435, 0.29162085, 0.31524068],
			// [0.25447053, 0.3274415 , 0.4559992 , 0.361889  , 0.4559992 , 0.        , 0.1401844 , 0.        , 0.44810057, 0.        ],
			// [0.1784125 , 0.32004   , 0.3747567 , 0.24765939, 0.        , 0.13770366, 0.        , 0.28615505, 0.3747567 , 0.09821558],
			// [0.24157673, 0.28480804, 0.20757711, 0.24765939, 0.12468535, 0.        , 0.28480804, 0.        , 0.12117732, 0.        ],
			// [0.        , 0.17564374, 0.22279435, 0.40529507, 0.2350465 , 0.40529507, 0.361889  , 0.14722657, 0.        , 0.32004   ],
			// [0.21925491, 0.29162085, 0.35807443, 0.2350465 , 0.35807443, 0.        , 0.07693398, 0.        , 0.28982067, 0.        ]],

			// positive mask
			// [[0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
			//  [0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
			//  [0., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
			//  [0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
			//  [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
			//  [0., 0., 0., 0., 0., 0., 0., 1., 0., 1.],
			//  [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
			//  [0., 0., 0., 0., 0., 1., 0., 0., 0., 1.],
			//  [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			//  [0., 0., 0., 0., 0., 1., 0., 1., 0., 0.]]
		}, -1)
}
