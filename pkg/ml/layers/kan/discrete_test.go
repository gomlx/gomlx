package kan

import (
	"fmt"
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/go-xla/pkg/types/dtypes"
)

func TestPiecewiseConstantFunctions(t *testing.T) {
	for distributionIdx, distribution := range []string{"standard", "soft-triangular", "soft-normal"} {
		fmt.Printf("\nPiecewiseConstantFunction: %s\n", distribution)
		pcfFn := func(input, controlPoints, splitPoints *Node) *Node {
			if distributionIdx == 0 {
				// Standard
				return PiecewiseConstantFunction(input, controlPoints, splitPoints)
			}

			// Return version with perturbation, but with softness == 0, which is equivalent to the constant function.
			p := ([...]PerturbationType{PerturbationTriangular, PerturbationNormal})[distributionIdx-1]
			return PiecewiseConstantFunctionWithInputPerturbation(
				input, controlPoints, splitPoints, p, ScalarZero(input.Graph(), input.DType()))
		}

		graphtest.RunTestGraphFn(t, "PiecewiseConstantFunction",
			func(g *Graph) (inputs, outputs []*Node) {
				dtype := dtypes.Float32
				batchSize := 1
				numInputNodes := 5
				numOutputNodes := 2
				numControlPoints := 3
				controlPoints := IotaFull(g, shapes.Make(dtype, numOutputNodes, numInputNodes, numControlPoints))
				splitPoints := Iota(g, shapes.Make(dtype, numOutputNodes, numInputNodes, numControlPoints-1), -1)
				input := Iota(g, shapes.Make(dtype, batchSize, numInputNodes), 1)
				input = AddScalar(MulScalar(DivScalar(input, float64(numInputNodes-1)), 1.2), -0.1)
				output := pcfFn(input, controlPoints, splitPoints)
				inputs = []*Node{input, controlPoints, splitPoints}
				outputs = []*Node{output}
				return
			}, []any{
				// Output: [batchSize, numOutputNodes, numInputNodes], and the selection of the control points
				// should be the one below:
				[][][]float32{{{0, 4, 7, 10, 14}, {15, 19, 22, 25, 29}}},
			}, 1e-4)

		graphtest.RunTestGraphFn(t, "PiecewiseConstantFunction Scalar Input",
			func(g *Graph) (inputs, outputs []*Node) {
				dtype := dtypes.Float32
				numControlPoints := 3
				controlPoints := IotaFull(g, shapes.Make(dtype, numControlPoints))
				splitPoints := Iota(g, shapes.Make(dtype, numControlPoints-1), -1)
				input := Scalar(g, dtype, 0.5)
				output := pcfFn(input, controlPoints, splitPoints)
				inputs = []*Node{input, controlPoints, splitPoints}
				outputs = []*Node{output}
				return
			}, []any{
				// Output: the value of the central control point (1).
				float32(1),
			}, 1e-4)

		graphtest.RunTestGraphFn(t, "PiecewiseConstantFunction with splits broadcast of inputs",
			func(g *Graph) (inputs, outputs []*Node) {
				dtype := dtypes.Float32
				batchSize := 1
				numInputNodes := 5
				numOutputNodes := 2
				numControlPoints := 3
				controlPoints := IotaFull(g, shapes.Make(dtype, numOutputNodes, numInputNodes, numControlPoints))
				splitPoints := Iota(g, shapes.Make(dtype, numOutputNodes, 1, numControlPoints-1), -1)
				input := Iota(g, shapes.Make(dtype, batchSize, numInputNodes), 1)
				input = AddScalar(MulScalar(DivScalar(input, float64(numInputNodes-1)), 1.2), -0.1)
				output := pcfFn(input, controlPoints, splitPoints)
				inputs = []*Node{input, controlPoints, splitPoints}
				outputs = []*Node{output}
				return
			}, []any{
				// Output: [batchSize, numOutputNodes, numInputNodes], and the selection of the control points
				// should be the one below:
				[][][]float32{{{0, 4, 7, 10, 14}, {15, 19, 22, 25, 29}}},
			}, 1e-4)

		graphtest.RunTestGraphFn(t, "PiecewiseConstantFunction with split broadcast of outputs",
			func(g *Graph) (inputs, outputs []*Node) {
				dtype := dtypes.Float32
				batchSize := 1
				numInputNodes := 5
				numOutputNodes := 2
				numControlPoints := 3
				controlPoints := IotaFull(g, shapes.Make(dtype, numOutputNodes, numInputNodes, numControlPoints))
				splitPoints := Iota(g, shapes.Make(dtype, 1, numInputNodes, numControlPoints-1), -1)
				input := Iota(g, shapes.Make(dtype, batchSize, numInputNodes), 1)
				input = AddScalar(MulScalar(DivScalar(input, float64(numInputNodes-1)), 1.2), -0.1)
				output := pcfFn(input, controlPoints, splitPoints)
				inputs = []*Node{input, controlPoints, splitPoints}
				outputs = []*Node{output}
				return
			}, []any{
				// Output: [batchSize, numOutputNodes, numInputNodes], and the selection of the control points
				// should be the one below:
				[][][]float32{{{0, 4, 7, 10, 14}, {15, 19, 22, 25, 29}}},
			}, 1e-4)

		graphtest.RunTestGraphFn(t, "PiecewiseConstantFunction with input grouping",
			func(g *Graph) (inputs, outputs []*Node) {
				dtype := dtypes.Float32
				batchSize := 1
				numInputNodes := 6
				numInputGroups := 2
				inputGroupSize := numInputNodes / numInputGroups
				numOutputNodes := 2
				numControlPoints := 3
				controlPoints := IotaFull(g, shapes.Make(dtype, numOutputNodes, numInputGroups, numControlPoints))
				splitPoints := Iota(g, shapes.Make(dtype, numOutputNodes, numInputGroups, numControlPoints-1), -1)
				input := Iota(g, shapes.Make(dtype, batchSize, numInputGroups, inputGroupSize), -1)
				input = AddScalar(MulScalar(DivScalar(input, float64(inputGroupSize-1)), 1.2), -0.1)
				output := pcfFn(input, controlPoints, splitPoints)
				inputs = []*Node{input, controlPoints, splitPoints}
				outputs = []*Node{output}
				return
			}, []any{
				// Output: [batchSize, numOutputNodes, numInputNodes], and the selection of the control points
				// should be the one below:
				[][][]float32{{{0, 1, 2, 3, 4, 5}, {6, 7, 8, 9, 10, 11}}},
			}, 1e-4)
	}
}
