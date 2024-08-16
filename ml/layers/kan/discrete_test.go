package kan

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"testing"
)

func TestPiecewiseConstantFunction(t *testing.T) {
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
			output := PiecewiseConstantFunction(input, controlPoints, splitPoints)
			inputs = []*Node{input, controlPoints, splitPoints}
			outputs = []*Node{output}
			return
		}, []any{
			// Output: [batchSize, numOutputNodes, numInputNodes], and the selection of the control points
			// should be the one below:
			[][][]float32{{{0, 4, 7, 10, 14}, {15, 19, 22, 25, 29}}},
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
			output := PiecewiseConstantFunction(input, controlPoints, splitPoints)
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
			output := PiecewiseConstantFunction(input, controlPoints, splitPoints)
			inputs = []*Node{input, controlPoints, splitPoints}
			outputs = []*Node{output}
			return
		}, []any{
			// Output: [batchSize, numOutputNodes, numInputNodes], and the selection of the control points
			// should be the one below:
			[][][]float32{{{0, 4, 7, 10, 14}, {15, 19, 22, 25, 29}}},
		}, 1e-4)

}
