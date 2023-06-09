package graph_test

import (
	"fmt"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/types/shapes"
	"testing"
)

func testRandomUniform[T interface{ float32 | float64 }](t *testing.T, manager *Manager) {
	dtype := shapes.DTypeGeneric[T]()
	graphtest.RunTestGraphFn(t, fmt.Sprintf("TestRandomUniform(%s)", dtype),
		func(g *Graph) (inputs []*Node, outputs []*Node) {
			state := Const(g, RngStateFromSeed(42))
			shape := shapes.Make(dtype, 100, 10000) // 1 million numbers.
			_, r := RandomUniform(state, shape)
			counts := make([]*Node, 10)
			for ii := range counts {
				from := 0.1 * float64(ii)
				to := from + 0.1
				includeSet := And(
					GreaterOrEqual(r, Scalar(g, dtype, from)),
					LessThan(r, Scalar(g, dtype, to)))
				count := ConvertType(includeSet, shapes.F32)
				count = ReduceAllSum(count)
				count = DivScalar(count, float64(shape.Size()))
				counts[ii] = count
			}
			outputs = []*Node{Concatenate(counts, 0)}
			return
		}, []any{
			[]float32{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
		}, 0.001)
}

func TestRandomUniform(t *testing.T) {
	manager := graphtest.BuildTestManager()
	testRandomUniform[float32](t, manager)
	testRandomUniform[float64](t, manager)
}

func testRandomNormal[T interface{ float32 | float64 }](t *testing.T, manager *Manager) {
	dtype := shapes.DTypeGeneric[T]()
	graphtest.RunTestGraphFn(t, fmt.Sprintf("TestRandomNormal(%s)", dtype),
		func(g *Graph) (inputs []*Node, outputs []*Node) {
			state := Const(g, RngStateFromSeed(42))
			shape := shapes.Make(dtype, 100, 10000) // 1 million numbers.
			_, r := RandomNormal(state, shape)
			mean := ReduceAllMean(r)
			r2 := Square(r)
			mean2 := ReduceAllMean(r2)
			variance := Sub(mean2, Square(mean))
			stddev := Sqrt(variance)
			outputs = []*Node{mean, stddev}
			return
		}, []any{
			T(0),
			T(1),
		}, 0.1)
}

func TestRandomNormal(t *testing.T) {
	manager := graphtest.BuildTestManager()
	testRandomNormal[float32](t, manager)
	testRandomNormal[float64](t, manager)
}
