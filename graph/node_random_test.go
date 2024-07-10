package graph_test

import (
	"fmt"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/x448/float16"
	"testing"
)

func testRandomUniform[T interface {
	float32 | float64 | float16.Float16 | complex64 | complex128
}](t *testing.T, manager *Manager) {
	dtype := shapes.FromGoType[T]()
	graphtest.RunTestGraphFn(t, fmt.Sprintf("TestRandomUniform(%s)", dtype),
		func(g *Graph) (inputs []*Node, outputs []*Node) {
			state := Const(g, RngStateFromSeed(42))
			shape := shapes.Make(dtype, 100, 5000) // 500k / 1 million numbers (for complex numbers).
			_, r := RandomUniform(state, shape)
			shapeSize := float64(shape.Size())
			if dtype.IsComplex() {
				// Split and concatenate real and imaginary part: they are sampled independently.
				shapeSize *= 2
				r = Concatenate([]*Node{Real(r), Imag(r)}, -1)
			}
			counts := make([]*Node, 10)
			for ii := range counts {
				from := 0.1 * float64(ii)
				to := from + 0.1
				includeSet := And(
					GreaterOrEqual(r, Scalar(g, r.DType(), from)),
					LessThan(r, Scalar(g, r.DType(), to)))
				count := ConvertType(includeSet, dtypes.Float32)
				count = ReduceAllSum(count)
				count = DivScalar(count, shapeSize)
				counts[ii] = count
			}
			outputs = []*Node{Concatenate(counts, 0)}
			return
		}, []any{
			[]float32{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
		}, 0.1)
}

func TestRandomUniform(t *testing.T) {
	manager := graphtest.BuildTestManager()
	testRandomUniform[float32](t, manager)
	testRandomUniform[float64](t, manager)
	testRandomUniform[float16.Float16](t, manager)
	testRandomUniform[complex64](t, manager)
	testRandomUniform[complex128](t, manager)
}

func testRandomNormal[T interface {
	float32 | float64 | float16.Float16
}](t *testing.T, manager *Manager) {
	dtype := shapes.FromGoType[T]()
	graphtest.RunTestGraphFn(t, fmt.Sprintf("TestRandomNormal(%s)", dtype),
		func(g *Graph) (inputs []*Node, outputs []*Node) {
			state := Const(g, RngStateFromSeed(42))
			shape := shapes.Make(dtype, 100, 10000) // 1 million numbers.
			_, r := RandomNormal(state, shape)
			if dtype == dtypes.Float16 {
				// 1M examples will overflow float16 resolution, so we convert to F32 to calculate the mean.
				r = ConvertType(r, dtypes.Float32)
			}
			mean := ReduceAllMean(r)
			r2 := Square(r)
			mean2 := ReduceAllMean(r2)
			variance := Sub(mean2, Square(mean))
			stddev := Sqrt(variance)

			// Convert to float64 for to get same type result for all.
			mean, stddev = ConvertType(mean, dtypes.Float64), ConvertType(stddev, dtypes.Float64)
			outputs = []*Node{mean, stddev}
			return
		},
		[]any{0.0, 1.0}, // mean / stddev
		0.1)
}

func TestRandomNormal(t *testing.T) {
	manager := graphtest.BuildTestManager()
	testRandomNormal[float32](t, manager)
	testRandomNormal[float64](t, manager)
	testRandomNormal[float16.Float16](t, manager)
}
