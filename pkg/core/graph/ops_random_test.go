package graph_test

import (
	"fmt"
	"math"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/x448/float16"

	"testing"
)

func testRandomUniform[T interface {
	float32 | float64 | float16.Float16 | bfloat16.BFloat16 | complex64 | complex128
}](t *testing.T) {
	dtype := dtypes.FromGenericsType[T]()
	graphtest.RunTestGraphFn(t, fmt.Sprintf("TestRandomUniform(%s)", dtype),
		func(g *Graph) (inputs []*Node, outputs []*Node) {
			state := Const(g, must1(RNGStateFromSeed(42)))
			shape := shapes.Make(dtype, 100, 5000) // 500k / 1 million numbers (for complex numbers).
			var r, sample *Node
			state, r = RandomUniform(state, shape)
			state, sample = RandomUniform(state, shapes.Make(dtype, 10))
			inputs = append(inputs, sample)
			sample = ConvertDType(sample, dtypes.Float32)
			inputs = append(inputs, sample)
			inputs = append(inputs, Scalar(g, r.DType(), 0.1))
			shapeSize := float64(shape.Size())
			if dtype.IsComplex() {
				// Split and concatenate real and imaginary part: they are sampled independently.
				shapeSize *= 2
				r = Concatenate([]*Node{Real(r), Imag(r)}, -1)
			}
			counts := make([]*Node, 12)
			for ii := range counts {
				from := 0.1*float64(ii) - 0.1
				if ii == 0 {
					from = math.Inf(-1)
				}
				to := from + 0.1
				if ii == 11 {
					to = math.Inf(1)
				}
				includeSet := LogicalAnd(
					GreaterOrEqual(r, Scalar(g, r.DType(), from)),
					LessThan(r, Scalar(g, r.DType(), to)))
				count := ConvertDType(includeSet, dtypes.Float32)
				count = ReduceAllSum(count)
				count = DivScalar(count, shapeSize)
				counts[ii] = count
			}
			outputs = []*Node{Concatenate(counts, 0)}
			return
		}, []any{
			[]float32{0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0},
		}, 0.1)
}

func TestRandomUniform(t *testing.T) {
	testRandomUniform[float32](t)
	testRandomUniform[float64](t)
	testRandomUniform[float16.Float16](t)
	testRandomUniform[bfloat16.BFloat16](t)
	testRandomUniform[complex64](t)
	testRandomUniform[complex128](t)
}

func testRandomNormal[T interface {
	float32 | float64 | float16.Float16
}](t *testing.T) {
	dtype := dtypes.FromGenericsType[T]()
	graphtest.RunTestGraphFn(t, fmt.Sprintf("TestRandomNormal(%s)", dtype),
		func(g *Graph) (inputs []*Node, outputs []*Node) {
			state := Const(g, must1(RNGStateFromSeed(42)))
			shape := shapes.Make(dtype, 100, 10000) // 1 million numbers.
			_, r := RandomNormal(state, shape)
			if dtype == dtypes.Float16 {
				// 1M examples will overflow float16 resolution, so we convert to F32 to calculate the mean.
				r = ConvertDType(r, dtypes.Float32)
			}
			mean := ReduceAllMean(r)
			r2 := Square(r)
			mean2 := ReduceAllMean(r2)
			variance := Sub(mean2, Square(mean))
			stddev := Sqrt(variance)

			// Convert to float64 for to get same type result for all.
			mean, stddev = ConvertDType(mean, dtypes.Float64), ConvertDType(stddev, dtypes.Float64)
			outputs = []*Node{mean, stddev}
			return
		},
		[]any{0.0, 1.0}, // mean / stddev
		0.1)
}

func TestRandomNormal(t *testing.T) {
	testRandomNormal[float32](t)
	testRandomNormal[float64](t)
	testRandomNormal[float16.Float16](t)
}

func testRandomIntN[T interface {
	uint8 | uint16 | uint32 | uint64 | int8 | int16 | int32 | int64
}](t *testing.T, useStatic bool) {
	dtype := dtypes.FromGenericsType[T]()
	graphtest.RunTestGraphFn(t, fmt.Sprintf("TestRandomIntN(%s, useStatic=%v)", dtype, useStatic),
		func(g *Graph) (inputs []*Node, outputs []*Node) {
			state := Const(g, must1(RNGStateFromSeed(42)))
			shape := shapes.Make(dtype, 100, 10000) // 1 million numbers.
			var r *Node
			if useStatic {
				state, r = RandomIntN(state, T(13), shape)
			} else {
				// Notice that the dtype of N shouldn't really matter, as long as it is an integer.
				state, r = RandomIntN(state, Const(g, int32(13)), shape)
			}
			var maxRatio, minRatio, totalCount *Node
			for ii := range 13 {
				includeSet := LogicalAnd(
					GreaterOrEqual(r, Scalar(g, r.DType(), ii)),
					LessThan(r, Scalar(g, r.DType(), ii+1)))
				count := ConvertDType(includeSet, dtypes.Float32)
				count = ReduceAllSum(count)
				count = DivScalar(count, shape.Size())
				if maxRatio == nil {
					maxRatio = count
					minRatio = count
					totalCount = count
				} else {
					maxRatio = Max(maxRatio, count)
					minRatio = Min(minRatio, count)
					totalCount = Add(totalCount, count)
				}
			}
			numInvalid := GreaterOrEqual(r, Scalar(g, r.DType(), 13))
			numInvalid = ReduceAllSum(ConvertDType(numInvalid, dtypes.Float32))
			outputs = []*Node{totalCount, minRatio, maxRatio, numInvalid}
			return
		}, []any{
			float32(1),
			float32(1) / 13,
			float32(1) / 13,
			float32(0.0),
		}, 0.02)
}

func TestRandomIntN(t *testing.T) {
	for _, useStatic := range []bool{false, true} {
		testRandomIntN[uint8](t, useStatic)
		testRandomIntN[uint16](t, useStatic)
		testRandomIntN[uint32](t, useStatic)
		testRandomIntN[uint64](t, useStatic)
		testRandomIntN[int8](t, useStatic)
		testRandomIntN[int16](t, useStatic)
		testRandomIntN[int32](t, useStatic)
		testRandomIntN[int64](t, useStatic)
	}
}

// TestMultiOutputs covers issue #197, about the execution nodes with multi-outputs (random number generator) on
// the SimpleGo backend.
//
// Kept here to eventually test new backends.
func TestMultiOutputs(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	_, err := MustNewExec(backend, func(x *Node) *Node {
		g := x.Graph()
		rngState := Const(g, must1(RNGStateFromSeed(42)))
		rngState, ws := RandomNormal(rngState, x.Shape())
		fmt.Printf("Graph:\n%s\n", g)
		return ws // Add(x, ws)
	}).Exec(0.0)
	if err != nil {
		fmt.Printf("Error: %+v\n", err)
		t.Fail()
	}
}
