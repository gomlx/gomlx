// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph_test

import (
	"fmt"
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/graphtest"
	"github.com/gomlx/gomlx/support/testutil"

	"testing"
)

func testRandomUniform[T interface {
	float32 | float64 | float16.Float16 | bfloat16.BFloat16 | complex64 | complex128
}](t *testing.T, backend compute.Backend) {
	dtype := dtypes.FromGenericsType[T]()
	testName := fmt.Sprintf("TestRandomUniform(%s)", dtype)
	if !backend.Capabilities().DTypes[dtype] {
		t.Run(testName, func(t *testing.T) {
			t.Skipf("Backend %q does not support %s", backend.Name(), dtype)
		})
		return
	}
	graphtest.RunTestGraphFnWithBackend(t, testName, backend,
		func(g *Graph) (inputs []*Node, outputs []*Node) {
			state := RNGStateFromSeedForGraph(g, 42)
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
			if dtype == dtypes.Float16 || dtype == dtypes.BFloat16 {
				r = ConvertDType(r, dtypes.Float32)
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
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		testRandomUniform[float32](t, backend)
		testRandomUniform[float64](t, backend)
		testRandomUniform[float16.Float16](t, backend)
		testRandomUniform[bfloat16.BFloat16](t, backend)
		testRandomUniform[complex64](t, backend)
		testRandomUniform[complex128](t, backend)
	})
}

func testRandomNormal[T interface {
	float32 | float64 | float16.Float16 | bfloat16.BFloat16
}](t *testing.T, backend compute.Backend) {
	dtype := dtypes.FromGenericsType[T]()
	testName := fmt.Sprintf("TestRandomNormal(%s)", dtype)
	if !backend.Capabilities().DTypes[dtype] {
		t.Run(testName, func(t *testing.T) {
			t.Skipf("Backend %q does not support %s", backend.Name(), dtype)
		})
		return
	}
	graphtest.RunTestGraphFnWithBackend(t, testName, backend,
		func(g *Graph) (inputs []*Node, outputs []*Node) {
			state := Const(g, must1(RNGStateFromSeed(42)))
			shape := shapes.Make(dtype, 100, 10000) // 1 million numbers.
			_, r := RandomNormal(state, shape)
			if dtype == dtypes.Float16 || dtype == dtypes.BFloat16 {
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
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		testRandomNormal[float32](t, backend)
		testRandomNormal[float64](t, backend)
		testRandomNormal[float16.Float16](t, backend)
		testRandomNormal[bfloat16.BFloat16](t, backend)
	})
}

func testRandomIntN[T interface {
	uint8 | uint16 | uint32 | uint64 | int8 | int16 | int32 | int64
}](t *testing.T, backend compute.Backend, useStatic bool) {
	dtype := dtypes.FromGenericsType[T]()
	testName := fmt.Sprintf("TestRandomIntN(%s, useStatic=%v)", dtype, useStatic)
	if !backend.Capabilities().DTypes[dtype] {
		t.Run(testName, func(t *testing.T) {
			t.Skipf("Backend %q does not support %s", backend.Name(), dtype)
		})
		return
	}
	graphtest.RunTestGraphFnWithBackend(t, testName, backend,
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
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		for _, useStatic := range []bool{false, true} {
			testRandomIntN[uint8](t, backend, useStatic)
			testRandomIntN[uint16](t, backend, useStatic)
			testRandomIntN[uint32](t, backend, useStatic)
			testRandomIntN[uint64](t, backend, useStatic)
			testRandomIntN[int8](t, backend, useStatic)
			testRandomIntN[int16](t, backend, useStatic)
			testRandomIntN[int32](t, backend, useStatic)
			testRandomIntN[int64](t, backend, useStatic)
		}
	})
}

// TestMultiOutputs covers issue #197, about the execution nodes with multi-outputs (random number generator) on
// the Go backend.
//
// Kept here to eventually test new compute.
func TestMultiOutputs(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		_, err := MustNewExec(backend, func(x *Node) *Node {
			g := x.Graph()
			rngState := Const(g, must1(RNGStateFromSeed(42)))
			rngState, ws := RandomNormal(rngState, x.Shape())
			fmt.Printf("Graph:\n%s\n", g)
			return ws // Add(x, ws)
		}).Call(0.0)
		if err != nil {
			fmt.Printf("Error: %+v\n", err)
			t.Fail()
		}
	})
}
