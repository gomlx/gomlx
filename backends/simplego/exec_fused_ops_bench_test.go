// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// benchMust panics on error, used in benchmark setup.
func benchMust[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

// benchExec holds a compiled executable and its input buffers for benchmarking.
type benchExec struct {
	exec   backends.Executable
	inputs []backends.Buffer
}

func (be *benchExec) run(b *testing.B) {
	b.Helper()
	for i := 0; i < b.N; i++ {
		outputs, err := be.exec.Execute(be.inputs, nil, 0)
		if err != nil {
			b.Fatal(err)
		}
		for _, buf := range outputs {
			buf.(*Buffer).flat = nil // release data
		}
	}
}

// buildBenchExec builds, compiles, and prepares inputs for a benchmark.
func buildBenchExec(inputShapes []shapes.Shape, inputDatas []any,
	buildFn func(f backends.Function, params []backends.Value) (backends.Value, error),
) *benchExec {
	exec, inputs, err := buildGraph(inputShapes, inputDatas, buildFn)
	if err != nil {
		panic(err)
	}
	return &benchExec{exec: exec, inputs: inputs}
}

// reduceAndKeep performs ReduceMax or ReduceSum and reshapes back to keep dims.
func reduceAndKeep(f backends.Function, x backends.Value, reduceFn func(backends.Value, ...int) (backends.Value, error), shape shapes.Shape, axis int) backends.Value {
	reduced := benchMust(reduceFn(x, axis))
	// Reshape to keep dimension: insert a size-1 at the axis position.
	keepDims := make([]int, shape.Rank())
	copy(keepDims, shape.Dimensions)
	keepDims[axis] = 1
	reshaped := benchMust(f.Reshape(reduced, keepDims...))
	// Broadcast back to original shape.
	broadcastAxes := make([]int, shape.Rank())
	for i := range broadcastAxes {
		broadcastAxes[i] = i
	}
	return benchMust(f.BroadcastInDim(reshaped, shape, broadcastAxes))
}

func randomFloat32(n int) []float32 {
	data := make([]float32, n)
	for i := range data {
		data[i] = rand.Float32()*2 - 1
	}
	return data
}

// --- Softmax Benchmarks ---

func BenchmarkSoftmax(b *testing.B) {
	sizes := []struct {
		name string
		dims []int
		axis int
	}{
		{"8x64_axis1", []int{8, 64}, 1},
		{"32x128_axis1", []int{32, 128}, 1},
		{"64x512_axis1", []int{64, 512}, 1},
		{"8x16x64_axis2", []int{8, 16, 64}, 2},
		{"4x8x32x128_axis3", []int{4, 8, 32, 128}, 3},
	}

	for _, sz := range sizes {
		shape := shapes.Make(dtypes.Float32, sz.dims...)
		data := randomFloat32(shape.Size())
		axis := sz.axis

		fused := buildBenchExec([]shapes.Shape{shape}, []any{data},
			func(f backends.Function, params []backends.Value) (backends.Value, error) {
				return f.FusedSoftmax(params[0], axis)
			})

		decomposed := buildBenchExec([]shapes.Shape{shape}, []any{data},
			func(f backends.Function, params []backends.Value) (backends.Value, error) {
				x := params[0]
				maxVal := reduceAndKeep(f, x, f.ReduceMax, shape, axis)
				shifted := benchMust(f.Sub(x, maxVal))
				exps := benchMust(f.Exp(shifted))
				sumExps := reduceAndKeep(f, exps, f.ReduceSum, shape, axis)
				return f.Div(exps, sumExps)
			})

		b.Run(fmt.Sprintf("Fused/%s", sz.name), func(b *testing.B) { fused.run(b) })
		b.Run(fmt.Sprintf("Decomposed/%s", sz.name), func(b *testing.B) { decomposed.run(b) })
	}
}

// --- GELU Benchmarks ---

func BenchmarkGelu(b *testing.B) {
	sizes := []struct {
		name string
		dims []int
	}{
		{"512", []int{512}},
		{"4096", []int{4096}},
		{"32x1024", []int{32, 1024}},
		{"64x4096", []int{64, 4096}},
	}

	for _, sz := range sizes {
		shape := shapes.Make(dtypes.Float32, sz.dims...)
		data := randomFloat32(shape.Size())

		fused := buildBenchExec([]shapes.Shape{shape}, []any{data},
			func(f backends.Function, params []backends.Value) (backends.Value, error) {
				return f.FusedGelu(params[0], true)
			})

		// Decomposed GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
		decomposed := buildBenchExec([]shapes.Shape{shape}, []any{data},
			func(f backends.Function, params []backends.Value) (backends.Value, error) {
				x := params[0]
				sqrt2Inv := benchMust(f.Constant([]float32{float32(1.0 / 1.4142135623730951)}, 1))
				sqrt2InvBroadcast := benchMust(f.BroadcastInDim(sqrt2Inv, shape, []int{0}))
				half := benchMust(f.Constant([]float32{0.5}, 1))
				halfBroadcast := benchMust(f.BroadcastInDim(half, shape, []int{0}))
				one := benchMust(f.Constant([]float32{1.0}, 1))
				oneBroadcast := benchMust(f.BroadcastInDim(one, shape, []int{0}))

				scaled := benchMust(f.Mul(x, sqrt2InvBroadcast))
				erfVal := benchMust(f.Erf(scaled))
				onePlusErf := benchMust(f.Add(oneBroadcast, erfVal))
				xHalf := benchMust(f.Mul(x, halfBroadcast))
				return f.Mul(xHalf, onePlusErf)
			})

		b.Run(fmt.Sprintf("Fused/%s", sz.name), func(b *testing.B) { fused.run(b) })
		b.Run(fmt.Sprintf("Decomposed/%s", sz.name), func(b *testing.B) { decomposed.run(b) })
	}
}

// --- LayerNorm Benchmarks ---

func BenchmarkLayerNorm(b *testing.B) {
	sizes := []struct {
		name string
		dims []int
		axis int
	}{
		{"8x64_axis1", []int{8, 64}, 1},
		{"32x256_axis1", []int{32, 256}, 1},
		{"64x768_axis1", []int{64, 768}, 1},
		{"8x16x64_axis2", []int{8, 16, 64}, 2},
	}

	for _, sz := range sizes {
		shape := shapes.Make(dtypes.Float32, sz.dims...)
		data := randomFloat32(shape.Size())
		normDim := sz.dims[sz.axis]
		gammaData := randomFloat32(normDim)
		betaData := randomFloat32(normDim)
		gammaShape := shapes.Make(dtypes.Float32, normDim)
		betaShape := shapes.Make(dtypes.Float32, normDim)
		axis := sz.axis

		allShapes := []shapes.Shape{shape, gammaShape, betaShape}
		allDatas := []any{data, gammaData, betaData}

		fused := buildBenchExec(allShapes, allDatas,
			func(f backends.Function, params []backends.Value) (backends.Value, error) {
				return f.FusedLayerNorm(params[0], []int{axis}, 1e-5, params[1], params[2])
			})

		// Decomposed: mean, variance, normalize, scale, offset.
		decomposed := buildBenchExec(allShapes, allDatas,
			func(f backends.Function, params []backends.Value) (backends.Value, error) {
				x := params[0]
				gamma := params[1]
				beta := params[2]

				// Compute normSize as float constant.
				normSizeF := float32(sz.dims[axis])
				normSizeConst := benchMust(f.Constant([]float32{normSizeF}, 1))
				normSizeBroadcast := benchMust(f.BroadcastInDim(normSizeConst, shape, []int{0}))

				// Mean.
				sum := reduceAndKeep(f, x, f.ReduceSum, shape, axis)
				mean := benchMust(f.Div(sum, normSizeBroadcast))

				// Variance.
				diff := benchMust(f.Sub(x, mean))
				diffSq := benchMust(f.Mul(diff, diff))
				varSum := reduceAndKeep(f, diffSq, f.ReduceSum, shape, axis)
				variance := benchMust(f.Div(varSum, normSizeBroadcast))

				// Normalize.
				epsConst := benchMust(f.Constant([]float32{1e-5}, 1))
				epsBroadcast := benchMust(f.BroadcastInDim(epsConst, shape, []int{0}))
				varPlusEps := benchMust(f.Add(variance, epsBroadcast))
				invStd := benchMust(f.Rsqrt(varPlusEps))
				normalized := benchMust(f.Mul(diff, invStd))

				// Scale and offset: gamma and beta have shape [normDim], need to broadcast.
				broadcastShape := shape.Clone()
				for i := range broadcastShape.Dimensions {
					broadcastShape.Dimensions[i] = 1
				}
				broadcastShape.Dimensions[axis] = normDim
				gammaReshaped := benchMust(f.Reshape(gamma, broadcastShape.Dimensions...))
				broadcastAxes := make([]int, shape.Rank())
				for i := range broadcastAxes {
					broadcastAxes[i] = i
				}
				gammaBroadcast := benchMust(f.BroadcastInDim(gammaReshaped, shape, broadcastAxes))
				scaled := benchMust(f.Mul(normalized, gammaBroadcast))

				betaReshaped := benchMust(f.Reshape(beta, broadcastShape.Dimensions...))
				betaBroadcast := benchMust(f.BroadcastInDim(betaReshaped, shape, broadcastAxes))
				return f.Add(scaled, betaBroadcast)
			})

		b.Run(fmt.Sprintf("Fused/%s", sz.name), func(b *testing.B) { fused.run(b) })
		b.Run(fmt.Sprintf("Decomposed/%s", sz.name), func(b *testing.B) { decomposed.run(b) })
	}
}

// --- Dense Benchmarks ---

func BenchmarkDense(b *testing.B) {
	sizes := []struct {
		name        string
		batch       int
		inFeatures  int
		outFeatures int
	}{
		{"1x64x64", 1, 64, 64},
		{"8x128x256", 8, 128, 256},
		{"32x512x1024", 32, 512, 1024},
	}

	for _, sz := range sizes {
		xShape := shapes.Make(dtypes.Float32, sz.batch, sz.inFeatures)
		wShape := shapes.Make(dtypes.Float32, sz.inFeatures, sz.outFeatures)
		bShape := shapes.Make(dtypes.Float32, sz.outFeatures)
		outShape := shapes.Make(dtypes.Float32, sz.batch, sz.outFeatures)

		xData := randomFloat32(xShape.Size())
		wData := randomFloat32(wShape.Size())
		biasData := randomFloat32(bShape.Size())

		allShapes := []shapes.Shape{xShape, wShape, bShape}
		allDatas := []any{xData, wData, biasData}

		fused := buildBenchExec(allShapes, allDatas,
			func(f backends.Function, params []backends.Value) (backends.Value, error) {
				return f.FusedDense(params[0], params[1], params[2], backends.ActivationNone)
			})

		// Decomposed: DotGeneral + bias add.
		decomposed := buildBenchExec(allShapes, allDatas,
			func(f backends.Function, params []backends.Value) (backends.Value, error) {
				x := params[0]
				weight := params[1]
				bias := params[2]

				// x @ weight via DotGeneral: contract x's axis 1 with weight's axis 0.
				y := benchMust(f.DotGeneral(x, []int{1}, nil, weight, []int{0}, nil))

				// Add bias: broadcast [outFeatures] -> [batch, outFeatures].
				biasBroadcast := benchMust(f.BroadcastInDim(bias, outShape, []int{1}))
				return f.Add(y, biasBroadcast)
			})

		b.Run(fmt.Sprintf("Fused/%s", sz.name), func(b *testing.B) { fused.run(b) })
		b.Run(fmt.Sprintf("Decomposed/%s", sz.name), func(b *testing.B) { decomposed.run(b) })
	}
}
