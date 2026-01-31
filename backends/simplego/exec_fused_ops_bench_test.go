// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// --- Decomposed implementations operating directly on float32 slices ---
// These mirror what the decomposed graph ops would compute, but without
// graph overhead, giving a fair comparison of the fused executor vs
// equivalent element-wise operations with intermediate allocations.

// decomposedSoftmaxFloat32 computes softmax using separate passes with intermediate allocations.
func decomposedSoftmaxFloat32(input []float32, shape shapes.Shape, axis int) []float32 {
	outerSize, axisSize, innerSize := computeAxisStrides(shape, axis)
	size := len(input)

	// Step 1: ReduceMax (intermediate allocation)
	maxVals := make([]float32, outerSize*innerSize)
	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			maxVal := float32(math.Inf(-1))
			for i := 0; i < axisSize; i++ {
				idx := outer*axisSize*innerSize + i*innerSize + inner
				if input[idx] > maxVal {
					maxVal = input[idx]
				}
			}
			maxVals[outer*innerSize+inner] = maxVal
		}
	}

	// Step 2: Sub x - max (intermediate allocation)
	shifted := make([]float32, size)
	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			maxVal := maxVals[outer*innerSize+inner]
			for i := 0; i < axisSize; i++ {
				idx := outer*axisSize*innerSize + i*innerSize + inner
				shifted[idx] = input[idx] - maxVal
			}
		}
	}

	// Step 3: Exp (intermediate allocation)
	expVals := make([]float32, size)
	for i, v := range shifted {
		expVals[i] = float32(math.Exp(float64(v)))
	}

	// Step 4: ReduceSum (intermediate allocation)
	sumVals := make([]float32, outerSize*innerSize)
	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			var sum float32
			for i := 0; i < axisSize; i++ {
				idx := outer*axisSize*innerSize + i*innerSize + inner
				sum += expVals[idx]
			}
			sumVals[outer*innerSize+inner] = sum
		}
	}

	// Step 5: Div (output)
	output := make([]float32, size)
	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			invSum := 1.0 / sumVals[outer*innerSize+inner]
			for i := 0; i < axisSize; i++ {
				idx := outer*axisSize*innerSize + i*innerSize + inner
				output[idx] = expVals[idx] * invSum
			}
		}
	}
	return output
}

// decomposedGeluFloat32 computes GELU using separate element-wise ops with intermediate allocations.
func decomposedGeluFloat32(input []float32) []float32 {
	sqrt2Inv := float32(1.0 / math.Sqrt(2.0))
	size := len(input)

	// Step 1: x / sqrt(2) (intermediate)
	scaled := make([]float32, size)
	for i, x := range input {
		scaled[i] = x * sqrt2Inv
	}

	// Step 2: erf (intermediate)
	erfVals := make([]float32, size)
	for i, v := range scaled {
		erfVals[i] = float32(math.Erf(float64(v)))
	}

	// Step 3: 1 + erf (intermediate)
	onePlusErf := make([]float32, size)
	for i, v := range erfVals {
		onePlusErf[i] = 1.0 + v
	}

	// Step 4: 0.5 * (1 + erf) (intermediate)
	cdf := make([]float32, size)
	for i, v := range onePlusErf {
		cdf[i] = 0.5 * v
	}

	// Step 5: x * cdf (output)
	output := make([]float32, size)
	for i, x := range input {
		output[i] = x * cdf[i]
	}
	return output
}

// decomposedLayerNormFloat32 computes layer norm using separate ops with intermediate allocations.
func decomposedLayerNormFloat32(input []float32, shape shapes.Shape, axis int, epsilon float64, gamma, beta []float32) []float32 {
	outerSize, normSize, _ := computeAxisStrides(shape, axis)
	normSizeF := float32(normSize)
	output := make([]float32, len(input))

	for outer := 0; outer < outerSize; outer++ {
		base := outer * normSize

		// Step 1: ReduceSum → mean (intermediate)
		var sum float32
		for i := 0; i < normSize; i++ {
			sum += input[base+i]
		}
		mean := sum / normSizeF

		// Step 2: Sub x - mean (intermediate)
		diff := make([]float32, normSize)
		for i := 0; i < normSize; i++ {
			diff[i] = input[base+i] - mean
		}

		// Step 3: Square (intermediate)
		diffSq := make([]float32, normSize)
		for i, d := range diff {
			diffSq[i] = d * d
		}

		// Step 4: ReduceSum → variance (intermediate)
		var varSum float32
		for _, v := range diffSq {
			varSum += v
		}
		variance := varSum / normSizeF

		// Step 5: Add epsilon, Sqrt, Div (intermediate)
		invStd := float32(1.0 / math.Sqrt(float64(variance)+epsilon))

		// Step 6: Normalize, scale, shift (output)
		for i := 0; i < normSize; i++ {
			normalized := diff[i] * invStd
			if gamma != nil {
				normalized *= gamma[i]
			}
			if beta != nil {
				normalized += beta[i]
			}
			output[base+i] = normalized
		}
	}
	return output
}

// decomposedDenseFloat32 computes y = x @ W + b using separate matmul and add steps.
// W is [in_features, out_features].
func decomposedDenseFloat32(xData, wData, biasData []float32, batchSize, inFeatures, outFeatures int) []float32 {
	// Step 1: MatMul x @ W (intermediate)
	matmulOut := make([]float32, batchSize*outFeatures)
	for b := 0; b < batchSize; b++ {
		for o := 0; o < outFeatures; o++ {
			var sum float32
			for i := 0; i < inFeatures; i++ {
				sum += xData[b*inFeatures+i] * wData[i*outFeatures+o]
			}
			matmulOut[b*outFeatures+o] = sum
		}
	}

	// Step 2: Add bias (output)
	output := make([]float32, batchSize*outFeatures)
	for b := 0; b < batchSize; b++ {
		for o := 0; o < outFeatures; o++ {
			output[b*outFeatures+o] = matmulOut[b*outFeatures+o] + biasData[o]
		}
	}
	return output
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
		data := make([]float32, shape.Size())
		for i := range data {
			data[i] = rand.Float32()*2 - 1
		}

		b.Run(fmt.Sprintf("Fused/%s", sz.name), func(b *testing.B) {
			be := backend.(*Backend)
			input := be.getBufferForShape(shape)
			copy(input.flat.([]float32), data)
			node := &Node{
				shape: shape,
				data:  &nodeFusedSoftmax{axis: sz.axis},
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				out, err := execFusedSoftmax(be, node, []*Buffer{input}, []bool{false})
				if err != nil {
					b.Fatal(err)
				}
				be.putBuffer(out)
			}
		})

		b.Run(fmt.Sprintf("Decomposed/%s", sz.name), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = decomposedSoftmaxFloat32(data, shape, sz.axis)
			}
		})
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
		data := make([]float32, shape.Size())
		for i := range data {
			data[i] = rand.Float32()*2 - 1
		}

		b.Run(fmt.Sprintf("Fused/%s", sz.name), func(b *testing.B) {
			be := backend.(*Backend)
			input := be.getBufferForShape(shape)
			copy(input.flat.([]float32), data)
			node := &Node{
				shape: shape,
				data:  &nodeFusedGelu{exact: true},
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				out, err := execFusedGelu(be, node, []*Buffer{input}, []bool{false})
				if err != nil {
					b.Fatal(err)
				}
				be.putBuffer(out)
			}
		})

		b.Run(fmt.Sprintf("Decomposed/%s", sz.name), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = decomposedGeluFloat32(data)
			}
		})
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
		data := make([]float32, shape.Size())
		for i := range data {
			data[i] = rand.Float32()*2 - 1
		}
		normDim := sz.dims[sz.axis]
		gamma := make([]float32, normDim)
		beta := make([]float32, normDim)
		for i := range gamma {
			gamma[i] = rand.Float32()*2 - 1
			beta[i] = rand.Float32()*2 - 1
		}
		gammaShape := shapes.Make(dtypes.Float32, normDim)
		betaShape := shapes.Make(dtypes.Float32, normDim)

		b.Run(fmt.Sprintf("Fused/%s", sz.name), func(b *testing.B) {
			be := backend.(*Backend)
			input := be.getBufferForShape(shape)
			copy(input.flat.([]float32), data)
			gammaBuf := be.getBufferForShape(gammaShape)
			copy(gammaBuf.flat.([]float32), gamma)
			betaBuf := be.getBufferForShape(betaShape)
			copy(betaBuf.flat.([]float32), beta)
			node := &Node{
				shape: shape,
				data:  &nodeFusedLayerNorm{axes: []int{sz.axis}, epsilon: 1e-5},
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				out, err := execFusedLayerNorm(be, node, []*Buffer{input, gammaBuf, betaBuf}, []bool{false, false, false})
				if err != nil {
					b.Fatal(err)
				}
				be.putBuffer(out)
			}
		})

		b.Run(fmt.Sprintf("Decomposed/%s", sz.name), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = decomposedLayerNormFloat32(data, shape, sz.axis, 1e-5, gamma, beta)
			}
		})
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

		xData := make([]float32, xShape.Size())
		wData := make([]float32, wShape.Size())
		biasData := make([]float32, bShape.Size())
		for i := range xData {
			xData[i] = rand.Float32()*2 - 1
		}
		for i := range wData {
			wData[i] = rand.Float32()*2 - 1
		}
		for i := range biasData {
			biasData[i] = rand.Float32()*2 - 1
		}

		b.Run(fmt.Sprintf("Fused/%s", sz.name), func(b *testing.B) {
			be := backend.(*Backend)
			xBuf := be.getBufferForShape(xShape)
			copy(xBuf.flat.([]float32), xData)
			wBuf := be.getBufferForShape(wShape)
			copy(wBuf.flat.([]float32), wData)
			bBuf := be.getBufferForShape(bShape)
			copy(bBuf.flat.([]float32), biasData)
			outShape := shapes.Make(dtypes.Float32, sz.batch, sz.outFeatures)
			node := &Node{shape: outShape}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				out, err := execFusedDense(be, node, []*Buffer{xBuf, wBuf, bBuf}, []bool{false, false, false})
				if err != nil {
					b.Fatal(err)
				}
				be.putBuffer(out)
			}
		})

		b.Run(fmt.Sprintf("Decomposed/%s", sz.name), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = decomposedDenseFloat32(xData, wData, biasData, sz.batch, sz.inFeatures, sz.outFeatures)
			}
		})
	}
}
