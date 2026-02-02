// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"math"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// tolerance for floating point comparison.
const fusedTestTol = 1e-6

func TestFusedSoftmax_1D(t *testing.T) {
	input := []float32{1.0, 2.0, 3.0, 4.0}
	shape := shapes.Make(dtypes.Float32, 4)

	result := testBackend(t, shape, input, func(f backends.Function, param backends.Value) (backends.Value, error) {
		return f.FusedSoftmax(param, 0)
	})

	got := result.flat.([]float32)
	// Known-correct softmax([1,2,3,4]).
	want := []float32{0.0320586, 0.0871443, 0.2368828, 0.6439143}
	require.Len(t, got, len(want))
	for i := range got {
		assert.InDelta(t, want[i], got[i], fusedTestTol, "index %d", i)
	}

	// Softmax output should sum to 1.
	var sum float32
	for _, v := range got {
		sum += v
	}
	assert.InDelta(t, 1.0, sum, fusedTestTol)
}

func TestFusedSoftmax_2D(t *testing.T) {
	// 2x3 matrix, softmax over axis 1 (last axis).
	input := []float32{1, 2, 3, 4, 5, 6}
	shape := shapes.Make(dtypes.Float32, 2, 3)

	result := testBackend(t, shape, input, func(f backends.Function, param backends.Value) (backends.Value, error) {
		return f.FusedSoftmax(param, 1)
	})

	got := result.flat.([]float32)
	// Each row should sum to 1.
	assert.InDelta(t, 1.0, got[0]+got[1]+got[2], fusedTestTol)
	assert.InDelta(t, 1.0, got[3]+got[4]+got[5], fusedTestTol)
	// Values within each row should be monotonically increasing.
	assert.Less(t, got[0], got[1])
	assert.Less(t, got[1], got[2])
}

func TestFusedSoftmax_Axis0(t *testing.T) {
	// 2x3 matrix, softmax over axis 0 (columns).
	input := []float32{1, 2, 3, 4, 5, 6}
	shape := shapes.Make(dtypes.Float32, 2, 3)

	result := testBackend(t, shape, input, func(f backends.Function, param backends.Value) (backends.Value, error) {
		return f.FusedSoftmax(param, 0)
	})

	got := result.flat.([]float32)
	// Each column should sum to 1.
	assert.InDelta(t, 1.0, got[0]+got[3], fusedTestTol) // col 0
	assert.InDelta(t, 1.0, got[1]+got[4], fusedTestTol) // col 1
	assert.InDelta(t, 1.0, got[2]+got[5], fusedTestTol) // col 2
}

func TestFusedSoftmax_NegativeAxis(t *testing.T) {
	// Negative axes should be rejected by FusedSoftmax (caller normalizes).
	shape := shapes.Make(dtypes.Float32, 2, 3)

	builder := backend.Builder("fused_test")
	mainFn := builder.Main()

	param, err := mainFn.Parameter("x", shape, nil)
	require.NoError(t, err)

	_, err = mainFn.FusedSoftmax(param, -1)
	assert.Error(t, err, "FusedSoftmax should reject negative axis")
}

func TestFusedSoftmax_Float64(t *testing.T) {
	input := []float64{1.0, 2.0, 3.0}
	shape := shapes.Make(dtypes.Float64, 3)

	result := testBackend(t, shape, input, func(f backends.Function, param backends.Value) (backends.Value, error) {
		return f.FusedSoftmax(param, 0)
	})

	got := result.flat.([]float64)
	var sum float64
	for _, v := range got {
		sum += v
	}
	assert.InDelta(t, 1.0, sum, fusedTestTol)
	// Values should be monotonically increasing.
	assert.Less(t, got[0], got[1])
	assert.Less(t, got[1], got[2])
}

func TestFusedGelu(t *testing.T) {
	input := []float32{-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0}
	shape := shapes.Make(dtypes.Float32, 7)

	result := testBackend(t, shape, input, func(f backends.Function, param backends.Value) (backends.Value, error) {
		return f.FusedGelu(param, true)
	})

	got := result.flat.([]float32)
	// Known-correct GELU values (computed at float32 precision).
	want := []float32{
		-0.04550028, // gelu(-2)
		-0.15865526, // gelu(-1)
		-0.15426877, // gelu(-0.5)
		0.0,         // gelu(0)
		0.34573123,  // gelu(0.5)
		0.84134474,  // gelu(1)
		1.9544997,   // gelu(2)
	}
	for i := range got {
		assert.InDelta(t, want[i], got[i], fusedTestTol, "index %d: gelu(%v)", i, input[i])
	}
}

func TestFusedGelu_Float64(t *testing.T) {
	input := []float64{-1.0, 0.0, 1.0}
	shape := shapes.Make(dtypes.Float64, 3)

	result := testBackend(t, shape, input, func(f backends.Function, param backends.Value) (backends.Value, error) {
		return f.FusedGelu(param, true)
	})

	got := result.flat.([]float64)
	// Known-correct GELU values.
	want := []float64{-0.15865525393145702, 0.0, 0.8413447460685429}
	for i := range got {
		assert.InDelta(t, want[i], got[i], fusedTestTol, "gelu(%v)", input[i])
	}
}

func TestFusedLayerNorm_Simple(t *testing.T) {
	// 2x4 input, normalize over last axis.
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	shape := shapes.Make(dtypes.Float32, 2, 4)
	epsilon := 1e-5

	result := testBackend(t, shape, input, func(f backends.Function, param backends.Value) (backends.Value, error) {
		return f.FusedLayerNorm(param, []int{1}, epsilon, nil, nil)
	})

	got := result.flat.([]float32)

	// Verify each row is normalized: mean ≈ 0, variance ≈ 1.
	for row := 0; row < 2; row++ {
		var sum float32
		for i := 0; i < 4; i++ {
			sum += got[row*4+i]
		}
		mean := sum / 4.0
		assert.InDelta(t, 0.0, mean, 1e-5, "row %d mean", row)

		var varSum float32
		for i := 0; i < 4; i++ {
			diff := got[row*4+i] - mean
			varSum += diff * diff
		}
		variance := varSum / 4.0
		assert.InDelta(t, 1.0, variance, 1e-3, "row %d variance", row)
	}
}

func TestFusedLayerNorm_WithGammaBeta(t *testing.T) {
	// 1x3 input with gamma and beta.
	input := []float32{1, 2, 3}
	gamma := []float32{2, 2, 2} // scale by 2
	beta := []float32{1, 1, 1}  // shift by 1
	shape := shapes.Make(dtypes.Float32, 1, 3)
	gammaShape := shapes.Make(dtypes.Float32, 3)
	betaShape := shapes.Make(dtypes.Float32, 3)
	epsilon := 1e-5

	result := testBackendMultiInput(t,
		[]shapes.Shape{shape, gammaShape, betaShape},
		[]any{input, gamma, beta},
		func(f backends.Function, params []backends.Value) (backends.Value, error) {
			return f.FusedLayerNorm(params[0], []int{1}, epsilon, params[1], params[2])
		},
	)

	got := result.flat.([]float32)

	// First normalize [1,2,3]: mean=2, var=2/3, std=sqrt(2/3)
	// normalized: [-1/std, 0, 1/std] where std=sqrt(2/3+eps)
	// Then multiply by gamma=2 and add beta=1.
	meanVal := float32(2.0)
	variance := float32((1.0 + 0.0 + 1.0) / 3.0) // sum of (x-mean)^2 / n
	invStd := float32(1.0 / math.Sqrt(float64(variance)+epsilon))

	for i, x := range input {
		normalized := (x - meanVal) * invStd
		want := normalized*gamma[i] + beta[i]
		assert.InDelta(t, want, got[i], 1e-4, "index %d", i)
	}
}

func TestFusedDense(t *testing.T) {
	// x: [2, 3] (batch=2, in_features=3)
	// w: [3, 4] (in_features=3, out_features=4)
	// b: [4]    (out_features=4)
	// output: [2, 4]
	x := []float32{1, 2, 3, 4, 5, 6}
	w := []float32{
		1, 0, 0, 1,
		0, 1, 0, 1,
		0, 0, 1, 1,
	}
	b := []float32{10, 20, 30, 40}

	xShape := shapes.Make(dtypes.Float32, 2, 3)
	wShape := shapes.Make(dtypes.Float32, 3, 4)
	bShape := shapes.Make(dtypes.Float32, 4)

	result := testBackendMultiInput(t,
		[]shapes.Shape{xShape, wShape, bShape},
		[]any{x, w, b},
		func(f backends.Function, params []backends.Value) (backends.Value, error) {
			return f.FusedDense(params[0], params[1], params[2], backends.ActivationNone)
		},
	)

	got := result.flat.([]float32)
	want := []float32{11, 22, 33, 46, 14, 25, 36, 55}
	for i := range got {
		assert.InDelta(t, want[i], got[i], fusedTestTol, "index %d", i)
	}
}

func TestFusedDense_NoBias(t *testing.T) {
	x := []float32{1, 2, 3}
	w := []float32{
		1, 2,
		1, 2,
		1, 2,
	}

	xShape := shapes.Make(dtypes.Float32, 1, 3)
	wShape := shapes.Make(dtypes.Float32, 3, 2)

	result := testBackendMultiInput(t,
		[]shapes.Shape{xShape, wShape},
		[]any{x, w},
		func(f backends.Function, params []backends.Value) (backends.Value, error) {
			return f.FusedDense(params[0], params[1], nil, backends.ActivationNone)
		},
	)

	got := result.flat.([]float32)
	want := []float32{6, 12}
	for i := range got {
		assert.InDelta(t, want[i], got[i], fusedTestTol, "index %d", i)
	}
}

func TestFusedDense_Relu(t *testing.T) {
	x := []float32{1, -1}
	w := []float32{
		1, 1,
		0, -1,
	}
	b := []float32{-1, -1}

	xShape := shapes.Make(dtypes.Float32, 1, 2)
	wShape := shapes.Make(dtypes.Float32, 2, 2)
	bShape := shapes.Make(dtypes.Float32, 2)

	result := testBackendMultiInput(t,
		[]shapes.Shape{xShape, wShape, bShape},
		[]any{x, w, b},
		func(f backends.Function, params []backends.Value) (backends.Value, error) {
			return f.FusedDense(params[0], params[1], params[2], backends.ActivationRelu)
		},
	)

	got := result.flat.([]float32)
	want := []float32{0, 1} // ReLU clamps negative to 0.
	for i := range got {
		assert.InDelta(t, want[i], got[i], fusedTestTol, "index %d", i)
	}
}

func TestFusedDense_Gelu(t *testing.T) {
	x := []float32{1, 0}
	w := []float32{1, 0, 0, 1} // identity [2,2]
	b := []float32{0, 0}

	xShape := shapes.Make(dtypes.Float32, 1, 2)
	wShape := shapes.Make(dtypes.Float32, 2, 2)
	bShape := shapes.Make(dtypes.Float32, 2)

	result := testBackendMultiInput(t,
		[]shapes.Shape{xShape, wShape, bShape},
		[]any{x, w, b},
		func(f backends.Function, params []backends.Value) (backends.Value, error) {
			return f.FusedDense(params[0], params[1], params[2], backends.ActivationGelu)
		},
	)

	got := result.flat.([]float32)
	// Known-correct: gelu(1) ≈ 0.8413447, gelu(0) = 0.
	assert.InDelta(t, 0.8413447, got[0], fusedTestTol)
	assert.InDelta(t, 0.0, got[1], fusedTestTol)
}

func TestFusedDense_Silu(t *testing.T) {
	x := []float32{2}
	w := []float32{1} // [1,1]
	b := []float32{0}

	xShape := shapes.Make(dtypes.Float32, 1, 1)
	wShape := shapes.Make(dtypes.Float32, 1, 1)
	bShape := shapes.Make(dtypes.Float32, 1)

	result := testBackendMultiInput(t,
		[]shapes.Shape{xShape, wShape, bShape},
		[]any{x, w, b},
		func(f backends.Function, params []backends.Value) (backends.Value, error) {
			return f.FusedDense(params[0], params[1], params[2], backends.ActivationSilu)
		},
	)

	got := result.flat.([]float32)
	want := float32(2.0 / (1.0 + math.Exp(-2.0)))
	assert.InDelta(t, want, got[0], fusedTestTol)
}

func TestFusedDense_Tanh(t *testing.T) {
	x := []float32{1}
	w := []float32{1}
	b := []float32{0}

	xShape := shapes.Make(dtypes.Float32, 1, 1)
	wShape := shapes.Make(dtypes.Float32, 1, 1)
	bShape := shapes.Make(dtypes.Float32, 1)

	result := testBackendMultiInput(t,
		[]shapes.Shape{xShape, wShape, bShape},
		[]any{x, w, b},
		func(f backends.Function, params []backends.Value) (backends.Value, error) {
			return f.FusedDense(params[0], params[1], params[2], backends.ActivationTanh)
		},
	)

	got := result.flat.([]float32)
	want := float32(math.Tanh(1.0))
	assert.InDelta(t, want, got[0], fusedTestTol)
}

func TestFusedSoftmax_LargeValues(t *testing.T) {
	// Test numerical stability with large values.
	input := []float32{1000, 1001, 1002}
	shape := shapes.Make(dtypes.Float32, 3)

	result := testBackend(t, shape, input, func(f backends.Function, param backends.Value) (backends.Value, error) {
		return f.FusedSoftmax(param, 0)
	})

	got := result.flat.([]float32)

	// Should still sum to 1 and not overflow.
	var sum float32
	for _, v := range got {
		sum += v
		assert.False(t, math.IsNaN(float64(v)), "softmax produced NaN")
		assert.False(t, math.IsInf(float64(v), 0), "softmax produced Inf")
	}
	assert.InDelta(t, 1.0, sum, fusedTestTol)
}

func TestFusedSoftmax_3D(t *testing.T) {
	// [2, 2, 3], softmax over axis 2 (last).
	input := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	}
	shape := shapes.Make(dtypes.Float32, 2, 2, 3)

	result := testBackend(t, shape, input, func(f backends.Function, param backends.Value) (backends.Value, error) {
		return f.FusedSoftmax(param, 2)
	})

	got := result.flat.([]float32)
	// Each group of 3 should sum to 1.
	for group := 0; group < 4; group++ {
		base := group * 3
		sum := got[base] + got[base+1] + got[base+2]
		assert.InDelta(t, 1.0, sum, fusedTestTol, "group %d", group)
	}
}
