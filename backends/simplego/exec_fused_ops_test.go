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
	for row := range 2 {
		var sum float32
		for i := range 4 {
			sum += got[row*4+i]
		}
		mean := sum / 4.0
		assert.InDelta(t, 0.0, mean, 1e-5, "row %d mean", row)

		var varSum float32
		for i := range 4 {
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
	for group := range 4 {
		base := group * 3
		sum := got[base] + got[base+1] + got[base+2]
		assert.InDelta(t, 1.0, sum, fusedTestTol, "group %d", group)
	}
}

// execFusedOpMultiOutput builds, compiles and executes a multi-output fused op graph.
// buildFn receives the Function and the parameter Values, and returns 3 output Values.
func execFusedOpMultiOutput3(t *testing.T, inputShapes []shapes.Shape, inputDatas []any,
	buildFn func(f backends.Function, params []backends.Value) (backends.Value, backends.Value, backends.Value, error),
) [3]*Buffer {
	t.Helper()
	builder := backend.Builder("fused_test_multiout")
	mainFn := builder.Main()

	params := make([]backends.Value, len(inputShapes))
	for i, s := range inputShapes {
		p, err := mainFn.Parameter("x"+string(rune('0'+i)), s, nil)
		require.NoError(t, err)
		params[i] = p
	}

	o0, o1, o2, err := buildFn(mainFn, params)
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{o0, o1, o2}, nil)
	require.NoError(t, err)

	exec, err := builder.Compile()
	require.NoError(t, err)

	inputBufs := make([]backends.Buffer, len(inputDatas))
	for i, data := range inputDatas {
		buf, err := backend.BufferFromFlatData(0, data, inputShapes[i])
		require.NoError(t, err)
		inputBufs[i] = buf
	}

	outputs, err := exec.Execute(inputBufs, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 3)
	return [3]*Buffer{outputs[0].(*Buffer), outputs[1].(*Buffer), outputs[2].(*Buffer)}
}

// ---- FusedMultiHeadSDPA tests ----

func TestFusedMultiHeadSDPA_SingleHead(t *testing.T) {
	// batch=1, numHeads=1, seqLen=2, headDim=2, kvLen=2
	// Q = [[1, 0], [0, 1]]
	// K = [[1, 0], [0, 1]]  (identity-like)
	// V = [[10, 20], [30, 40]]
	q := []float32{1, 0, 0, 1}
	k := []float32{1, 0, 0, 1}
	v := []float32{10, 20, 30, 40}

	qShape := shapes.Make(dtypes.Float32, 1, 1, 2, 2)
	kShape := shapes.Make(dtypes.Float32, 1, 1, 2, 2)
	vShape := shapes.Make(dtypes.Float32, 1, 1, 2, 2)

	scale := float64(1.0 / math.Sqrt(2.0)) // 1/sqrt(headDim)

	result := testBackendMultiInput(t,
		[]shapes.Shape{qShape, kShape, vShape},
		[]any{q, k, v},
		func(f backends.Function, params []backends.Value) (backends.Value, error) {
			return f.FusedMultiHeadSDPA(params[0], params[1], params[2], nil, 1, 1, scale, false)
		},
	)

	got := result.flat.([]float32)
	// scores[0][0] = (1*1+0*0)*scale = scale, scores[0][1] = (1*0+0*1)*scale = 0
	// softmax([scale, 0]) = [exp(scale)/(exp(scale)+1), 1/(exp(scale)+1)]
	// Output row 0 = softmax_weights @ V
	// Similarly for row 1.
	require.Len(t, got, 4)
	for _, val := range got {
		assert.False(t, math.IsNaN(float64(val)), "output contains NaN")
	}
	// Output should be a weighted avg of V rows, so between min and max of V.
	for i := range got {
		assert.GreaterOrEqual(t, got[i], float32(10.0)-1e-3)
		assert.LessOrEqual(t, got[i], float32(40.0)+1e-3)
	}
}

func TestFusedMultiHeadSDPA_Causal(t *testing.T) {
	// batch=1, numHeads=1, seqLen=2, headDim=1, kvLen=2
	// With causal mask: position 0 can only attend to position 0.
	q := []float32{1, 1}
	k := []float32{1, 1}
	v := []float32{10, 20}

	qShape := shapes.Make(dtypes.Float32, 1, 1, 2, 1)
	kShape := shapes.Make(dtypes.Float32, 1, 1, 2, 1)
	vShape := shapes.Make(dtypes.Float32, 1, 1, 2, 1)

	result := testBackendMultiInput(t,
		[]shapes.Shape{qShape, kShape, vShape},
		[]any{q, k, v},
		func(f backends.Function, params []backends.Value) (backends.Value, error) {
			return f.FusedMultiHeadSDPA(params[0], params[1], params[2], nil, 1, 1, 1.0, true)
		},
	)

	got := result.flat.([]float32)
	// Position 0 can only see position 0 → output = V[0] = 10
	assert.InDelta(t, 10.0, got[0], fusedTestTol)
	// Position 1 can see both → softmax([1, 1]) = [0.5, 0.5] → output = 0.5*10+0.5*20 = 15
	assert.InDelta(t, 15.0, got[1], fusedTestTol)
}

func TestFusedMultiHeadSDPA_MultiHead(t *testing.T) {
	// batch=1, numHeads=2, seqLen=1, headDim=1, kvLen=1
	// Simple case: each head attends to a single key/value.
	q := []float32{1, 2}     // 2 heads, each with seqLen=1, headDim=1
	k := []float32{1, 1}     // 2 heads
	v := []float32{100, 200} // 2 heads

	qShape := shapes.Make(dtypes.Float32, 1, 2, 1, 1)
	kShape := shapes.Make(dtypes.Float32, 1, 2, 1, 1)
	vShape := shapes.Make(dtypes.Float32, 1, 2, 1, 1)

	result := testBackendMultiInput(t,
		[]shapes.Shape{qShape, kShape, vShape},
		[]any{q, k, v},
		func(f backends.Function, params []backends.Value) (backends.Value, error) {
			return f.FusedMultiHeadSDPA(params[0], params[1], params[2], nil, 2, 2, 1.0, false)
		},
	)

	got := result.flat.([]float32)
	// With kvLen=1, attention is just V itself (softmax of single element = 1).
	assert.InDelta(t, 100.0, got[0], fusedTestTol) // head 0
	assert.InDelta(t, 200.0, got[1], fusedTestTol) // head 1
}

func TestFusedMultiHeadSDPA_GQA(t *testing.T) {
	// batch=1, numHeads=2, numKVHeads=1 (GQA: 2 query heads share 1 KV head)
	// seqLen=1, kvLen=1, headDim=1
	q := []float32{1, 2} // 2 heads
	k := []float32{1}    // 1 KV head
	v := []float32{42}   // 1 KV head

	qShape := shapes.Make(dtypes.Float32, 1, 2, 1, 1)
	kShape := shapes.Make(dtypes.Float32, 1, 1, 1, 1)
	vShape := shapes.Make(dtypes.Float32, 1, 1, 1, 1)

	result := testBackendMultiInput(t,
		[]shapes.Shape{qShape, kShape, vShape},
		[]any{q, k, v},
		func(f backends.Function, params []backends.Value) (backends.Value, error) {
			return f.FusedMultiHeadSDPA(params[0], params[1], params[2], nil, 2, 1, 1.0, false)
		},
	)

	got := result.flat.([]float32)
	// Both heads attend to the same single KV → output = V = 42 for both.
	assert.InDelta(t, 42.0, got[0], fusedTestTol)
	assert.InDelta(t, 42.0, got[1], fusedTestTol)
}

func TestFusedMultiHeadSDPA_WithMask(t *testing.T) {
	// batch=1, numHeads=1, seqLen=1, kvLen=2, headDim=1
	// mask blocks second key position with -inf.
	q := []float32{1}
	k := []float32{1, 1}
	v := []float32{10, 20}
	mask := []float32{0, float32(math.Inf(-1))} // block second position

	qShape := shapes.Make(dtypes.Float32, 1, 1, 1, 1)
	kShape := shapes.Make(dtypes.Float32, 1, 1, 2, 1)
	vShape := shapes.Make(dtypes.Float32, 1, 1, 2, 1)
	maskShape := shapes.Make(dtypes.Float32, 1, 2)

	result := testBackendMultiInput(t,
		[]shapes.Shape{qShape, kShape, vShape, maskShape},
		[]any{q, k, v, mask},
		func(f backends.Function, params []backends.Value) (backends.Value, error) {
			return f.FusedMultiHeadSDPA(params[0], params[1], params[2], params[3], 1, 1, 1.0, false)
		},
	)

	got := result.flat.([]float32)
	// Only first position visible → output = V[0] = 10
	assert.InDelta(t, 10.0, got[0], fusedTestTol)
}

// ---- FusedQKVDense tests ----

func TestFusedQKVDense_Identity(t *testing.T) {
	// batch=1, inFeatures=3, qDim=2, kvDim=1
	// wQKV: [inFeatures, qDim+2*kvDim] = [3, 4]
	// Use identity-like weights for easy verification.
	x := []float32{1, 2, 3}
	// wQKV columns: Q[0], Q[1], K[0], V[0]
	// Row 0 (x[0]): contributes to Q[0]=1, Q[1]=0, K[0]=0, V[0]=1
	// Row 1 (x[1]): contributes to Q[0]=0, Q[1]=1, K[0]=0, V[0]=1
	// Row 2 (x[2]): contributes to Q[0]=0, Q[1]=0, K[0]=1, V[0]=1
	wQKV := []float32{
		1, 0, 0, 1, // row 0: Q[0]=1, Q[1]=0, K[0]=0, V[0]=1
		0, 1, 0, 1, // row 1: Q[0]=0, Q[1]=1, K[0]=0, V[0]=1
		0, 0, 1, 1, // row 2: Q[0]=0, Q[1]=0, K[0]=1, V[0]=1
	}
	biasQ := []float32{10, 20}
	biasK := []float32{100}
	biasV := []float32{1000}

	xShape := shapes.Make(dtypes.Float32, 1, 3)
	wShape := shapes.Make(dtypes.Float32, 3, 4)
	bqShape := shapes.Make(dtypes.Float32, 2)
	bkShape := shapes.Make(dtypes.Float32, 1)
	bvShape := shapes.Make(dtypes.Float32, 1)

	results := execFusedOpMultiOutput3(t,
		[]shapes.Shape{xShape, wShape, bqShape, bkShape, bvShape},
		[]any{x, wQKV, biasQ, biasK, biasV},
		func(f backends.Function, params []backends.Value) (backends.Value, backends.Value, backends.Value, error) {
			return f.FusedQKVDense(params[0], params[1], params[2], params[3], params[4], 2, 1)
		},
	)

	qGot := results[0].flat.([]float32)
	kGot := results[1].flat.([]float32)
	vGot := results[2].flat.([]float32)

	// Q = x @ wQ^T + biasQ = [1+10, 2+20] = [11, 22]
	assert.InDelta(t, 11.0, qGot[0], fusedTestTol)
	assert.InDelta(t, 22.0, qGot[1], fusedTestTol)
	// K = x @ wK^T + biasK = [3+100] = [103]
	assert.InDelta(t, 103.0, kGot[0], fusedTestTol)
	// V = x @ wV^T + biasV = [6+1000] = [1006]
	assert.InDelta(t, 1006.0, vGot[0], fusedTestTol)
}

func TestFusedQKVDense_NoBias(t *testing.T) {
	// batch=2, inFeatures=2, qDim=2, kvDim=1
	x := []float32{
		1, 0, // batch 0
		0, 1, // batch 1
	}
	// wQKV: [2, 4] (inFeatures=2, totalOut=4)
	// Columns: Q[0], Q[1], K[0], V[0]
	wQKV := []float32{
		1, 3, 5, 7, // row 0 (x[0]): Q[0]=1, Q[1]=3, K[0]=5, V[0]=7
		2, 4, 6, 8, // row 1 (x[1]): Q[0]=2, Q[1]=4, K[0]=6, V[0]=8
	}

	xShape := shapes.Make(dtypes.Float32, 2, 2)
	wShape := shapes.Make(dtypes.Float32, 2, 4)

	results := execFusedOpMultiOutput3(t,
		[]shapes.Shape{xShape, wShape},
		[]any{x, wQKV},
		func(f backends.Function, params []backends.Value) (backends.Value, backends.Value, backends.Value, error) {
			return f.FusedQKVDense(params[0], params[1], nil, nil, nil, 2, 1)
		},
	)

	qGot := results[0].flat.([]float32)
	kGot := results[1].flat.([]float32)
	vGot := results[2].flat.([]float32)

	// Batch 0: x=[1,0]
	// Q = [1*1+0*2, 1*3+0*4] = [1, 3]
	// K = [1*5+0*6] = [5]
	// V = [1*7+0*8] = [7]
	assert.InDelta(t, 1.0, qGot[0], fusedTestTol)
	assert.InDelta(t, 3.0, qGot[1], fusedTestTol)
	assert.InDelta(t, 5.0, kGot[0], fusedTestTol)
	assert.InDelta(t, 7.0, vGot[0], fusedTestTol)

	// Batch 1: x=[0,1]
	// Q = [0*1+1*2, 0*3+1*4] = [2, 4]
	// K = [0*5+1*6] = [6]
	// V = [0*7+1*8] = [8]
	assert.InDelta(t, 2.0, qGot[2], fusedTestTol)
	assert.InDelta(t, 4.0, qGot[3], fusedTestTol)
	assert.InDelta(t, 6.0, kGot[1], fusedTestTol)
	assert.InDelta(t, 8.0, vGot[1], fusedTestTol)
}

func TestFusedQKVDense_EqualDims(t *testing.T) {
	// When qDim == kvDim, equivalent to 3 separate dense ops.
	// batch=1, inFeatures=2, qDim=2, kvDim=2
	x := []float32{1, 1}
	// wQKV: [2, 6] (inFeatures=2, totalOut=qDim+2*kvDim=6)
	// Columns: Q[0], Q[1], K[0], K[1], V[0], V[1]
	wQKV := []float32{
		1, 0, 2, 0, 3, 0, // row 0 (x[0])
		0, 1, 0, 2, 0, 3, // row 1 (x[1])
	}

	xShape := shapes.Make(dtypes.Float32, 1, 2)
	wShape := shapes.Make(dtypes.Float32, 2, 6)

	results := execFusedOpMultiOutput3(t,
		[]shapes.Shape{xShape, wShape},
		[]any{x, wQKV},
		func(f backends.Function, params []backends.Value) (backends.Value, backends.Value, backends.Value, error) {
			return f.FusedQKVDense(params[0], params[1], nil, nil, nil, 2, 2)
		},
	)

	qGot := results[0].flat.([]float32)
	kGot := results[1].flat.([]float32)
	vGot := results[2].flat.([]float32)

	// x=[1,1]
	// Q = [1, 1], K = [2, 2], V = [3, 3]
	assert.InDelta(t, 1.0, qGot[0], fusedTestTol)
	assert.InDelta(t, 1.0, qGot[1], fusedTestTol)
	assert.InDelta(t, 2.0, kGot[0], fusedTestTol)
	assert.InDelta(t, 2.0, kGot[1], fusedTestTol)
	assert.InDelta(t, 3.0, vGot[0], fusedTestTol)
	assert.InDelta(t, 3.0, vGot[1], fusedTestTol)
}
