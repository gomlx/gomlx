// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package highway

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/simplego"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/stretchr/testify/assert"
)

// backend is initialized in TestMain after highway registers itself.
var backend *simplego.Backend

func setup() {
	fmt.Printf("Available backends: %q\n", backends.List())
	// Force use of the simplego backend
	if os.Getenv(backends.ConfigEnvVar) == "" {
		must.M(os.Setenv(backends.ConfigEnvVar, "go"))
	}
	b := backends.MustNew()
	backend = b.(*simplego.Backend)
	fmt.Printf("Backend: %s, %s\n", backend.Name(), backend.Description())
}

func teardown() {
	backend.Finalize()
}

func TestMain(m *testing.M) {
	setup()
	code := m.Run()
	teardown()
	os.Exit(code)
}

func TestHighwayUnaryExp(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Exp)

	// Test float32
	y0 := exec.MustExec(float32(1.0))[0]
	assert.InDelta(t, float32(math.E), y0.Value(), 1e-5)

	// Test with array
	y1 := exec.MustExec([]float32{0.0, 1.0, -1.0, 2.0})[0]
	got := y1.Value().([]float32)
	want := []float32{1.0, float32(math.E), float32(1.0 / math.E), float32(math.E * math.E)}
	for i := range want {
		assert.InDelta(t, want[i], got[i], 1e-5, "Exp mismatch at index %d", i)
	}

	// Test float64 (SIMD has ~4 ULP error, so use looser tolerance)
	y2 := exec.MustExec(float64(1.0))[0]
	assert.InDelta(t, math.E, y2.Value(), 1e-6)

	// Test bfloat16
	y3 := exec.MustExec(bfloat16.FromFloat32(1.0))[0]
	assert.InDelta(t, float32(math.E), y3.Value().(bfloat16.BFloat16).Float32(), 1e-2)
}

func TestHighwayUnaryLog(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Log)

	// Test float32
	y0 := exec.MustExec(float32(math.E))[0]
	assert.InDelta(t, float32(1.0), y0.Value(), 1e-5)

	// Test with array
	y1 := exec.MustExec([]float32{1.0, float32(math.E), 10.0, 0.1})[0]
	got := y1.Value().([]float32)
	want := []float32{0.0, 1.0, float32(math.Log(10.0)), float32(math.Log(0.1))}
	for i := range want {
		assert.InDelta(t, want[i], got[i], 1e-5, "Log mismatch at index %d", i)
	}

	// Test float64 (SIMD has ~4 ULP error, so use looser tolerance)
	y2 := exec.MustExec(math.E)[0]
	assert.InDelta(t, 1.0, y2.Value(), 1e-6)
}

func TestHighwayUnarySin(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Sin)

	// Test float32
	y0 := exec.MustExec(float32(math.Pi / 2))[0]
	assert.InDelta(t, float32(1.0), y0.Value(), 1e-5)

	// Test with array
	y1 := exec.MustExec([]float32{0.0, float32(math.Pi / 2), float32(math.Pi), float32(-math.Pi / 2)})[0]
	got := y1.Value().([]float32)
	want := []float32{0.0, 1.0, 0.0, -1.0}
	for i := range want {
		assert.InDelta(t, want[i], got[i], 1e-5, "Sin mismatch at index %d", i)
	}

	// Test float64 (SIMD has ~4 ULP error, so use looser tolerance)
	y2 := exec.MustExec(math.Pi / 2)[0]
	assert.InDelta(t, 1.0, y2.Value(), 1e-6)
}

func TestHighwayUnaryCos(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Cos)

	// Test float32
	y0 := exec.MustExec(float32(0.0))[0]
	assert.InDelta(t, float32(1.0), y0.Value(), 1e-5)

	// Test with array
	y1 := exec.MustExec([]float32{0.0, float32(math.Pi / 2), float32(math.Pi)})[0]
	got := y1.Value().([]float32)
	want := []float32{1.0, 0.0, -1.0}
	for i := range want {
		assert.InDelta(t, want[i], got[i], 1e-5, "Cos mismatch at index %d", i)
	}

	// Test float64 (SIMD has ~4 ULP error, so use looser tolerance)
	y2 := exec.MustExec(0.0)[0]
	assert.InDelta(t, 1.0, y2.Value(), 1e-6)
}

func TestHighwayUnaryTanh(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Tanh)

	// Test float32
	y0 := exec.MustExec(float32(0.0))[0]
	assert.InDelta(t, float32(0.0), y0.Value(), 1e-5)

	// Test with array
	y1 := exec.MustExec([]float32{0.0, 1.0, -1.0, 2.0})[0]
	got := y1.Value().([]float32)
	want := []float32{0.0, float32(math.Tanh(1.0)), float32(math.Tanh(-1.0)), float32(math.Tanh(2.0))}
	for i := range want {
		assert.InDelta(t, want[i], got[i], 1e-5, "Tanh mismatch at index %d", i)
	}

	// Test float64 (SIMD has ~4 ULP error, so use looser tolerance)
	y2 := exec.MustExec(1.0)[0]
	assert.InDelta(t, math.Tanh(1.0), y2.Value(), 1e-6)
}

func TestHighwayUnaryLogistic(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Logistic)

	// Test float32
	y0 := exec.MustExec(float32(0.0))[0]
	assert.InDelta(t, float32(0.5), y0.Value(), 1e-5)

	// Test with array
	y1 := exec.MustExec([]float32{0.0, 1.0, -1.0, 5.0, -5.0})[0]
	got := y1.Value().([]float32)
	sigmoid := func(x float64) float32 { return float32(1.0 / (1.0 + math.Exp(-x))) }
	want := []float32{sigmoid(0), sigmoid(1), sigmoid(-1), sigmoid(5), sigmoid(-5)}
	for i := range want {
		assert.InDelta(t, want[i], got[i], 1e-5, "Logistic mismatch at index %d", i)
	}

	// Test float64 (SIMD has ~4 ULP error, so use looser tolerance)
	y2 := exec.MustExec(0.0)[0]
	assert.InDelta(t, 0.5, y2.Value(), 1e-6)
}

func TestHighwayUnaryErf(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Erf)

	// Test float32
	y0 := exec.MustExec(float32(1.0))[0]
	assert.InDelta(t, float32(math.Erf(1.0)), y0.Value(), 1e-4)

	// Test with array
	y1 := exec.MustExec([]float32{0.0, 0.5, 1.0, -0.5, -1.0})[0]
	got := y1.Value().([]float32)
	want := []float32{0.0, float32(math.Erf(0.5)), float32(math.Erf(1.0)), float32(math.Erf(-0.5)), float32(math.Erf(-1.0))}
	for i := range want {
		assert.InDelta(t, want[i], got[i], 1e-4, "Erf mismatch at index %d", i)
	}

	// Test float64 (SIMD has ~4 ULP error, so use looser tolerance)
	y2 := exec.MustExec(1.0)[0]
	assert.InDelta(t, math.Erf(1.0), y2.Value(), 1e-6)
}

// TestHighwayUnaryLargeArray tests that SIMD operations work correctly with larger arrays
// where vectorization provides real benefits.
func TestHighwayUnaryLargeArray(t *testing.T) {
	const size = 1024
	input := make([]float32, size)
	for i := range input {
		input[i] = float32(i) * 0.01
	}

	exec := graph.MustNewExec(backend, graph.Exp)
	y := exec.MustExec(input)[0]
	got := y.Value().([]float32)

	for i := range input {
		expected := float32(math.Exp(float64(input[i])))
		// Use relative tolerance for large values, absolute for small values
		relTol := float64(expected) * 1e-5
		absTol := 1e-5
		tol := relTol
		if absTol > tol {
			tol = absTol
		}
		assert.InDelta(t, expected, got[i], tol, "Exp mismatch at index %d for input %v", i, input[i])
	}
}
