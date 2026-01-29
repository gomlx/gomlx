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
	"github.com/gomlx/gomlx/pkg/core/dtypes"
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

// TestHighwayTranspose2D tests that SIMD transpose works correctly.
func TestHighwayTranspose2D(t *testing.T) {
	// Test float32 transpose
	m, k := 4, 8
	src := make([]float32, m*k)
	for i := range src {
		src[i] = float32(i)
	}
	dst := make([]float32, k*m)

	// Transpose using highway
	ok := Transpose2D(dtypes.Float32, src, m, k, dst)
	assert.True(t, ok, "Transpose2D should succeed for float32")

	// Verify: dst[j*m + i] should equal src[i*k + j]
	for i := 0; i < m; i++ {
		for j := 0; j < k; j++ {
			expected := src[i*k+j]
			got := dst[j*m+i]
			assert.Equal(t, expected, got, "Mismatch at (%d,%d)", i, j)
		}
	}
}

// TestHighwayDotGeneralWithTranspose tests DotGeneral when matrices need transposing.
func TestHighwayDotGeneralWithTranspose(t *testing.T) {
	// Create a DotGeneral where RHS needs transpose:
	// LHS: [2, 3] (M=2, K=3) - contracting axis is last, OK
	// RHS: [4, 3] (N=4, K=3) - contracting axis is last, needs transpose to [3, 4]
	// Result: [2, 4]

	lhsData := []float32{1, 2, 3, 4, 5, 6}          // 2x3
	rhsData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} // 4x3

	// Expected result: LHS @ RHS^T
	// [1,2,3] @ [1,4,7,10]^T = 1*1 + 2*2 + 3*3 = 14
	//                         1*4 + 2*5 + 3*6 = 32
	//                         1*7 + 2*8 + 3*9 = 50
	//                         1*10 + 2*11 + 3*12 = 68
	// [4,5,6] @ ... = 32, 77, 122, 167

	execFn := func(g *graph.Graph) *graph.Node {
		lhs := graph.Reshape(graph.Const(g, lhsData), 2, 3)
		rhs := graph.Reshape(graph.Const(g, rhsData), 4, 3)
		// DotGeneral with RHS having contracting axis as last (not first)
		// LHS: contracting=1 (last), batch=[]
		// RHS: contracting=1 (last, needs transpose), batch=[]
		return graph.Einsum("mk,nk->mn", lhs, rhs)
	}

	exec := graph.MustNewExec(backend, execFn)
	result := exec.MustExec()[0]
	got := result.Value().([][]float32)

	// Expected result is 2x4 matrix
	want := [][]float32{
		{14, 32, 50, 68},
		{32, 77, 122, 167},
	}
	for i := range want {
		for j := range want[i] {
			assert.InDelta(t, want[i][j], got[i][j], 1e-4, "Result mismatch at (%d,%d)", i, j)
		}
	}
}

// TestHighwayNormalizedPathWithSIMDTranspose tests that the normalized DotGeneral path
// uses SIMD transpose when the operand needs a simple 2D transpose.
// This exercises the dgNormalizeShape SIMD fast path.
func TestHighwayNormalizedPathWithSIMDTranspose(t *testing.T) {
	// Test case: LHS needs transpose [K, M] -> [M, K]
	// LHS: [3, 2] with contracting axis 0 (first)
	// RHS: [3, 4] with contracting axis 0 (first, standard order)
	// Result: [2, 4]
	//
	// This forces LHS through dgNormalizeShape which should use SIMD transpose.

	lhsData := []float32{1, 2, 3, 4, 5, 6} // [3, 2] = K x M
	rhsData := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	} // [3, 4] = K x N

	execFn := func(g *graph.Graph) *graph.Node {
		lhs := graph.Reshape(graph.Const(g, lhsData), 3, 2) // [K=3, M=2]
		rhs := graph.Reshape(graph.Const(g, rhsData), 3, 4) // [K=3, N=4]
		// Einsum "km,kn->mn" means:
		//   - LHS contracting axis is 0 (k)
		//   - RHS contracting axis is 0 (k)
		//   - Result has dimensions [m, n]
		// LHS is [K, M] and needs transpose to [M, K] for normalized path.
		return graph.Einsum("km,kn->mn", lhs, rhs)
	}

	exec := graph.MustNewExec(backend, execFn)
	result := exec.MustExec()[0]
	got := result.Value().([][]float32)

	// Manual calculation: LHS^T @ RHS
	// LHS^T = [[1, 3, 5], [2, 4, 6]] (2x3)
	// RHS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] (3x4)
	// Result[i,j] = sum_k LHS^T[i,k] * RHS[k,j]
	// Result[0,0] = 1*1 + 3*5 + 5*9 = 1 + 15 + 45 = 61
	// Result[0,1] = 1*2 + 3*6 + 5*10 = 2 + 18 + 50 = 70
	// Result[0,2] = 1*3 + 3*7 + 5*11 = 3 + 21 + 55 = 79
	// Result[0,3] = 1*4 + 3*8 + 5*12 = 4 + 24 + 60 = 88
	// Result[1,0] = 2*1 + 4*5 + 6*9 = 2 + 20 + 54 = 76
	// Result[1,1] = 2*2 + 4*6 + 6*10 = 4 + 24 + 60 = 88
	// Result[1,2] = 2*3 + 4*7 + 6*11 = 6 + 28 + 66 = 100
	// Result[1,3] = 2*4 + 4*8 + 6*12 = 8 + 32 + 72 = 112
	want := [][]float32{
		{61, 70, 79, 88},
		{76, 88, 100, 112},
	}

	for i := range want {
		for j := range want[i] {
			assert.InDelta(t, want[i][j], got[i][j], 1e-4, "Result mismatch at (%d,%d)", i, j)
		}
	}
}

// TestHighwayNormalizedPathBatchedTranspose tests batched SIMD transpose in dgNormalizeShape.
func TestHighwayNormalizedPathBatchedTranspose(t *testing.T) {
	// Test case: Batched LHS needs transpose [B, K, M] -> [B, M, K]
	// LHS: [2, 3, 2] with batch axis 0, contracting axis 1 (first non-batch)
	// RHS: [2, 3, 4] with batch axis 0, contracting axis 1 (first non-batch, standard)
	// Result: [2, 2, 4]

	lhsData := []float32{
		// Batch 0: [3, 2]
		1, 2, 3, 4, 5, 6,
		// Batch 1: [3, 2]
		7, 8, 9, 10, 11, 12,
	}
	rhsData := []float32{
		// Batch 0: [3, 4]
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		// Batch 1: [3, 4]
		13, 14, 15, 16,
		17, 18, 19, 20,
		21, 22, 23, 24,
	}

	execFn := func(g *graph.Graph) *graph.Node {
		lhs := graph.Reshape(graph.Const(g, lhsData), 2, 3, 2) // [B=2, K=3, M=2]
		rhs := graph.Reshape(graph.Const(g, rhsData), 2, 3, 4) // [B=2, K=3, N=4]
		// Einsum "bkm,bkn->bmn"
		return graph.Einsum("bkm,bkn->bmn", lhs, rhs)
	}

	exec := graph.MustNewExec(backend, execFn)
	result := exec.MustExec()[0]
	got := result.Value().([][][]float32)

	// Batch 0: same as unbatched test
	want := [][][]float32{
		{
			{61, 70, 79, 88},
			{76, 88, 100, 112},
		},
		{
			// Batch 1: LHS^T @ RHS
			// LHS^T = [[7, 9, 11], [8, 10, 12]] (2x3)
			// RHS = [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]] (3x4)
			// Result[0,0] = 7*13 + 9*17 + 11*21 = 91 + 153 + 231 = 475
			// Result[0,1] = 7*14 + 9*18 + 11*22 = 98 + 162 + 242 = 502
			// Result[1,0] = 8*13 + 10*17 + 12*21 = 104 + 170 + 252 = 526
			// etc.
			{475, 502, 529, 556},
			{526, 556, 586, 616},
		},
	}

	for b := range want {
		for i := range want[b] {
			for j := range want[b][i] {
				assert.InDelta(t, want[b][i][j], got[b][i][j], 1e-4, "Result mismatch at batch %d (%d,%d)", b, i, j)
			}
		}
	}
}

// TestHighwayLinearLayerPattern tests the common linear layer pattern:
// [batch, seq, features] @ [features, hidden] → [batch, seq, hidden]
// This has LHS with 3 dims (multi-cross) and RHS with 2 dims.
func TestHighwayLinearLayerPattern(t *testing.T) {
	// Linear layer: input [1, 11, 1024] @ weights [1024, 1024] → output [1, 11, 1024]
	// This is the pattern that was failing before the fix.

	inputData := make([]float32, 1*11*1024)
	weightsData := make([]float32, 1024*1024)
	for i := range inputData {
		inputData[i] = float32(i%100) * 0.01
	}
	for i := range weightsData {
		weightsData[i] = float32(i%100) * 0.001
	}

	execFn := func(g *graph.Graph) *graph.Node {
		input := graph.Reshape(graph.Const(g, inputData), 1, 11, 1024)
		weights := graph.Reshape(graph.Const(g, weightsData), 1024, 1024)
		// This is: input @ weights, contracting on last dim of input and first dim of weights
		return graph.Einsum("bsk,kh->bsh", input, weights)
	}

	exec := graph.MustNewExec(backend, execFn)
	result := exec.MustExec()[0]
	got := result.Value().([][][]float32)

	// Verify shape
	assert.Equal(t, 1, len(got), "batch dim")
	assert.Equal(t, 11, len(got[0]), "seq dim")
	assert.Equal(t, 1024, len(got[0][0]), "hidden dim")

	// Verify the result is non-zero (basic sanity check)
	assert.NotEqual(t, float32(0), got[0][0][0], "result should be non-zero")
}

// TestHighwayDenseLayerPyTorchWeights tests the dense layer pattern with PyTorch-style weights:
// "bsi,oi->bso" - LHS has contracting axis last, RHS has contracting axis last (not first!)
// This is the pattern that caused the original regression when highway needed to transpose RHS.
// With MatMulKLast, we can handle this directly without transpose.
func TestHighwayDenseLayerPyTorchWeights(t *testing.T) {
	// Dense layer: input [batch, seq, in] @ weights [out, in] → output [batch, seq, out]
	// Note: PyTorch stores weights as [out, in], not [in, out]
	// This means the Einsum is "bsi,oi->bso" where contracting axis is last in both!
	batch, seq, inFeatures, outFeatures := 4, 128, 768, 3072 // Realistic transformer MLP sizes

	inputData := make([]float32, batch*seq*inFeatures)
	weightsData := make([]float32, outFeatures*inFeatures) // [out, in] PyTorch format
	for i := range inputData {
		inputData[i] = float32(i%100) * 0.01
	}
	for i := range weightsData {
		weightsData[i] = float32(i%100) * 0.001
	}

	execFn := func(g *graph.Graph) *graph.Node {
		input := graph.Reshape(graph.Const(g, inputData), batch, seq, inFeatures)
		weights := graph.Reshape(graph.Const(g, weightsData), outFeatures, inFeatures)
		// Einsum "bsi,oi->bso": contract on 'i' (last dim of both)
		return graph.Einsum("bsi,oi->bso", input, weights)
	}

	exec := graph.MustNewExec(backend, execFn)
	result := exec.MustExec()[0]
	got := result.Value().([][][]float32)

	// Verify shape
	assert.Equal(t, batch, len(got), "batch dim")
	assert.Equal(t, seq, len(got[0]), "seq dim")
	assert.Equal(t, outFeatures, len(got[0][0]), "out dim")

	// Verify the result is non-zero (basic sanity check)
	assert.NotEqual(t, float32(0), got[0][0][0], "result should be non-zero")
}

// TestHighwayBERTAttentionPattern tests the exact pattern used in BERT attention:
// "bhqd,bhkd->bhqk" - both LHS and RHS have contracting axis as last dimension.
func TestHighwayBERTAttentionPattern(t *testing.T) {
	// BERT attention scores: query @ key^T
	// query: [batch, heads, query_len, depth] = [1, 2, 3, 4]
	// key:   [batch, heads, key_len, depth] = [1, 2, 5, 4]
	// result: [batch, heads, query_len, key_len] = [1, 2, 3, 5]
	//
	// Einsum "bhqd,bhkd->bhqk" means:
	//   - batch axes: [0, 1] (b, h)
	//   - LHS contracting: 3 (d)
	//   - RHS contracting: 3 (d)

	// Simple values for verification
	queryData := make([]float32, 1*2*3*4) // [1, 2, 3, 4]
	keyData := make([]float32, 1*2*5*4)   // [1, 2, 5, 4]
	for i := range queryData {
		queryData[i] = float32(i + 1)
	}
	for i := range keyData {
		keyData[i] = float32(i + 1)
	}

	execFn := func(g *graph.Graph) *graph.Node {
		query := graph.Reshape(graph.Const(g, queryData), 1, 2, 3, 4)
		key := graph.Reshape(graph.Const(g, keyData), 1, 2, 5, 4)
		// This is BERT's attention score pattern
		return graph.Einsum("bhqd,bhkd->bhqk", query, key)
	}

	exec := graph.MustNewExec(backend, execFn)
	result := exec.MustExec()[0]
	got := result.Value().([][][][]float32)

	// Verify shape
	assert.Equal(t, 1, len(got), "batch dim")
	assert.Equal(t, 2, len(got[0]), "heads dim")
	assert.Equal(t, 3, len(got[0][0]), "query_len dim")
	assert.Equal(t, 5, len(got[0][0][0]), "key_len dim")

	// Verify a few values manually
	// For batch=0, head=0, query[0] @ key[0..4]^T
	// query[0,0,0,:] = [1, 2, 3, 4]
	// key[0,0,0,:] = [1, 2, 3, 4] → dot = 1+4+9+16 = 30
	// key[0,0,1,:] = [5, 6, 7, 8] → dot = 5+12+21+32 = 70
	assert.InDelta(t, float32(30), got[0][0][0][0], 1e-4, "score[0,0,0,0]")
	assert.InDelta(t, float32(70), got[0][0][0][1], 1e-4, "score[0,0,0,1]")
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
