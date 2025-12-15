package simplego

import (
	"testing"

	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/stretchr/testify/require"

	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// naiveMatMul computes matrix multiplication naively for verification
func naiveMatMul(lhs []float32, rhs []float32, M, K, N int) []float32 {
	result := make([]float32, M*N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float32
			for k := 0; k < K; k++ {
				sum += lhs[i*K+k] * rhs[k*N+j]
			}
			result[i*N+j] = sum
		}
	}
	return result
}

// naiveBatchedMatMul computes batched matrix multiplication for [B, M, K] × [K, N] → [B, M, N]
func naiveBatchedMatMul(lhs []float32, rhs []float32, B, M, K, N int) []float32 {
	result := make([]float32, B*M*N)
	for b := 0; b < B; b++ {
		lhsOffset := b * M * K
		outputOffset := b * M * N
		for i := 0; i < M; i++ {
			for j := 0; j < N; j++ {
				var sum float32
				for k := 0; k < K; k++ {
					sum += lhs[lhsOffset+i*K+k] * rhs[k*N+j]
				}
				result[outputOffset+i*N+j] = sum
			}
		}
	}
	return result
}

// naiveBatchedMatMulFloat64 computes batched matrix multiplication for float64
func naiveBatchedMatMulFloat64(lhs []float64, rhs []float64, B, M, K, N int) []float64 {
	result := make([]float64, B*M*N)
	for b := 0; b < B; b++ {
		lhsOffset := b * M * K
		outputOffset := b * M * N
		for i := 0; i < M; i++ {
			for j := 0; j < N; j++ {
				var sum float64
				for k := 0; k < K; k++ {
					sum += lhs[lhsOffset+i*K+k] * rhs[k*N+j]
				}
				result[outputOffset+i*N+j] = sum
			}
		}
	}
	return result
}

// TestPreBlockedWeights_Unbatched tests the pre-blocked weights path for
// standard 2D matmul: [M, K] × [K, N] → [M, N]
func TestPreBlockedWeights_Unbatched(t *testing.T) {
	be, ok := backend.(*Backend)
	if !ok {
		t.Skip("Skipping test because backend is not a SimpleGo Backend")
	}

	M, K, N := 64, 128, 96 // Sizes that work well with blocking

	// Create LHS [M, K] and RHS [K, N]
	lhsShape := shapes.Make(dtypes.Float32, M, K)
	rhsShape := shapes.Make(dtypes.Float32, K, N)
	// Output for pre-blocked path uses normalized shape [batchSize, lhsCrossSize, rhsCrossSize]
	normalizedOutputShape := shapes.Make(dtypes.Float32, 1, M, N)

	lhs := be.NewBuffer(lhsShape)
	rhs := be.NewBuffer(rhsShape)

	// Fill with test data
	lhsFlat := lhs.flat.([]float32)
	rhsFlat := rhs.flat.([]float32)
	for i := range lhsFlat {
		lhsFlat[i] = float32(i%10) * 0.1
	}
	for i := range rhsFlat {
		rhsFlat[i] = float32(i%10) * 0.1
	}

	// Compute expected result using naive implementation
	expected := naiveMatMul(lhsFlat, rhsFlat, M, K, N)

	// Set up params for 2D matmul
	params := &dotGeneralNodeData{
		lhsContractingAxes: []int{1},
		rhsContractingAxes: []int{0},
		lhsBatchAxes:       []int{},
		rhsBatchAxes:       []int{},
		batchSize:          1,
		lhsCrossSize:       M,
		rhsCrossSize:       N,
		contractingSize:    K,
	}
	blockLog2Dim := DotGeneralTargetBlockLog2Dim[dtypes.Float32]
	params.lhsBlockedShape = dgCreateBlockedShape(dtypes.Float32, 1, M, K, blockLog2Dim)
	params.rhsBlockedShape = dgCreateBlockedShape(dtypes.Float32, 1, N, K, blockLog2Dim)
	params.outputBlockedShape = dgCreateBlockedShape(dtypes.Float32, 1, M, N, blockLog2Dim)

	// Test with pre-blocked weights
	pbw := PreBlockWeightForMatMul(rhs)
	require.NotNil(t, pbw, "Pre-blocking should succeed for [K, N] weight")

	actualOutput := be.NewBuffer(normalizedOutputShape)
	actualOutput.Zeros()
	err := execDotGeneralWithPreBlockedRHS(be, lhs, pbw, params, actualOutput)
	require.NoError(t, err)

	// Compare results
	actualFlat := actualOutput.flat.([]float32)
	for i := range expected {
		require.InDelta(t, expected[i], actualFlat[i], 1e-4,
			"Mismatch at index %d: expected %f, got %f", i, expected[i], actualFlat[i])
	}
}

// TestPreBlockedWeights_Batched tests the pre-blocked weights path for
// batched matmul: [B, M, K] × [K, N] → [B, M, N]
// This is the common pattern in neural networks where weights are shared across batch.
func TestPreBlockedWeights_Batched(t *testing.T) {
	be, ok := backend.(*Backend)
	if !ok {
		t.Skip("Skipping test because backend is not a SimpleGo Backend")
	}

	B, M, K, N := 4, 32, 64, 48 // Batch size 4

	// Create LHS [B, M, K] and RHS [K, N]
	lhsShape := shapes.Make(dtypes.Float32, B, M, K)
	rhsShape := shapes.Make(dtypes.Float32, K, N)
	// Output uses normalized shape [batchSize, lhsCrossSize, rhsCrossSize]
	outputShape := shapes.Make(dtypes.Float32, B, M, N)

	lhs := be.NewBuffer(lhsShape)
	rhs := be.NewBuffer(rhsShape)

	// Fill with test data
	lhsFlat := lhs.flat.([]float32)
	rhsFlat := rhs.flat.([]float32)
	for i := range lhsFlat {
		lhsFlat[i] = float32(i%10) * 0.1
	}
	for i := range rhsFlat {
		rhsFlat[i] = float32(i%10) * 0.1
	}

	// Compute expected result using naive implementation
	expected := naiveBatchedMatMul(lhsFlat, rhsFlat, B, M, K, N)

	// Set up params for batched matmul
	// LHS shape [B, M, K]: batch axis 0, cross axis 1, contracting axis 2
	// RHS shape [K, N]: no batch, cross axis 1, contracting axis 0
	params := &dotGeneralNodeData{
		lhsContractingAxes: []int{2}, // Last axis of LHS
		rhsContractingAxes: []int{0}, // First axis of RHS
		lhsBatchAxes:       []int{0}, // First axis of LHS is batch
		rhsBatchAxes:       []int{},  // RHS has no batch (shared weights)
		batchSize:          B,
		lhsCrossSize:       M,
		rhsCrossSize:       N,
		contractingSize:    K,
	}
	blockLog2Dim := DotGeneralTargetBlockLog2Dim[dtypes.Float32]
	params.lhsBlockedShape = dgCreateBlockedShape(dtypes.Float32, B, M, K, blockLog2Dim)
	// RHS blocked shape has batchSize=1 since weights are shared
	params.rhsBlockedShape = dgCreateBlockedShape(dtypes.Float32, 1, N, K, blockLog2Dim)
	params.outputBlockedShape = dgCreateBlockedShape(dtypes.Float32, B, M, N, blockLog2Dim)

	// Now test with pre-blocked weights
	pbw := PreBlockWeightForMatMul(rhs)
	require.NotNil(t, pbw, "Pre-blocking should succeed for [K, N] weight")

	// Verify the pre-blocked path is used
	require.True(t, canUsePreBlockedPath(lhs, rhs, params),
		"canUsePreBlockedPath should return true for batched LHS with unbatched RHS")
	require.True(t, CanUsePreBlockedWeight(pbw, rhsShape, params),
		"CanUsePreBlockedWeight should return true for batched matmul")

	actualOutput := be.NewBuffer(outputShape)
	actualOutput.Zeros()
	err := execDotGeneralWithPreBlockedRHS(be, lhs, pbw, params, actualOutput)
	require.NoError(t, err)

	// Compare results
	actualFlat := actualOutput.flat.([]float32)
	for i := range expected {
		require.InDelta(t, expected[i], actualFlat[i], 1e-4,
			"Mismatch at index %d: expected %f, got %f", i, expected[i], actualFlat[i])
	}
}

// TestPreBlockedWeights_LargeBatch tests with a larger batch size to ensure
// parallelization across batches works correctly.
func TestPreBlockedWeights_LargeBatch(t *testing.T) {
	be, ok := backend.(*Backend)
	if !ok {
		t.Skip("Skipping test because backend is not a SimpleGo Backend")
	}

	B, M, K, N := 16, 64, 128, 64 // Larger batch

	lhsShape := shapes.Make(dtypes.Float32, B, M, K)
	rhsShape := shapes.Make(dtypes.Float32, K, N)
	outputShape := shapes.Make(dtypes.Float32, B, M, N)

	lhs := be.NewBuffer(lhsShape)
	rhs := be.NewBuffer(rhsShape)

	lhsFlat := lhs.flat.([]float32)
	rhsFlat := rhs.flat.([]float32)
	for i := range lhsFlat {
		lhsFlat[i] = float32(i%7) * 0.1
	}
	for i := range rhsFlat {
		rhsFlat[i] = float32(i%11) * 0.1
	}

	// Compute expected result using naive implementation
	expected := naiveBatchedMatMul(lhsFlat, rhsFlat, B, M, K, N)

	params := &dotGeneralNodeData{
		lhsContractingAxes: []int{2},
		rhsContractingAxes: []int{0},
		lhsBatchAxes:       []int{0},
		rhsBatchAxes:       []int{},
		batchSize:          B,
		lhsCrossSize:       M,
		rhsCrossSize:       N,
		contractingSize:    K,
	}
	blockLog2Dim := DotGeneralTargetBlockLog2Dim[dtypes.Float32]
	params.lhsBlockedShape = dgCreateBlockedShape(dtypes.Float32, B, M, K, blockLog2Dim)
	params.rhsBlockedShape = dgCreateBlockedShape(dtypes.Float32, 1, N, K, blockLog2Dim)
	params.outputBlockedShape = dgCreateBlockedShape(dtypes.Float32, B, M, N, blockLog2Dim)

	// Test with pre-blocked weights
	pbw := PreBlockWeightForMatMul(rhs)
	require.NotNil(t, pbw)

	actualOutput := be.NewBuffer(outputShape)
	actualOutput.Zeros()
	err := execDotGeneralWithPreBlockedRHS(be, lhs, pbw, params, actualOutput)
	require.NoError(t, err)

	// Compare results
	actualFlat := actualOutput.flat.([]float32)
	for i := range expected {
		require.InDelta(t, expected[i], actualFlat[i], 1e-4,
			"Mismatch at index %d: expected %f, got %f", i, expected[i], actualFlat[i])
	}
}

// TestPreBlockedWeights_CacheInvalidation tests that cache invalidation works.
func TestPreBlockedWeights_CacheInvalidation(t *testing.T) {
	be, ok := backend.(*Backend)
	if !ok {
		t.Skip("Skipping test because backend is not a SimpleGo Backend")
	}

	K, N := 64, 32
	rhsShape := shapes.Make(dtypes.Float32, K, N)
	rhs := be.NewBuffer(rhsShape)

	// Pre-block the weight
	pbw := be.preBlockedWeightCache.GetOrCreate(rhs)
	require.NotNil(t, pbw)

	// Verify it's in the cache
	cached := be.preBlockedWeightCache.Get(rhs)
	require.NotNil(t, cached)
	require.Equal(t, pbw, cached)

	// Invalidate
	be.preBlockedWeightCache.Invalidate(rhs)

	// Verify it's gone
	cached = be.preBlockedWeightCache.Get(rhs)
	require.Nil(t, cached)

	// Clear should also work
	pbw = be.preBlockedWeightCache.GetOrCreate(rhs)
	require.NotNil(t, pbw)
	be.preBlockedWeightCache.Clear()
	cached = be.preBlockedWeightCache.Get(rhs)
	require.Nil(t, cached)
}

// TestPreBlockedWeights_Float64 tests pre-blocked weights with float64.
func TestPreBlockedWeights_Float64(t *testing.T) {
	be, ok := backend.(*Backend)
	if !ok {
		t.Skip("Skipping test because backend is not a SimpleGo Backend")
	}

	B, M, K, N := 2, 32, 48, 24

	lhsShape := shapes.Make(dtypes.Float64, B, M, K)
	rhsShape := shapes.Make(dtypes.Float64, K, N)
	outputShape := shapes.Make(dtypes.Float64, B, M, N)

	lhs := be.NewBuffer(lhsShape)
	rhs := be.NewBuffer(rhsShape)

	lhsFlat := lhs.flat.([]float64)
	rhsFlat := rhs.flat.([]float64)
	for i := range lhsFlat {
		lhsFlat[i] = float64(i%10) * 0.1
	}
	for i := range rhsFlat {
		rhsFlat[i] = float64(i%10) * 0.1
	}

	// Compute expected result using naive implementation
	expected := naiveBatchedMatMulFloat64(lhsFlat, rhsFlat, B, M, K, N)

	params := &dotGeneralNodeData{
		lhsContractingAxes: []int{2},
		rhsContractingAxes: []int{0},
		lhsBatchAxes:       []int{0},
		rhsBatchAxes:       []int{},
		batchSize:          B,
		lhsCrossSize:       M,
		rhsCrossSize:       N,
		contractingSize:    K,
	}
	blockLog2Dim := DotGeneralTargetBlockLog2Dim[dtypes.Float64]
	params.lhsBlockedShape = dgCreateBlockedShape(dtypes.Float64, B, M, K, blockLog2Dim)
	params.rhsBlockedShape = dgCreateBlockedShape(dtypes.Float64, 1, N, K, blockLog2Dim)
	params.outputBlockedShape = dgCreateBlockedShape(dtypes.Float64, B, M, N, blockLog2Dim)

	// Test with pre-blocked weights
	pbw := PreBlockWeightForMatMul(rhs)
	require.NotNil(t, pbw)

	actualOutput := be.NewBuffer(outputShape)
	actualOutput.Zeros()
	err := execDotGeneralWithPreBlockedRHS(be, lhs, pbw, params, actualOutput)
	require.NoError(t, err)

	// Compare results
	actualFlat := actualOutput.flat.([]float64)
	for i := range expected {
		require.InDelta(t, expected[i], actualFlat[i], 1e-10,
			"Mismatch at index %d: expected %f, got %f", i, expected[i], actualFlat[i])
	}
}
