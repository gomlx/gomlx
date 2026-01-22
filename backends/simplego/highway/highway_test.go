// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package highway

import (
	"testing"

	"github.com/gomlx/gomlx/backends/simplego"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHighwayRegistration(t *testing.T) {
	// Verify that importing this package registers the highway implementation
	assert.True(t, simplego.Highway.HasDTypeSupport(dtypes.Float32, dtypes.Float32),
		"Highway should support Float32 after import")
	assert.True(t, simplego.Highway.HasDTypeSupport(dtypes.Float64, dtypes.Float64),
		"Highway should support Float64 after import")
	assert.True(t, simplego.Highway.HasDTypeSupport(dtypes.Float16, dtypes.Float16),
		"Highway should support Float16 after import")
	assert.True(t, simplego.Highway.HasDTypeSupport(dtypes.BFloat16, dtypes.BFloat16),
		"Highway should support BFloat16 after import")
}

func TestHighwayMatMul(t *testing.T) {
	// Simple 2x2 matrix multiplication test
	// A = [[1, 2], [3, 4]]
	// B = [[5, 6], [7, 8]]
	// C = A * B = [[19, 22], [43, 50]]
	lhs := []float32{1, 2, 3, 4}
	rhs := []float32{5, 6, 7, 8}
	output := make([]float32, 4)

	err := simplego.Highway.MatMulDynamic(
		dtypes.Float32, dtypes.Float32,
		lhs, rhs,
		1,    // batchSize
		2,    // lhsCrossSize (M)
		2,    // rhsCrossSize (N)
		2,    // contractingSize (K)
		output,
		nil, nil, nil,
	)
	require.NoError(t, err)

	expected := []float32{19, 22, 43, 50}
	assert.InDeltaSlice(t, expected, output, 1e-5, "MatMul result mismatch")
}
