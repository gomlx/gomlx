// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"github.com/gomlx/gomlx/backends/simplego/packgemm"
	"github.com/gomlx/gomlx/internal/workerspool"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/pkg/errors"
)

// HighwayMatMul defines the interface for highway-accelerated matrix multiplication.
// The default implementation returns "not supported" for all dtypes.
// To enable highway support, import the highway submodule which requires Go 1.26+:
//
//	import _ "github.com/gomlx/gomlx/backends/simplego/highway"
type HighwayMatMul interface {
	// HasDTypeSupport returns true if MatMulDynamic is available for the given dtypes.
	HasDTypeSupport(input, output dtypes.DType) bool

	// MatMulDynamic performs batched matrix multiplication.
	// C = A * B where:
	//   - A is [batchSize, lhsCrossSize, contractingSize] (M x K per batch)
	//   - B is [batchSize, contractingSize, rhsCrossSize] (K x N per batch)
	//   - C is [batchSize, lhsCrossSize, rhsCrossSize] (M x N per batch)
	MatMulDynamic(inputDType, outputDType dtypes.DType,
		lhsFlat, rhsFlat any, batchSize,
		lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat any,
		bufAllocAnyFn packgemm.BufAllocAnyFn, bufReleaseFn packgemm.BufReleaseFn,
		pool *workerspool.Pool) error

	// MatMulKLast performs batched matrix multiplication where both matrices have K as last dimension.
	// C = A * B^T where:
	//   - A is [batchSize, lhsCrossSize, contractingSize] (M x K per batch)
	//   - B is [batchSize, rhsCrossSize, contractingSize] (N x K per batch)
	//   - C is [batchSize, lhsCrossSize, rhsCrossSize] (M x N per batch)
	// This is more efficient than MatMulDynamic when B is stored in [N, K] order (PyTorch convention)
	// because it avoids the need to transpose B.
	MatMulKLast(inputDType, outputDType dtypes.DType,
		lhsFlat, rhsFlat any, batchSize,
		lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat any,
		pool *workerspool.Pool) error

	// Transpose2D transposes an M×K row-major matrix to K×M using SIMD.
	// Returns false if the dtype is not supported.
	Transpose2D(dtype dtypes.DType, src any, m, k int, dst any) bool
}

// Highway is the registered highway implementation.
// Default is a stub that returns "not supported".
// Import the highway submodule to register the real implementation.
var Highway HighwayMatMul = stubHighway{}

// RegisterHighway registers a highway implementation.
// This is called by the highway submodule's init() function.
func RegisterHighway(impl HighwayMatMul) {
	Highway = impl
}

// stubHighway is the default implementation that reports no dtype support.
type stubHighway struct{}

func (stubHighway) HasDTypeSupport(input, output dtypes.DType) bool {
	return false
}

func (stubHighway) MatMulDynamic(inputDType, outputDType dtypes.DType,
	lhsFlat, rhsFlat any, batchSize,
	lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat any,
	bufAllocAnyFn packgemm.BufAllocAnyFn, bufReleaseFn packgemm.BufReleaseFn,
	pool *workerspool.Pool) error {
	return errors.New("highway matmul not available: requires Go 1.26+ and importing the highway submodule")
}

func (stubHighway) MatMulKLast(inputDType, outputDType dtypes.DType,
	lhsFlat, rhsFlat any, batchSize,
	lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat any,
	pool *workerspool.Pool) error {
	return errors.New("highway matmul not available: requires Go 1.26+ and importing the highway submodule")
}

func (stubHighway) Transpose2D(dtype dtypes.DType, src any, m, k int, dst any) bool {
	return false
}
