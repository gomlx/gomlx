// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package highway

import (
	"errors"

	"github.com/gomlx/gomlx/backends/simplego/packgemm"
	"github.com/gomlx/gomlx/internal/workerspool"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
)

// HasDTypeSupport returns true if a MatMulDynamic function is registered for the given dtypes.
func HasDTypeSupport(input, output dtypes.DType) bool {
	return false
}

// MatMulDynamic dispatches the MatMul function for the given dtypes.
func MatMulDynamic(inputDType, outputDType dtypes.DType,
	lhsFlat, rhsFlat any, batchSize,
	lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat any,
	bufAllocAnyFn packgemm.BufAllocAnyFn, bufReleaseFn packgemm.BufReleaseFn, pool *workerspool.Pool) error {
	return errors.New("not implemented")
}
