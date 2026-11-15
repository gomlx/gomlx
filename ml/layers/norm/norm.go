// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package norm

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/pkg/errors"
)

// resolveNormalizationDType determines the precision dtype used for normalization
// computations, checking paramKey in scope and otherwise promoting 16-bit floats
// (Float16, BFloat16) to Float32.
func resolveNormalizationDType(scope *model.Scope, paramKey string, inputDType dtypes.DType) dtypes.DType {
	normDTypeStr := model.GetParamOr(scope, paramKey, "")
	if normDTypeStr == "" {
		if inputDType == dtypes.Float16 || inputDType == dtypes.BFloat16 {
			return dtypes.Float32
		}
		return inputDType
	}
	normDType, err := dtypes.DTypeString(normDTypeStr)
	if err != nil {
		panic(errors.WithMessagef(err, "Invalid dtype configured for hyperparameter %q", paramKey))
	}
	return normDType
}

// resolveVariableDType returns varDType if explicitly configured (non-InvalidDType),
// or defaults to inputDType so created and pre-existing variables must match the
// expected dtype.
func resolveVariableDType(varDType, inputDType dtypes.DType) dtypes.DType {
	if varDType != dtypes.InvalidDType {
		return varDType
	}
	return inputDType
}
