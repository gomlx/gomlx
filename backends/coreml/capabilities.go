//go:build darwin && cgo

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package coreml

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
)

// Capabilities defines what operations and dtypes are supported by the CoreML backend.
// This is a minimal set to start; more operations will be added as they are implemented.
var Capabilities = backends.Capabilities{
	// Functions indicates support for the new function/closure interface.
	Functions: true,

	Operations: map[backends.OpType]bool{
		// Graph inputs (leaf nodes)
		backends.OpTypeParameter: true,
		backends.OpTypeConstant:  true,

		// Basic unary operations
		backends.OpTypeAbs:      true,
		backends.OpTypeNeg:      true,
		backends.OpTypeExp:      true,
		backends.OpTypeExpm1:    true,
		backends.OpTypeLog:      true,
		backends.OpTypeLog1p:    true,
		backends.OpTypeSqrt:     true,
		backends.OpTypeRsqrt:    true,
		backends.OpTypeTanh:     true,
		backends.OpTypeFloor:    true,
		backends.OpTypeCeil:     true,
		backends.OpTypeRound:    true,
		backends.OpTypeSign:     true,
		backends.OpTypeLogistic: true,
		backends.OpTypeCos:      true,
		backends.OpTypeSin:      true,
		backends.OpTypeErf:      true,

		// Basic binary operations
		backends.OpTypeAdd: true,
		backends.OpTypeSub: true,
		backends.OpTypeMul: true,
		backends.OpTypeDiv: true,
		backends.OpTypePow: true,
		backends.OpTypeMax: true,
		backends.OpTypeMin: true,

		// Comparison operations
		backends.OpTypeEqual:          true,
		backends.OpTypeNotEqual:       true,
		backends.OpTypeLessThan:       true,
		backends.OpTypeLessOrEqual:    true,
		backends.OpTypeGreaterThan:    true,
		backends.OpTypeGreaterOrEqual: true,

		// Matrix operations
		backends.OpTypeDot:        true,
		backends.OpTypeDotGeneral: true,

		// Reduce operations
		backends.OpTypeReduceSum:     true,
		backends.OpTypeReduceMax:     true,
		backends.OpTypeReduceMin:     true,
		backends.OpTypeReduceProduct: true,
		backends.OpTypeArgMinMax:     true,
		backends.OpTypeReduceWindow:  true,

		// Slice operations
		backends.OpTypeSlice:              true,
		backends.OpTypeDynamicUpdateSlice: true,

		// Shape manipulation operations
		backends.OpTypeReshape:   true,
		backends.OpTypeTranspose: true,

		// Convolution operations
		backends.OpTypeConvGeneral: true,

		// Array operations
		backends.OpTypeConcatenate:  true,
		backends.OpTypeGather:       true,
		backends.OpTypePad:          true,
		backends.OpTypeReverse:      true,
		backends.OpTypeConvertDType: true,
		backends.OpTypeWhere:        true,
		backends.OpTypeIota:         true,
	},

	DTypes: map[dtypes.DType]bool{
		dtypes.Float16: true,
		dtypes.Float32: true,
		dtypes.Float64: true,
		dtypes.Int8:    true,
		dtypes.Int16:   true,
		dtypes.Int32:   true,
		dtypes.Int64:   true,
		dtypes.Bool:    true,
	},
}
