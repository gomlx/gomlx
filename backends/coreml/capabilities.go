//go:build darwin

package coreml

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
)

// Capabilities defines what operations and dtypes are supported by the CoreML backend.
// This is a minimal set to start; more operations will be added as they are implemented.
var Capabilities = backends.Capabilities{
	Operations: map[backends.OpType]bool{
		// Graph inputs (leaf nodes)
		backends.OpTypeParameter: true,
		backends.OpTypeConstant:  true,

		// Basic unary operations
		backends.OpTypeAbs:    true,
		backends.OpTypeNeg:    true,
		backends.OpTypeExp:    true,
		backends.OpTypeLog:   true,
		backends.OpTypeSqrt:  true,
		backends.OpTypeRsqrt: true,
		backends.OpTypeTanh:  true,
		backends.OpTypeFloor: true,
		backends.OpTypeCeil:  true,
		backends.OpTypeRound: true,
		backends.OpTypeSign:  true,

		// Basic binary operations
		backends.OpTypeAdd:      true,
		backends.OpTypeSub:      true,
		backends.OpTypeMul:      true,
		backends.OpTypeDiv:      true,
		backends.OpTypePow:      true,
		backends.OpTypeMax:      true,
		backends.OpTypeMin:      true,

		// Matrix operations
		backends.OpTypeDot: true,

		// Reduce operations
		backends.OpTypeReduceSum:     true,
		backends.OpTypeReduceMax:     true,
		backends.OpTypeReduceMin:     true,
		backends.OpTypeReduceProduct: true,
		backends.OpTypeArgMinMax:     true,

		// Slice operations
		backends.OpTypeSlice: true,
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
