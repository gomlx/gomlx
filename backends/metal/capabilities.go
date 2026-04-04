//go:build darwin && cgo

package metal

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
)

// MetalCapabilities tracks backend-level op/dtype availability.
// Individual ops still impose dtype/layout restrictions in the execution path.
var MetalCapabilities = backends.Capabilities{
	Functions: true,

	Operations: map[backends.OpType]bool{
		// Graph inputs
		backends.OpTypeParameter: true,
		backends.OpTypeConstant:  true,
		backends.OpTypeIdentity:  true,
		backends.OpTypeCall:      true,
		backends.OpTypeIf:        true,
		backends.OpTypeWhile:     true,
		backends.OpTypeSort:      true,

		// Unary operations
		backends.OpTypeAbs:        true,
		backends.OpTypeBitCount:   true,
		backends.OpTypeBitwiseNot: true,
		backends.OpTypeCeil:       true,
		backends.OpTypeClz:        true,
		backends.OpTypeCos:        true,
		backends.OpTypeErf:        true,
		backends.OpTypeExp:        true,
		backends.OpTypeExpm1:      true,
		backends.OpTypeFloor:      true,
		backends.OpTypeIsFinite:   true,
		backends.OpTypeIsNaN:      true,
		backends.OpTypeLog1p:      true,
		backends.OpTypeLog:        true,
		backends.OpTypeLogicalNot: true,
		backends.OpTypeLogistic:   true,
		backends.OpTypeNeg:        true,
		backends.OpTypeRound:      true,
		backends.OpTypeRsqrt:      true,
		backends.OpTypeSign:       true,
		backends.OpTypeSin:        true,
		backends.OpTypeSqrt:       true,
		backends.OpTypeTanh:       true,

		// Binary operations
		backends.OpTypeAdd:        true,
		backends.OpTypeAtan2:      true,
		backends.OpTypeBitwiseAnd: true,
		backends.OpTypeBitwiseOr:  true,
		backends.OpTypeBitwiseXor: true,
		backends.OpTypeDiv:        true,
		backends.OpTypeLogicalAnd: true,
		backends.OpTypeLogicalOr:  true,
		backends.OpTypeLogicalXor: true,
		backends.OpTypeMax:        true,
		backends.OpTypeMin:        true,
		backends.OpTypeMul:        true,
		backends.OpTypePow:        true,
		backends.OpTypeRem:        true,
		backends.OpTypeSub:        true,

		// Comparisons
		backends.OpTypeEqual:                    true,
		backends.OpTypeEqualTotalOrder:          true,
		backends.OpTypeNotEqual:                 true,
		backends.OpTypeNotEqualTotalOrder:       true,
		backends.OpTypeGreaterOrEqual:           true,
		backends.OpTypeGreaterOrEqualTotalOrder: true,
		backends.OpTypeGreaterThan:              true,
		backends.OpTypeGreaterThanTotalOrder:    true,
		backends.OpTypeLessOrEqual:              true,
		backends.OpTypeLessOrEqualTotalOrder:    true,
		backends.OpTypeLessThan:                 true,
		backends.OpTypeLessThanTotalOrder:       true,

		// Tensor ops
		backends.OpTypeArgMinMax:             true,
		backends.OpTypeBatchNormForInference: true,
		backends.OpTypeBatchNormForTraining:  true,
		backends.OpTypeBatchNormGradient:     true,
		backends.OpTypeBitcast:               true,
		backends.OpTypeClamp:                 true,
		backends.OpTypeBroadcast:             true,
		backends.OpTypeBroadcastInDim:        true,
		backends.OpTypeConcatenate:           true,
		backends.OpTypeConvGeneral:           true,
		backends.OpTypeConvertDType:          true,
		backends.OpTypeDot:                   true,
		backends.OpTypeDotGeneral:            true,
		backends.OpTypeGather:                true,
		backends.OpTypeIota:                  true,
		backends.OpTypePad:                   true,
		backends.OpTypeReduceWindow:          true,
		backends.OpTypeReshape:               true,
		backends.OpTypeReverse:               true,
		backends.OpTypeRNGBitGenerator:       true,
		backends.OpTypeScatterSum:            true,
		backends.OpTypeScatterMax:            true,
		backends.OpTypeScatterMin:            true,
		backends.OpTypeSlice:                 true,
		backends.OpTypeTranspose:             true,
		backends.OpTypeWhere:                 true,

		// Reductions
		backends.OpTypeReduceSum:        true,
		backends.OpTypeReduceProduct:    true,
		backends.OpTypeReduceMax:        true,
		backends.OpTypeReduceMin:        true,
		backends.OpTypeReduceBitwiseAnd: true,
		backends.OpTypeReduceBitwiseOr:  true,
		backends.OpTypeReduceBitwiseXor: true,
		backends.OpTypeReduceLogicalAnd: true,
		backends.OpTypeReduceLogicalOr:  true,
		backends.OpTypeReduceLogicalXor: true,

		// Fused operations
		backends.OpTypeFusedSoftmax:                   true,
		backends.OpTypeFusedGelu:                      true,
		backends.OpTypeFusedLayerNorm:                 true,
		backends.OpTypeFusedDense:                     true,
		backends.OpTypeFusedScaledDotProductAttention: true,
		backends.OpTypeFusedAttentionQKVProjection:    true,
		backends.OpTypeFusedQuantizedDense:            true,

		backends.OpTypeAllReduce: true,
	},

	DTypes: map[dtypes.DType]bool{
		dtypes.Bool:    true,
		dtypes.Int8:    true,
		dtypes.Int16:   true,
		dtypes.Int32:   true,
		dtypes.Int64:   true,
		dtypes.Uint8:   true,
		dtypes.Uint16:  true,
		dtypes.Uint32:  true,
		dtypes.Uint64:  true,
		dtypes.Float16: true,
		dtypes.Float32: true,
		// Native float64/double is not available in Metal compute on Apple GPUs.
		dtypes.Float64: false,
	},
}
