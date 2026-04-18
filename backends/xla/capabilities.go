// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xla

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
)

// Capabilities of the SimpleGo backends: the set of supported operations and data types.
var Capabilities = compute.Capabilities{
	Operations: map[compute.OpType]bool{
		// Graph inputs (leaf nodes)
		compute.OpTypeParameter: true,
		compute.OpTypeConstant:  true,

		// Standard unary operations:
		compute.OpTypeAbs:        true,
		compute.OpTypeBitCount:   true,
		compute.OpTypeBitwiseNot: true,
		compute.OpTypeCeil:       true,
		compute.OpTypeClz:        true,
		compute.OpTypeCos:        true,
		compute.OpTypeErf:        true,
		compute.OpTypeExp:        true,
		compute.OpTypeExpm1:      true,
		compute.OpTypeFloor:      true,
		compute.OpTypeIsFinite:   true,
		compute.OpTypeIsNaN:      true,
		compute.OpTypeLog1p:      true,
		compute.OpTypeLog:        true,
		compute.OpTypeLogicalNot: true,
		compute.OpTypeLogistic:   true,
		compute.OpTypeNeg:        true,
		compute.OpTypeRound:      true,
		compute.OpTypeRsqrt:      true,
		compute.OpTypeSign:       true,
		compute.OpTypeSin:        true,
		compute.OpTypeSqrt:       true,
		compute.OpTypeTanh:       true,

		// Standard binary operations:
		compute.OpTypeAdd:        true,
		compute.OpTypeBitwiseAnd: true,
		compute.OpTypeBitwiseOr:  true,
		compute.OpTypeBitwiseXor: true,
		compute.OpTypeDiv:        true,
		compute.OpTypeLogicalAnd: true,
		compute.OpTypeLogicalOr:  true,
		compute.OpTypeLogicalXor: true,
		compute.OpTypeMax:        true,
		compute.OpTypeMin:        true,
		compute.OpTypeMul:        true,
		compute.OpTypePow:        true,
		compute.OpTypeRem:        true,
		compute.OpTypeSub:        true,

		// Comparison operators.
		compute.OpTypeEqual:                    true,
		compute.OpTypeEqualTotalOrder:          true,
		compute.OpTypeGreaterOrEqual:           true,
		compute.OpTypeGreaterOrEqualTotalOrder: true,
		compute.OpTypeGreaterThan:              true,
		compute.OpTypeGreaterThanTotalOrder:    true,
		compute.OpTypeLessOrEqual:              true,
		compute.OpTypeLessOrEqualTotalOrder:    true,
		compute.OpTypeLessThan:                 true,
		compute.OpTypeLessThanTotalOrder:       true,
		compute.OpTypeNotEqual:                 true,
		compute.OpTypeNotEqualTotalOrder:       true,

		// Complex operations:
		compute.OpTypeComplex: true,
		compute.OpTypeConj:    true,
		compute.OpTypeImag:    true,
		compute.OpTypeReal:    true,

		// Other operations:
		compute.OpTypeArgMinMax:             true,
		compute.OpTypeBatchNormForInference: true,
		compute.OpTypeBatchNormForTraining:  true,
		compute.OpTypeBatchNormGradient:     true,
		compute.OpTypeBitcast:               true,
		compute.OpTypeBroadcast:             true,
		compute.OpTypeBroadcastInDim:        true,
		compute.OpTypeClamp:                 true,
		compute.OpTypeConcatenate:           true,
		compute.OpTypeConvertDType:          true,
		compute.OpTypeConvGeneral:           true,
		compute.OpTypeDynamicSlice:          true,
		compute.OpTypeDynamicUpdateSlice:    true,
		compute.OpTypeDot:                   true,
		compute.OpTypeDotGeneral:            true,
		compute.OpTypeFFT:                   true,
		compute.OpTypeGather:                true,
		compute.OpTypeIdentity:              true,
		compute.OpTypeIota:                  true,
		compute.OpTypePad:                   true,
		compute.OpTypeReduceBitwiseAnd:      true,
		compute.OpTypeReduceBitwiseOr:       true,
		compute.OpTypeReduceBitwiseXor:      true,
		compute.OpTypeReduceLogicalAnd:      true,
		compute.OpTypeReduceLogicalOr:       true,
		compute.OpTypeReduceLogicalXor:      true,
		compute.OpTypeReduceMax:             true,
		compute.OpTypeReduceMin:             true,
		compute.OpTypeReduceProduct:         true,
		compute.OpTypeReduceSum:             true,
		compute.OpTypeReduceWindow:          true,
		compute.OpTypeReshape:               true,
		compute.OpTypeReverse:               true,
		compute.OpTypeRNGBitGenerator:       true,
		compute.OpTypeScatterSum:            true,
		compute.OpTypeScatterMax:            true,
		compute.OpTypeScatterMin:            true,
		compute.OpTypeSelectAndScatterMax:   true,
		compute.OpTypeSelectAndScatterMin:   true,
		compute.OpTypeShiftLeft:             true,
		compute.OpTypeShiftRightArithmetic:  true,
		compute.OpTypeShiftRightLogical:     true,
		compute.OpTypeSlice:                 true,
		compute.OpTypeTranspose:             true,
		compute.OpTypeWhere:                 true,

		// Collective (distributed across devices) operations:
		compute.OpTypeAllReduce: true,
	},

	Functions: true,

	DTypes: map[dtypes.DType]bool{
		dtypes.Bool:       true,
		dtypes.Int8:       true,
		dtypes.Int16:      true,
		dtypes.Int32:      true,
		dtypes.Int64:      true,
		dtypes.Uint8:      true,
		dtypes.Uint16:     true,
		dtypes.Uint32:     true,
		dtypes.Uint64:     true,
		dtypes.Float32:    true,
		dtypes.Float64:    true,
		dtypes.BFloat16:   true,
		dtypes.Complex64:  true,
		dtypes.Complex128: true,
	},
}
