package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gopjrt/dtypes"
)

// TODO:
// BroadcastInDims
// Broadcast
// DotGeneral
// ...

// Capabilities of the SimpleGo backends: the set of supported operations and data types.
var Capabilities = backends.Capabilities{
	Operations: map[backends.OpType]bool{
		// Graph inputs (leaf nodes)
		backends.OpTypeParameter: true,
		backends.OpTypeConstant:  true,

		// Other operations:
		backends.OpTypeArgMinMax:       true,
		backends.OpTypeBroadcast:       true,
		backends.OpTypeBroadcastInDim:  true,
		backends.OpTypeConcatenate:     true,
		backends.OpTypeConvertDType:    true,
		backends.OpTypeDot:             true,
		backends.OpTypeDotGeneral:      true,
		backends.OpTypeGather:          true,
		backends.OpTypeIdentity:        true,
		backends.OpTypeReduceMax:       true,
		backends.OpTypeReduceMin:       true,
		backends.OpTypeReduceProduct:   true,
		backends.OpTypeReduceSum:       true,
		backends.OpTypeReshape:         true,
		backends.OpTypeRngBitGenerator: true,
		backends.OpTypeScatterMax:      true,
		backends.OpTypeScatterMin:      true,
		backends.OpTypeScatterSum:      true,
		backends.OpTypeSlice:           true,
		backends.OpTypeTranspose:       true,
		backends.OpTypeWhere:           true,

		// Standard unary operations:
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

		// Standard binary operations:
		backends.OpTypeAdd:        true,
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

		// Comparison operators.
		backends.OpTypeEqual:          true,
		backends.OpTypeGreaterOrEqual: true,
		backends.OpTypeGreaterThan:    true,
		backends.OpTypeLessOrEqual:    true,
		backends.OpTypeLessThan:       true,
	},

	DTypes: map[dtypes.DType]bool{
		dtypes.Bool:     true,
		dtypes.Int8:     true,
		dtypes.Int16:    true,
		dtypes.Int32:    true,
		dtypes.Int64:    true,
		dtypes.Uint8:    true,
		dtypes.Uint16:   true,
		dtypes.Uint32:   true,
		dtypes.Uint64:   true,
		dtypes.Float32:  true,
		dtypes.Float64:  true,
		dtypes.BFloat16: true,
	},
}
