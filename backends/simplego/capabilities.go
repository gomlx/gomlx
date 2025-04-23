package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gopjrt/dtypes"
)

// TODO:
// Identity
// Where
// ReduceWindow
// BroadcastInDims
// Broadcast
// DotGeneral
// ...

// Capabilities of the SimpleGo backends: the set of supported operations and data types.
var Capabilities = backends.Capabilities{
	Operations: map[backends.OpType]bool{
		backends.OpTypeParameter: true,
		backends.OpTypeConstant:  true,

		// Standard unary operations:
		backends.OpTypeAbs:        true,
		backends.OpTypeBitCount:   true,
		backends.OpTypeBitwiseNot: true,
		backends.OpTypeCeil:       true,
		backends.OpTypeClz:        true,
		backends.OpTypeCos:        true,
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

// podNumericConstrains are used for generics for the Golang pod (plain-old-data) types.
// BFloat16 is not included because it is a specialized type, not natively supported by Go.
type podNumericConstraints interface {
	int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float32 | float64
}

// podSignedNumericConstraints are used for generics for the Golang pod (plain-old-data) types.
// BFloat16 is not included because it is a specialized type, not natively supported by Go.
type podSignedNumericPODConstraints interface {
	int8 | int16 | int32 | int64 | float32 | float64
}

// podIntegerConstraints are used for generics for the Golang pod (plain-old-data) types.
type podIntegerConstraints interface {
	int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64
}

// podUnsignedConstraints are used for generics for the Golang pod (plain-old-data) types.
type podUnsignedConstraints interface {
	uint8 | uint16 | uint32 | uint64
}

// podFloatConstrains are used for generics for the Golang pod (plain-old-data) types.
// BFloat16 is not included because it is a specialized type, not natively supported by Go.
type podFloatConstraints interface {
	float32 | float64
}

// podIntegerConstrains are used for generics for the Golang pod (plain-old-data) types.
type integerPODConstraints interface {
	int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64
}

// podBooleanConstraints is a simple placeholder for the gen_exec_binary.go generated code.
type podBooleanConstraints interface {
	bool
}
