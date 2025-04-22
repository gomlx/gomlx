package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gopjrt/dtypes"
)

// Capabilities of the SimpleGo backends.
var Capabilities = backends.Capabilities{
	Operations: map[backends.OpType]bool{
		backends.OpTypeParameter: true,
		backends.OpTypeConstant:  true,
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

// numericConstrains are used for generics for the Golang pod (plain-old-data) types.
// BFloat16 is not included because it is a specialized type, not natively supported by Go.
type numericPODConstraints interface {
	int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float32 | float64
}

// signedNumericConstraints are used for generics for the Golang pod (plain-old-data) types.
// BFloat16 is not included because it is a specialized type, not natively supported by Go.
type signedNumericPODConstraints interface {
	int8 | int16 | int32 | int64 | float32 | float64
}

// unsignedConstraints are used for generics for the Golang pod (plain-old-data) types.
// BFloat16 is not included because it is a specialized type, not natively supported by Go.
type unsignedPODConstraints interface {
	uint8 | uint16 | uint32 | uint64
}

// floatConstrains are used for generics for the Golang pod (plain-old-data) types.
// BFloat16 is not included because it is a specialized type, not natively supported by Go.
type floatConstraints interface {
	float32 | float64
}

// integerConstrains are used for generics for the Golang pod (plain-old-data) types.
type integerConstraints interface {
	int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64
}
