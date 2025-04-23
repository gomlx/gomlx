package simplego

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
)

// FuncForDispatcher is type of functions that the DTypeDispatcher can handle.
type FuncForDispatcher func(params ...any)

const MaxDTypes = 32

// DTypeDispatcher will call
type DTypeDispatcher struct {
	Name  string
	fnMap [MaxDTypes]FuncForDispatcher
}

// NewDTypeDispatcher creates a new dispatcher for a class of functions.
func NewDTypeDispatcher(name string) *DTypeDispatcher {
	return &DTypeDispatcher{
		Name: name,
	}
}

// Dispatch call the function that matches the dtype.
func (d *DTypeDispatcher) Dispatch(dtype dtypes.DType, params ...any) {
	if dtype >= MaxDTypes {
		exceptions.Panicf("dtype %s not supported by %s", dtype, d.Name)
	}
	fn := d.fnMap[dtype]
	if fn == nil {
		exceptions.Panicf("dtype %s not supported by %s", dtype, d.Name)
	}
	fn(params...)
}

// Register a function to handle a specific dtype.
// This overwrites any previous setting for the same dtype.
func (d *DTypeDispatcher) Register(dtype dtypes.DType, fn FuncForDispatcher) {
	if dtype >= MaxDTypes {
		exceptions.Panicf("dtype %s not supported by %s", dtype, d.Name)
	}
	d.fnMap[dtype] = fn
}

// RegisterIfNotSet a function to handle a specific dtype.
func (d *DTypeDispatcher) RegisterIfNotSet(dtype dtypes.DType, fn FuncForDispatcher) {
	if dtype >= MaxDTypes {
		exceptions.Panicf("dtype %s not supported by %s", dtype, d.Name)
	}
	if d.fnMap[dtype] != nil {
		return
	}
	d.fnMap[dtype] = fn
}

// SupportedTypesConstraints enumerates the types supported by SimpleGo.
type SupportedTypesConstraints interface {
	bool | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float32 | float64 | bfloat16.BFloat16
}

// PODNumericConstraints are used for generics for the Golang pod (plain-old-data) types.
// BFloat16 is not included because it is a specialized type, not natively supported by Go.
type PODNumericConstraints interface {
	int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float32 | float64
}

// PODSignedNumericConstraints are used for generics for the Golang pod (plain-old-data) types.
// BFloat16 is not included because it is a specialized type, not natively supported by Go.
type PODSignedNumericConstraints interface {
	int8 | int16 | int32 | int64 | float32 | float64
}

// PODIntegerConstraints are used for generics for the Golang pod (plain-old-data) types.
type PODIntegerConstraints interface {
	int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64
}

// PODUnsignedConstraints are used for generics for the Golang pod (plain-old-data) types.
type PODUnsignedConstraints interface {
	uint8 | uint16 | uint32 | uint64
}

// PODFloatConstraints are used for generics for the Golang pod (plain-old-data) types.
// BFloat16 is not included because it is a specialized type, not natively supported by Go.
type PODFloatConstraints interface {
	float32 | float64
}

// PODBooleanConstraints is a simple placeholder for the gen_exec_binary.go generated code.
type PODBooleanConstraints interface {
	bool
}
