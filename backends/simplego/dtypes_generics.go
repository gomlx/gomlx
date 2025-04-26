package simplego

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
)

// FuncForDispatcher is type of functions that the DTypeDispatcher can handle.
type FuncForDispatcher func(params ...any) any

const MaxDTypes = 32

// DTypeDispatcher manages dispatching functions to handle specific DTypes.
// Often, these functions will be instances of a generic function.
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
func (d *DTypeDispatcher) Dispatch(dtype dtypes.DType, params ...any) any {
	if dtype >= MaxDTypes {
		exceptions.Panicf("dtype %s not supported by %s", dtype, d.Name)
	}
	fn := d.fnMap[dtype]
	if fn == nil {
		exceptions.Panicf("dtype %s not supported by %s -- "+
			"if you need it, consider creating an issue to add support in github.com/gomlx/gomlx",
			dtype, d.Name)
	}
	return fn(params...)
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

// DType2Dispatcher manages dispatching functions to handle specific pair of DTypes.
// Often, these functions will be instances of a generic function.
type DType2Dispatcher struct {
	Name  string
	fnMap [MaxDTypes][MaxDTypes]FuncForDispatcher
}

// NewDType2Dispatcher creates a new dispatcher for a class of functions.
func NewDType2Dispatcher(name string) *DType2Dispatcher {
	return &DType2Dispatcher{
		Name: name,
	}
}

// Dispatch call the function that matches the dtype.
func (d *DType2Dispatcher) Dispatch(dtype1, dtype2 dtypes.DType, params ...any) {
	if dtype1 >= MaxDTypes || dtype2 >= MaxDTypes {
		exceptions.Panicf("dtypes %s or %s not supported by %s", dtype1, dtype2, d.Name)
	}
	fn := d.fnMap[dtype1][dtype2]
	if fn == nil {
		exceptions.Panicf("dtype pair (%s, %s) not supported by %s -- "+
			"if you need it, consider creating an issue to add support in github.com/gomlx/gomlx",
			dtype1, dtype2, d.Name)
	}
	fn(params...)
}

// Register a function to handle a specific dtype.
// This overwrites any previous setting for the same dtype.
func (d *DType2Dispatcher) Register(dtype1, dtype2 dtypes.DType, fn FuncForDispatcher) {
	if dtype1 >= MaxDTypes || dtype2 >= MaxDTypes {
		exceptions.Panicf("dtypes %s or %s not supported by %s", dtype1, dtype2, d.Name)
	}
	d.fnMap[dtype1][dtype2] = fn
}

// RegisterIfNotSet a function to handle a specific dtype.
func (d *DType2Dispatcher) RegisterIfNotSet(dtype1, dtype2 dtypes.DType, fn FuncForDispatcher) {
	if dtype1 >= MaxDTypes || dtype2 >= MaxDTypes {
		exceptions.Panicf("dtypes %s or %s not supported by %s", dtype1, dtype2, d.Name)
	}
	if d.fnMap[dtype1][dtype2] != nil {
		return
	}
	d.fnMap[dtype1][dtype2] = fn
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
