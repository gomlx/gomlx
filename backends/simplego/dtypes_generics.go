// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/gomlx/gomlx/pkg/support/exceptions"
	"github.com/x448/float16"
)

const MaxDTypes = 32

// DTypeMap --------------------------------------------------------------------------------------------------

// DTypeMap manages registering of an arbitrary value per dtype.
type DTypeMap struct {
	Name     string
	Map      [MaxDTypes]any
	Priority [MaxDTypes]registerPriority
}

// NewDTypeMap creates a new DTypeMap.
func NewDTypeMap(name string) *DTypeMap {
	return &DTypeMap{
		Name: name,
	}
}

// Get retrieves the value for the given dtype, or throw an exception if none was registered.
func (d *DTypeMap) Get(dtype dtypes.DType) any {
	if dtype >= MaxDTypes {
		exceptions.Panicf("dtype %s not supported by %s", dtype, d.Name)
	}
	value := d.Map[dtype]
	if value == nil {
		exceptions.Panicf("dtype %s not supported by %s -- "+
			"if you need it, consider creating an issue to add support in github.com/gomlx/gomlx",
			dtype, d.Name)
	}
	return value
}

// Register a value for a dtype with the specified priority.
// If the priority is lower than the current priority for the dtype, the value is ignored.
func (d *DTypeMap) Register(dtype dtypes.DType, priority registerPriority, value any) {
	if dtype >= MaxDTypes {
		exceptions.Panicf("dtype %s not supported by %s", dtype, d.Name)
	}
	if priority < d.Priority[dtype] {
		// We have something registered with higher priority, ignore.
		return
	}
	d.Priority[dtype] = priority
	d.Map[dtype] = value
}

// DTypeDispatcher --------------------------------------------------------------------------------------------------

// FuncForDispatcher is type of functions that the DTypeDispatcher can handle.
type FuncForDispatcher func(params ...any) any

// DTypeDispatcher manages dispatching functions to handle specific DTypes.
// Often, these functions will be instances of a generic function.
type DTypeDispatcher struct {
	Name     string
	fnMap    [MaxDTypes]FuncForDispatcher
	Priority [MaxDTypes]registerPriority
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

// Register a function to handle a specific dtype with the specified priority.
// If the priority is lower than the current priority for the dtype, the function is ignored.
func (d *DTypeDispatcher) Register(dtype dtypes.DType, priority registerPriority, fn FuncForDispatcher) {
	if dtype >= MaxDTypes {
		exceptions.Panicf("dtype %s not supported by %s", dtype, d.Name)
	}
	if priority < d.Priority[dtype] {
		// We have something registered with higher priority, ignore.
		return
	}
	d.Priority[dtype] = priority
	d.fnMap[dtype] = fn
}

// DTypePairMap --------------------------------------------------------------------------------------------------

// DTypePairMap manages registering of an arbitrary value per dtype pair.
type DTypePairMap struct {
	Name     string
	Map      [MaxDTypes][MaxDTypes]any
	Priority [MaxDTypes][MaxDTypes]registerPriority
}

// NewDTypePairMap creates a new DTypePairMap.
func NewDTypePairMap(name string) *DTypePairMap {
	return &DTypePairMap{
		Name: name,
	}
}

// Get retrieves the value for the given dtype pair, or throw an exception if none was registered.
func (d *DTypePairMap) Get(dtype1, dtype2 dtypes.DType) any {
	if dtype1 >= MaxDTypes || dtype2 >= MaxDTypes {
		exceptions.Panicf("dtypes %s or %s not supported by %s", dtype1, dtype2, d.Name)
	}
	value := d.Map[dtype1][dtype2]
	if value == nil {
		exceptions.Panicf("dtype pair (%s, %s) not supported by %s -- "+
			"if you need it, consider creating an issue to add support in github.com/gomlx/gomlx",
			dtype1, dtype2, d.Name)
	}
	return value
}

// Register a value for a dtype pair with the specified priority.
// If the priority is lower than the current priority for the dtype pair, the value is ignored.
func (d *DTypePairMap) Register(dtype1, dtype2 dtypes.DType, priority registerPriority, value any) {
	if dtype1 >= MaxDTypes || dtype2 >= MaxDTypes {
		exceptions.Panicf("dtypes %s or %s not supported by %s", dtype1, dtype2, d.Name)
	}
	if priority < d.Priority[dtype1][dtype2] {
		// We have something registered with higher priority, ignore.
		return
	}
	d.Priority[dtype1][dtype2] = priority
	d.Map[dtype1][dtype2] = value
}

// Constraints --------------------------------------------------------------------------------------------------------

// SupportedTypesConstraints enumerates the types supported by SimpleGo.
type SupportedTypesConstraints interface {
	bool | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float32 | float64 |
		bfloat16.BFloat16 | float16.Float16
}

// PODNumericConstraints are used for generics for the Golang pod (plain-old-data) types.
// BFloat16 is not included because it is a specialized type, not natively supported by Go.
type PODNumericConstraints interface {
	int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float32 | float64
}

// PODSignedNumericConstraints are used for generics for the Golang pod (plain-old-data) types.
// BFloat16 and Float16 are not included because they are specialized types, not natively supported by Go.
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
// BFloat16 and Float16 are not included because they are specialized types, not natively supported by Go.
type PODFloatConstraints interface {
	float32 | float64
}

// PODBooleanConstraints is a simple placeholder for the gen_exec_binary.go generated code.
type PODBooleanConstraints interface {
	bool
}
