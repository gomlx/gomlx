// Package dtypes includes the DType enum for all supported data types for GoMLX.
//
// It is forked from XLA based github.com/gomlx/go-xla/pkg/types/dtypes, and likely it will almost
// always be mostly the same (maybe with a lag), but may diverge if new backends emerge with different
// data-types not supported by XLA.
//
// It includes several converters to/from Go native types (and reflect.Type), and constants for min/max values
// for types, etc. It also includes some constraint interfaces to be used with generics (Number, NumberNotComplex, GoFloat).
package dtypes

import (
	"maps"
	"math"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/pkg/errors"
	"github.com/x448/float16"
)

// panicf panics with the formatted description.
//
// It is only used for "bugs in the code" -- when parameters don't follow the specifications.
// In principle, it should never happen -- the same way nil-pointer panics should never happen.
func panicf(format string, args ...any) {
	panic(errors.Errorf(format, args...))
}

func init() {
	// Only works for 32 and 64 bits platforms.
	// TODO: find some compile-time check.
	if strconv.IntSize != 32 && strconv.IntSize != 64 {
		panicf("cannot use int of %d bits with gopjrt -- only platforms with int32 or int64 are supported", strconv.IntSize)
	}

	// Add a mapping to the lower-case version of dtypes.
	keys := slices.Collect(maps.Keys(MapOfNames))
	for _, key := range keys {
		lowerKey := strings.ToLower(key)
		if lowerKey == key {
			continue
		}
		if _, found := MapOfNames[lowerKey]; found {
			continue
		}
		MapOfNames[lowerKey] = MapOfNames[key]
	}
}

// Generate automatic C-to-Go boilerplate code for pjrt_c_api.h.
//go:generate go run ../../../internal/cmd/dtypes_codegen

// FromGenericsType returns the DType enum for the given type that this package knows about.
func FromGenericsType[T Supported]() DType {
	var t T
	switch (any(t)).(type) {
	case float64:
		return Float64
	case float32:
		return Float32
	case float16.Float16:
		return Float16
	case bfloat16.BFloat16:
		return BFloat16
	case int:
		switch strconv.IntSize {
		case 32:
			return Int32
		case 64:
			return Int64
		default:
			panicf("Cannot use int of %d bits with gopjrt -- try using int32 or int64", strconv.IntSize)
		}
	case int64:
		return Int64
	case int32:
		return Int32
	case int16:
		return Int16
	case int8:
		return Int8
	case bool:
		return Bool
	case uint8:
		return Uint8
	case uint16:
		return Uint16
	case uint32:
		return Uint32
	case uint64:
		return Uint64
	case complex64:
		return Complex64
	case complex128:
		return Complex128
	}
	return InvalidDType
}

// FromGoType returns the DType for the given "reflect.Type".
// It panics for unknown DType values.
func FromGoType(t reflect.Type) DType {
	if t == float16Type {
		return Float16
	} else if t == bfloat16Type {
		return BFloat16
	}
	switch t.Kind() {
	case reflect.Int:
		switch strconv.IntSize {
		case 32:
			return Int32
		case 64:
			return Int64
		default:
			panicf("cannot use int of %d bits with GoMLX -- try using int32 or int64", strconv.IntSize)
		}
	case reflect.Int64:
		return Int64
	case reflect.Int32:
		return Int32
	case reflect.Int16:
		return Int16
	case reflect.Int8:
		return Int8

	case reflect.Uint64:
		return Uint64
	case reflect.Uint32:
		return Uint32
	case reflect.Uint16:
		return Uint16
	case reflect.Uint8:
		return Uint8

	case reflect.Bool:
		return Bool

	case reflect.Float32:
		return Float32
	case reflect.Float64:
		return Float64

	case reflect.Complex64:
		return Complex64
	case reflect.Complex128:
		return Complex128
	default:
		return InvalidDType
	}
	return InvalidDType
}

// FromAny introspects the underlying type of any and returns the corresponding DType.
// Non-scalar types, or unsupported types return an InvalidType.
func FromAny(value any) DType {
	return FromGoType(reflect.TypeOf(value))
}

// Size returns the number of bytes for the given DType, or 0 if the dtype uses fraction(s) of bytes.
// If the size is 0 (like a 4-bits quantity), consider the Bits or SizeForDimensions method.
func (dtype DType) Size() int {
	return int(dtype.GoType().Size())
}

// Bits returns the number of bits for the given DType.
func (dtype DType) Bits() int {
	return dtype.Size() * 8
}

// SizeForDimensions returns the size in bytes used for the given dimensions.
// This is a safer method than Size in case the dtype uses an underlying size that is not multiple of 8 bits.
//
// It works also for scalar (one element) shapes where the list of dimensions is empty.
func (dtype DType) SizeForDimensions(dimensions ...int) int {
	numElements := 1
	for _, dim := range dimensions {
		if dim < 0 {
			panicf("dim cannot be neative for SizeForDimensions, got %v", dimensions)
		}
		numElements *= dim
	}

	// Switch case for dtypes with size not multiple of 8 bits (1 byte).

	// Default is simply the number of elements times the size in bytes per element.
	return numElements * dtype.Size()
}

// Memory returns the number of bytes for the given DType.
// It's an alias to Size, converted to uintptr.
func (dtype DType) Memory() uintptr {
	return uintptr(dtype.Size())
}

// Pre-generate constant reflect.TypeOf for convenience.
var (
	float32Type  = reflect.TypeOf(float32(0))
	float64Type  = reflect.TypeOf(float64(0))
	float16Type  = reflect.TypeOf(float16.Float16(0))
	bfloat16Type = reflect.TypeOf(bfloat16.BFloat16(0))
)

// GoType returns the Go `reflect.Type` corresponding to the tensor DType.
func (dtype DType) GoType() reflect.Type {
	switch dtype {
	case Int64:
		return reflect.TypeOf(int64(0))
	case Int32:
		return reflect.TypeOf(int32(0))
	case Int16:
		return reflect.TypeOf(int16(0))
	case Int8:
		return reflect.TypeOf(int8(0))

	case Uint64:
		return reflect.TypeOf(uint64(0))
	case Uint32:
		return reflect.TypeOf(uint32(0))
	case Uint16:
		return reflect.TypeOf(uint16(0))
	case Uint8:
		return reflect.TypeOf(uint8(0))

	case Bool:
		return reflect.TypeOf(true)

	case Float16:
		return float16Type
	case BFloat16:
		return bfloat16Type
	case Float32:
		return float32Type
	case Float64:
		return float64Type

	case Complex64:
		return reflect.TypeOf(complex64(0))
	case Complex128:
		return reflect.TypeOf(complex128(0))

	default:
		// This should never happen, except if someone entered an invalid DType number beyond the values
		// defined.
		panicf("unknown dtype %q (%d) in DType.GoType", dtype, dtype)
		panic(nil)
	}
}

// GoStr converts dtype to the corresponding Go type and convert that to string.
// Notice the names are different from the Dtype (so `Int64` dtype is simply `int` in Go).
func (dtype DType) GoStr() string {
	return dtype.GoType().Name()
}

// LowestValue for dtype converted to the corresponding Go type.
// For float values it will return negative infinite.
// There is no lowest value for complex numbers, since they are not ordered.
func (dtype DType) LowestValue() any {
	switch dtype {
	case Int64:
		return int64(math.MinInt64)
	case Int32:
		return int32(math.MinInt32)
	case Int16:
		return int16(math.MinInt16)
	case Int8:
		return int16(math.MinInt8)

	case Uint64:
		return uint64(0)
	case Uint32:
		return uint32(0)
	case Uint16:
		return uint16(0)
	case Uint8:
		return uint8(0)

	case Bool:
		return false

	case Float32:
		return float32(math.Inf(-1))
	case Float64:
		return math.Inf(-1)
	case Float16:
		return float16.Inf(-1)
	case BFloat16:
		return bfloat16.Inf(-1)

	default:
		// For invalid dtypes (like complex numbers), return zero.
		return reflect.New(dtype.GoType()).Elem().Interface()
	}
}

// HighestValue for dtype converted to the corresponding Go type.
// For float values it will return infinite.
// There is no lowest value for complex numbers, since they are not ordered.
func (dtype DType) HighestValue() any {
	switch dtype {
	case Int64:
		return int64(math.MaxInt64)
	case Int32:
		return int32(math.MaxInt32)
	case Int16:
		return int16(math.MaxInt16)
	case Int8:
		return int8(math.MaxInt8)

	case Uint64:
		return uint64(math.MaxUint64)
	case Uint32:
		return uint32(math.MaxUint32)
	case Uint16:
		return uint16(math.MaxUint16)
	case Uint8:
		return uint8(math.MaxUint8)

	case Bool:
		return true

	case Float32:
		return float32(math.Inf(1))
	case Float64:
		return math.Inf(1)
	case Float16:
		return float16.Inf(1)
	case BFloat16:
		return bfloat16.Inf(1)

	default:
		// For invalid dtypes (like complex numbers), return zero.
		return reflect.New(dtype.GoType()).Elem().Interface()
	}
}

// SmallestNonZeroValueForDType is the smallest non-zero-value dtypes.
// Only useful for float types.
// The return value is converted to the corresponding Go type.
// There is no smallest non-zero value for complex numbers, since they are not ordered.
func (dtype DType) SmallestNonZeroValueForDType() any {
	switch dtype {
	case Int64:
		return int64(1)
	case Int32:
		return int32(1)
	case Int16:
		return int16(1)
	case Int8:
		return int8(1)

	case Uint64:
		return uint64(1)
	case Uint32:
		return uint32(1)
	case Uint16:
		return uint16(1)
	case Uint8:
		return uint8(1)

	case Bool:
		return true

	case Float32:
		return float32(math.SmallestNonzeroFloat32)
	case Float64:
		return math.SmallestNonzeroFloat64
	case Float16:
		return float16.Float16(0x0001) // 1p-24, see discussion in https://github.com/x448/float16/pull/46
	case BFloat16:
		return bfloat16.SmallestNonzero // 1p-24, see discussion in https://github.com/x448/float16/pull/46

	default:
		// For invalid dtypes (like complex numbers), return zero.
		return reflect.New(dtype.GoType()).Elem().Interface()
	}
}

// IsFloat returns whether dtype is a supported float -- float types not yet supported will return false.
// It returns false for complex numbers.
func (dtype DType) IsFloat() bool {
	return dtype == Float32 || dtype == Float64 || dtype == Float16 || dtype == BFloat16
}

// IsFloat16 returns whether dtype is a supported float with 16 bits: [Float16] or [BFloat16].
func (dtype DType) IsFloat16() bool {
	return dtype == Float16 || dtype == BFloat16
}

// IsComplex returns whether dtype is a supported complex number type.
func (dtype DType) IsComplex() bool {
	return dtype == Complex64 || dtype == Complex128
}

// RealDType returns the real component of complex dtypes.
// For float dtypes, it returns itself.
//
// It returns InvalidDType for other non-(complex or float) dtypes.
func (dtype DType) RealDType() DType {
	if dtype.IsFloat() {
		return dtype
	}
	switch dtype {
	case Complex64:
		return Float32
	case Complex128:
		return Float64
	default:
		// RealDType is not defined for other dtypes.
		return InvalidDType
	}
}

// IsInt returns whether dtype is a supported integer type -- float types not yet supported will return false.
func (dtype DType) IsInt() bool {
	return dtype == Int64 || dtype == Int32 || dtype == Int16 || dtype == Int8 ||
		dtype == Uint8 || dtype == Uint16 || dtype == Uint32 || dtype == Uint64
}

// IsUnsigned returns whether dtype is one of the unsigned (only int for now) types.
func (dtype DType) IsUnsigned() bool {
	return dtype == Uint8 || dtype == Uint16 || dtype == Uint32 || dtype == Uint64
}

// IsSupported returns whether dtype is supported by `gopjrt`.
func (dtype DType) IsSupported() bool {
	return dtype == Bool || dtype == Float16 || dtype == BFloat16 || dtype == Float32 || dtype == Float64 || dtype == Int64 || dtype == Int32 || dtype == Int16 || dtype == Int8 || dtype == Uint32 || dtype == Uint16 || dtype == Uint8 || dtype == Complex64 || dtype == Complex128
}

// IsPromotableTo returns whether dtype can be promoted to target.
//
// For example, Int32 can be promoted to Int64, but not to Uint64.
//
// See https://openxla.org/stablehlo/spec#functions_on_types for reference.
//
//goland:noinspection ALL
func (dtype DType) IsPromotableTo(target DType) bool {
	if dtype == target {
		return true
	}

	// Check for same dtypeV category:
	isSameType := (dtype == Bool && target == Bool) ||
		(dtype.IsInt() && target.IsInt()) ||
		(dtype.IsFloat() && target.IsFloat()) ||
		(dtype.IsComplex() && target.IsComplex())

	if !isSameType {
		return false
	}

	// For integer and float types, check bitwidth
	if dtype.IsInt() || dtype.IsFloat() || dtype.IsComplex() {
		return dtype.Bits() <= target.Bits()
	}
	return false
}

// Supported lists the Go types that `gopjrt` knows how to convert -- there are more types that can be manually
// converted.
// Used as traits for generics.
//
// Notice Go's `int` type is not portable, since it may translate to dtypes Int32 or Int64 depending
// on the platform.
type Supported interface {
	bool | float16.Float16 | bfloat16.BFloat16 |
		float32 | float64 | int | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 |
		complex64 | complex128
}

// Number represents the Go numeric types corresponding to supported DType's.
// Used as traits for generics.
//
// It includes complex numbers.
// It doesn't include float16.Float16 or bfloat16.BFloat16 because they are not native number types.
type Number interface {
	float32 | float64 | int | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | complex64 | complex128
}

// NumberNotComplex represents the Go numeric types corresponding to supported DType's.
// Used as a Generics constraint.
//
// See also Number.
type NumberNotComplex interface {
	float32 | float64 | int | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64
}

// GoFloat represent a continuous Go numeric type, supported by GoMLX.
// It doesn't include complex numbers.
type GoFloat interface {
	float32 | float64
}
