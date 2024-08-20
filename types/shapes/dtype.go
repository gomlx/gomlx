/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package shapes

import (
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"reflect"
	"unsafe"

	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gopjrt/dtypes"
	"github.com/x448/float16"
)

// ConvertTo converts any scalar (typically returned by `tensor.Local.Value()`) of the
// supported dtypes to `T`.
// Returns 0 if value is not a scalar or not a supported number (e.g: bool).
// It doesn't work for if T (the output type) is a complex number.
// If value is a complex number, it converts by taking the real part of the number and
// discarding the imaginary part.
func ConvertTo[T NumberNotComplex](value any) T {
	t, ok := value.(T)
	if ok {
		return t
	}
	if reflect.TypeOf(t) == float16Type {
		v32 := ConvertTo[float32](value)
		return T(float16.Fromfloat32(v32))
	}

	switch v := value.(type) {
	case float64:
		return T(v)
	case float32:
		return T(v)
	case float16.Float16:
		return T(v.Float32())
	case bfloat16.BFloat16:
		return T(v.Float32())
	case int:
		return T(v)
	case int64:
		return T(v)
	case int32:
		return T(v)
	case int16:
		return T(v)
	case int8:
		return T(v)
	case uint64:
		return T(v)
	case uint32:
		return T(v)
	case uint16:
		return T(v)
	case uint8:
		return T(v)
	case complex64:
		return T(real(v))
	case complex128:
		return T(real(v))
	}
	return T(0)
}

// UnsafeSliceForDType creates a slice of the corresponding dtype
// and casts it to any.
// It uses unsafe.Slice.
// Set `len` to the number of `DType` elements (not the number of bytes).
func UnsafeSliceForDType(dtype DType, unsafePtr unsafe.Pointer, len int) any {
	switch dtype {
	case Int64:
		return unsafe.Slice((*int64)(unsafePtr), len)
	case Int32:
		return unsafe.Slice((*int32)(unsafePtr), len)
	case Int16:
		return unsafe.Slice((*int16)(unsafePtr), len)
	case Int8:
		return unsafe.Slice((*int8)(unsafePtr), len)

	case Uint64:
		return unsafe.Slice((*uint64)(unsafePtr), len)
	case Uint32:
		return unsafe.Slice((*uint32)(unsafePtr), len)
	case Uint16:
		return unsafe.Slice((*uint16)(unsafePtr), len)
	case Uint8:
		return unsafe.Slice((*uint8)(unsafePtr), len)

	case Bool:
		return unsafe.Slice((*bool)(unsafePtr), len)

	case Float16:
		return unsafe.Slice((*float16.Float16)(unsafePtr), len)
	case BFloat16:
		return unsafe.Slice((*bfloat16.BFloat16)(unsafePtr), len)
	case Float32:
		return unsafe.Slice((*float32)(unsafePtr), len)
	case Float64:
		return unsafe.Slice((*float64)(unsafePtr), len)

	case Complex64:
		return unsafe.Slice((*complex64)(unsafePtr), len)
	case Complex128:
		return unsafe.Slice((*complex128)(unsafePtr), len)
	default:
		exceptions.Panicf("unknown dtype %q (%d) in UnsafeSliceForDType", dtype, dtype)
		panic(nil) // Quiet lint warning.
	}
}

// Pre-generate constant reflect.TypeOf for convenience.
var (
	float32Type  = reflect.TypeOf(float32(0))
	float64Type  = reflect.TypeOf(float64(0))
	float16Type  = reflect.TypeOf(float16.Float16(0))
	bfloat16Type = reflect.TypeOf(bfloat16.BFloat16(0))
)

// CastAsDType casts a numeric value to the corresponding for the DType.
// If the value is a slice it will convert to a newly allocated slice of
// the given DType.
//
// It doesn't work for complex numbers.
func CastAsDType(value any, dtype DType) any {
	typeOf := reflect.TypeOf(value)
	valueOf := reflect.ValueOf(value)
	newTypeOf := typeForSliceDType(typeOf, dtype)
	if typeOf.Kind() != reflect.Slice && typeOf.Kind() != reflect.Array {
		// Scalar value.
		if dtype == Bool {
			return !valueOf.IsZero()
		}
		if dtype == Complex64 {
			r := valueOf.Convert(float32Type).Interface().(float32)
			return complex(r, float32(0))
		}
		if dtype == Complex128 {
			r := valueOf.Convert(float64Type).Interface().(float64)
			return complex(r, float64(0))
		}
		if dtype == Float16 {
			v32 := valueOf.Convert(float32Type).Interface().(float32)
			return float16.Fromfloat32(v32)
		}
		if dtype == BFloat16 {
			v32 := valueOf.Convert(float32Type).Interface().(float32)
			return bfloat16.FromFloat32(v32)
		}
		// TODO: if adding support for non-native Go types (e.g: B16), we need
		//       to write our own conversion here.
		return valueOf.Convert(newTypeOf).Interface()
	}

	newValueOf := reflect.MakeSlice(newTypeOf, valueOf.Len(), valueOf.Len())
	for ii := 0; ii < valueOf.Len(); ii++ {
		elem := CastAsDType(valueOf.Index(ii).Interface(), dtype)
		newValueOf.Index(ii).Set(reflect.ValueOf(elem))
	}
	return newValueOf.Interface()
}

// typeForSliceDType recursively converts a type that is a (multi-dimension-) slice
// of some type, to a `reflect.Type` of a (multi-dimension-) slice of `dtype`.
// Arrays are converted to slices.
func typeForSliceDType(valueType reflect.Type, dtype DType) reflect.Type {
	if valueType.Kind() != reflect.Slice && valueType.Kind() != reflect.Array {
		// Base case for recursion, simply return the `reflect.Type` for the DType.
		return dtype.GoType()
	}
	subType := typeForSliceDType(valueType.Elem(), dtype)
	return reflect.SliceOf(subType) // Return a slice of the recursively converted type.
}
