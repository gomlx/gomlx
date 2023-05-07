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
	"math"
	"reflect"
)

// DType indicates the type of the unit element of a Tensor (or its representation in
// a computation graph). It enumerates the known data types. So far only
// Bool, Int32, Int64, Float32 and Float64 are supported.
//
// The values of DType must match "tensorflow/compiler/xla/xla_data.pb.h", hence it needs
// to be an int32.
// TODO: do a little generate script to generate these automatically.
//
// See example in package shapes documentation.
type DType int32

//go:generate stringer -type=DType

// DType constants must match `tensorflow/compiler/xla/xla_data.proto`.
const (
	InvalidDType DType = iota
	Bool               // Bool, but also known as PRED in `xla_data.proto`.
	Int8               // S8
	Int16              // S16
	Int32              // S32
	Int64              // S64, in Go represented as int
	UInt8              // U8
	UInt16             // U16
	UInt32             // U32
	UInt64             // U64
	Float16            // F16
	Float32            // F32
	Float64            // F64

	BFloat16   DType = 16 // BF16
	Complex64  DType = 15 // C64
	Complex128 DType = 18 // C128

	Tuple      DType = 13
	OpaqueType DType = 14
	Token      DType = 17
)

// PRED type is an alias to Bool, used in `tensorflow/compiler/xla/xla_data.proto`.
const PRED = Bool

const (
	I32 = Int32
	I64 = Int64
	F32 = Float32
	F64 = Float64
)

// IsFloat returns whether dtype is a supported float -- float types not yet supported will return false.
func (dtype DType) IsFloat() bool {
	return dtype == Float32 || dtype == Float64
}

// IsInt returns whether dtype is a supported integer type -- float types not yet supported will return false.
func (dtype DType) IsInt() bool {
	return dtype == Int64 || dtype == Int32
}

func (dtype DType) IsSupported() bool {
	return dtype == Bool || dtype == Float32 || dtype == Float64 || dtype == Int64 || dtype == Int32
}

// Supported represents the Go types that are supported by the graph package. Used as a Generics constraint.
// See also Number.
type Supported interface {
	bool | float32 | float64 | int | int32
}

// Number represents the Go numeric types that are supported by graph package. Used as a Generics constraint.
// Notice that "int" becomes int64 in the implementation. Since it needs a 1:1 mapping, it doesn't support the native
// (Go) int64 type.
type Number interface {
	float32 | float64 | int | int32
}

// GoFloat represent a continuous Go numeric type, supported by GoMLX.
type GoFloat interface {
	float32 | float64
}

// MultiDimensionSlice lists the Go types a Tensor can be converted to/from. There are no recursions in
// generics constraints definitions, so we enumerate up to 7 levels of slices. Feel free to add
// more if needed, the implementation will work with any arbitrary number.
type MultiDimensionSlice interface {
	bool | int | int32 | float32 | float64 |
		[]bool | []int | []int32 | []float32 | []float64 |
		[][]bool | [][]int | [][]int32 | [][]float32 | [][]float64 |
		[][][]bool | [][][]int | [][][]int32 | [][][]float32 | [][][]float64 |
		[][][][]bool | [][][][]int | [][][][]int32 | [][][][]float32 | [][][][]float64 |
		[][][][][]bool | [][][][][]int | [][][][][]int32 | [][][][][]float32 | [][][][][]float64 |
		[][][][][][]bool | [][][][][][]int | [][][][][][]int32 | [][][][][][]float32 | [][][][][][]float64 |
		[][][][][][][]bool | [][][][][][][]int | [][][][][][][]int32 | [][][][][][][]float32 | [][][][][][][]float64 |
		[][][][][][][][]bool | [][][][][][][][]int | [][][][][][][][]int32 | [][][][][][][][]float32 | [][][][][][][][]float64 |
		[][][][][][][][][]bool | [][][][][][][][][]int | [][][][][][][][][]int32 | [][][][][][][][][]float32 | [][][][][][][][][]float64
}

func DTypeGeneric[T Supported]() DType {
	var t T
	switch (any(t)).(type) {
	case float32:
		return Float32
	case float64:
		return Float64
	case int:
		return Int64
	case int32:
		return Int32
	case bool:
		return Bool
	}
	return InvalidDType
}

func DTypeForType(t reflect.Type) DType {
	switch t.Kind() {
	case reflect.Int:
		return Int64
	case reflect.Int32:
		return Int32
	case reflect.Float32:
		return Float32
	case reflect.Float64:
		return Float64
	case reflect.Bool:
		return Bool
	}
	return InvalidDType
}

func TypeForDType(dtype DType) reflect.Type {
	switch dtype {
	case Int64:
		return reflect.TypeOf(int(0))
	case Int32:
		return reflect.TypeOf(int32(0))
	case Float32:
		return reflect.TypeOf(float32(0))
	case Float64:
		return reflect.TypeOf(float64(0))
	case Bool:
		return reflect.TypeOf(true)
	}
	return reflect.TypeOf(nil)
}

// CastAsDType casts a numeric value to the corresponding for the DType.
// If the value is a slice it will convert to a newly allocated slice of
// the given DType.
func CastAsDType(value any, dtype DType) any {
	typeOf := reflect.TypeOf(value)
	valueOf := reflect.ValueOf(value)
	newTypeOf := typeForSliceDType(typeOf, dtype)
	if typeOf.Kind() != reflect.Slice && typeOf.Kind() != reflect.Array {
		// Scalar value.
		if newTypeOf.Kind() == reflect.Bool {
			return !valueOf.IsZero()
		}
		// TODO: if adding support for non-native Go types (e.g: B16), we need
		// to write our own conversion here.
		return valueOf.Convert(newTypeOf).Interface()
	}

	newValueOf := reflect.MakeSlice(newTypeOf, valueOf.Len(), valueOf.Len())
	for ii := 0; ii < valueOf.Len(); ii++ {
		elem := CastAsDType(valueOf.Index(ii).Interface(), dtype)
		newValueOf.Index(ii).Set(reflect.ValueOf(elem))
	}
	return newValueOf.Interface()
}

func typeForSliceDType(valueType reflect.Type, dtype DType) reflect.Type {
	if valueType.Kind() != reflect.Slice && valueType.Kind() != reflect.Array {
		return TypeForDType(dtype)
	}
	subType := typeForSliceDType(valueType.Elem(), dtype)
	return reflect.SliceOf(subType)
}

func LowestValueForDType(dtype DType) any {
	switch dtype {
	case Int64:
		return math.MinInt64
	case Int32:
		return math.MinInt32
	case Float32:
		return float32(math.Inf(-1))
	case Float64:
		return math.Inf(-1)
	case Bool:
		return false
	}
	return math.NaN()
}
