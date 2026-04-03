// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dtypes

import (
	"testing"
	"unsafe"

	"github.com/stretchr/testify/require"
)

func TestUnsafeByteSlice(t *testing.T) {
	// Test with float32
	f32s := []float32{1.0, 2.0, 3.0}
	bytesF32 := UnsafeByteSlice(f32s)
	require.Len(t, bytesF32, len(f32s)*4)

	// Test with int64
	i64s := []int64{10, 20, 30}
	bytesI64 := UnsafeByteSlice(i64s)
	require.Len(t, bytesI64, len(i64s)*8)
}

func TestUnsafeByteSliceFromAny(t *testing.T) {
	// Test with float64
	f64s := []float64{1.5, 2.5}
	bytesF64 := UnsafeByteSliceFromAny(f64s)
	expectedBytesF64 := UnsafeByteSlice(f64s)
	require.Equal(t, expectedBytesF64, bytesF64)

	// Test with int32
	i32s := []int32{1, 2, 3, 4}
	bytesI32 := UnsafeByteSliceFromAny(i32s)
	expectedBytesI32 := UnsafeByteSlice(i32s)
	require.Equal(t, expectedBytesI32, bytesI32)

	// Test panic on unsupported slice
	require.Panics(t, func() { UnsafeByteSliceFromAny([]string{"a"}) })
}

func TestUnsafeSliceFromBytes(t *testing.T) {
	// Test with float32
	f32s := []float32{1.0, 2.0, 3.0}
	bytesF32 := UnsafeByteSlice(f32s)
	recoveredF32s := UnsafeSliceFromBytes[float32](unsafe.Pointer(&bytesF32[0]), len(f32s))
	require.Equal(t, f32s, recoveredF32s)

	// Test with int64
	i64s := []int64{10, 20, 30}
	bytesI64 := UnsafeByteSlice(i64s)
	recoveredI64s := UnsafeSliceFromBytes[int64](unsafe.Pointer(&bytesI64[0]), len(i64s))
	require.Equal(t, i64s, recoveredI64s)
}

func TestUnsafeAnySliceFromBytes(t *testing.T) {
	// Test with float64
	f64s := []float64{1.5, 2.5}
	bytesF64 := UnsafeByteSlice(f64s)
	recoveredF64sAny := UnsafeAnySliceFromBytes(unsafe.Pointer(&bytesF64[0]), Float64, len(f64s))
	recoveredF64s, ok := recoveredF64sAny.([]float64)
	require.True(t, ok)
	require.Equal(t, f64s, recoveredF64s)

	// Test with int32
	i32s := []int32{1, 2, 3, 4}
	bytesI32 := UnsafeByteSlice(i32s)
	recoveredI32sAny := UnsafeAnySliceFromBytes(unsafe.Pointer(&bytesI32[0]), Int32, len(i32s))
	recoveredI32s, ok := recoveredI32sAny.([]int32)
	require.True(t, ok)
	require.Equal(t, i32s, recoveredI32s)
}
