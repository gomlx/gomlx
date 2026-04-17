// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/stretchr/testify/assert"
)

func TestConvertPackedInt4ToInt8(t *testing.T) {
	// Packed Int4 → Int8: unpacks nibbles with sign extension.
	// Byte 0xF0 = low nibble 0x0 (0), high nibble 0xF (-1).
	// Byte 0x87 = low nibble 0x7 (7), high nibble 0x8 (-8).
	srcData := []byte{0xF0, 0x87}
	srcShape := shapes.Make(dtypes.Int4, 4) // 4 Int4 elements packed in 2 bytes
	srcBuf := &Buffer{shape: srcShape, flat: srcData, inUse: true}

	dstShape := shapes.Make(dtypes.Int8, 4)
	dstBuf := &Buffer{shape: dstShape, flat: make([]int8, 4), inUse: true}

	tmpAny, tmpErr := convertDTypePairMap.Get(dtypes.Int4, dtypes.Int8)
	if tmpErr != nil {
		panic(tmpErr)
	}
	convertFn := tmpAny.(convertFnType)
	convertFn(srcBuf, dstBuf)

	result := dstBuf.flat.([]int8)
	assert.Equal(t, int8(0), result[0])
	assert.Equal(t, int8(-1), result[1])
	assert.Equal(t, int8(7), result[2])
	assert.Equal(t, int8(-8), result[3])
}

func TestConvertPackedUint4ToUint8(t *testing.T) {
	// Packed Uint4 → Uint8: unpacks nibbles (no sign extension).
	// Byte 0xF0 = low nibble 0, high nibble 15.
	// Byte 0x87 = low nibble 7, high nibble 8.
	srcData := []byte{0xF0, 0x87}
	srcShape := shapes.Make(dtypes.Uint4, 4) // 4 Uint4 elements packed in 2 bytes
	srcBuf := &Buffer{shape: srcShape, flat: srcData, inUse: true}

	dstShape := shapes.Make(dtypes.Uint8, 4)
	dstBuf := &Buffer{shape: dstShape, flat: make([]uint8, 4), inUse: true}

	tmpAny, tmpErr := convertDTypePairMap.Get(dtypes.Uint4, dtypes.Uint8)
	if tmpErr != nil {
		panic(tmpErr)
	}
	convertFn := tmpAny.(convertFnType)
	convertFn(srcBuf, dstBuf)

	result := dstBuf.flat.([]uint8)
	assert.Equal(t, uint8(0), result[0])
	assert.Equal(t, uint8(15), result[1])
	assert.Equal(t, uint8(7), result[2])
	assert.Equal(t, uint8(8), result[3])
}

func TestConvertPackedInt4ToFloat32(t *testing.T) {
	// Packed Int4 → Float32: unpacks and converts.
	srcData := []byte{0xF0, 0x87}
	srcShape := shapes.Make(dtypes.Int4, 4)
	srcBuf := &Buffer{shape: srcShape, flat: srcData, inUse: true}

	dstShape := shapes.Make(dtypes.Float32, 4)
	dstBuf := &Buffer{shape: dstShape, flat: make([]float32, 4), inUse: true}

	tmpAny, tmpErr := convertDTypePairMap.Get(dtypes.Int4, dtypes.Float32)
	if tmpErr != nil {
		panic(tmpErr)
	}
	convertFn := tmpAny.(convertFnType)
	convertFn(srcBuf, dstBuf)

	result := dstBuf.flat.([]float32)
	assert.Equal(t, float32(0), result[0])
	assert.Equal(t, float32(-1), result[1])
	assert.Equal(t, float32(7), result[2])
	assert.Equal(t, float32(-8), result[3])
}

func TestConvertPackedInt2ToInt8(t *testing.T) {
	// Packed Int2 → Int8: unpacks 2-bit values with sign extension.
	// Byte 0b11_10_01_00 = 0xE4: values 0, 1, -2, -1.
	srcData := []byte{0xE4}
	srcShape := shapes.Make(dtypes.Int2, 4) // 4 Int2 elements packed in 1 byte
	srcBuf := &Buffer{shape: srcShape, flat: srcData, inUse: true}

	dstShape := shapes.Make(dtypes.Int8, 4)
	dstBuf := &Buffer{shape: dstShape, flat: make([]int8, 4), inUse: true}

	tmpAny, tmpErr := convertDTypePairMap.Get(dtypes.Int2, dtypes.Int8)
	if tmpErr != nil {
		panic(tmpErr)
	}
	convertFn := tmpAny.(convertFnType)
	convertFn(srcBuf, dstBuf)

	result := dstBuf.flat.([]int8)
	assert.Equal(t, int8(0), result[0])
	assert.Equal(t, int8(1), result[1])
	assert.Equal(t, int8(-2), result[2])
	assert.Equal(t, int8(-1), result[3])
}

func TestConvertPackedUint2ToUint8(t *testing.T) {
	// Packed Uint2 → Uint8: unpacks 2-bit values (no sign extension).
	// Byte 0b11_10_01_00 = 0xE4: values 0, 1, 2, 3.
	srcData := []byte{0xE4}
	srcShape := shapes.Make(dtypes.Uint2, 4) // 4 Uint2 elements packed in 1 byte
	srcBuf := &Buffer{shape: srcShape, flat: srcData, inUse: true}

	dstShape := shapes.Make(dtypes.Uint8, 4)
	dstBuf := &Buffer{shape: dstShape, flat: make([]uint8, 4), inUse: true}

	tmpAny, tmpErr := convertDTypePairMap.Get(dtypes.Uint2, dtypes.Uint8)
	if tmpErr != nil {
		panic(tmpErr)
	}
	convertFn := tmpAny.(convertFnType)
	convertFn(srcBuf, dstBuf)

	result := dstBuf.flat.([]uint8)
	assert.Equal(t, uint8(0), result[0])
	assert.Equal(t, uint8(1), result[1])
	assert.Equal(t, uint8(2), result[2])
	assert.Equal(t, uint8(3), result[3])
}

func TestExecSpecialOps_ConvertDType(t *testing.T) {
	// Test int32 to float32
	y0 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ConvertDType(x, dtypes.Float32)
	}, int32(42))
	// fmt.Printf("\ty0=%s\n", y0.GoStr())
	assert.Equal(t, float32(42.0), y0.Value())

	// Test float32 to bfloat16
	y1 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ConvertDType(x, dtypes.BFloat16)
	}, float32(3.14))
	// fmt.Printf("\ty1=%s\n", y1.GoStr())
	assert.Equal(t, bf16(3.14), y1.Value())

	// Test bfloat16 to int32
	y2 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ConvertDType(x, dtypes.Int32)
	}, bf16(7.8))
	// fmt.Printf("\ty2=%s\n", y2.GoStr())
	assert.Equal(t, int32(7), y2.Value())

	// Test bool to int32
	y3 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ConvertDType(x, dtypes.Int32)
	}, true)
	// fmt.Printf("\ty3=%s\n", y3.GoStr())
	assert.Equal(t, int32(1), y3.Value())

	// Test float32 to bool
	y4 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ConvertDType(x, dtypes.Bool)
	}, float32(1.0))
	// fmt.Printf("\ty4=%s\n", y4.GoStr())
	assert.Equal(t, true, y4.Value())
}
