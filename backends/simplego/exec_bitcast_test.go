// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBitcast_Uint8ToUint4_PureReinterpret(t *testing.T) {
	// Bitcast uint8[2] → Uint4[4]: raw bytes stay the same.
	// Byte 0xF0 = low nibble 0, high nibble 15.
	// Byte 0x87 = low nibble 7, high nibble 8.
	srcData := []uint8{0xF0, 0x87}
	srcShape := shapes.Make(dtypes.Uint8, 2)
	srcBuf := &Buffer{shape: srcShape, flat: srcData, inUse: true}

	dstShape := shapes.Make(dtypes.Uint4, 4)
	node := &Node{shape: dstShape}

	// Not owned: should copy bytes without unpacking.
	result, err := execBitcast(nil, node, []*Buffer{srcBuf}, []bool{false})
	require.NoError(t, err)
	assert.Equal(t, dstShape, result.shape)

	// Raw bytes should be identical to source.
	resultData := result.flat.([]uint8)
	assert.Equal(t, srcData, resultData)
}

func TestBitcast_Uint8ToInt4_PureReinterpret(t *testing.T) {
	// Bitcast uint8[2] → Int4[4]: raw bytes stay the same.
	srcData := []uint8{0xF0, 0x87}
	srcShape := shapes.Make(dtypes.Uint8, 2)
	srcBuf := &Buffer{shape: srcShape, flat: srcData, inUse: true}

	dstShape := shapes.Make(dtypes.Int4, 4)
	node := &Node{shape: dstShape}

	result, err := execBitcast(nil, node, []*Buffer{srcBuf}, []bool{false})
	require.NoError(t, err)
	assert.Equal(t, dstShape, result.shape)

	// Raw bytes should be identical — Bitcast doesn't unpack.
	resultData := result.flat.([]uint8)
	assert.Equal(t, srcData, resultData)
}

func TestBitcast_Uint8ToInt4_OwnedReuse(t *testing.T) {
	// When owned, Bitcast should reuse the buffer.
	srcData := []uint8{0xAB, 0xCD}
	srcShape := shapes.Make(dtypes.Uint8, 2)
	srcBuf := &Buffer{shape: srcShape, flat: srcData, inUse: true}

	dstShape := shapes.Make(dtypes.Int4, 4)
	node := &Node{shape: dstShape}

	result, err := execBitcast(nil, node, []*Buffer{srcBuf}, []bool{true})
	require.NoError(t, err)
	assert.Equal(t, dstShape, result.shape)
	// Should be the exact same buffer (reused).
	assert.Same(t, srcBuf, result)
	assert.Equal(t, srcData, result.flat.([]uint8))
}

func TestBitcast_SameSize_Uint8ToInt8(t *testing.T) {
	// Same bit-width, different Go type: should copy bytes.
	backend, err := New("")
	require.NoError(t, err)
	defer backend.Finalize()

	srcData := []uint8{0xFF, 0x80, 0x01}
	srcShape := shapes.Make(dtypes.Uint8, 3)
	srcBuf := &Buffer{shape: srcShape, flat: srcData, inUse: true}

	dstShape := shapes.Make(dtypes.Int8, 3)
	node := &Node{shape: dstShape}

	result, err := execBitcast(backend.(*Backend), node, []*Buffer{srcBuf}, []bool{false})
	require.NoError(t, err)
	assert.Equal(t, dstShape, result.shape)

	// Verify byte-level identity.
	resultData := result.flat.([]int8)
	assert.Equal(t, int8(-1), resultData[0])  // 0xFF
	assert.Equal(t, int8(-128), resultData[1]) // 0x80
	assert.Equal(t, int8(1), resultData[2])    // 0x01
}
