// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBitcastNibbleRoundtrip_Uint4(t *testing.T) {
	// Pack nibbles into uint8, then unpack back; verify roundtrip.
	// Logical nibbles: [0, 15, 7, 8] → packed: [0xF0, 0x87] → unpacked: [0, 15, 7, 8]
	nibbles := []uint8{0, 15, 7, 8}
	nibbleShape := shapes.Make(dtypes.Uint4, 4)

	srcBuf := &Buffer{shape: nibbleShape, flat: nibbles}
	packShape := shapes.Make(dtypes.Uint8, 2)
	packBuf := &Buffer{shape: packShape, flat: make([]uint8, 2)}
	packed, err := execBitcastNibblesToUint8(srcBuf, packBuf, dtypes.Uint4)
	require.NoError(t, err)

	// Verify packed bytes.
	packedData := packed.flat.([]uint8)
	assert.Equal(t, uint8(0xF0), packedData[0]) // low=0, high=15
	assert.Equal(t, uint8(0x87), packedData[1]) // low=7, high=8

	// Unpack back.
	unpackShape := shapes.Make(dtypes.Uint4, 4)
	unpackBuf := &Buffer{shape: unpackShape, flat: make([]uint8, 4)}
	unpacked, err := execBitcastUint8ToNibbles(packed, unpackBuf, dtypes.Uint4)
	require.NoError(t, err)

	unpackedData := unpacked.flat.([]uint8)
	for i := range nibbles {
		assert.Equal(t, nibbles[i], unpackedData[i], "index %d", i)
	}
}

func TestBitcastNibbleRoundtrip_Int4(t *testing.T) {
	// Int4 nibbles: [0, -1, 7, -8] → packed → unpacked should roundtrip.
	nibbles := []int8{0, -1, 7, -8}
	nibbleShape := shapes.Make(dtypes.Int4, 4)

	srcBuf := &Buffer{shape: nibbleShape, flat: nibbles}
	packShape := shapes.Make(dtypes.Uint8, 2)
	packBuf := &Buffer{shape: packShape, flat: make([]uint8, 2)}
	packed, err := execBitcastNibblesToUint8(srcBuf, packBuf, dtypes.Int4)
	require.NoError(t, err)

	// Verify packed bytes.
	packedData := packed.flat.([]uint8)
	// nibble 0 → 0x0, nibble -1 → 0xF (unsigned representation): byte = 0xF0
	assert.Equal(t, uint8(0xF0), packedData[0])
	// nibble 7 → 0x7, nibble -8 → 0x8: byte = 0x87
	assert.Equal(t, uint8(0x87), packedData[1])

	// Unpack back.
	unpackShape := shapes.Make(dtypes.Int4, 4)
	unpackBuf := &Buffer{shape: unpackShape, flat: make([]int8, 4)}
	unpacked, err := execBitcastUint8ToNibbles(packed, unpackBuf, dtypes.Int4)
	require.NoError(t, err)

	unpackedData := unpacked.flat.([]int8)
	for i := range nibbles {
		assert.Equal(t, nibbles[i], unpackedData[i], "index %d", i)
	}
}
