// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package tensors

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestUnpackSubByteAt(t *testing.T) {
	// Uint4
	{
		packed := []uint8{0x12, 0x34}
		assert.Equal(t, int8(0x02), UnpackSubByteAt(packed, dtypes.Uint4, 0))
		assert.Equal(t, int8(0x01), UnpackSubByteAt(packed, dtypes.Uint4, 1))
		assert.Equal(t, int8(0x04), UnpackSubByteAt(packed, dtypes.Uint4, 2))
		assert.Equal(t, int8(0x03), UnpackSubByteAt(packed, dtypes.Uint4, 3))
	}

	// Int4
	{
		// 0xF = -1, 0x8 = -8, 0x7 = 7, 0x0 = 0
		packed := []uint8{0x8F, 0x07}
		assert.Equal(t, int8(-1), UnpackSubByteAt(packed, dtypes.Int4, 0))
		assert.Equal(t, int8(-8), UnpackSubByteAt(packed, dtypes.Int4, 1))
		assert.Equal(t, int8(7), UnpackSubByteAt(packed, dtypes.Int4, 2))
		assert.Equal(t, int8(0), UnpackSubByteAt(packed, dtypes.Int4, 3))
	}

	// Uint2
	{
		// 0b1101_1001 = 0xD9
		// Indices: 0: 01 (1), 1: 10 (2), 2: 01 (1), 3: 11 (3)
		packed := []uint8{0xD9}
		assert.Equal(t, int8(1), UnpackSubByteAt(packed, dtypes.Uint2, 0))
		assert.Equal(t, int8(2), UnpackSubByteAt(packed, dtypes.Uint2, 1))
		assert.Equal(t, int8(1), UnpackSubByteAt(packed, dtypes.Uint2, 2))
		assert.Equal(t, int8(3), UnpackSubByteAt(packed, dtypes.Uint2, 3))
	}

	// Int2
	{
		// 0b1110_0100 = 0xE4
		// Indices: 0: 00 (0), 1: 01 (1), 2: 10 (-2), 3: 11 (-1)
		packed := []uint8{0xE4}
		assert.Equal(t, int8(0), UnpackSubByteAt(packed, dtypes.Int2, 0))
		assert.Equal(t, int8(1), UnpackSubByteAt(packed, dtypes.Int2, 1))
		assert.Equal(t, int8(-2), UnpackSubByteAt(packed, dtypes.Int2, 2))
		assert.Equal(t, int8(-1), UnpackSubByteAt(packed, dtypes.Int2, 3))
	}

	// Uint1
	{
		// 0b1101_1001 = 0xD9
		// Indices: 0: 1, 1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 1, 7: 1
		packed := []uint8{0xD9}
		assert.Equal(t, int8(1), UnpackSubByteAt(packed, dtypes.Uint1, 0))
		assert.Equal(t, int8(0), UnpackSubByteAt(packed, dtypes.Uint1, 1))
		assert.Equal(t, int8(0), UnpackSubByteAt(packed, dtypes.Uint1, 2))
		assert.Equal(t, int8(1), UnpackSubByteAt(packed, dtypes.Uint1, 3))
		assert.Equal(t, int8(1), UnpackSubByteAt(packed, dtypes.Uint1, 4))
		assert.Equal(t, int8(0), UnpackSubByteAt(packed, dtypes.Uint1, 5))
		assert.Equal(t, int8(1), UnpackSubByteAt(packed, dtypes.Uint1, 6))
		assert.Equal(t, int8(1), UnpackSubByteAt(packed, dtypes.Uint1, 7))
	}

	// Int1
	{
		// 0b1110_0100 = 0xE4
		// Indices: 0: 0, 1: 0, 2: 1 (-1), 3: 0, 4: 0, 5: 1 (-1), 6: 1 (-1), 7: 1 (-1)
		packed := []uint8{0xE4}
		assert.Equal(t, int8(0), UnpackSubByteAt(packed, dtypes.Int1, 0))
		assert.Equal(t, int8(0), UnpackSubByteAt(packed, dtypes.Int1, 1))
		assert.Equal(t, int8(-1), UnpackSubByteAt(packed, dtypes.Int1, 2))
		assert.Equal(t, int8(0), UnpackSubByteAt(packed, dtypes.Int1, 3))
		assert.Equal(t, int8(0), UnpackSubByteAt(packed, dtypes.Int1, 4))
		assert.Equal(t, int8(-1), UnpackSubByteAt(packed, dtypes.Int1, 5))
		assert.Equal(t, int8(-1), UnpackSubByteAt(packed, dtypes.Int1, 6))
		assert.Equal(t, int8(-1), UnpackSubByteAt(packed, dtypes.Int1, 7))
	}

	// Out-of-bounds
	{
		packed := []uint8{0x12}
		assert.Panics(t, func() {
			UnpackSubByteAt(packed, dtypes.Uint4, 2)
		}, "Expected panic for out-of-bounds index (one-past-end)")
		assert.Panics(t, func() {
			UnpackSubByteAt(packed, dtypes.Uint4, -1)
		}, "Expected panic for out-of-bounds index (-1)")
	}
}

func TestUnpackSubBytes(t *testing.T) {
	// Test Int1
	{
		packed := []uint8{0x01} // 0b0000_0001 -> [1, 0, 0, 0, 0, 0, 0, 0] if Uint1, but [ -1, 0, 0, 0, 0, 0, 0, 0] if Int1
		unpacked := UnpackSubBytes(packed, dtypes.Int1, 0)
		assert.Equal(t, []int8{-1, 0, 0, 0, 0, 0, 0, 0}, unpacked)
	}

	// Test numValues truncation
	{
		packed := []uint8{0x01}
		unpacked := UnpackSubBytes(packed, dtypes.Int1, 2)
		assert.Equal(t, []int8{-1, 0}, unpacked)
	}
}
