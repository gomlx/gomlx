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
