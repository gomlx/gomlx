// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package packgemm

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCast(t *testing.T) {
	// Test int16 <-> uint16
	i16 := []int16{-1, 0, 1}
	u16 := castHalfPrecisionSlice[uint16](i16)
	assert.Equal(t, len(i16), len(u16))
	assert.Equal(t, uint16(0xffff), u16[0]) // -1 in 2's complement is 0xffff
	assert.Equal(t, uint16(0), u16[1])
	assert.Equal(t, uint16(1), u16[2])

	// Test round trip
	i16Back := castHalfPrecisionSlice[int16](u16)
	assert.Equal(t, i16, i16Back)
	// Verify memory sharing
	u16[0] = 0x1234
	assert.Equal(t, int16(0x1234), i16[0])
}
