// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/stretchr/testify/require"
)

func TestBuffers_Bytes(t *testing.T) {
	buf, err := backend.(*Backend).getBuffer(dtypes.Int32, 3)
	require.NoError(t, err)
	buf.shape = shapes.Make(dtypes.Int32, 3)
	buf.Zeros()
	require.Len(t, buf.flat.([]int32), 3)
	flatBytes, err := buf.mutableBytes()
	require.NoError(t, err)
	require.Len(t, flatBytes, 3*int(dtypes.Int32.Size()))
	flatBytes[0] = 1
	flatBytes[4] = 7
	flatBytes[8] = 3
	require.Equal(t, []int32{1, 7, 3}, buf.flat.([]int32))
	runtime.KeepAlive(buf)
}

func TestBuffers_Fill(t *testing.T) {
	buf, err := backend.(*Backend).getBuffer(dtypes.Int32, 3)
	require.NoError(t, err)
	buf.shape = shapes.Make(dtypes.Int32, 3)
	require.Len(t, buf.flat.([]int32), 3)
	require.NoError(t, buf.Fill(int32(3)))
	require.Equal(t, []int32{3, 3, 3}, buf.flat.([]int32))

	buf.Zeros()
	require.Equal(t, []int32{0, 0, 0}, buf.flat.([]int32))
}

func TestBucketSize(t *testing.T) {
	tests := []struct {
		input, expected int
	}{
		{0, 0},
		{1, 1},
		{2, 2},
		{3, 4},
		{4, 4},
		{5, 8},
		{7, 8},
		{8, 8},
		{9, 16},
		{33, 64},
		{45, 64},
		{64, 64},
		{65, 128},
		{100, 128},
		{1023, 1024},
		{1024, 1024},
		{1025, 2048},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("%d", tt.input), func(t *testing.T) {
			require.Equal(t, tt.expected, bucketSize(tt.input))
		})
	}
}

func TestSubSliceFlat(t *testing.T) {
	original := make([]float32, 8)
	for i := range original {
		original[i] = float32(i)
	}
	sub := subSliceFlat(original, 5).([]float32)
	require.Len(t, sub, 5)
	require.Equal(t, 8, cap(sub))
	require.Equal(t, []float32{0, 1, 2, 3, 4}, sub)

	// Bool type.
	boolSlice := make([]bool, 4)
	boolSlice[0] = true
	boolSub := subSliceFlat(boolSlice, 2).([]bool)
	require.Len(t, boolSub, 2)
	require.Equal(t, []bool{true, false}, boolSub)
}

func TestFullCapFlat(t *testing.T) {
	original := make([]float32, 8)
	for i := range original {
		original[i] = float32(i)
	}
	sub := original[:3]
	require.Len(t, sub, 3)

	restored := fullCapFlat(sub).([]float32)
	require.Len(t, restored, 8)
	require.Equal(t, 8, cap(restored))
	// All 8 elements from the original backing array are preserved.
	for i := 0; i < 8; i++ {
		require.Equal(t, float32(i), restored[i])
	}
}

func TestBufferPoolBucketing(t *testing.T) {
	b := backend.(*Backend)

	// Get a buffer of size 3 (buckets to 4), verify flat len and cap.
	buf1, err := b.getBuffer(dtypes.Float32, 3)
	require.NoError(t, err)
	require.Len(t, buf1.flat.([]float32), 3)
	require.Equal(t, 4, cap(buf1.flat.([]float32)), "flat capacity should be bucketed to 4")
	buf1.shape = shapes.Make(dtypes.Float32, 3)
	b.putBuffer(buf1)

	// Get a buffer of size 4 (same bucket). Verify it has correct len/cap.
	buf2, err := b.getBuffer(dtypes.Float32, 4)
	require.NoError(t, err)
	require.Len(t, buf2.flat.([]float32), 4)
	require.Equal(t, 4, cap(buf2.flat.([]float32)), "flat capacity should be bucketed to 4")
	buf2.shape = shapes.Make(dtypes.Float32, 4)
	b.putBuffer(buf2)

	// Get a buffer of size 3 again (same bucket 4). Verify correct len/cap.
	buf3, err := b.getBuffer(dtypes.Float32, 3)
	require.NoError(t, err)
	require.Len(t, buf3.flat.([]float32), 3)
	require.Equal(t, 4, cap(buf3.flat.([]float32)), "flat capacity should be bucketed to 4")
	buf3.shape = shapes.Make(dtypes.Float32, 3)
	b.putBuffer(buf3)

	// Different bucket: size 5 buckets to 8.
	buf4, err := b.getBuffer(dtypes.Float32, 5)
	require.NoError(t, err)
	require.Len(t, buf4.flat.([]float32), 5)
	require.Equal(t, 8, cap(buf4.flat.([]float32)), "flat capacity should be bucketed to 8")
	buf4.shape = shapes.Make(dtypes.Float32, 5)
	b.putBuffer(buf4)
}

func TestBufferPoolBucketing_MutableBytes(t *testing.T) {
	b := backend.(*Backend)

	// Size 3 buckets to 4, but flat is sub-sliced to 3.
	buf, err := b.getBuffer(dtypes.Int32, 3)
	require.NoError(t, err)
	buf.shape = shapes.Make(dtypes.Int32, 3)
	buf.Zeros()

	// mutableBytes should return 3*4=12 bytes, not 4*4=16.
	bytes, err2 := buf.mutableBytes()
	require.NoError(t, err2)
	require.Len(t, bytes, 3*int(dtypes.Int32.Size()))
	b.putBuffer(buf)
}
