package simplego

import (
	"runtime"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
)

func TestBuffers_Bytes(t *testing.T) {
	buf := backend.(*Backend).getBuffer(dtypes.Int32, 3)
	buf.shape = shapes.Make(dtypes.Int32, 3)
	buf.Zeros()
	require.Len(t, buf.flat.([]int32), 3)
	flatBytes := buf.mutableBytes()
	require.Len(t, flatBytes, 3*int(dtypes.Int32.Size()))
	flatBytes[0] = 1
	flatBytes[4] = 7
	flatBytes[8] = 3
	require.Equal(t, []int32{1, 7, 3}, buf.flat.([]int32))
	runtime.KeepAlive(buf)
}

func TestBuffers_Fill(t *testing.T) {
	buf := backend.(*Backend).getBuffer(dtypes.Int32, 3)
	buf.shape = shapes.Make(dtypes.Int32, 3)
	require.Len(t, buf.flat.([]int32), 3)

	err := buf.Fill(int32(3))
	require.NoError(t, err)
	require.Equal(t, []int32{3, 3, 3}, buf.flat.([]int32))

	buf.Zeros()
	require.Equal(t, []int32{0, 0, 0}, buf.flat.([]int32))
}
