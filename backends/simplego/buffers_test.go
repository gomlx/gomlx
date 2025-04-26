package simplego

import (
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestBuffers_Bytes(t *testing.T) {
	buf := backend.(*Backend).getBuffer(dtypes.Int32, 3)
	buf.shape = shapes.Make(dtypes.Int32, 3)
	require.Len(t, buf.flat.([]int32), 3)
	flatBytes := buf.mutableBytes()
	require.Len(t, flatBytes, 3*int(dtypes.Int32.Size()))
	flatBytes[0] = 1
	flatBytes[4] = 7
	flatBytes[8] = 3
	require.Equal(t, []int32{1, 7, 3}, buf.flat.([]int32))
}
