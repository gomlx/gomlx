package xla

import (
	"fmt"
	"os"
	"testing"

	"google3/third_party/golang/testify/assert/assert"
	"github.com/stretchr/testify/require"
)

func TestLibDevice(t *testing.T) {
	const defaultCudaDir = "/foo"
	require.NoError(t, os.Setenv(CudaDirKey, defaultCudaDir))

	// If XLA_FLAGS doesn't set any cuda directory, and fall back to CUDA_DIR:
	wantDefault := fmt.Sprintf("%s=%s", XlaFlagGpuCudaDataDir, defaultCudaDir)

	xlaFlagstests := []struct {
		Input, Want string
	}{
		// Set flag from CUDA_DIR
		{"", wantDefault},
		{"   ", wantDefault},
		{"--help", "--help " + wantDefault},

		// Remove spurious space, but preserve flags.
		{XlaFlagGpuCudaDataDir + "=/abc  --help", XlaFlagGpuCudaDataDir + "=/abc --help"},

		// Flag overwritten, when it's defined more than once, and value separated by space (instead of "=")
		{XlaFlagGpuCudaDataDir + "=/abc  --help " + XlaFlagGpuCudaDataDir + " ./def", "--help " + XlaFlagGpuCudaDataDir + "=./def"},
	}

	for _, tt := range xlaFlagstests {
		require.NoError(t, os.Setenv(XlaFlagsKey, tt.Input))
		PresetXlaFlagsCudaDir()
		got := os.Getenv(XlaFlagsKey)
		assert.Equalf(t, tt.Want, got, "For PresetXlaFlagsCudaDir() with XLA_FLAGS=%q", tt.Input)
	}
}
