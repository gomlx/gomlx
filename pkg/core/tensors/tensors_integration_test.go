package tensors_test

import (
	"fmt"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/simplego"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/stretchr/testify/require"
)

// TestFinalizeAfterBackend checks that it's ok to finalize a Tensor after its backend.
func TestFinalizeAfterBackend(t *testing.T) {
	// We create a temporary SimpleGo backend for this test.
	deviceNum := backends.DeviceNum(0) // For `go` backend, there is only one device.
	for _, share := range []bool{false, true} {
		t.Run(fmt.Sprintf("Shared=%v", share), func(t *testing.T) {
			backend, err := simplego.New("")
			require.NoError(t, err)
			x := tensors.FromAnyValue([][]float32{{1}, {2}})
			err = x.MaterializeOnDevice(backend, share, deviceNum)
			require.NoError(t, err)
			backend.Finalize()
			require.NoError(t, x.FinalizeAll())
		})
	}
}

// TestInvalidAccessAfterBackendFinalized checks that it panics/returns an error if one attempts to access the
// tensor after its device was destroyed.
func TestInvalidAccessAfterBackendFinalized(t *testing.T) {
	// We create a temporary SimpleGo backend for this test.
	deviceNum := backends.DeviceNum(0) // For `go` backend, there is only one device.
	for _, share := range []bool{false, true} {
		t.Run(fmt.Sprintf("Shared=%v", share), func(t *testing.T) {
			backend, err := simplego.New("")
			require.NoError(t, err)
			x := tensors.FromAnyValue([][]float32{{1}, {2}})
			require.NoError(t, x.MaterializeOnDevice(backend, share, deviceNum))
			x.FinalizeLocal()
			backend.Finalize()
			require.Panics(t, func() {
				_ = x.Value()
				fmt.Println("x shouldn't be accessible after its backend was finalized!")
			})
		})
	}
}
