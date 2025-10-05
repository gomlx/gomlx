package tensors_test

import (
	"fmt"
	"testing"

	"github.com/gomlx/gomlx/backends/simplego"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/stretchr/testify/require"
)

// TestFinalizeAfterBackend checks that it's ok to finalize a Tensor after its backend.
func TestFinalizeAfterBackend(t *testing.T) {
	// We create a temporary SimpleGo backend for this test.
	for _, share := range []bool{false, true} {
		t.Run(fmt.Sprintf("Shared=%v", share), func(t *testing.T) {
			backend, err := simplego.New("")
			require.NoError(t, err)
			x := tensors.FromAnyValue([][]float32{{1}, {2}})
			x.MaterializeOnDevices(backend, share)
			backend.Finalize()
			require.NotPanics(t, func() {
				// Finalizing the tensor after the backend should not panic.
				x.FinalizeAll()
			})
		})
	}
}

// TestInvalidAccessAfterBackendFinalized checks that it panicks if one attempts to access the
// tensor after its device was destroyed.
func TestInvalidAccessAfterBackendFinalized(t *testing.T) {
	// We create a temporary SimpleGo backend for this test.
	for _, share := range []bool{false, true} {
		t.Run(fmt.Sprintf("Shared=%v", share), func(t *testing.T) {
			backend, err := simplego.New("")
			require.NoError(t, err)
			x := tensors.FromAnyValue([][]float32{{1}, {2}})
			x.MaterializeOnDevices(backend, share)
			x.FinalizeLocal()
			backend.Finalize()
			require.Panics(t, func() {
				_ = x.Value()
				fmt.Println("x shouldn't be accessible after its backend was finalized!")
			})
		})
	}
}
