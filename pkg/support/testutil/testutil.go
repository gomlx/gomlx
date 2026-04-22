// Package testutil provides utility functions for testing.
//
// They are used within GoMLX for testing, but may be useful for other GoMLX-based library
// developers.
package testutil

import (
	"os"
	"slices"
	"sync"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/go-xla/installer"
	"github.com/gomlx/gomlx/backends/xla"
	"k8s.io/klog/v2"
)

var (
	OfficialTestBackendNames []string = []string{
		"xla:cpu",
		"go",
	}
	OfficialTestBackends = make(map[string]compute.Backend)
)

func init() {
	if selectedBackendName := os.Getenv(compute.ConfigEnvVar); selectedBackendName != "" {
		OfficialTestBackendNames = []string{selectedBackendName}
		return
	}

	// Include CUDA PJRT test if available.
	if installer.HasNvidiaGPU() {
		OfficialTestBackendNames = append(OfficialTestBackendNames, "xla:cuda")
	}
}

var (
	backendOnce sync.Once
)

// BuildTestBackend and sets compute.DefaultConfig to "xla:cpu" -- it can be overwritten by GOMLX_BACKEND environment variable.
func BuildTestBackend() compute.Backend {
	compute.DefaultConfig = OfficialTestBackendNames[0]
	backendOnce.Do(func() {
		err := xla.AutoInstall()
		if err != nil {
			klog.Fatalf("Failed to auto-install XLA PJRT: %+v", err)
		}
		for i, backendName := range OfficialTestBackendNames {
			backend, err := compute.NewWithConfig(backendName)
			if err != nil {
				if i == 0 {
					klog.Fatalf("Failed to create backend %q: %+v", backendName, err)
				}
				klog.Errorf("Failed to create backend %q: %+v", backendName, err)
				continue
			}
			OfficialTestBackends[backendName] = backend
		}
	})
	return OfficialTestBackends[compute.DefaultConfig]
}

// TestOfficialBackends iterates over list of backends and calls testFn for each of them.
// If GOMLX_BACKEND environment variable is set, it will only iterate over the one set.
// If GOMLX_BACKEND is not set, it will iterate over all official backends, except those in excludecompute.
// (for tests known not to work on those backends)
func TestOfficialBackends(t *testing.T, testFn func(t *testing.T, backend compute.Backend), excludeBackends ...string) {
	BuildTestBackend()
	for backendName, backend := range OfficialTestBackends {
		if slices.Contains(excludeBackends, backendName) {
			continue
		}
		if backend == nil {
			// This happens if the backend already failed to initialize, no need to report it more than once.
			continue
		}
		t.Run(backendName, func(t *testing.T) {
			testFn(t, backend)
		})
	}
}
