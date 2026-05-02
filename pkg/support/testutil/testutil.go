// Package testutil provides utility functions for testing.
//
// They are used within GoMLX for testing, but may be useful for other GoMLX-based library
// developers.
package testutil

import (
	"math"
	"os"
	"slices"
	"sync"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/go-xla/compute/xla"
	"github.com/google/go-cmp/cmp"
	"k8s.io/klog/v2"
)

var (
	OfficialTestBackendNames []string = []string{
		"go",
	}
	OfficialTestBackends = make(map[string]compute.Backend)
)

func init() {
	if selectedBackendName := os.Getenv(compute.ConfigEnvVar); selectedBackendName != "" {
		OfficialTestBackendNames = []string{selectedBackendName}
		return
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
			if i == 0 {
				klog.Infof("Default backend %q: %s", backendName, backend.Description())
			} else {
				klog.Infof("Extra backend #%d %q: %s", i, backendName, backend.Description())
			}
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

func withinDeltaBase[T ~float32 | ~float64](a, b T, delta float64) bool {
	return math.Abs(float64(a-b)) < delta
}

type halfFloat interface {
	float16.Float16 | bfloat16.BFloat16
	Float64() float64
}

func withinDeltaHalfPrecision[T halfFloat](a, b T, delta float64) bool {
	return withinDeltaBase(a.Float64(), b.Float64(), delta)
}

// IsInDelta reports if want and got are equal within a given absolute delta.
// If they are not equal, it returns the diff using the format "-want +got").
func IsInDelta(want, got any, delta float64) (ok bool, diff string) {
	opts := []cmp.Option{
		cmp.Comparer(func(a, b float32) bool { return withinDeltaBase(a, b, delta) }),
		cmp.Comparer(func(a, b float64) bool { return withinDeltaBase(a, b, delta) }),
		cmp.Comparer(func(a, b float16.Float16) bool { return withinDeltaHalfPrecision(a, b, delta) }),
		cmp.Comparer(func(a, b bfloat16.BFloat16) bool { return withinDeltaHalfPrecision(a, b, delta) }),
	}
	if cmp.Equal(want, got, opts...) {
		return true, ""
	}
	return false, cmp.Diff(want, got, opts...)
}

func withinRelativeDeltaBase[T ~float32 | ~float64](a, b T, relDelta float64) bool {
	delta := math.Abs(float64(a) - float64(b))
	mean := math.Abs(float64(b)) + math.Abs(float64(b))
	return delta/mean < relDelta
}

func withinRelativeDeltaHalfPrecision[T halfFloat](a, b T, relDelta float64) bool {
	return withinRelativeDeltaBase(a.Float64(), b.Float64(), relDelta)
}

// IsInRelativeDelta reports if want and got are equal within a given relative delta.
// If they are not equal, it returns the diff using the format "-want +got").
func IsInRelativeDelta(want, got any, relDelta float64) (ok bool, diff string) {
	opts := []cmp.Option{
		cmp.Comparer(func(a, b float32) bool { return withinRelativeDeltaBase(a, b, relDelta) }),
		cmp.Comparer(func(a, b float64) bool { return withinRelativeDeltaBase(a, b, relDelta) }),
		cmp.Comparer(func(a, b float16.Float16) bool { return withinRelativeDeltaHalfPrecision(a, b, relDelta) }),
		cmp.Comparer(func(a, b bfloat16.BFloat16) bool { return withinRelativeDeltaHalfPrecision(a, b, relDelta) }),
	}
	if cmp.Equal(want, got, opts...) {
		return true, ""
	}
	return false, cmp.Diff(want, got, opts...)
}
