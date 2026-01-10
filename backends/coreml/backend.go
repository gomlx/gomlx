//go:build darwin && cgo

// Package coreml implements a CoreML backend for GoMLX targeting Apple Silicon.
//
// This backend leverages Apple's CoreML framework to execute ML models on Apple Silicon
// devices, utilizing the Neural Engine (ANE), GPU, and CPU compute units.
package coreml

import (
	"os"
	"strings"
	"sync"

	"github.com/gomlx/go-coreml/model"
	"github.com/gomlx/go-coreml/runtime"
	"github.com/gomlx/gomlx/backends"
	"github.com/pkg/errors"
)

// BackendName to be used in GOMLX_BACKEND to specify this backend.
const BackendName = "coreml"

// Backend implements the backends.Backend interface for CoreML.
type Backend struct {
	// runtime is the CoreML runtime for compilation and execution
	runtime *runtime.Runtime

	// cacheDir is the directory where compiled CoreML models are cached.
	cacheDir string

	// bufferPools are a map to pools of buffers that can be reused.
	// The underlying type is map[bufferPoolKey]*sync.Pool.
	bufferPools sync.Map

	// isFinalized is true if the backend has been finalized.
	isFinalized bool
}

// Compile-time check that coreml.Backend implements backends.Backend.
var _ backends.Backend = &Backend{}

// New constructs a new CoreML Backend.
//
// The config string can specify options (currently unused, reserved for future use):
//   - "" (empty) - use default settings
//
// Example: backends.NewWithConfig("coreml")
func New(config string) (backends.Backend, error) {
	cacheDir := os.TempDir()

	// Create CoreML runtime with default options
	rt := runtime.New(
		runtime.WithCacheDir(cacheDir),
	)

	b := &Backend{
		runtime:  rt,
		cacheDir: cacheDir,
	}

	// Parse configuration options (reserved for future use)
	if config != "" {
		parts := strings.Split(config, ",")
		for _, part := range parts {
			part = strings.TrimSpace(part)
			if part == "" {
				continue
			}
			return nil, errors.Errorf("unknown configuration option %q for CoreML (coreml) backend", part)
		}
	}

	return b, nil
}

// Name returns the short name of the backend.
func (b *Backend) Name() string {
	return "CoreML (coreml)"
}

// String implements backends.Backend.
func (b *Backend) String() string {
	return BackendName
}

// Description is a longer description of the Backend that can be used to pretty-print.
func (b *Backend) Description() string {
	return "CoreML Backend for Apple Silicon"
}

// NumDevices returns the number of devices available for this Backend.
// CoreML abstracts the underlying hardware, so we report 1 logical device.
func (b *Backend) NumDevices() int {
	return 1
}

// DeviceDescription returns a description of the device with the given deviceNum.
func (b *Backend) DeviceDescription(deviceNum backends.DeviceNum) string {
	if deviceNum != 0 {
		return "invalid device"
	}
	return "device#0 (CoreML)"
}

// Capabilities returns information about what is supported by this backend.
func (b *Backend) Capabilities() backends.Capabilities {
	return Capabilities
}

// Builder creates a new builder used to construct a named computation.
func (b *Backend) Builder(name string) backends.Builder {
	// CoreML requires the function name to be "main"
	builder := &Builder{
		backend:    b,
		name:       name,
		milBuilder: model.NewBuilder("main"),
		nodes:      make([]*Node, 0),
		inputs:     make([]*Node, 0),
		outputs:    make([]*Node, 0),
		nodeMap:    make(map[backends.Value]*model.Value),
	}
	return builder
}

// Finalize releases all the associated resources immediately, and makes the backend invalid.
func (b *Backend) Finalize() {
	b.isFinalized = true
	b.bufferPools.Clear()
}

// IsFinalized returns true if the backend has been finalized.
func (b *Backend) IsFinalized() bool {
	return b.isFinalized
}
