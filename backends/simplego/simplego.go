// Package simplego implements a simple, and not very fast, but very portable backend for GoMLX.
//
// It only implements the most popular dtypes and operations.
// But generally, it's easy to add new ops, if you need, just open an issue in GoMLX.
package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"sync"
)

//go:generate go run ../../internal/cmd/simplego_generator

// Registers the various generics function instances.
//go:generate go run ../../internal/cmd/simplego_dispatcher

// BackendName to be used in GOMLX_BACKEND to specify this backend.
const BackendName = "go"

// Registers New() as the default constructor for "xla" backend.
func init() {
	backends.Register(BackendName, New)
}

// New constructs a new SimpleGo Backend.
// There are no configurations, the string is simply ignored.
func New(_ string) backends.Backend {
	return newBackend()
}

func newBackend() *Backend {
	return &Backend{}
}

// Backend implements the backends.Backend interface.
type Backend struct {
	// bufferPools are a map to pools of buffers that can be reused.
	// The underlying type is map[bufferPoolKey]*sync.Pool.
	bufferPools sync.Map
}

// Compile-time check that simplego.Backend implements backends.Backend.
var _ backends.Backend = &Backend{}

// Name returns the short name of the backend. E.g.: "xla" for the Xla/PJRT plugin.
func (b *Backend) Name() string {
	return "SimpleGo (go)"
}

// String implement backends.Backend.
func (b *Backend) String() string { return BackendName }

// Description is a longer description of the Backend that can be used to pretty-print.
func (b *Backend) Description() string {
	return "Simple Go Portable Backend"
}

// NumDevices return the number of devices available for this Backend.
func (b *Backend) NumDevices() backends.DeviceNum {
	return 1
}

// Capabilities returns information about what is supported by this backend.
func (b *Backend) Capabilities() backends.Capabilities {
	return Capabilities
}

// Builder creates a new builder used to define a new named computation.
func (b *Backend) Builder(name string) backends.Builder {
	return &Builder{
		backend: b,
		name:    name,
	}
}

// Finalize releases all the associated resources immediately, and makes the backend invalid.
func (b *Backend) Finalize() {}
