// Package simplego implements a simple, and not very fast, but very portable backend for GoMLX.
//
// It only implements the most popular dtypes and operations.
// But generally, it's easy to add new ops, if you need, just open an issue in GoMLX.
package simplego

import (
	"fmt"
	"strconv"
	"strings"
	"sync/atomic"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/notimplemented"
	"github.com/pkg/errors"

	"sync"
)

// Generates some trivial functions (binary and unary operators) automatically.
//go:generate go run ../../internal/cmd/simplego_generator

// Registers the various generics function instances.
//go:generate go run ../../internal/cmd/simplego_dispatcher

// Auto-generate alternate versions of functions, with small changes.
// (that can't easily be refactored into smaller functions due to latency penalities)
//go:generate go run ../../internal/cmd/alternates_generator -base=convgeneral_exec.go -tags=bf16,full,full_bf16

// BackendName to be used in GOMLX_BACKEND to specify this backend.
const BackendName = "go"

// Registers New() as the default constructor for "xla" backend.
func init() {
	backends.Register(BackendName, New)
}

// GetBackend returns a singleton backend for SimpleGo, created with the default configuration.
// The backend is only created at the first call of the function.
//
// The singleton is never destroyed.
var GetBackend = sync.OnceValue(func() backends.Backend {
	backend, err := New("")
	if err != nil {
		panic(err)
	}
	return backend
})

// New constructs a new SimpleGo Backend.
// There are no configurations, the string is simply ignored.
func New(config string) (backends.Backend, error) {
	b := newDefaultBackend()
	parts := strings.Split(config, ",")
	for _, part := range parts {
		key := part
		var value string
		if eqPos := strings.Index(part, "="); eqPos != -1 {
			key, value = part[0:eqPos], part[eqPos+1:]
		}
		switch key {
		case "parallelism":
			vInt, err := strconv.Atoi(value)
			if err != nil {
				return nil, errors.Wrapf(err, "invalid value for %q in SimpleGo backend config: needs an int, got %q", key, value)
			}
			b.workers.SetMaxParallelism(vInt)
			fmt.Printf("SimpleGo backend: parallelism set to %d\n", vInt)
		case "dotgeneral_small":
			// This will force DotGeneral operation to use the version designed for smaller matrices.
			b.dotGeneralForceProblemSize = smallProblemSize
		case "dotgeneral_large":
			// This will force DotGeneral operation to use the version designed for large matrices.
			b.dotGeneralForceProblemSize = largeProblemSize
		case "dotgeneral_check":
			// This will force every DotGeneral operation to be executed with both versions and the outputs compared.
			// This is used exclusively for debugging purposes.
			b.dotGeneralForceProblemSize = checkProblemSize
		case "ops_sequential":
			// This will force the ops to be executed sequentially.
			// The default is running parallel if it's the only thing executing, otherwise sequentially.
			b.opsExecutionType = opsExecutionSequential
		case "ops_parallel":
			// This will force the ops to be executed in parallel where possible.
			// The default is running parallel if it's the only thing executing, otherwise sequentially.
			b.opsExecutionType = opsExecutionParallel
		case "":
			// No-op, just skip.
		default:
			return nil, errors.Errorf("unknown configuration option %q for SimpleGo (go) backend -- valid configuration options are: "+
				"parallelism=#workers, dotgeneral_small, dotgeneral_large, dotgeneral_check, ops_sequential, ops_parallel; see code for documentation", key)
		}
	}
	return b, nil
}

func newDefaultBackend() *Backend {
	b := &Backend{}
	b.workers.Initialize()
	b.preBlockedWeightCache = NewPreBlockedWeightCache()
	return b
}

// Backend implements the backends.Backend interface.
type Backend struct {
	// bufferPools are a map to pools of buffers that can be reused.
	// The underlying type is map[bufferPoolKey]*sync.Pool.
	bufferPools sync.Map
	workers     workersPool

	numLiveExecutions atomic.Int32

	// dotGeneralForceProblemSize allows a DotGeneral algorithm to always be used.
	dotGeneralForceProblemSize dotGeneralProblemSizeType

	// opsExecutionType defines how to execute the ops of a computation.
	opsExecutionType opsExecutionType

	// preBlockedWeightCache caches pre-blocked weight tensors for efficient matmul.
	// This allows skipping the blocking step for constant weights (model parameters).
	preBlockedWeightCache *PreBlockedWeightCache

	// isFinalized is true if the backend has been isFinalized.
	isFinalized bool
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
func (b *Backend) NumDevices() int {
	return 1
}

// DeviceDescription returns a description of the device with the given deviceNum.
func (b *Backend) DeviceDescription(deviceNum backends.DeviceNum) string {
	return "device#0"
}

// Capabilities returns information about what is supported by this backend.
func (b *Backend) Capabilities() backends.Capabilities {
	return Capabilities
}

// Builder creates a new builder used to construct a named computation.
func (b *Backend) Builder(name string) backends.Builder {
	builder := &Builder{
		backend: b,
		name:    name,
	}
	// Set the "not implemented" custom message:
	builder.Builder.ErrFn = notImplementedError
	return builder
}

func notImplementedError(opType backends.OpType) error {
	return errors.Wrapf(notimplemented.NotImplementedError, "sorry, op %q not implemented in SimpleGo yet "+
		"-- reach out to github.com/gomlx/gomlx and open an issue if you need this op, this helps us prioritize the work",
		opType)
}

// Finalize releases all the associated resources immediately, and makes the backend invalid.
func (b *Backend) Finalize() {
	b.isFinalized = true
	b.bufferPools.Clear()
}

// IsFinalized returns true if the backend has been isFinalized.
func (b *Backend) IsFinalized() bool {
	return b.isFinalized
}
