// Package backends defines the interface to a computation building and execution system needs to implement to be used by GoMLX.
//
// It is based on 3 interfaces:
//
//   - DataInterface: handles how data (Tensors) is stored in buffers for the backend. These things are handled
//     differently by different backends and even by different accelerators with the same backend.
//   - Builder: how computation graphs are built.
//   - Executable: how executable computations are executed.
//
// It is based on OpenXLA's API for now.
//
// A backend that doesn't implement every operation can simply return a "Not implemented" error for any op, and
// it would still work for computations that don't require those operations.
// The backend/notimplemented package helps bootstrap any new backend implementation by providing
// a "Not Implemented" implementation for all methods of the Builder interface.
//
// To simplify error handling, all functions are expected to throw (panic) with a stack trace in case of errors.
// See package github.com/gomlx/exceptions.
package backends

import (
	"github.com/gomlx/exceptions"
	"golang.org/x/exp/maps"
	"os"
	"strings"
)

//go:generate go run ../internal/cmd/backends_generator

// DeviceNum represents which device holds a buffer or should execute a computation.
// It's up to the backend to interpret it, but it should be between 0 and Backend.NumDevices.
type DeviceNum int

// Backend is the API that needs to be implemented by a GoMLX backend.
type Backend interface {
	// Name returns the short name of the backend. E.g.: "xla" for the Xla/PJRT plugin.
	Name() string

	// Description is a longer description of the Backend that can be used to pretty-print.
	Description() string

	// NumDevices return the number of devices available for this Backend.
	NumDevices() DeviceNum

	// Capabilities returns information about what is supported by this backend.
	Capabilities() Capabilities

	// Builder creates a new builder used to define a new named computation.
	Builder(name string) Builder

	// DataInterface is the sub-interface that defines the API to transfer Buffer to/from accelerators for the backend.
	DataInterface

	// Finalize releases all the associated resources immediately and makes the backend invalid.
	Finalize()
}

// Constructor takes a config string (optionally empty) and returns a Backend.
type Constructor func(config string) Backend

var (
	registeredConstructors = make(map[string]Constructor)
	firstRegistered        string
)

// Register backend with the given name and a default constructor that takes as input a configuration string that is
// passed along to the backend constructor.
//
// To be safe, call Register during initialization of a package.
func Register(name string, constructor Constructor) {
	if len(registeredConstructors) == 0 {
		firstRegistered = name
	}
	registeredConstructors[name] = constructor
}

// DefaultConfig is the name of the default backend configuration to use if specified.
//
// See NewWithConfig for the format of the configuration string.
var DefaultConfig string

// ConfigEnvVar is the name of the environment variable with the default backend configuration to use:
// "GOMLX_BACKEND".
//
// The format of the configuration is "<backend_name>:<backend_configuration>".
// The "<backend_name>" is the name of a registered backend (e.g.: "xla") and
// "<backend_configuration>" is backend-specific (e.g.: for xla backend, it is the pjrt plugin name).
const ConfigEnvVar = "GOMLX_BACKEND"

// GOMLX_BACKEND is deprecated and will be removed in future versions -- it is an alias to ConfigEnvVar
// Deprecated: use ConfigEnvVar.
const GOMLX_BACKEND = ConfigEnvVar

// New returns a new default Backend.
//
// The default is:
//
// 1. The environment ConfigEnvVar is used as a configuration if defined.
// 2. Next the variable DefaultConfig is used as a configuration if defined.
// 3. The first registered backend is used with an empty configuration.
//
// It panics if not the backend was registered.
func New() Backend {
	config, found := os.LookupEnv(ConfigEnvVar)
	if found {
		return NewWithConfig(config)
	}
	if DefaultConfig != "" {
		return NewWithConfig(DefaultConfig)
	}
	return NewWithConfig("")
}

// NewWithConfig takes a configuration string formated as
//
// The format of config is "<backend_name>:<backend_configuration>".
// The "<backend_name>" is the name of a registered backend (e.g.: "xla") and
// "<backend_configuration>" is backend-specific (e.g.: for xla backend, it is the pjrt plugin name).
func NewWithConfig(config string) Backend {
	if len(registeredConstructors) == 0 {
		exceptions.Panicf(`no registered backends for GoMLX -- maybe import the default XLA one with import _ "github.com/gomlx/gomlx/backends/xla"?`)
	}
	backendName := firstRegistered
	backendConfig := config
	if idx := strings.Index(config, ":"); idx != -1 {
		backendName = config[:idx]
		backendConfig = config[idx+1:]
	}
	constructor, found := registeredConstructors[backendName]
	if !found {
		exceptions.Panicf("can't find backend %q for configuration %q given, backends available: \"%s\"",
			backendName, config, strings.Join(List(), "\", \""))
	}
	return constructor(backendConfig)
}

// List the registered (compiled-in) backends.
func List() []string {
	return maps.Keys(registeredConstructors)
}
