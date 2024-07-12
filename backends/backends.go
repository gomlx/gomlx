// Package backends defines the interface to a computation building and execution system needs to implement to be used by GoMLX.
//
// It is based on OpenXLA's API for now, since it's the only implementation.
//
// A backend that doesn't implement every operation, can simply return a "Not implemented" error for any op, and
// it would still work for computations that don't require those operations.
//
// To simplify error handling, all functions are expected to throw (panic) with a stack trace in case of errors.
// See package github.com/gomlx/exceptions.
package backends

import (
	"github.com/gomlx/exceptions"
	"os"
	"strings"
)

//go:generate go run ../cmd/backends_generator

// DeviceNum represents which device holds a buffer, or should execute a computation.
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

	// Builder creates a new builder used to define a new named computation.
	Builder(name string) Builder

	// DataInterface is the sub-interface that defines the API to transfer Buffer to/from accelerators for the backend.
	DataInterface

	// Finalize releases all the associated resources immediately, and makes the backend invalid.
	Finalize()
}

// Constructor takes a config string (optionally empty) and returns a Backend.
type Constructor func(config string) Backend

var (
	registeredConstructors = make(map[string]Constructor)
	firstRegistered        string
)

// Register backend with the given name, and a default constructor that takes as input a configuration string that is
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

// GOMLX_BACKEND is the environment variable with the default backend configuration to use.
//
// The format of config is "<backend_name>:<backend_configuration>".
// The "<backend_name>" is the name of a registered backend (e.g.: "xla") and
// "<backend_configuration>" is backend specific (e.g.: for xla backend, it is the pjrt plugin name).
const GOMLX_BACKEND = "GOMLX_BACKEND"

// New returns a new default Backend.
//
// The default is:
//
// 1. The environment GOMLX_BACKEND is used as a configuration if defined.
// 2. Next the variable DefaultConfig is used as a configuration if defined.
// 3. The first registered backend is used with an empty configuration.
//
// It panics if not backend was registered.
func New() Backend {
	config, found := os.LookupEnv(GOMLX_BACKEND)
	if found {
		return NewWithConfig(config)
	}
	if DefaultConfig != "" {
		return NewWithConfig(DefaultConfig)
	}
	return NewWithConfig("")
}

// NewWithConfig takes a configurations string formated as
//
// The format of config is "<backend_name>:<backend_configuration>".
// The "<backend_name>" is the name of a registered backend (e.g.: "xla") and
// "<backend_configuration>" is backend specific (e.g.: for xla backend, it is the pjrt plugin name).
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
		exceptions.Panicf("can't find backend %q for configuration %q given", backendName, config)
	}
	return constructor(backendConfig)
}
