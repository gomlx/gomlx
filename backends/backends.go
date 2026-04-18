// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package backends is an alias to compute package. It's here for historical reasons.
//
// Deprecated: use the [compute] package instead.
package backends

import "github.com/gomlx/compute"

// Backend represents a compute backend, capabable of building, compiling, transferring data to/from and executing a
// computation graph.
//
// Deprecated: it's just an alias to [compute.Backend], use that instead.
type Backend = compute.Backend

// DeviceNum represents which device holds a buffer or should execute a computation. It's up to the backend to interpret
// it, but it should be between 0 and Backend.NumDevices.
//
// Deprecated: it's just an alias to [compute.DeviceNum], use that instead.
type DeviceNum = compute.DeviceNum

// Buffer represents actual data (a tensor) stored in the accelerator that is actually going to execute the graph.
// It's used as input/output of computation execution. A Buffer is always associated to a DeviceNum, even if there is
// only one.
//
// It is opaque from GoMLX perspective, but it cannot be mixed -- a Buffer returned by one backend can't be used with
// another backend.
//
// Deprecated: it's just an alias to [compute.Buffer], use that instead.
type Buffer = compute.Buffer

// DefaultConfig is the name of the default backend configuration to use if specified.
//
// Deprecated: it's just an alias to [compute.DefaultConfig], use that instead.
var DefaultConfig = compute.DefaultConfig

// MustNew returns a new default Backend or panics if it fails.
//
// The default is:
//
// 1. The environment $GOMLX_BACKEND (ConfigEnvVar) is used as a configuration if defined.
// 2. Next, it uses the variable DefaultConfig as the configuration.
// 3. The first registered backend is used with an empty configuration.
//
// It fails if no backends were registered.
//
// Deprecated: use [compute.MustNew] instead.
func MustNew() compute.Backend {
	return compute.MustNew()
}

// New returns a new default backend.
//
// Deprecated: use [compute.New] instead.
func New() (compute.Backend, error) {
	return compute.New()
}

// NewWithConfig takes a configuration string formated as
//
// The format of config is "<backend_name>:<backend_configuration>".
// The "<backend_name>" is the name of a registered backend (e.g.: "xla") and
// "<backend_configuration>" is backend-specific (e.g.: for xla backend, it is the PJRT plugin name).
//
// Deprecated: use [compute.NewWithConfig] instead.
func NewWithConfig(config string) (compute.Backend, error) {
	return compute.NewWithConfig(config)
}
