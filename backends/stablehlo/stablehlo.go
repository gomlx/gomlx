// Package stablehlo has been renamed to backends/xla.
//
// Deprecated: Use the backends/xla package instead.
package stablehlo

import "github.com/gomlx/gomlx/backends/xla"

type Backend = xla.Backend

var (
	New                 = xla.New
	NewWithOptions      = xla.NewWithOptions
	GetAvailablePlugins = xla.GetAvailablePlugins

	BackendName = xla.BackendName
)
