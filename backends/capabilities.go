package backends

import "github.com/gomlx/gomlx/pkg/core/dtypes"

// Capabilities holds mappings of what is supported by a backend.
type Capabilities struct {
	// Operations supported by a backend.
	// If not listed, it's assumed to be false, hence not supported.
	Operations map[OpType]bool

	// DTypes list the data types supported by a backend.
	// If not listed, it's assumed to be false, hence not supported.
	DTypes map[dtypes.DType]bool

	// SupportsDynamicShapes indicates whether the backend can execute graphs
	// with different input shapes without requiring expensive recompilation.
	// When true, Exec can skip pattern caching and bucketing since creating
	// new graphs is cheap. This is typically true for interpreted backends
	// like SimpleGo, but false for compiled backends like XLA.
	// Default is false (zero value).
	SupportsDynamicShapes bool
}

// Clone makes a deep copy of the Capabilities.
func (c Capabilities) Clone() Capabilities {
	var c2 Capabilities
	c2.Operations = make(map[OpType]bool, len(c.Operations))
	for k, v := range c.Operations {
		c2.Operations[k] = v
	}
	c2.DTypes = make(map[dtypes.DType]bool, len(c.DTypes))
	for k, v := range c.DTypes {
		c2.DTypes[k] = v
	}
	c2.SupportsDynamicShapes = c.SupportsDynamicShapes
	return c2
}
