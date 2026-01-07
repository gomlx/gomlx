package backends

import "maps"

import "github.com/gomlx/gomlx/pkg/core/dtypes"

// Capabilities holds mappings of what is supported by a backend.
type Capabilities struct {
	// Operations supported by a backend.
	// If not listed, it's assumed to be false, hence not supported.
	Operations map[OpType]bool

	// DTypes list the data types supported by a backend.
	// If not listed, it's assumed to be false, hence not supported.
	DTypes map[dtypes.DType]bool
}

// Clone makes a deep copy of the Capabilities.
func (c Capabilities) Clone() Capabilities {
	var c2 Capabilities
	c2.Operations = make(map[OpType]bool, len(c.Operations))
	maps.Copy(c2.Operations, c.Operations)
	c2.DTypes = make(map[dtypes.DType]bool, len(c.DTypes))
	maps.Copy(c2.DTypes, c.DTypes)
	return c2
}
