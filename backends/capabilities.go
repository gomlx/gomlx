package backends

import "github.com/gomlx/gopjrt/dtypes"

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
	for k, v := range c.Operations {
		c2.Operations[k] = v
	}
	c2.DTypes = make(map[dtypes.DType]bool, len(c.DTypes))
	for k, v := range c.DTypes {
		c2.DTypes[k] = v
	}
	return c2
}
