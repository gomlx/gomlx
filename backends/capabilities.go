package backends

import "github.com/gomlx/gopjrt/dtypes"

// Capabilities holds mappings of what is supported by a backend.
type Capabilities struct {
	// Operations supported (or not) by a backend. If not listed, it is assumed it is not supported.
	Operations map[OpType]bool

	// DTypes list the data types supported by a backend. If not listed, it is assumed it is not supported.
	DTypes map[dtypes.DType]bool
}
