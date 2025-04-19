package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/notimplemented"
)

// Builder keeps track of the computation graph being defined.
type Builder struct {
	notimplemented.Builder
}

// Compile-time check.
var _ backends.Builder = (*Builder)(nil)
