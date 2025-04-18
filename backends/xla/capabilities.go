package xla

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gopjrt/dtypes"
)

// CPUCapabilities supported by XLA CPU backends.
//
// This is the base value, and can be copied and specialized by specific PJRT that may not
// support everything.
var CPUCapabilities = backends.Capabilities{
	Operations: map[backends.OpType]bool{},
	DTypes:     map[dtypes.DType]bool{},
}
