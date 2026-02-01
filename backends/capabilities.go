// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backends

import (
	"maps"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
)

// Capabilities holds mappings of what is supported by a backend.
type Capabilities struct {
	// Operations supported by a backend.
	// If not listed, it's assumed to be false, hence not supported.
	Operations map[OpType]bool

	// Functions indicates whether the backend supports functions (top-level functions or closures).
	// Without functions, it's not possible to support Call() op or any other
	// op that takes as input a closure (While, If, etc.)
	Functions bool

	// DTypes list the data types supported by a backend.
	// If not listed, it's assumed to be false, hence not supported.
	DTypes map[dtypes.DType]bool

	// PreferConstantsForVariables indicates whether the backend prefers using constants
	// instead of variables for model weights. This is used by ONNX model conversion.
	PreferConstantsForVariables bool
}

// Clone makes a deep copy of the Capabilities.
func (c Capabilities) Clone() Capabilities {
	var c2 Capabilities
	c2 = c
	c2.Operations = make(map[OpType]bool, len(c.Operations))
	maps.Copy(c2.Operations, c.Operations)
	c2.DTypes = make(map[dtypes.DType]bool, len(c.DTypes))
	maps.Copy(c2.DTypes, c.DTypes)
	return c2
}
