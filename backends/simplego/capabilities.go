package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gopjrt/dtypes"
)

// Capabilities of the SimpleGo backends.
var Capabilities = backends.Capabilities{
	Operations: map[backends.OpType]bool{
		backends.OpTypeParameter: true,
		backends.OpTypeConstant:  true,
	},
	DTypes: map[dtypes.DType]bool{
		dtypes.Bool:     true,
		dtypes.Int8:     true,
		dtypes.Int16:    true,
		dtypes.Int32:    true,
		dtypes.Int64:    true,
		dtypes.Uint8:    true,
		dtypes.Uint16:   true,
		dtypes.Uint32:   true,
		dtypes.Uint64:   true,
		dtypes.Float32:  true,
		dtypes.Float64:  true,
		dtypes.BFloat16: true,
	},
}
