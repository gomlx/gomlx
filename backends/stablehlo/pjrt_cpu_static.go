//go:build pjrt_cpu_static

// Set `pjrt_cpu_static` to include the package that statically links PJRT CPU plugin.

package stablehlo

import (
	// Link CPU PJRT statically.
	_ "github.com/gomlx/gopjrt/pjrt/cpu/static"
)
