//go:build pjrt_cpu_dynamic

// Set `pjrt_cpu_dynamic` to include the package that statically links PJRT CPU plugin.

package stablehlo

import (
	// Link CPU PJRT statically: slower but works on Mac.
	_ "github.com/gomlx/gopjrt/pjrt/cpu/dynamic"
)
