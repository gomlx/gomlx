// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package dynamic links the XLA/PJRT CPU plugin dynamically (as in ".so" libraries) with your binary.
//
// The default is to load PJRT plugins only after the program starts, using unix `dlopen`.
// A binary build in this mode will fail to execute if the PJRT library is not available where the binary is
// being executed.
//
// To use it, import it:
//
//	import _ "github.com/gomlx/gomlx/backends/stablehlo/cpu/dynamic"
//
// It also automatically includes the XLA engine ("github.com/gomlx/gomlx/backends/stablehlo").
//
// See also github.com/gomlx/gomlx/backends/stablehlo/cpu/static for static linking.
package dynamic

import (
	// Link XLA engine.
	_ "github.com/gomlx/gomlx/backends/xla"

	// Link CPU PJRT statically.
	_ "github.com/gomlx/go-xla/pkg/pjrt/cpu/dynamic"
)
