//go:build pjrt_cpu_static

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0


// Set `pjrt_cpu_static` to include the package that statically links PJRT CPU plugin.

package xla

import (
	// Link CPU PJRT statically.
	_ "github.com/gomlx/go-xla/pkg/pjrt/cpu/static"
)
