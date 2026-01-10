//go:build ((linux && amd64) || darwin) && !noxla

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0


package _default

import _ "github.com/gomlx/gomlx/backends/xla"
