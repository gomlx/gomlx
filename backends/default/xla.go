// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build ((linux && amd64) || darwin) && !noxla

package _default

import _ "github.com/gomlx/go-xla/compute/xla"
