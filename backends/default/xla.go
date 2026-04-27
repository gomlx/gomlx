// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build ((linux && (amd64 || arm64)) || (darwin && arm64) || (windows && amd64)) && !noxla

package _default

import _ "github.com/gomlx/go-xla/compute/xla"
