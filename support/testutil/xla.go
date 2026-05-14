// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build ((linux && (amd64 || arm64)) || (darwin && arm64) || (windows && amd64)) && !noxla

package testutil

// Included only for platforms where XLA is supported. It can be disabled with --tags=noxla.

import (
	"os"
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/go-xla/installer"
)

func init() {
	// If no explicit backend is selected, we add xla backends (cpu and cuda if available) to the front of the test list.
	if os.Getenv(compute.ConfigEnvVar) == "" {
		OfficialTestBackendNames = slices.Insert(OfficialTestBackendNames, 0, "xla:cpu")
		if installer.HasNvidiaGPU() {
			OfficialTestBackendNames = slices.Insert(OfficialTestBackendNames, 0, "xla:cuda")
		}
	}
}
