// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build ((linux && amd64) || darwin) && !noxla

package testutil

// Included only for platforms where XLA is supported. It can be disabled with --tags=noxla.

import (
	"os"

	"github.com/gomlx/compute"
	"github.com/gomlx/go-xla/installer"
)

func init() {
	// If no explicit backend is selected, we add xla backends (cpu and cuda if available) to the tests.
	if os.Getenv(compute.ConfigEnvVar) == "" {
		OfficialTestBackendNames = append(OfficialTestBackendNames, "xla:cpu")
		if installer.HasNvidiaGPU() {
			OfficialTestBackendNames = append(OfficialTestBackendNames, "xla:cuda")
		}
	}
}
