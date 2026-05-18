// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"sync"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/stretchr/testify/require"
)

var (
	flagSettings *string
	muDemo       sync.Mutex
)

func init() {
	scope := createDefaultContext()
	flagSettings = commandline.CreateSettingsFlag(scope, "")
	if _, found := os.LookupEnv(compute.ConfigEnvVar); !found {
		// For testing, we use the CPU backend (and avoid GPU if not explicitly requested).
		check(os.Setenv(compute.ConfigEnvVar, "xla:cpu"))
	}
}

func TestMainFunc(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping testing in short mode")
		return
	}
	scope := createDefaultContext()
	scope.SetParam("train_steps", 10)
	paramsSet := check1(commandline.ParseSettings(scope, *flagSettings))
	err := mainWithContext(scope, *flagDataDir, *flagCheckpoint, paramsSet)
	require.NoError(t, err, "failed to train Adult model for 10 steps")
}
