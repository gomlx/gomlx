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
	store := createModelStore()
	flagSettings = commandline.CreateSettingsFlag(store, "")
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
	store := createModelStore()
	store.SetParam("train_steps", 10)
	paramsSet := must1(commandline.ParseSettings(store, *flagSettings))
	err := mainWithStore(store, *flagDataDir, *flagCheckpoint, paramsSet)
	require.NoError(t, err, "failed to train Adult model for 10 steps")
}
