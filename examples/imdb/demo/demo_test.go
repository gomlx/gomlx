// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"sync"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/examples/imdb"
	"github.com/gomlx/gomlx/ui/commandline"

	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2"
)

var (
	flagSettings *string
	muTrain      sync.Mutex
)

func init() {
	scope := imdb.CreateDefaultContext()
	flagSettings = commandline.CreateSettingsFlag(scope, "")
	klog.InitFlags(nil)
	if _, found := os.LookupEnv(compute.ConfigEnvVar); !found {
		// For testing, we use the CPU backend (and avoid GPU if not explicitly requested).
		check(os.Setenv(compute.ConfigEnvVar, "xla:cpu"))
	}
}

func TestDemo(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping testing in short mode")
		return
	}

	scope := imdb.CreateDefaultContext()
	scope.SetParam("train_steps", 10)
	paramsSet := check1(commandline.ParseSettings(scope, *flagSettings))

	muTrain.Lock()
	defer muTrain.Unlock()
	require.NotPanics(t, func() {
		imdb.TrainModel(scope, *flagDataDir, *flagCheckpoint, paramsSet, *flagEval, *flagVerbosity)
	})
}
