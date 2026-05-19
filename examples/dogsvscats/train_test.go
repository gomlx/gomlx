// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dogsvscats

import (
	"os"
	"sync"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ui/commandline"
	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/default"
)

var (
	flagSettings *string
	muTrain      sync.Mutex
)

func init() {
	store := CreateModelStore()
	scope := store.RootScope()
	flagSettings = commandline.CreateSettingsFlag(scope, "")
	klog.InitFlags(nil)
	if _, found := os.LookupEnv(compute.ConfigEnvVar); !found {
		// For testing, we use the CPU backend (and avoid GPU if not explicitly requested).
		check(os.Setenv(compute.ConfigEnvVar, "xla:cpu"))
	}
}

// TestTrain train the default model for 50 steps.
func TestTrain(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping testing in short mode")
		return
	}
	store := CreateModelStore()
	store.SetParam("train_steps", 10)
	store.SetParam("plots", false)
	store.SetParam(layers.ParamNormalization, "layer")
	paramsSet := check1(commandline.ParseSettings(store.RootScope(), *flagSettings))
	TrainWithStore(store, *flagDataDir, "", false, paramsSet)
}
