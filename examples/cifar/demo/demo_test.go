// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"sync"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/examples/cifar"
	"github.com/gomlx/gomlx/ui/commandline"
	"k8s.io/klog/v2"
)

var (
	flagSettings *string
	muDemo       sync.Mutex
)

func init() {
	klog.InitFlags(nil)
	ctx := createDefaultContext()
	flagSettings = commandline.CreateContextSettingsFlag(ctx, "")
	if _, found := os.LookupEnv(backends.ConfigEnvVar); !found {
		// For testing, we use the CPU backend (and avoid GPU if not explicitly requested).
		check(os.Setenv(backends.ConfigEnvVar, "xla:cpu"))
	}
}

// TestDemo trains the model for 10 steps, not generating any checkpoints.
//
// Still it has to download the training data, and it will use the flag *flagDataDir (--data)
// as the location to store the training data.
//
// It is disabled for short tests.
func TestDemo(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping testing in short mode")
		return
	}

	// Run at most one demo training at a time:
	muDemo.Lock()
	defer muDemo.Unlock()

	ctx := createDefaultContext()
	ctx.SetParam("train_steps", 10) // Only 10 steps.
	paramsSet := check1(commandline.ParseContextSettings(ctx, *flagSettings))
	cifar.TrainCifar10Model(ctx, *flagDataDir, "", true, 1, paramsSet)
}
