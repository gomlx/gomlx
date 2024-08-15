package main

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/examples/cifar"
	"github.com/gomlx/gomlx/ml/train/commandline"
	"github.com/janpfeifer/must"
	"os"
	"sync"
	"testing"
)

var (
	flagSettings *string
	muDemo       sync.Mutex
)

func init() {
	ctx := createDefaultContext()
	flagSettings = commandline.CreateContextSettingsFlag(ctx, "")
	if _, found := os.LookupEnv(backends.GOMLX_BACKEND); !found {
		// For testing, we use the CPU backend (and avoid GPU if not explicitly requested).
		must.M(os.Setenv(backends.GOMLX_BACKEND, "cpu"))
	}
}

func TestMainFunc(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping testing in short mode")
		return
	}
	ctx := createDefaultContext()
	ctx.SetParam("train_steps", 10)
	paramsSet := must.M1(commandline.ParseContextSettings(ctx, *flagSettings))
	cifar.TrainCifar10Model(ctx, *flagDataDir, "", true, 1, paramsSet)
	mainWithContext(ctx, *flagDataDir, *flagCheckpoint, paramsSet)
}
