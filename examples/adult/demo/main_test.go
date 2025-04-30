package main

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/janpfeifer/must"
	"github.com/stretchr/testify/require"
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
	if _, found := os.LookupEnv(backends.ConfigEnvVar); !found {
		// For testing, we use the CPU backend (and avoid GPU if not explicitly requested).
		must.M(os.Setenv(backends.ConfigEnvVar, "xla:cpu"))
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
	//cifar.TrainCifar10Model(ctx, *flagDataDir, "", true, 1, paramsSet)
	err := mainWithContext(ctx, *flagDataDir, *flagCheckpoint, paramsSet)
	require.NoError(t, err, "failed to train Adult model for 10 steps")
}
