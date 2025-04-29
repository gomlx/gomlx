package main

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/examples/imdb"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/janpfeifer/must"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2"
	"os"
	"sync"
	"testing"
)

var (
	flagSettings *string
	muTrain      sync.Mutex
)

func init() {
	ctx := imdb.CreateDefaultContext()
	flagSettings = commandline.CreateContextSettingsFlag(ctx, "")
	klog.InitFlags(nil)
	if _, found := os.LookupEnv(backends.ConfigEnvVar); !found {
		// For testing, we use the CPU backend (and avoid GPU if not explicitly requested).
		must.M(os.Setenv(backends.ConfigEnvVar, "xla:cpu"))
	}
}

func TestDemo(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping testing in short mode")
		return
	}

	ctx := imdb.CreateDefaultContext()
	ctx.SetParam("train_steps", 10)
	paramsSet := must.M1(commandline.ParseContextSettings(ctx, *flagSettings))

	muTrain.Lock()
	defer muTrain.Unlock()
	require.NotPanics(t, func() {
		imdb.TrainModel(ctx, *flagDataDir, *flagCheckpoint, paramsSet, *flagEval, *flagVerbosity)
	})
}
