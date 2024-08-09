package main

import (
	"flag"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/examples/imdb"
	"github.com/gomlx/gomlx/ml/train/commandline"
	"github.com/janpfeifer/must"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2"
	"os"
	"testing"
)

func TestDemo(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping testing in short mode")
		return
	}

	if _, found := os.LookupEnv(backends.GOMLX_BACKEND); !found {
		// For testing, we use the CPU backend (and avoid GPU if not explicitly requested).
		require.NoError(t, os.Setenv(backends.GOMLX_BACKEND, "cpu"))
	}

	ctx := imdb.CreateDefaultContext()
	ctx.SetParam("train_steps", 10)
	settings := commandline.CreateContextSettingsFlag(ctx, "")
	klog.InitFlags(nil)
	flag.Parse()
	paramsSet := must.M1(commandline.ParseContextSettings(ctx, *settings))
	require.NotPanics(t, func() {
		imdb.TrainModel(ctx, *flagDataDir, *flagCheckpoint, paramsSet, *flagEval, *flagVerbosity)
	})
}
