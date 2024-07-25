package main

import (
	"flag"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/train/commandline"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"
	"os"
	"testing"
)

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

	ctx := createDefaultContext()
	settings := commandline.CreateContextSettingsFlag(ctx, "")
	klog.InitFlags(nil)
	flag.Parse()
	must.M(commandline.ParseContextSettings(ctx, *settings))
	ctx.SetParam("train_steps", 10) // Only 10 steps.

	*flagDataDir = data.ReplaceTildeInDir(*flagDataDir)
	if !data.FileExists(*flagDataDir) {
		must.M(os.MkdirAll(*flagDataDir, 0777))
	}
	must.M(commandline.ParseContextSettings(ctx, *settings))
	*flagEval = false
	trainModel(ctx)
}
