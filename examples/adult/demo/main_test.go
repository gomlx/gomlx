package main

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/stretchr/testify/require"
	"os"
	"testing"
)

func TestMainFunc(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping testing in short mode")
		return
	}
	ctx := createDefaultContext()
	ctx.SetParam("train_steps", 10)
	if _, found := os.LookupEnv(backends.GOMLX_BACKEND); !found {
		// For testing, we use the CPU backend (and avoid GPU if not explicitly requested).
		require.NoError(t, os.Setenv(backends.GOMLX_BACKEND, "cpu"))
	}
	mainWithContext(ctx, *flagDataDir, *flagCheckpoint, []string{"train_steps"})
}
