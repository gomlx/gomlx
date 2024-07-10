package main

import (
	"flag"
	"github.com/gomlx/gomlx/graph"
	"github.com/stretchr/testify/require"
	"os"
	"testing"
)

func TestMainFunc(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping testing in short mode")
		return
	}
	flag.Parse()
	*flagNumSteps = 10
	if _, found := os.LookupEnv(graph.DefaultPluginEnv); !found { // "GOMLX_PJRT_PLUGIN"
		err := os.Setenv(graph.DefaultPluginEnv, "cpu") // Force default to CPU.
		require.NoError(t, err)
	}
	main()
}
