package main

import (
	"flag"
	"os"
	"testing"

	"github.com/gomlx/gomlx/graph"
	"github.com/stretchr/testify/require"
)

func TestMainFunc(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping testing in short mode")
		return
	}
	flag.Parse()
	*flagNumSteps = 10
	if _, found := os.LookupEnv(graph.DefaultPlatformEnv); !found { // "GOMLX_PLATFORM"
		err := os.Setenv("GOMLX_PLATFORM", "Host") // Force default to CPU.
		require.NoError(t, err)
	}
	main()
}
