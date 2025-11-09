package stablehlo_test

import (
	"fmt"
	"os"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/stablehlo"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/stretchr/testify/assert"
	"k8s.io/klog/v2"
)

var backend backends.Backend

func init() {
	klog.InitFlags(nil)
}

func setup() {
	fmt.Printf("Available backends: %q\n", backends.List())
	if os.Getenv(backends.ConfigEnvVar) == "" {
		must.M(os.Setenv(backends.ConfigEnvVar, stablehlo.BackendName))
	} else {
		fmt.Printf("\t$%s=%q\n", backends.ConfigEnvVar, os.Getenv(backends.ConfigEnvVar))
	}
	backend = backends.MustNew()
	fmt.Printf("Backend: %s, %s\n", backend.Name(), backend.Description())
	fmt.Printf("\t- Add flag -vmodule=executable=2 to log the StableHLO program being executed.\n")
	for deviceNum := range backends.DeviceNum(backend.NumDevices()) {
		fmt.Printf("\t- Device #%d: %s\n", deviceNum, backend.DeviceDescription(deviceNum))
	}
}

func teardown() {
	backend.Finalize()
}

func TestMain(m *testing.M) {
	setup()
	code := m.Run() // Run all tests in the file
	teardown()
	os.Exit(code)
}

func TestCompileAndRun(t *testing.T) {
	// Just return a constant.
	exec := graph.MustNewExec(backend, func(g *graph.Graph) *graph.Node { return graph.Const(g, float32(-7)) })
	y0 := exec.MustExec()[0]
	assert.Equal(t, float32(-7), y0.Value())
}
