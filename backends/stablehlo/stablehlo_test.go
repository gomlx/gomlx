package stablehlo_test

import (
	"fmt"
	"os"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/stablehlo"
	"github.com/gomlx/gomlx/graph"
	"github.com/janpfeifer/must"
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
	exec := graph.NewExec(backend, func(g *graph.Graph) *graph.Node { return graph.Const(g, float32(-7)) })
	y0 := exec.Call()[0]
	assert.Equal(t, float32(-7), y0.Value())
}
