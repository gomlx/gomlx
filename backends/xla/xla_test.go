// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xla_test

import (
	"fmt"
	"os"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/backends/xla"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/stretchr/testify/assert"
	"k8s.io/klog/v2"
)

var backend compute.Backend

func init() {
	klog.InitFlags(nil)
}

func setup() {
	fmt.Printf("Available backends: %q\n", compute.List())
	if os.Getenv(compute.ConfigEnvVar) == "" {
		must.M(os.Setenv(compute.ConfigEnvVar, xla.BackendName))
	} else {
		fmt.Printf("\t$%s=%q\n", compute.ConfigEnvVar, os.Getenv(compute.ConfigEnvVar))
	}
	backend = compute.MustNew()
	fmt.Printf("Backend: %s, %s\n", backend.Name(), backend.Description())
	fmt.Printf("\t- Add flag -vmodule=executable=2 to log the StableHLO program being executed.\n")
	for deviceNum := range compute.DeviceNum(backend.NumDevices()) {
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
