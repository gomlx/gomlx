// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package graphtest holds test utilities for packages that depend on the graph package.
package graphtest

import (
	"fmt"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/testutil"
	"github.com/stretchr/testify/require"
)

// TestGraphFn should build its own inputs, and return both inputs and outputs
type TestGraphFn func(g *graph.Graph) (inputs, outputs []*graph.Node)

// BuildTestBackend and sets compute.DefaultConfig to "xla:cpu" -- it can be overwritten by GOMLX_BACKEND environment variable.
//
// Deprecated: use testutil.BuildTestBackend instead.
func BuildTestBackend() compute.Backend {
	return testutil.BuildTestBackend()
}

// TestOfficialBackends iterates over list of backends and calls testFn for each of them.
// If GOMLX_BACKEND environment variable is set, it will only iterate over the one set.
// If GOMLX_BACKEND is not set, it will iterate over all official backends, except those in excludecompute.
// (for tests known not to work on those backends)
//
// Deprecated: use testutil.TestOfficialBackends instead.
func TestOfficialBackends(t *testing.T, testFn func(t *testing.T, backend compute.Backend), excludeBackends ...string) {
	testutil.TestOfficialBackends(t, testFn, excludeBackends...)
}

// RunTestGraphFn tests a graph building function graphFn by executing it and comparing
// its output(s) to the values in want, reporting back any errors in t.
//
// delta is the margin of value on the difference of output and want values that are acceptable.
// Values of delta <= 0 means only exact equality is accepted.
func RunTestGraphFn(t *testing.T, testName string, graphFn TestGraphFn, want []any, delta float64) {
	RunTestGraphFnWithBackend(t, testName, BuildTestBackend(), graphFn, want, delta)
}

// RunTestGraphFnWithBackend tests a graph building function graphFn by executing it and comparing
// its output(s) to the values in want, reporting back any errors in t.
//
// delta is the margin of value on the difference of output and want values that are acceptable.
// Values of delta <= 0 means only exact equality is accepted.
func RunTestGraphFnWithBackend(t *testing.T, testName string, backend compute.Backend, graphFn TestGraphFn, want []any, delta float64) {
	t.Run(testName, func(t *testing.T) {
		wantTensors := xslices.Map(want, func(value any) *tensors.Tensor {
			if s, ok := value.(shapes.Shape); ok {
				return tensors.FromShape(s)
			}
			return tensors.FromAnyValue(value)
		})

		var numInputs, numOutputs int
		wrapperFn := func(g *graph.Graph) []*graph.Node {
			i, o := graphFn(g)
			numInputs, numOutputs = len(i), len(o)
			all := append(i, o...)
			return all
		}
		exec := graph.MustNewExec(backend, wrapperFn)
		inputsAndOutputs, err := exec.Exec()
		require.NoErrorf(t, err, "%s: failed to execute graph", testName)
		require.NotPanicsf(t, func() { inputsAndOutputs = exec.MustExec() }, "%s: failed to execute graph", testName)
		inputs := inputsAndOutputs[:numInputs]
		for ii, input := range inputs {
			if input == nil {
				t.Fatalf("%q: inputs[%d] is nil!?", testName, ii)
			}
		}
		outputs := inputsAndOutputs[numInputs:]
		for ii, input := range inputs {
			if input == nil {
				t.Fatalf("%q: outputs[%d] is nil!?", testName, ii)
			}
		}

		fmt.Printf("\n%s:\n", testName)
		for ii, input := range inputs {
			fmt.Printf("\tInput %d: %s\n", ii, input.GoStr())
		}
		if numInputs > 0 {
			fmt.Printf("\t======\n")
		}
		for ii, output := range outputs {
			fmt.Printf("\tOutput %d: %s\n", ii, output.GoStr())
		}
		require.Equalf(t, len(want), numOutputs, "%s: number of wanted results different from number of outputs", testName)

		for ii, output := range outputs {
			require.Truef(t, wantTensors[ii].InDelta(output, delta), "%s: output #%d doesn't match wanted value %v",
				testName, ii, want[ii])
		}
	})
}
