// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package modeltest holds test utilities for packages that depend on model package. It allows
// for easy running tests on graph building functions that depends on model.Scope objects.
package modeltest

import (
	"fmt"
	"testing"

	"github.com/gomlx/compute/support/xslices"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/require"
)

// TestScopeGraphFn should build its own inputs, and return both inputs and outputs
type TestScopeGraphFn func(scope *model.Scope, g *Graph) (inputs, outputs []*Node)

// RunTestGraphFn tests a graph building function graphFn by executing it and comparing
// its output(s) to the values in want, reporting back any errors in t.
//
// delta is the margin of value on the difference of output and want values that are acceptable.
// Values of delta <= 0 means only exact equality is accepted.
func RunTestGraphFn(t *testing.T, testName string, graphFn TestScopeGraphFn, want []any, delta float64) {
	store := model.NewStore()
	var numInputs, numOutputs int
	wrapperFn := func(scope *model.Scope, g *Graph) []*Node {
		i, o := graphFn(scope, g)
		numInputs, numOutputs = len(i), len(o)
		all := append(i, o...)
		return all
	}

	backend := testutil.BuildTestBackend()
	exec := model.MustNewExec(backend, store, wrapperFn)
	var inputsAndOutputs []*tensors.Tensor
	require.NotPanicsf(t, func() { inputsAndOutputs = exec.MustCall() },
		"%s: failed to run graph", testName)
	inputs := inputsAndOutputs[:numInputs]
	outputs := inputsAndOutputs[numInputs:]

	fmt.Printf("\n%s:\n", testName)
	for ii, input := range inputs {
		fmt.Printf("\tInput %d: %s\n", ii, input.GoStr())
	}
	if numInputs > 0 {
		fmt.Printf("\t======\n")
	}
	for v := range store.IterVariables() {
		fmt.Printf("\tVar %s: ", v.Path())
		if v.Shape().Size() > 16 {
			fmt.Printf("%s\n", v.Shape())
		} else {
			fmt.Printf("%s\n", v.MustValue().GoStr())
		}
	}
	fmt.Printf("\t======\n")
	for ii, output := range outputs {
		fmt.Printf("\tOutput %d: %s\n", ii, output.GoStr())
	}
	require.Equalf(t, len(want), numOutputs, "%s: number of wanted results different from number of outputs", testName)

	for ii, output := range outputs {
		require.Truef(t, xslices.SlicesInDelta(output.Value(), want[ii], delta), "%s: output #%d doesn't match wanted value %v",
			testName, ii, want[ii])
	}
}
