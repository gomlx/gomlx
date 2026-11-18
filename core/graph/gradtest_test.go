// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph_test

import (
	"fmt"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/autodiff"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/require"
)

type gradTestFunc func(g *Graph) (output *Node, nodesForGrad []*Node)

func testGradients(t *testing.T, name string, testFn gradTestFunc, wantForGrad []any) {
	backend := testutil.BuildTestBackend()
	testGradientsInDelta(t, name, backend, testFn, wantForGrad, Epsilon)
}

func testGradientsWithBackend(t *testing.T, name string, backend compute.Backend, testFn gradTestFunc, wantForGrad []any) {
	testGradientsInDelta(t, name, backend, testFn, wantForGrad, Epsilon)
}

func testGradientsInDelta(t *testing.T, name string, backend compute.Backend, testFn gradTestFunc, wantForGrad []any, delta float64) {
	t.Run(name, func(t *testing.T) {
		fn := func(g *Graph) []*Node {
			output, nodesForGrad := testFn(g)
			grads := autodiff.Gradient(ReduceAllSum(output), nodesForGrad...)
			return append([]*Node{output}, grads...)
		}
		exec, err := NewExec(backend, fn)
		require.NoError(t, err)
		results, err := exec.Call()
		if compute.IsNotImplemented(err) {
			t.Skipf("Backend %q does not implement op: %v", backend.Name(), err)
		}
		require.NoError(t, err)
		fmt.Printf("\toutput=%v\n", results[0].GoStr())
		gradients := results[1:]
		for ii, output := range gradients {
			fmt.Printf("\tGradient #%d: %s\n", ii, output.GoStr())
		}
		require.Equalf(t, len(wantForGrad), len(gradients), "%s: number of wanted results different from number of gradients", name)
		for ii, output := range gradients {
			if wantShape, ok := wantForGrad[ii].(shapes.Shape); ok {
				require.Truef(t, output.Shape().Equal(wantShape), "Wanted shape %s, got %s", wantShape, output.Shape())
			} else {
				require.Truef(t, tensors.MustFromAnyValue(wantForGrad[ii]).InDelta(output, delta),
					"%s: gradient #%d doesn't match wanted value (within %g delta/margin)\n\t%v", name, ii, delta, wantForGrad[ii])
			}
		}
	})
}
