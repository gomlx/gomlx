/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

// Package ctxtest holds test utilities for packages that depend on context package. It allows
// for easy running tests on graph building functions that depends on context.Context objects.
package ctxtest

import (
	"fmt"
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/stretchr/testify/require"
)

// TestContextGraphFn should build its own inputs, and return both inputs and outputs
type TestContextGraphFn func(ctx *context.Context, g *Graph) (inputs, outputs []*Node)

// RunTestGraphFn tests a graph building function graphFn by executing it and comparing
// its output(s) to the values in want, reporting back any errors in t.
//
// delta is the margin of value on the difference of output and want values that are acceptable.
// Values of delta <= 0 means only exact equality is accepted.
func RunTestGraphFn(t *testing.T, testName string, graphFn TestContextGraphFn, want []any, delta float64) {
	ctx := context.New()
	var numInputs, numOutputs int
	wrapperFn := func(ctx *context.Context, g *Graph) []*Node {
		i, o := graphFn(ctx, g)
		numInputs, numOutputs = len(i), len(o)
		all := append(i, o...)
		return all
	}

	backend := graphtest.BuildTestBackend()
	exec := context.MustNewExec(backend, ctx, wrapperFn)
	var inputsAndOutputs []*tensors.Tensor
	require.NotPanicsf(t, func() { inputsAndOutputs = exec.MustExec() },
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
	ctx.EnumerateVariables(func(v *context.Variable) {
		fmt.Printf("\tVar %s: ", v.ParameterName())
		if v.Shape().Size() > 16 {
			fmt.Printf("%s\n", v.Shape())
		} else {
			fmt.Printf("%s\n", v.MustValue().GoStr())
		}
	})
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
