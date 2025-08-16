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

// Package graphtest holds test utilities for packages that depend on the graph package.
package graphtest

import (
	"fmt"
	"sync"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/stretchr/testify/require"
)

// TestGraphFn should build its own inputs, and return both inputs and outputs
type TestGraphFn func(g *graph.Graph) (inputs, outputs []*graph.Node)

var (
	backendOnce   sync.Once
	cachedBackend backends.Backend
)

// BuildTestBackend and sets backends.DefaultConfig to "xla:cpu" -- it can be overwritten by GOMLX_BACKEND environment variable.
func BuildTestBackend() backends.Backend {
	backends.DefaultConfig = "xla:cpu"
	backendOnce.Do(func() {
		cachedBackend = backends.MustNew()
		fmt.Printf("Backend: %s\n", cachedBackend.Description())
	})
	return cachedBackend
}

// RunTestGraphFn tests a graph building function graphFn by executing it and comparing
// its output(s) to the values in want, reporting back any errors in t.
//
// delta is the margin of value on the difference of output and want values that are acceptable.
// Values of delta <= 0 means only exact equality is accepted.
func RunTestGraphFn(t *testing.T, testName string, graphFn TestGraphFn, want []any, delta float64) {
	t.Run(testName, func(t *testing.T) {
		backend := BuildTestBackend()
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
		exec := graph.NewExec(backend, wrapperFn)
		var inputsAndOutputs []*tensors.Tensor
		require.NotPanicsf(t, func() { inputsAndOutputs = exec.Call() }, "%s: failed to execute graph", testName)
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
