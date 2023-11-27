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

package context_test

import (
	"fmt"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func oneLayerGraph(ctx *context.Context, x *Node) *Node {
	return layers.DenseWithBias(ctx.In("dense0"), x, 1)
}

func TestExec(t *testing.T) {
	manager := graphtest.BuildTestManager()
	oneLayer, err := context.NewExecAny(manager, nil, oneLayerGraph)
	if err != nil {
		t.Fatalf("Failed to create context.Exec for oneLayer: %+v", err)
	}

	// First graph build, variable should be initialized.
	x := [][]float64{{1, 1, 1}}
	got := oneLayer.Call(x)[0].Value().([][]float64)[0][0]
	assert.NotEqual(t, 0.0, got, "Failed evaluating oneLayer(%v) returned 0, but weights should have been randomly initialized", x)

	// Second call should reuse the graph, and yield same result.
	got2 := oneLayer.Call(x)[0].Value().([][]float64)[0][0]
	assert.Equal(t, got, got2, "Second call to oneLayer(%v) returned %f, but first returned %f", x, got, got2)

	// Different batch size: it should generate a new graph, but yield the same result.
	x = [][]float64{{1, 1, 1}, {2, 2, 2}}
	got3 := oneLayer.Call(x)[0].Value().([][]float64)[0][0]
	assert.Equal(t, got, got3, "Call to oneLayer(%v) on a larger batch returned %f, but original call returned %f",
		x, got, got3)

	// If X changes the inner dimension, then the layers.DenseWithBias would require different shaped variables, which should
	// fail to build.
	x = [][]float64{{1, 1, 1, 1}}
	require.Panics(t, func() { _ = oneLayer.Call(x) },
		"oneLayer(%v) leads to a graph with differently shaped variables should have failed",
		x)
}

func oneLayerManyInputsGraph(ctx *context.Context, inputs []*Node) *Node {
	return layers.DenseWithBias(ctx.In("dense0"), Concatenate(inputs, -1), 1)
}

func TestExecWithSlices(t *testing.T) {
	manager := graphtest.BuildTestManager()
	oneLayer, err := context.NewExecAny(manager, nil, oneLayerManyInputsGraph)
	if err != nil {
		t.Fatalf("Failed to create context.Exec for oneLayer: %+v", err)
	}

	// First execution builds graph and should initialize variables (weights) randomly.
	x := [][]float64{{1, 1, 1}}
	got := oneLayer.Call(x)[0].Value().([][]float64)
	fmt.Printf("\toneLayer(%v)=%v\n", x, got)
	if got[0][0] == 0 {
		t.Fatalf("Failed evaluating oneLayer(%v) returned 0, but weights should have been randomly initialized", x)
	}

	// Second execution: two tensors that if concatenated will be the same as the previous run. Since the context
	// and hence the variables are the same, it should evaluate to exact same value.
	oneElementOfX := [][]float64{{1}}
	got2 := oneLayer.Call(oneElementOfX, oneElementOfX, oneElementOfX)[0].Value().([][]float64)
	fmt.Printf("\toneLayer(%v, %v, %v)=%v\n", oneElementOfX, oneElementOfX, oneElementOfX, got)
	if !slices.DeepSliceCmp(got, got2, slices.Equal[float64]) {
		t.Fatalf("oneLayer(%v) should have been the same as oneLayer(%v, %v, %v), but got %v and %v",
			x, oneElementOfX, oneElementOfX, oneElementOfX, got, got2)
	}
}

func TestExecWithVariableUpdates(t *testing.T) {
	manager := graphtest.BuildTestManager()
	ctx := context.NewContext(manager)
	counter := context.NewExec(manager, ctx, func(ctx *context.Context, g *Graph) *Node {
		dtype := shapes.Int64
		counterVar := ctx.WithInitializer(initializers.Zero).VariableWithShape("counter", shapes.Make(dtype))
		counterNode := counterVar.ValueGraph(g)
		counterNode = Add(counterNode, OnesLike(counterNode))
		counterVar.SetValueGraph(counterNode)
		return counterNode
	})
	results := counter.Call()
	gotTensor := results[0]
	got := gotTensor.Value().(int64)
	if got != 1 {
		t.Fatalf("Wanted first counter value to be 1, got %d instead", got)
	}
	results = counter.Call()
	gotTensor = results[0]
	got = gotTensor.Value().(int64)
	if got != 2 {
		t.Fatalf("Wanted second counter value to be 2, got %d instead", got)
	}
	fmt.Printf("%s\n", ctx.InspectVariable(ctx.Scope(), "counter").Value()) // 2
}
