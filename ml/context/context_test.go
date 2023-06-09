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

package context

import (
	"fmt"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestContextVariables(t *testing.T) {
	manager := graphtest.BuildTestManager()
	ctx := NewContext(manager)
	ctx2 := ctx.In("a")

	if ctx.Scope() != ScopeSeparator {
		t.Fatalf("Expected scope to be %q, got %q instead", ScopeSeparator, ctx.Scope())
	}
	want := fmt.Sprintf("%s%s", ScopeSeparator, "a")
	if ctx2.Scope() != want {
		t.Fatalf("Expected scope to be %q, got %q instead", want, ctx2.Scope())
	}

	// Same variable name, but different scopes.
	ctx3 := ctx.In("b")
	v0 := ctx2.VariableWithShape("x", shapes.Make(shapes.Float32))
	_ = ctx3.VariableWithShape("x", shapes.Make(shapes.Float64))

	// Try to reuse, without the context being set for that:
	require.Panicsf(t, func() { v0 = ctx2.VariableWithShape("x", shapes.Make(shapes.Float32)) },
		"Allowed re-creating variable without context set to reuse. v0=%+v", v0)

	// Create another variable, different name.
	require.NotPanics(t, func() { v0 = ctx2.VariableWithShape("y", shapes.Make(shapes.Int64)) })

	// Try to reuse:
	ctx2 = ctx2.Reuse()
	v0 = ctx2.VariableWithShape("x", shapes.Make(shapes.Float32))

	// Try to reuse with a different shape:
	require.Panicsf(t, func() { v0 = ctx2.VariableWithShape("x", shapes.Make(shapes.Float32, 1, 1)) },
		"Allowed re-using variable %q in scope %q with a different shape context set to reuse.", v0.Name(), v0.Scope())
}

func TestContextVariablesInitialization(t *testing.T) {
	manager := graphtest.BuildTestManager()
	ctx := NewContext(manager)
	ctx0 := ctx.In("a").WithInitializer(initializers.RandomUniformFn(42, 1.5, 2.5))
	v0 := ctx0.VariableWithShape("x", shapes.Make(shapes.Float32))
	ctx1 := ctx.In("b").WithInitializer(initializers.RandomNormalFn(42, 1.0))
	v1 := ctx1.VariableWithShape("y", shapes.Make(shapes.Float64, 2))
	ctx2 := ctx1.In("c").WithInitializer(initializers.Zero)
	v2 := ctx2.VariableWithShape("z", shapes.Make(shapes.Int64, 3, 1))
	ctx.InitializeVariables()

	fmt.Printf("\tv0=%v\n", v0.Value().Local())
	fmt.Printf("\tv1=%v\n", v1.Value().Local())
	fmt.Printf("\tv2=%v\n", v2.Value().Local())
	t0 := v0.Value().Local().Value().(float32)
	if t0 < 1.5 || t0 > 2.5 {
		t.Errorf("Expected RandomUniformFn initialization > 1.5, < 2.5, instead got %f", t0)
	}
	t1 := v1.Value().Local().Value().([]float64)
	if t1[0] == 0 || t1[1] == 0 {
		t.Errorf("Expected RandomNormalFn initialization to be random, got %v intead", t1)
	}
	t2 := v2.Value().Local().Value().([][]int)
	if !slices.DeepSliceCmp([][]int{{0}, {0}, {0}}, t2, slices.Equal[int]) {
		t.Errorf("Expected Zeros initialization to yield zeros, got %v instead", t2)
	}
}

type ConstantLoader struct {
	Values map[string]map[string]tensor.Tensor
}

// LoadVariable implements Loader.
func (l *ConstantLoader) LoadVariable(ctx *Context, v *Variable) (value tensor.Tensor, found bool) {
	if l.Values == nil {
		return
	}
	var nameToValue map[string]tensor.Tensor
	nameToValue, found = l.Values[v.Scope()]
	if !found {
		return
	}
	value, found = nameToValue[v.Name()]
	return
}

func TestContext_SetLoader(t *testing.T) {
	manager := graphtest.BuildTestManager()
	ctx := NewContext(manager)
	ctx.SetLoader(&ConstantLoader{
		Values: map[string]map[string]tensor.Tensor{
			"/": {
				"x": tensor.FromValue(float32(2)),
				"y": tensor.FromValue(int(3)),
			},
		},
	})
	e := NewExec(manager, ctx, func(ctx *Context, g *Graph) (*Node, *Node) {
		v0 := ctx.WithInitializer(initializers.Zero).VariableWithShape("x", shapes.Make(shapes.Float32))
		v1 := ctx.VariableWithValue("y", 1)
		return v0.ValueGraph(g), v1.ValueGraph(g)
	})
	var results []tensor.Tensor
	require.NotPanics(t, func() { results = e.Call() }, "Failed to run context.Exec")
	gotV0 := results[0].Value().(float32)
	gotV1 := results[1].Value().(int)
	if gotV0 != 2 || gotV1 != 3 {
		t.Errorf("Got x,y = (%f, %d), wanted (2.0, 3)", gotV0, gotV1)
	}
}
