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
	"github.com/gomlx/gomlx/graph/graphtest"
	. "github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"

	_ "github.com/gomlx/gomlx/backends/xla"
)

func TestContextVariables(t *testing.T) {
	ctx := New()
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
	v0 := ctx2.VariableWithShape("x", shapes.Make(dtypes.Float32))
	_ = ctx3.VariableWithShape("x", shapes.Make(dtypes.Float64))

	// Try to reuse, without the context being set for that:
	require.Panicsf(t, func() { v0 = ctx2.VariableWithShape("x", shapes.Make(dtypes.Float32)) },
		"Allowed re-creating variable without context set to reuse. v0=%+v", v0)

	// Create another variable, different name.
	require.NotPanics(t, func() { v0 = ctx2.VariableWithShape("y", shapes.Make(dtypes.Int64)) })

	// Try to reuse:
	ctx2 = ctx2.Reuse()
	v0 = ctx2.VariableWithShape("x", shapes.Make(dtypes.Float32))

	// Try to reuse with a different shape:
	require.Panicsf(t, func() { v0 = ctx2.VariableWithShape("x", shapes.Make(dtypes.Float32, 1, 1)) },
		"Allowed re-using variable %q in scope %q with a different shape context set to reuse.", v0.Name(), v0.Scope())
}

func TestContextVariablesInitialization(t *testing.T) {
	ctx := New()
	ctx.SetParam(initializers.ParamInitialSeed, int64(42))
	ctx0 := ctx.In("a").WithInitializer(initializers.RandomUniformFn(ctx, 1.5, 2.5))
	v0 := ctx0.VariableWithShape("x", shapes.Make(dtypes.Float32))
	ctx1 := ctx.In("b").WithInitializer(initializers.RandomNormalFn(ctx, 1.0))
	v1 := ctx1.VariableWithShape("y", shapes.Make(dtypes.Float64, 2))
	ctx2 := ctx1.In("c").WithInitializer(initializers.Zero)
	v2 := ctx2.VariableWithShape("z", shapes.Make(dtypes.Int64, 3, 1))

	backend := graphtest.BuildTestBackend()
	ctx.InitializeVariables(backend)

	fmt.Printf("\tv0=%v\n", v0)
	fmt.Printf("\tv1=%v\n", v1)
	fmt.Printf("\tv2=%v\n", v2)
	t0 := v0.Value().Value().(float32)
	if t0 < 1.5 || t0 > 2.5 {
		t.Errorf("Expected RandomUniformFn initialization > 1.5, < 2.5, instead got %f", t0)
	}
	t1 := v1.Value().Value().([]float64)
	if t1[0] == 0 || t1[1] == 0 {
		t.Errorf("Expected RandomNormalFn initialization to be random, got %v intead", t1)
	}
	t2 := v2.Value().Value().([][]int64)
	require.Equalf(t, [][]int64{{0}, {0}, {0}}, t2, "Expected Zeros initialization to yield zeros, got %v instead", t2)
}

func TestParams(t *testing.T) {
	ctx := New()
	ctx.SetParam("x", 7.0)
	got, found := ctx.GetParam("x")
	assert.True(t, found)
	assert.Equal(t, 7.0, got)
	assert.Equal(t, 7.0, GetParamOr(ctx, "x", 0.0))
	assert.Equal(t, 0.0, GetParamOr(ctx, "foo", 0.0))
	// Wrong type should panic.
	assert.Panics(t, func() { _ = GetParamOr(ctx, "x", "string value") })

	// Check correct search to root node.
	ctx.SetParam("y", 11.0)
	ctx0 := ctx.In("0")
	ctx0.SetParam("y", 13.0)
	got, found = ctx0.GetParam("x") // Takes value from root scope.
	assert.True(t, found)
	assert.Equal(t, 7.0, got)
	got, found = ctx0.GetParam("y") // Takes value from "/0" scope.
	assert.True(t, found)
	assert.Equal(t, 13.0, got)
	assert.Equal(t, 7.0, GetParamOr(ctx, "x", 0.0))
}

func TestEnumerateVariables(t *testing.T) {
	ctx := New()
	ctx0 := ctx.In("a")
	_ = ctx0.VariableWithShape("x", shapes.Make(dtypes.Float32))
	ctx1 := ctx.In("b")
	_ = ctx1.VariableWithShape("y", shapes.Make(dtypes.Float64, 2))
	ctx2 := ctx1.In("c")
	_ = ctx2.VariableWithShape("z", shapes.Make(dtypes.Float32, 3, 1))

	backend := graphtest.BuildTestBackend()
	ctx.InitializeVariables(backend)

	// Checks EnumerateVariables lists all variables:
	got := types.MakeSet[string]()
	setGotFn := func(v *Variable) { got.Insert(v.Name()) }
	ctx.EnumerateVariables(setGotFn)
	assert.Equal(t, 3, len(got))
	assert.True(t, got.Has("x") && got.Has("y") && got.Has("z"))

	// Checks EnumerateVariables lists all variables, even if starting from a different scope:
	got = types.MakeSet[string]()
	ctx0.EnumerateVariables(setGotFn)
	assert.Equal(t, 3, len(got))
	assert.True(t, got.Has("x") && got.Has("y") && got.Has("z"))

	// Checks EnumerateVariablesInScope:
	got = types.MakeSet[string]()
	ctx1.EnumerateVariablesInScope(setGotFn)
	assert.Equal(t, 2, len(got))
	assert.True(t, got.Has("y") && got.Has("z"))
}

func TestDeleteVariable(t *testing.T) {
	loader := &ConstantLoader{
		Values: map[string]*tensors.Tensor{
			"/a/x":   tensors.FromValue(float32(2)),
			"/y":     tensors.FromValue(int64(3)),
			"/b/c/z": tensors.FromValue([][]float32{{7}}),
			"/b/c/w": tensors.FromValue(int64(11)),
		},
	}
	ctx := New().Checked(false)
	ctx.SetLoader(loader)
	ctx0 := ctx.In("a")
	_ = ctx0.VariableWithShape("x", shapes.Make(dtypes.Float32))
	ctx1 := ctx.In("b")
	_ = ctx1.VariableWithShape("y", shapes.Make(dtypes.Float64, 2))
	ctx2 := ctx1.In("c")
	_ = ctx2.VariableWithShape("z", shapes.Make(dtypes.Float32, 1, 1))

	backend := graphtest.BuildTestBackend()
	ctx.InitializeVariables(backend)

	assert.Equal(t, 3, ctx.NumVariables())
	ctx.DeleteVariable("/foo", "x")
	assert.Equal(t, 3, ctx.NumVariables())
	assert.Len(t, loader.Values, 4)

	ctx.DeleteVariable("/b", "y")
	assert.Equal(t, 2, ctx.NumVariables())
	assert.Len(t, loader.Values, 4)
	assert.NotNil(t, ctx.InspectVariable("/b/c", "z")) // Check "z" hasn't been deleted.

	ctx.DeleteVariable("/b/c", "z")
	assert.Equal(t, 1, ctx.NumVariables())
	assert.Len(t, loader.Values, 3)

	ctx.DeleteVariable("/a", "x")
	assert.Equal(t, 0, ctx.NumVariables())
	assert.Len(t, loader.Values, 2)
}

func TestDeleteVariablesInScope(t *testing.T) {
	ctx := New().Checked(false)
	loader := &ConstantLoader{
		Values: map[string]*tensors.Tensor{
			"/a/x":   tensors.FromValue(float32(2)),
			"/y":     tensors.FromValue(int64(3)),
			"/b/c/z": tensors.FromValue([][]float32{{7}}),
			"/b/c/w": tensors.FromValue(int64(11)),
		},
	}
	ctx.SetLoader(loader)
	ctx0 := ctx.In("a")
	_ = ctx0.VariableWithShape("x", shapes.Make(dtypes.Float32))
	ctx1 := ctx.In("b")
	_ = ctx1.VariableWithShape("y", shapes.Make(dtypes.Float64, 2))
	ctx2 := ctx1.In("c")
	_ = ctx2.VariableWithShape("z", shapes.Make(dtypes.Float32, 1, 1))

	backend := graphtest.BuildTestBackend()
	ctx.InitializeVariables(backend)

	// Remove all under scope "/b"
	ctx1.DeleteVariablesInScope()
	assert.Equal(t, 1, ctx.NumVariables())
	assert.NotNil(t, ctx.InspectVariable("/a", "x")) // Check "x" hasn't been deleted.
	assert.Len(t, loader.Values, 3)                  // Only "/b/c/z" must have been deleted -- but notice that /b/c/w is not affected.

	// Check that deleting an empty scope is a no-op.
	ctx.In("foo").DeleteVariablesInScope()
	assert.Equal(t, 1, ctx.NumVariables())
	assert.NotNil(t, ctx.InspectVariable("/a", "x")) // Check "x" hasn't been deleted.

	// Delete everything.
	ctx.DeleteVariablesInScope()
	assert.Equal(t, 0, ctx.NumVariables())
}

// ConstantLoader implements a hard-coded loader of values.
type ConstantLoader struct {
	Values map[string]*tensors.Tensor
}

// LoadVariable implements Loader.
func (l *ConstantLoader) LoadVariable(_ *Context, scope, name string) (value *tensors.Tensor, found bool) {
	if l.Values == nil {
		return
	}
	value, found = l.Values[JoinScope(scope, name)]
	return
}

func (l *ConstantLoader) DeleteVariable(_ *Context, scope, name string) {
	delete(l.Values, JoinScope(scope, name))
}

func TestContext_SetLoader(t *testing.T) {
	ctx := New()
	ctx.SetLoader(&ConstantLoader{
		Values: map[string]*tensors.Tensor{
			"/x": tensors.FromValue(float32(2)),
			"/y": tensors.FromValue(int64(3)),
			"/z": tensors.FromValue(int64(7)),
		},
	})
	ctx = ctx.Reuse()

	backend := graphtest.BuildTestBackend()
	e := NewExec(backend, ctx, func(ctx *Context, g *Graph) (*Node, *Node) {
		v0 := ctx.WithInitializer(initializers.Zero).VariableWithShape("x", shapes.Make(dtypes.Float32))
		v1 := ctx.VariableWithValue("y", 1)
		return v0.ValueGraph(g), v1.ValueGraph(g)
	})
	var results []*tensors.Tensor
	require.NotPanics(t, func() { results = e.Call() }, "Failed to run context.Exec")
	gotV0 := tensors.ToScalar[float32](results[0])
	gotV1 := tensors.ToScalar[int64](results[1])
	if gotV0 != 2 || gotV1 != 3 {
		t.Errorf("Got x,y = (%f, %d), wanted (2.0, 3)", gotV0, gotV1)
	}
}

func TestJoinAndSplitScope(t *testing.T) {
	assert.Equal(t, "a", JoinScope("", "a"))
	assert.Equal(t, "/a", JoinScope("/", "a"))
	assert.Equal(t, "/b/a", JoinScope("/b", "a"))
	assert.Equal(t, "/b/a", JoinScope("/b/", "a"))
	assert.Equal(t, "/c/b/a", JoinScope("/c/b/", "a"))

	testSplit := func(scopeAndName, wantScope, wantName string) {
		gotScope, gotName := SplitScope(scopeAndName)
		assert.Equal(t, []string{wantScope, wantName}, []string{gotScope, gotName})
	}
	testSplit("a", "", "a")
	testSplit("/a", "/", "a")
	testSplit("/b/a", "/b", "a")
	testSplit("/c/b/a", "/c/b", "a")
	testSplit("/c/b/", "/c/b", "")
	testSplit("a/b", "", "a/b") // Notice that something that doesn't start with "/" doesn't have a scope.
}
