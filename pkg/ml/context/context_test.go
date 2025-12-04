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
	"runtime"
	"strings"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	. "github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/initializers"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/default"
)

func init() {
	klog.InitFlags(nil)
}

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
	ctx.InitializeVariables(backend, nil)

	fmt.Printf("\tv0=%v\n", v0)
	fmt.Printf("\tv1=%v\n", v1)
	fmt.Printf("\tv2=%v\n", v2)
	t0 := v0.MustValue().Value().(float32)
	if t0 < 1.5 || t0 > 2.5 {
		t.Errorf("Expected RandomUniformFn initialization > 1.5, < 2.5, instead got %f", t0)
	}
	t1 := v1.MustValue().Value().([]float64)
	if t1[0] == 0 || t1[1] == 0 {
		t.Errorf("Expected RandomNormalFn initialization to be random, got %v instead", t1)
	}
	t2 := v2.MustValue().Value().([][]int64)
	require.Equalf(t, [][]int64{{0}, {0}, {0}}, t2, "Expected Zeros initialization to yield zeros, got %v instead", t2)
}

// Iota type parameter using Enumer
type ParamType uint

const (
	ParamTypeA ParamType = iota
	ParamTypeB
)

func (i *ParamType) UnmarshalText(text []byte) error {
	switch string(text) {
	case "a", "type A", "type a":
		*i = ParamTypeA
	case "b", "type B", "type b":
		*i = ParamTypeB
	default:
		return fmt.Errorf(" type %s don't exist", string(text))
	}
	return nil
}

func TestParams(t *testing.T) {
	ctx := New()
	ctx.SetParam("x", 7.0)
	ctx.SetParam("nil", nil)
	got, found := ctx.GetParam("x")
	assert.True(t, found)
	assert.Equal(t, 7.0, got)
	assert.Equal(t, 7.0, GetParamOr(ctx, "x", 0.0))
	assert.Equal(t, 0.0, GetParamOr(ctx, "foo", 0.0))
	assert.Equal(t, 7.0, MustGetParam[float64](ctx, "x"))
	assert.Equal(t, 7, MustGetParam[int](ctx, "x")) // Auto-conversion float64 -> int

	// If set to nil, GetParamOr will return the default.
	assert.Equal(t, float32(11), GetParamOr(ctx, "nil", float32(11)))
	assert.Equal(t, "blah", GetParamOr(ctx, "nil", "blah"))

	// The wrong type should panic.
	assert.Panics(t, func() { _ = GetParamOr(ctx, "x", "string value") })
	// Missing value should panic for MustGetParam.
	assert.Panics(t, func() { MustGetParam[float64](ctx, "foo") })

	// String type parameter
	ctx.SetParam("a", "type A")
	assert.Equal(t, "type A", GetParamOr(ctx, "a", "type A"))

	// Iota type parameter using Enumer
	ctx.SetParam("param_type", ParamTypeA)
	assert.Equal(t, ParamTypeA, GetParamOr(ctx, "param_type", ParamTypeA))

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
	ctx.InitializeVariables(backend, nil)

	// Checks EnumerateVariables lists all variables:
	got := sets.Make[string]()
	setGotFn := func(v *Variable) {
		if strings.HasPrefix(v.Name(), "#") {
			// Skip internal variables, like #rngstate.
			return
		}
		got.Insert(v.Name())
	}
	ctx.EnumerateVariables(setGotFn)
	assert.Equal(t, 3, len(got))
	assert.True(t, got.Has("x") && got.Has("y") && got.Has("z"))

	// Checks EnumerateVariables lists all variables, even if starting from a different scope:
	got = sets.Make[string]()
	ctx0.EnumerateVariables(setGotFn)
	assert.Equal(t, 3, len(got))
	assert.True(t, got.Has("x") && got.Has("y") && got.Has("z"))

	// Checks EnumerateVariablesInScope:
	got = sets.Make[string]()
	ctx1.EnumerateVariablesInScope(setGotFn)
	assert.Equal(t, 2, len(got))
	assert.True(t, got.Has("y") && got.Has("z"))
}

func TestIterVariables(t *testing.T) {
	ctx := New()
	ctx0 := ctx.In("a")
	_ = ctx0.VariableWithShape("x", shapes.Make(dtypes.Float32))
	ctx1 := ctx.In("b")
	_ = ctx1.VariableWithShape("y", shapes.Make(dtypes.Float64, 2))
	ctx2 := ctx1.In("c")
	_ = ctx2.VariableWithShape("z", shapes.Make(dtypes.Float32, 3, 1))

	backend := graphtest.BuildTestBackend()
	ctx.InitializeVariables(backend, nil)

	// Checks IterVariables lists all variables:
	got := sets.Make[string]()
	for v := range ctx.IterVariables() {
		if strings.HasPrefix(v.Name(), "#") {
			// Skip internal variables, like #rngstate.
			continue
		}
		got.Insert(v.Name())
	}
	assert.Equal(t, 3, len(got))
	assert.True(t, got.Has("x") && got.Has("y") && got.Has("z"))

	// Checks IterVariables lists all variables, even if starting from a different scope:
	got = sets.Make[string]()
	for v := range ctx.IterVariables() {
		if strings.HasPrefix(v.Name(), "#") {
			// Skip internal variables, like #rngstate.
			continue
		}
		got.Insert(v.Name())
	}
	assert.Equal(t, 3, len(got))
	assert.True(t, got.Has("x") && got.Has("y") && got.Has("z"))

	// Checks IterVariablesInScope:
	got = sets.Make[string]()
	for v := range ctx1.IterVariablesInScope() {
		if strings.HasPrefix(v.Name(), "#") {
			// Skip internal variables, like #rngstate.
			continue
		}
		got.Insert(v.Name())
	}
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
	ctx.InitializeVariables(backend, nil)

	assert.Equal(t, 3, ctx.NumVariables())
	require.NoError(t, ctx.DeleteVariable("/foo", "x"))
	assert.Equal(t, 3, ctx.NumVariables())
	assert.Len(t, loader.Values, 4)

	require.NoError(t, ctx.DeleteVariable("/b", "y"))
	assert.Equal(t, 2, ctx.NumVariables())
	assert.Len(t, loader.Values, 4)
	assert.NotNil(t, ctx.GetVariableByScopeAndName("/b/c", "z")) // Check "z" hasn't been deleted.

	require.NoError(t, ctx.DeleteVariable("/b/c", "z"))
	assert.Equal(t, 1, ctx.NumVariables())
	assert.Len(t, loader.Values, 3)

	require.NoError(t, ctx.DeleteVariable("/a", "x"))
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
	ctx.InitializeVariables(backend, nil)

	// Remove all under scope "/b"
	require.NoError(t, ctx1.DeleteVariablesInScope())
	assert.Equal(t, 1, ctx.NumVariables())
	assert.NotNil(t, ctx.GetVariableByScopeAndName("/a", "x")) // Check "x" hasn't been deleted.
	assert.Len(
		t,
		loader.Values,
		3,
	) // Only "/b/c/z" must have been deleted -- but notice that /b/c/w is not affected.

	// Check that deleting an empty scope is a no-op.
	require.NoError(t, ctx.In("foo").DeleteVariablesInScope())
	assert.Equal(t, 1, ctx.NumVariables())
	assert.NotNil(t, ctx.GetVariableByScopeAndName("/a", "x")) // Check "x" hasn't been deleted.

	// Delete everything.
	require.NoError(t, ctx.DeleteVariablesInScope())
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

func (l *ConstantLoader) DeleteVariable(_ *Context, scope, name string) error {
	delete(l.Values, JoinScope(scope, name))
	return nil
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
	e := MustNewExec(backend, ctx, func(ctx *Context, g *Graph) (*Node, *Node) {
		v0 := ctx.WithInitializer(initializers.Zero).VariableWithShape("x", shapes.Make(dtypes.Float32))
		v1 := ctx.VariableWithValue("y", 1)
		return v0.ValueGraph(g), v1.ValueGraph(g)
	})
	var results []*tensors.Tensor
	require.NotPanics(t, func() { results = e.MustExec() }, "Failed to run context.Exec")
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

func TestContext_Clone(t *testing.T) {
	value := []float32{3, 5, 7, 11, 13}
	ctx0 := New()
	ctx0.SetParam("initial_seed", int64(42))
	v0x := ctx0.In("a").In("b").VariableWithValue("x", value)
	// Uninitialized variable:
	v0y := ctx0.In("a").In("b").VariableWithShape("y", shapes.Make(dtypes.Int8, 2, 3, 4))

	ctx1, err := ctx0.In("a").In("b").Reuse().Clone()
	require.NoError(t, err)
	require.True(t, ctx1.IsChecked())
	require.True(t, ctx1.IsReuse())
	require.Equal(t, "/a/b", ctx1.Scope())

	require.Equal(t, 2, ctx1.NumVariables())
	v1x := ctx1.GetVariableByScopeAndName("/a/b", "x")
	require.NotNil(t, v1x)
	fmt.Printf("Cloned variable %q: %s\n", v1x.ScopeAndName(), v1x.MustValue())
	v1y := ctx1.GetVariable("y")
	require.NotNil(t, v1y)
	_, err = v1y.Value()
	require.Error(t, err, "/a/b/y was created uninitialized, it should have no value")
	require.True(t, v1y.Shape().Equal(v0y.Shape()))

	// Check the new variable value is independent of the old one.
	ctx0 = nil
	v0x.MustValue().MustFinalizeAll()
	for range 5 {
		runtime.GC()
	}
	require.Equal(t, value, tensors.MustCopyFlatData[float32](v1x.MustValue()))
	// GetParam should back-search to the "initial_seed" at the root scope, and find it.
	require.Equal(t, int64(42), GetParamOr(ctx1, "initial_seed", int64(0)))
}
