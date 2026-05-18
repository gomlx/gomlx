// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package model_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	graph "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	. "github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/initializer"
	"github.com/gomlx/gomlx/support/sets"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/default"
)

func init() {
	klog.InitFlags(nil)
}

func TestScopeVariables(t *testing.T) {
	store := NewStore()
	root := store.RootScope()
	sA := root.In("a")

	if root.Scope() != "/" {
		t.Fatalf("Expected scope to be %q, got %q instead", "/", root.Scope())
	}
	want := "/a"
	if sA.Scope() != want {
		t.Fatalf("Expected scope to be %q, got %q instead", want, sA.Scope())
	}

	// Same variable name, but different scopes.
	sB := root.In("b")
	v0 := sA.VariableWithShape("x", shapes.Make(dtypes.Float32))
	v1 := sB.VariableWithShape("x", shapes.Make(dtypes.Float64))

	assert.Equal(t, "/a/x", v0.Path())
	assert.Equal(t, "/b/x", v1.Path())

	// Try to visit same scope again: should panic.
	require.Panicsf(t, func() { _ = root.In("a") }, "Allowed re-visiting scope %q with In()", "a")

	// Use Shared to visit it again:
	require.NotPanics(t, func() { _ = root.Shared("a") })

	// Create another variable, different name.
	require.NotPanics(t, func() { _ = sA.VariableWithShape("y", shapes.Make(dtypes.Int64)) })

	// Try to reuse x in sA:
	v0Reuse := sA.VariableWithShape("x", shapes.Make(dtypes.Float32))
	assert.Equal(t, v0, v0Reuse)

	// Try to reuse with a different shape:
	require.Panicsf(t, func() { _ = sA.VariableWithShape("x", shapes.Make(dtypes.Float32, 1, 1)) },
		"Allowed re-using variable %q in scope %q with a different shape.", v0.Name(), v0.Scope())
}

func TestScopeVariablesInitialization(t *testing.T) {
	store := NewStore()
	root := store.RootScope()
	root.SetParam(initializer.ParamInitialSeed, int64(42))
	sA := root.In("a").WithInitializer(initializer.RandomUniformFn(root, 1.5, 2.5))
	v0 := sA.VariableWithShape("x", shapes.Make(dtypes.Float32))
	sB := root.In("b").WithInitializer(initializer.RandomNormalFn(root, 1.0))
	v1 := sB.VariableWithShape("y", shapes.Make(dtypes.Float64, 2))
	sC := sB.In("c").WithInitializer(initializer.Zero)
	v2 := sC.VariableWithShape("z", shapes.Make(dtypes.Int64, 3, 1))

	backend := testutil.BuildTestBackend()
	require.NoError(t, store.InitializeVariables(backend, nil))

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

func TestParams(t *testing.T) {
	store := NewStore()
	root := store.RootScope()
	root.SetParam("x", 7.0)
	root.SetParam("nil", nil)
	got, found := root.GetParam("x")
	assert.True(t, found)
	assert.Equal(t, 7.0, got)
	assert.Equal(t, 7.0, GetParamOr(root, "x", 0.0))
	assert.Equal(t, 0.0, GetParamOr(root, "foo", 0.0))
	assert.Equal(t, 7.0, MustGetParam[float64](root, "x"))
	assert.Equal(t, 7, MustGetParam[int](root, "x")) // Auto-conversion float64 -> int

	// If set to nil, GetParamOr will return the default.
	assert.Equal(t, float32(11), GetParamOr(root, "nil", float32(11)))
	assert.Equal(t, "blah", GetParamOr(root, "nil", "blah"))

	// The wrong type should panic.
	assert.Panics(t, func() { _ = GetParamOr(root, "x", "string value") })
	// Missing value should panic for MustGetParam.
	assert.Panics(t, func() { MustGetParam[float64](root, "foo") })

	// Check correct search to root node.
	root.SetParam("y", 11.0)
	s0 := root.In("0")
	s0.SetParam("y", 13.0)
	got, found = s0.GetParam("x") // Takes value from root scope.
	assert.True(t, found)
	assert.Equal(t, 7.0, got)
	got, found = s0.GetParam("y") // Takes value from "/0" scope.
	assert.True(t, found)
	assert.Equal(t, 13.0, got)
}

func TestIterVariables(t *testing.T) {
	store := NewStore()
	root := store.RootScope()
	sA := root.In("a")
	_ = sA.VariableWithShape("x", shapes.Make(dtypes.Float32))
	sB := root.In("b")
	_ = sB.VariableWithShape("y", shapes.Make(dtypes.Float64, 2))
	sC := sB.In("c")
	_ = sC.VariableWithShape("z", shapes.Make(dtypes.Float32, 3, 1))

	backend := testutil.BuildTestBackend()
	require.NoError(t, store.InitializeVariables(backend, nil))

	// Checks IterVariables lists all variables:
	got := sets.Make[string]()
	for v := range store.IterVariables() {
		if strings.HasPrefix(v.Name(), "#") {
			// Skip internal variables, like #rngstate.
			continue
		}
		got.Insert(v.Name())
	}
	assert.Equal(t, 3, len(got))
	assert.True(t, got.Has("x") && got.Has("y") && got.Has("z"))
}

func TestDeleteVariable(t *testing.T) {
	loader := &ConstantLoader{
		Values: map[string]*tensors.Tensor{
			"/a/x":   tensors.FromValue(float32(2)),
			"/y":     tensors.FromValue(int64(3)),
			"/b/c/z": tensors.FromValue([][]float32{{7}}),
		},
	}
	store := NewStore()
	store.SetLoader(loader)
	root := store.RootScope()
	sA := root.In("a")
	_ = sA.VariableWithShape("x", shapes.Make(dtypes.Float32))
	sB := root.In("b")
	_ = sB.VariableWithShape("y", shapes.Make(dtypes.Float64, 2))
	sC := sB.In("c")
	_ = sC.VariableWithShape("z", shapes.Make(dtypes.Float32, 1, 1))

	backend := testutil.BuildTestBackend()
	require.NoError(t, store.InitializeVariables(backend, nil))

	assert.Equal(t, 3, store.NumVariables())
	require.NoError(t, store.DeleteVariable("/foo/x"))
	assert.Equal(t, 3, store.NumVariables())
	assert.Len(t, loader.Values, 3)

	require.NoError(t, store.DeleteVariable("/b/y"))
	assert.Equal(t, 2, store.NumVariables())
	assert.Len(t, loader.Values, 3)

	require.NoError(t, store.DeleteVariable("/b/c/z"))
	assert.Equal(t, 1, store.NumVariables())
	assert.Len(t, loader.Values, 2)

	require.NoError(t, store.DeleteVariable("/a/x"))
	assert.Equal(t, 0, store.NumVariables())
	assert.Len(t, loader.Values, 1)
}

// ConstantLoader implements a hard-coded loader of values.
type ConstantLoader struct {
	Values map[string]*tensors.Tensor
}

// LoadVariable implements Loader.
func (l *ConstantLoader) LoadVariable(_ *Store, fullPath string) (value *tensors.Tensor, found bool) {
	if l.Values == nil {
		return
	}
	value, found = l.Values[fullPath]
	return
}

func (l *ConstantLoader) DeleteVariable(_ *Store, fullPath string) error {
	delete(l.Values, fullPath)
	return nil
}

func TestStore_SetLoader(t *testing.T) {
	store := NewStore()
	store.SetLoader(&ConstantLoader{
		Values: map[string]*tensors.Tensor{
			"/x": tensors.FromValue(float32(2)),
			"/y": tensors.FromValue(int32(3)),
		},
	})

	backend := testutil.BuildTestBackend()
	e := MustNewExec(backend, store, func(s *Scope, g *graph.Graph) (*graph.Node, *graph.Node) {
		v0 := s.WithInitializer(initializer.Zero).VariableWithShape("x", shapes.Make(dtypes.Float32))
		v1 := s.VariableWithValue("y", int32(1))
		return v0.NodeValue(g), v1.NodeValue(g)
	})
	results := e.MustExec()
	gotV0 := tensors.ToScalar[float32](results[0])
	gotV1 := tensors.ToScalar[int32](results[1])
	assert.Equal(t, float32(2), gotV0)
	assert.Equal(t, int32(3), gotV1)
}

func TestStore_Clone(t *testing.T) {
	value := []float32{3, 5, 7, 11, 13}
	store := NewStore()
	root := store.RootScope()
	root.SetParam("initial_seed", int64(42))
	_ = root.In("a").In("b").VariableWithValue("x", value)
	_ = root.Shared("a").In("b").VariableWithShape("y", shapes.Make(dtypes.Int8, 2, 3, 4))

	store2, err := store.Clone()
	require.NoError(t, err)

	require.Equal(t, 2, store2.NumVariables())
	v1x := store2.GetVariable("/a/b/x")
	require.NotNil(t, v1x)
	v1y := store2.GetVariable("/a/b/y")
	require.NotNil(t, v1y)
	require.True(t, v1y.Shape().Equal(shapes.Make(dtypes.Int8, 2, 3, 4)))

	// GetParam should back-search to the "initial_seed" at the root scope, and find it.
	root2 := store2.RootScope()
	require.Equal(t, int64(42), GetParamOr(root2.In("a").In("b"), "initial_seed", int64(0)))
}
