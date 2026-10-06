// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package model_test

import (
	"maps"
	"testing"

	graph "github.com/gomlx/gomlx/core/graph"
	. "github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

func TestStore_SetParam(t *testing.T) {
	t.Run("RootScopeWithAndWithoutSlash", func(t *testing.T) {
		store := NewStore()
		store.SetParam("/p1", 10)
		store.SetParam("p2", 20)

		// Retrieval via Store.GetParam:
		v, found := store.GetParam("/p1")
		assert.True(t, found)
		assert.Equal(t, 10, v)
		v, found = store.GetParam("p1")
		assert.True(t, found)
		assert.Equal(t, 10, v)

		v, found = store.GetParam("/p2")
		assert.True(t, found)
		assert.Equal(t, 20, v)
		v, found = store.GetParam("p2")
		assert.True(t, found)
		assert.Equal(t, 20, v)

		// Both are accessible in the root scope and inherited by sub-scopes.
		root := store.RootScope()
		assert.Equal(t, 10, MustGetParam[int](root, "p1"))
		assert.Equal(t, 20, MustGetParam[int](root, "p2"))

		sub := root.In("sub")
		assert.Equal(t, 10, MustGetParam[int](sub, "p1"))
		assert.Equal(t, 20, MustGetParam[int](sub, "p2"))
	})

	t.Run("SubScopeWithAndWithoutSlash", func(t *testing.T) {
		store := NewStore()
		store.SetParam("/a/b/p1", "val1")
		store.SetParam("a/b/p2", "val2")

		// Retrieval via Store.GetParam with and without leading slash:
		v, found := store.GetParam("/a/b/p1")
		assert.True(t, found)
		assert.Equal(t, "val1", v)
		v, found = store.GetParam("a/b/p1")
		assert.True(t, found)
		assert.Equal(t, "val1", v)

		v, found = store.GetParam("/a/b/p2")
		assert.True(t, found)
		assert.Equal(t, "val2", v)
		v, found = store.GetParam("a/b/p2")
		assert.True(t, found)
		assert.Equal(t, "val2", v)

		// Accessible in scope /a/b:
		sAB := store.Scope("/a/b")
		assert.Equal(t, "val1", MustGetParam[string](sAB, "p1"))
		assert.Equal(t, "val2", MustGetParam[string](sAB, "p2"))

		// Inherited by descendant scope /a/b/c:
		sABC := store.Scope("/a/b/c")
		assert.Equal(t, "val1", MustGetParam[string](sABC, "p1"))
		assert.Equal(t, "val2", MustGetParam[string](sABC, "p2"))

		// NOT accessible in parent scope /a or root scope /:
		sA := store.Scope("/a")
		_, found = sA.GetParam("p1")
		assert.False(t, found)
		_, found = store.RootScope().GetParam("p1")
		assert.False(t, found)

		// NOT accessible in sibling scope /a/sibling:
		sSibling := store.Scope("/a/sibling")
		_, found = sSibling.GetParam("p1")
		assert.False(t, found)
	})

	t.Run("ShadowingAndOverwriting", func(t *testing.T) {
		store := NewStore()
		store.SetParam("/lr", 0.01)
		store.SetParam("/model/lr", 0.001)

		root := store.RootScope()
		sModel := store.Scope("/model")
		sLayer := store.Scope("/model/layer")
		sOther := store.Scope("/other")

		// Root and sibling should see root lr:
		assert.Equal(t, 0.01, MustGetParam[float64](root, "lr"))
		assert.Equal(t, 0.01, MustGetParam[float64](sOther, "lr"))

		// /model and /model/layer should see shadowed lr:
		assert.Equal(t, 0.001, MustGetParam[float64](sModel, "lr"))
		assert.Equal(t, 0.001, MustGetParam[float64](sLayer, "lr"))

		// Overwrite parameter in /model:
		store.SetParam("/model/lr", 0.002)
		assert.Equal(t, 0.002, MustGetParam[float64](sModel, "lr"))
		assert.Equal(t, 0.002, MustGetParam[float64](sLayer, "lr"))
		assert.Equal(t, 0.01, MustGetParam[float64](root, "lr"))
	})
}

func TestStore_SetParams(t *testing.T) {
	store := NewStore()
	store.SetParams(map[string]any{
		"/global_lr":         0.1,
		"seed":               int64(42),
		"/cnn/num_layers":    4,
		"cnn/kernel_size":    3,
		"/cnn/conv1/filters": 32,
	})

	root := store.RootScope()
	sCNN := store.Scope("/cnn")
	sConv1 := store.Scope("/cnn/conv1")
	sOther := store.Scope("/rnn")

	// Root parameters:
	assert.Equal(t, 0.1, MustGetParam[float64](root, "global_lr"))
	assert.Equal(t, int64(42), MustGetParam[int64](root, "seed"))

	// Inherited by /cnn and /rnn:
	assert.Equal(t, 0.1, MustGetParam[float64](sCNN, "global_lr"))
	assert.Equal(t, int64(42), MustGetParam[int64](sCNN, "seed"))
	assert.Equal(t, 0.1, MustGetParam[float64](sOther, "global_lr"))
	assert.Equal(t, int64(42), MustGetParam[int64](sOther, "seed"))

	// Scope /cnn parameters:
	assert.Equal(t, 4, MustGetParam[int](sCNN, "num_layers"))
	assert.Equal(t, 3, MustGetParam[int](sCNN, "kernel_size"))

	// Sibling /rnn does NOT see /cnn parameters:
	_, found := sOther.GetParam("num_layers")
	assert.False(t, found)
	_, found = sOther.GetParam("kernel_size")
	assert.False(t, found)

	// Child /cnn/conv1 inherits /cnn and root parameters, and has filters:
	assert.Equal(t, 32, MustGetParam[int](sConv1, "filters"))
	assert.Equal(t, 4, MustGetParam[int](sConv1, "num_layers"))
	assert.Equal(t, 3, MustGetParam[int](sConv1, "kernel_size"))
	assert.Equal(t, 0.1, MustGetParam[float64](sConv1, "global_lr"))

	// Parent /cnn does NOT have filters:
	_, found = sCNN.GetParam("filters")
	assert.False(t, found)

	// Overwrite via SetParams:
	store.SetParams(map[string]any{
		"/cnn/num_layers": 8,
		"seed":            int64(100),
	})
	assert.Equal(t, 8, MustGetParam[int](sCNN, "num_layers"))
	assert.Equal(t, int64(100), MustGetParam[int64](root, "seed"))
}

func TestStore_SetParamsInScope(t *testing.T) {
	t.Run("ScopeFormats", func(t *testing.T) {
		// Test different scope string formats:
		testCases := []struct {
			name          string
			scope         string
			expectedScope string
		}{
			{name: "RootSlash", scope: "/", expectedScope: "/"},
			{name: "RootEmpty", scope: "", expectedScope: "/"},
			{name: "SubClean", scope: "/a/b", expectedScope: "/a/b"},
			{name: "SubTrailingSlash", scope: "/a/b/", expectedScope: "/a/b"},
			{name: "SubNoLeadingSlash", scope: "a/b", expectedScope: "/a/b"},
			{name: "SubNoLeadingAndTrailingSlash", scope: "a/b/", expectedScope: "/a/b"},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				store := NewStore()
				store.SetParamsInScope(tc.scope, map[string]any{
					"param1": "val1",
					"param2": 42,
				})

				scopeObj := store.Scope(tc.expectedScope)
				assert.Equal(t, "val1", MustGetParam[string](scopeObj, "param1"))
				assert.Equal(t, 42, MustGetParam[int](scopeObj, "param2"))

				// Check through Store.GetParam:
				fullPath1 := tc.expectedScope
				if fullPath1 == "/" {
					fullPath1 = "/param1"
				} else {
					fullPath1 += "/param1"
				}
				v, found := store.GetParam(fullPath1)
				assert.True(t, found)
				assert.Equal(t, "val1", v)
			})
		}
	})

	t.Run("ScopeIsolationAndInheritance", func(t *testing.T) {
		store := NewStore()
		store.SetParamsInScope("/", map[string]any{
			"root_cfg": "root",
		})
		store.SetParamsInScope("module/encoder", map[string]any{
			"heads": 8,
			"dim":   512,
		})

		sRoot := store.RootScope()
		sMod := store.Scope("/module")
		sEnc := store.Scope("/module/encoder")
		sLayer := store.Scope("/module/encoder/layer0")
		sDec := store.Scope("/module/decoder")

		// Scope /module/encoder should see its own params and root params:
		assert.Equal(t, 8, MustGetParam[int](sEnc, "heads"))
		assert.Equal(t, 512, MustGetParam[int](sEnc, "dim"))
		assert.Equal(t, "root", MustGetParam[string](sEnc, "root_cfg"))

		// Child /module/encoder/layer0 inherits:
		assert.Equal(t, 8, MustGetParam[int](sLayer, "heads"))
		assert.Equal(t, 512, MustGetParam[int](sLayer, "dim"))
		assert.Equal(t, "root", MustGetParam[string](sLayer, "root_cfg"))

		// Parent /module does NOT see encoder params:
		_, found := sMod.GetParam("heads")
		assert.False(t, found)
		assert.Equal(t, "root", MustGetParam[string](sMod, "root_cfg"))

		// Sibling /module/decoder does NOT see encoder params:
		_, found = sDec.GetParam("heads")
		assert.False(t, found)
		assert.Equal(t, "root", MustGetParam[string](sDec, "root_cfg"))

		// Root does NOT see encoder params:
		_, found = sRoot.GetParam("heads")
		assert.False(t, found)
	})
}

func TestScope_SetParam(t *testing.T) {
	store := NewStore()
	root := store.RootScope()
	root.SetParam("root_val", "hello")

	sA := root.In("a")
	sA.SetParam("a_val", 123)

	sB := sA.In("b")
	sB.SetParam("b_val", 456.0)

	sSibling := root.In("sibling")

	// Root scope checks:
	assert.Equal(t, "hello", MustGetParam[string](root, "root_val"))
	_, found := root.GetParam("a_val")
	assert.False(t, found)

	// Scope A checks:
	assert.Equal(t, "hello", MustGetParam[string](sA, "root_val"))
	assert.Equal(t, 123, MustGetParam[int](sA, "a_val"))
	_, found = sA.GetParam("b_val")
	assert.False(t, found)

	// Scope B checks:
	assert.Equal(t, "hello", MustGetParam[string](sB, "root_val"))
	assert.Equal(t, 123, MustGetParam[int](sB, "a_val"))
	assert.Equal(t, 456.0, MustGetParam[float64](sB, "b_val"))

	// Sibling checks:
	assert.Equal(t, "hello", MustGetParam[string](sSibling, "root_val"))
	_, found = sSibling.GetParam("a_val")
	assert.False(t, found)

	// Typed helper functions:
	assert.Equal(t, "hello", MustGetRootParam[string](store, "root_val"))
	assert.Equal(t, "hello", GetRootParamOr(store, "root_val", "default"))
	assert.Equal(t, "default", GetRootParamOr(store, "missing_param", "default"))
	assert.Equal(t, 123, GetParamOr(sA, "a_val", 0))
	assert.Equal(t, 999, GetParamOr(sA, "missing_param", 999))
}

func TestScope_SetParams(t *testing.T) {
	store := NewStore()
	root := store.RootScope()

	sA := root.In("layer")
	sA.SetParams(map[string]any{
		"activation": "gelu",
		"dropout":    0.1,
	})

	sChild := sA.In("dense")
	sOther := root.In("other")

	// Check sA:
	assert.Equal(t, "gelu", MustGetParam[string](sA, "activation"))
	assert.Equal(t, 0.1, MustGetParam[float64](sA, "dropout"))

	// Child inherits:
	assert.Equal(t, "gelu", MustGetParam[string](sChild, "activation"))
	assert.Equal(t, 0.1, MustGetParam[float64](sChild, "dropout"))

	// Parent and sibling do not have these params:
	_, found := root.GetParam("activation")
	assert.False(t, found)
	_, found = sOther.GetParam("activation")
	assert.False(t, found)
}

func TestSetGraphParam(t *testing.T) {
	backend := testutil.BuildTestBackend()
	g := graph.NewGraph(backend, "test_graph")

	SetGraphParam(g, "/root_gp", 100)
	SetGraphParam(g, "/enc/gp_enc", "enc_value")
	SetGraphParam(g, "/enc/attn/heads", 12)

	// Direct retrieval with GetGraphParam:
	v, found := GetGraphParam(g, "/root_gp")
	assert.True(t, found)
	assert.Equal(t, 100, v)

	v, found = GetGraphParam(g, "/enc/gp_enc")
	assert.True(t, found)
	assert.Equal(t, "enc_value", v)

	v, found = GetGraphParam(g, "/enc/attn/heads")
	assert.True(t, found)
	assert.Equal(t, 12, v)

	// Retrieval via Scope.GetGraphParam:
	store := NewStore()
	root := store.RootScope()
	sEnc := root.In("enc")
	sAttn := sEnc.In("attn")
	sDec := root.In("dec")

	// Root graph param is inherited by all scopes:
	v, found = root.GetGraphParam(g, "root_gp")
	assert.True(t, found)
	assert.Equal(t, 100, v)

	v, found = sEnc.GetGraphParam(g, "root_gp")
	assert.True(t, found)
	assert.Equal(t, 100, v)

	v, found = sAttn.GetGraphParam(g, "root_gp")
	assert.True(t, found)
	assert.Equal(t, 100, v)

	// Scope /enc:
	v, found = sEnc.GetGraphParam(g, "gp_enc")
	assert.True(t, found)
	assert.Equal(t, "enc_value", v)

	// Child /enc/attn inherits /enc param:
	v, found = sAttn.GetGraphParam(g, "gp_enc")
	assert.True(t, found)
	assert.Equal(t, "enc_value", v)

	// Scope /enc/attn has heads:
	v, found = sAttn.GetGraphParam(g, "heads")
	assert.True(t, found)
	assert.Equal(t, 12, v)

	// Parent /enc does NOT see heads:
	_, found = sEnc.GetGraphParam(g, "heads")
	assert.False(t, found)

	// Sibling /dec does NOT see /enc params:
	_, found = sDec.GetGraphParam(g, "gp_enc")
	assert.False(t, found)
	_, found = sDec.GetGraphParam(g, "heads")
	assert.False(t, found)

	// Graph isolation:
	g2 := graph.NewGraph(backend, "test_graph_2")
	_, found = GetGraphParam(g2, "/root_gp")
	assert.False(t, found)
	_, found = GetGraphParam(g2, "/enc/gp_enc")
	assert.False(t, found)
}

func TestScope_SetGraphParam(t *testing.T) {
	backend := testutil.BuildTestBackend()
	g := graph.NewGraph(backend, "test_scope_graph")

	store := NewStore()
	root := store.RootScope()
	root.SetGraphParam(g, "global_flag", true)

	sModule := root.In("module")
	sModule.SetGraphParam(g, "step", 1)

	sBlock := sModule.In("block")
	sBlock.SetGraphParam(g, "step", 2) // Shadow step in sub-scope.

	sOther := root.In("other")

	// Global flag visible everywhere:
	assert.True(t, GetGraphParamOr(root, g, "global_flag", false))
	assert.True(t, GetGraphParamOr(sModule, g, "global_flag", false))
	assert.True(t, GetGraphParamOr(sBlock, g, "global_flag", false))
	assert.True(t, GetGraphParamOr(sOther, g, "global_flag", false))

	// Step in module:
	assert.Equal(t, 1, GetGraphParamOr(sModule, g, "step", 0))

	// Step shadowed in block:
	assert.Equal(t, 2, GetGraphParamOr(sBlock, g, "step", 0))

	// Root and other do not have step:
	assert.Equal(t, -1, GetGraphParamOr(root, g, "step", -1))
	assert.Equal(t, -1, GetGraphParamOr(sOther, g, "step", -1))
}

func TestTrainingParams(t *testing.T) {
	backend := testutil.BuildTestBackend()
	g1 := graph.NewGraph(backend, "train_graph")
	g2 := graph.NewGraph(backend, "eval_graph")

	store := NewStore()
	root := store.RootScope()
	sub := root.In("sub")

	// Initially neither graph is training:
	assert.False(t, store.IsTraining(g1))
	assert.False(t, root.IsTraining(g1))
	assert.False(t, sub.IsTraining(g1))
	assert.False(t, store.IsTraining(g2))

	// Set training via Store:
	store.SetTraining(g1, true)
	assert.True(t, store.IsTraining(g1))
	assert.True(t, root.IsTraining(g1))
	assert.True(t, sub.IsTraining(g1))

	// g2 remains false:
	assert.False(t, store.IsTraining(g2))
	assert.False(t, root.IsTraining(g2))

	// Set training via Scope:
	sub.SetTraining(g1, false)
	assert.False(t, store.IsTraining(g1))
	assert.False(t, root.IsTraining(g1))
	assert.False(t, sub.IsTraining(g1))

	// Set training for g2:
	sub.SetTraining(g2, true)
	assert.True(t, store.IsTraining(g2))
	assert.True(t, root.IsTraining(g2))
	assert.True(t, sub.IsTraining(g2))
	assert.False(t, store.IsTraining(g1))
}

func TestParamIterators(t *testing.T) {
	t.Run("StoreAndScopeIterParams", func(t *testing.T) {
		store := NewStore()
		store.SetParam("/p_root", "root")
		store.SetParam("/a/p_a", "a")
		store.SetParam("/a/b/p_ab", "ab")
		store.SetParam("/c/p_c", "c")

		// Store.IterParams:
		allParams := make(map[string]any)
		for k, v := range store.IterParams() {
			allParams[k] = v
		}
		require.Len(t, allParams, 4)
		assert.Equal(t, "root", allParams["/p_root"])
		assert.Equal(t, "a", allParams["/a/p_a"])
		assert.Equal(t, "ab", allParams["/a/b/p_ab"])
		assert.Equal(t, "c", allParams["/c/p_c"])

		// Scope /a should yield /a/p_a and /a/b/p_ab:
		sA := store.Scope("/a")
		scopeAParams := make(map[string]any)
		for k, v := range sA.IterParams() {
			scopeAParams[k] = v
		}
		assert.Len(t, scopeAParams, 2)
		assert.Equal(t, "a", scopeAParams["/a/p_a"])
		assert.Equal(t, "ab", scopeAParams["/a/b/p_ab"])

		// Scope /a/b should yield /a/b/p_ab:
		sAB := store.Scope("/a/b")
		scopeABParams := make(map[string]any)
		for k, v := range sAB.IterParams() {
			scopeABParams[k] = v
		}
		assert.Len(t, scopeABParams, 1)
		assert.Equal(t, "ab", scopeABParams["/a/b/p_ab"])

		// Scope /d has no params:
		sD := store.Scope("/d")
		scopeDParams := maps.Collect(sD.IterParams())
		assert.Empty(t, scopeDParams)
	})

	t.Run("StoreAndScopeIterGraphParams", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		g := graph.NewGraph(backend, "iter_graph")

		store := NewStore()
		SetGraphParam(g, "/gp_root", 1)
		SetGraphParam(g, "/x/gp_x", 2)
		SetGraphParam(g, "/x/y/gp_xy", 3)
		SetGraphParam(g, "/z/gp_z", 4)

		// Store.IterGraphParams:
		storeCount := 0
		for range store.IterGraphParams(g) {
			storeCount++
		}
		assert.Equal(t, 4, storeCount)

		// Scope /x should yield /x/gp_x and /x/y/gp_xy:
		sX := store.Scope("/x")
		xCount := 0
		for range sX.IterGraphParams(g) {
			xCount++
		}
		assert.Equal(t, 2, xCount)

		// Scope /x/y should yield 1:
		sXY := store.Scope("/x/y")
		xyCount := 0
		for range sXY.IterGraphParams(g) {
			xyCount++
		}
		assert.Equal(t, 1, xyCount)

		// Scope /other should yield 0:
		sOther := store.Scope("/other")
		otherCount := 0
		for range sOther.IterGraphParams(g) {
			otherCount++
		}
		assert.Equal(t, 0, otherCount)
	})
}
