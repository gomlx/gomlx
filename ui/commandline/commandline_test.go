// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package commandline

import (
	"fmt"
	"testing"

	"github.com/gomlx/gomlx/ml/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func createTestModelStore() *model.Store {
	store := model.NewStore()
	store.SetParam("x", 11.0)
	store.SetParam("y", 7)
	store.SetParam("z", false)
	store.SetParam("s", "foo")
	store.SetParam("list_int", []int{})
	store.SetParam("list_float", []float64{})
	store.SetParam("list_str", []string{})
	return store
}

func TestParseScopeSettings(t *testing.T) {
	store := createTestModelStore()
	modifiedParams, err := ParseSettings(store, "x=13;/a/z=true;/a/b/y=3;s=bar;list_int=1,3,7;list_float=0.1,1.2,3e3;list_str=a,b;")
	require.NoError(t, err)
	require.Equal(t, []string{"x", "/a/z", "/a/b/y", "s", "list_int", "list_float", "list_str"}, modifiedParams)

	fmt.Printf("- Parameters after parsing:\n%s\n", SprintSettings(store))

	x, found := store.GetParam("x")
	assert.True(t, found)
	assert.Equal(t, 13.0, x.(float64))

	rootScope := store.RootScope()
	y, found := store.GetParam("y")
	assert.True(t, found)
	assert.Equal(t, 7, y)
	y, _ = rootScope.At("a").GetParam("y")
	assert.Equal(t, 7, y)
	y, _ = rootScope.At("a").At("b").GetParam("y")
	assert.Equal(t, 3, y)

	z, found := rootScope.GetParam("z")
	assert.True(t, found)
	assert.False(t, z.(bool))
	z, _ = rootScope.At("a").GetParam("z")
	assert.True(t, z.(bool))

	s, found := rootScope.GetParam("s")
	assert.True(t, found)
	assert.Equal(t, "bar", s.(string))

	assert.Equal(t, []int{1, 3, 7}, model.GetParamOr(rootScope, "list_int", []int{}))
	assert.Equal(t, []float64{0.1, 1.2, 3e3}, model.GetParamOr(rootScope, "list_float", []float64{}))
	assert.Equal(t, []string{"a", "b"}, model.GetParamOr(rootScope, "list_str", []string{}))

	// Parameter "q" is unknown.
	_, err = ParseSettings(store, "q=3")
	require.Error(t, err)

	// Parameter "q" is still unknown in root.
	rootScope.In("c").SetParam("q", 13)
	_, err = ParseSettings(store, "q=3")
	require.Error(t, err)

	// Cannot set the wrong type of value.
	_, err = ParseSettings(store, "y=3.14")
	require.Error(t, err)

	// Cannot parse setting with scope not absolute.
	_, err = ParseSettings(store, "a/abc=3.14")
	require.Error(t, err)
}
