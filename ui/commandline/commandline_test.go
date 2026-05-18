// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package commandline

import (
	"testing"

	"github.com/gomlx/gomlx/ml/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func createTestContext() *model.Scope {
	scope := model.NewStore()
	scope.SetParam("x", 11.0)
	scope.SetParam("y", 7)
	scope.SetParam("z", false)
	scope.SetParam("s", "foo")
	scope.SetParam("list_int", []int{})
	scope.SetParam("list_float", []float64{})
	scope.SetParam("list_str", []string{})
	return scope
}

func TestParseContextSettings(t *testing.T) {
	scope := createTestContext()

	paramsSet, err := ParseSettings(scope, "x=13;/a/z=true;/a/b/y=3;s=bar;list_int=1,3,7;list_float=0.1,1.2,3e3;list_str=a,b;")
	require.NoError(t, err)
	require.Equal(t, []string{"x", "/a/z", "/a/b/y", "s", "list_int", "list_float", "list_str"}, paramsSet)
	x, found := scope.GetParam("x")
	assert.True(t, found)
	assert.Equal(t, 13.0, x.(float64))

	y, found := scope.GetParam("y")
	assert.True(t, found)
	assert.Equal(t, 7, y)
	y, _ = scope.In("a").GetParam("y")
	assert.Equal(t, 7, y)
	y, _ = scope.In("a").In("b").GetParam("y")
	assert.Equal(t, 3, y)

	z, found := scope.GetParam("z")
	assert.True(t, found)
	assert.False(t, z.(bool))
	z, _ = scope.In("a").GetParam("z")
	assert.True(t, z.(bool))

	s, found := scope.GetParam("s")
	assert.True(t, found)
	assert.Equal(t, "bar", s.(string))

	assert.Equal(t, []int{1, 3, 7}, model.GetParamOr(scope, "list_int", []int{}))
	assert.Equal(t, []float64{0.1, 1.2, 3e3}, model.GetParamOr(scope, "list_float", []float64{}))
	assert.Equal(t, []string{"a", "b"}, model.GetParamOr(scope, "list_str", []string{}))

	// Parameter "q" is unknown.
	_, err = ParseSettings(scope, "q=3")
	require.Error(t, err)

	// Parameter "q" is still unknown in root.
	scope.In("c").SetParam("q", 13)
	_, err = ParseSettings(scope, "q=3")
	require.Error(t, err)

	// Cannot set the wrong type of value.
	_, err = ParseSettings(scope, "y=3.14")
	require.Error(t, err)

	// Cannot parse setting with scope not absolute.
	_, err = ParseSettings(scope, "a/abc=3.14")
	require.Error(t, err)
}
