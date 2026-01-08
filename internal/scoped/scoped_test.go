// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package scoped_test

import (
	"testing"

	"github.com/gomlx/gomlx/internal/scoped"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2"
)

func init() {
	klog.InitFlags(nil)
}

func TestScopedParams(t *testing.T) {
	p := scoped.New("/")

	//	Scope: "/": { "x":10, "y": 20, "z": 40 }
	//	Scope: "/a": { "y": 30 }
	//	Scope: "/a/b": { "x": 100 }
	p.Set("/", "x", 10)
	p.Set("/", "y", 20)
	p.Set("/", "z", 40)
	p.Set("/a", "y", 30)
	p.Set("/a/b", "x", 100)

	// scopedParams.Get("/a/b", "x") -> 100
	value, found := p.Get("/a/b", "x")
	assert.True(t, found, "/a/b:x should be set")
	assert.Equal(t, 100, value.(int), "scopedParams.Get(\"/a/b\", \"x\") -> 100")

	//	scopedParams.Get("/a/b", "y") -> 30
	value, found = p.Get("/a/b", "y")
	assert.True(t, found, "/a:y should be set and found")
	assert.Equal(t, 30, value.(int), "scopedParams.Get(\"/a/b\", \"y\") -> 30")

	//	scopedParams.Get("/a/b", "z") -> 40
	value, found = p.Get("/a/b", "z")
	assert.True(t, found, "/:z should be set and found")
	assert.Equal(t, 40, value.(int), "scopedParams.Get(\"/a/b\", \"z\") -> 40")

	//	scopedParams.Get("/a/b", "w") -> Not found.
	value, found = p.Get("/a/b", "w")
	assert.False(t, found, "/a/b:w should not be set and not found")

	//	scopedParams.Get("/d/e/f", "z") -> 40
	value, found = p.Get("/d/e/f", "z")
	assert.True(t, found, "/:z should be set and found")
	assert.Equal(t, 40, value.(int), "scopedParams.Get(\"/d/e/f\", \"z\") -> 40")

	want := []struct {
		scope string
		key   string
		value int
	}{
		{"/", "x", 10},
		{"/", "y", 20},
		{"/", "z", 40},
		{"/a", "y", 30},
		{"/a/b", "x", 100},
	}
	pos := 0
	p.Enumerate(func(scope, key string, valueAny any) {
		value := valueAny.(int)
		require.Lessf(t, pos, len(want), "Enumerate returned more elements (%d at least) than listed in `want` array: "+
			"scope=%q, key=%q, value=%d", pos, scope, key, value)
		require.Equalf(t, want[pos].scope, scope,
			"Enumerating element %d: wanted %+v, got: {scope=%q, key=%q, value=%d}", pos, want[pos], scope, key, value)
		require.Equalf(t, want[pos].key, key,
			"Enumerating element %d: wanted %+v, got: {scope=%q, key=%q, value=%d}", pos, want[pos], scope, key, value)
		require.Equalf(t, want[pos].value, value,
			"Enumerating element %d: wanted %+v, got: {scope=%q, key=%q, value=%d}", pos, want[pos], scope, key, value)
		pos += 1
	})
	require.Equal(t, len(want), pos, "Enumerate returned fewer elements (%d) than listed in `want` array (%d elements)", pos, len(want))
}
