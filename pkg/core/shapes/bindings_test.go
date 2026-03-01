// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package shapes

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/stretchr/testify/require"
)

func TestAxisBindings_Key(t *testing.T) {
	b := AxisBindings{"batch": 32, "seq_len": 128}
	require.Equal(t, "batch=32,seq_len=128", b.Key())

	// Deterministic ordering.
	b2 := AxisBindings{"seq_len": 128, "batch": 32}
	require.Equal(t, b.Key(), b2.Key())

	// Empty.
	require.Equal(t, "", AxisBindings{}.Key())
}

func TestResolve(t *testing.T) {
	s := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
	bindings := AxisBindings{"batch": 32}
	resolved := s.Resolve(bindings)

	require.Equal(t, []int{32, 512}, resolved.Dimensions)
	require.Equal(t, []string{"batch", ""}, resolved.AxisNames)
	require.False(t, resolved.HasDynamicDims())
}

func TestResolve_MultipleAxes(t *testing.T) {
	s := MakeDynamic(dtypes.Float32, []int{-1, -1, 768}, []string{"batch", "seq_len", ""})
	bindings := AxisBindings{"batch": 8, "seq_len": 256}
	resolved := s.Resolve(bindings)

	require.Equal(t, []int{8, 256, 768}, resolved.Dimensions)
}

func TestResolve_StaticShape(t *testing.T) {
	s := Make(dtypes.Float32, 32, 512)
	// Resolve on static shape returns same shape (no-op).
	resolved := s.Resolve(AxisBindings{"batch": 64})
	require.True(t, s.Equal(resolved))
}

func TestResolve_MissingBinding(t *testing.T) {
	s := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
	require.Panics(t, func() { s.Resolve(AxisBindings{}) })
}

func TestResolve_NonPositiveBinding(t *testing.T) {
	s := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
	require.Panics(t, func() { s.Resolve(AxisBindings{"batch": 0}) })
	require.Panics(t, func() { s.Resolve(AxisBindings{"batch": -5}) })
}

func TestExtractBindings(t *testing.T) {
	template := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
	concrete := Make(dtypes.Float32, 32, 512)

	bindings, err := ExtractBindings(template, concrete)
	require.NoError(t, err)
	require.Equal(t, AxisBindings{"batch": 32}, bindings)
}

func TestExtractBindings_MultipleAxes(t *testing.T) {
	template := MakeDynamic(dtypes.Float32, []int{-1, -1, 768}, []string{"batch", "seq_len", ""})
	concrete := Make(dtypes.Float32, 8, 128, 768)

	bindings, err := ExtractBindings(template, concrete)
	require.NoError(t, err)
	require.Equal(t, AxisBindings{"batch": 8, "seq_len": 128}, bindings)
}

func TestExtractBindings_ConsistencyCheck(t *testing.T) {
	// Same axis name appears multiple times with different values.
	template := MakeDynamic(dtypes.Float32, []int{-1, -1}, []string{"n", "n"})
	concrete := Make(dtypes.Float32, 5, 5)

	// Same value → OK.
	bindings, err := ExtractBindings(template, concrete)
	require.NoError(t, err)
	require.Equal(t, AxisBindings{"n": 5}, bindings)

	// Different values → error.
	concrete2 := Make(dtypes.Float32, 5, 10)
	_, err = ExtractBindings(template, concrete2)
	require.Error(t, err)
	require.Contains(t, err.Error(), "conflicting")
}

func TestExtractBindings_RankMismatch(t *testing.T) {
	template := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
	concrete := Make(dtypes.Float32, 32, 512, 3)

	_, err := ExtractBindings(template, concrete)
	require.Error(t, err)
	require.Contains(t, err.Error(), "rank")
}

func TestExtractBindings_StaticDimMismatch(t *testing.T) {
	template := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
	concrete := Make(dtypes.Float32, 32, 256)

	_, err := ExtractBindings(template, concrete)
	require.Error(t, err)
	require.Contains(t, err.Error(), "mismatch")
}

func TestMergeBindings(t *testing.T) {
	b1 := AxisBindings{"batch": 32}
	b2 := AxisBindings{"seq_len": 128}

	merged, err := MergeBindings(b1, b2)
	require.NoError(t, err)
	require.Equal(t, AxisBindings{"batch": 32, "seq_len": 128}, merged)
}

func TestMergeBindings_SameValue(t *testing.T) {
	b1 := AxisBindings{"batch": 32}
	b2 := AxisBindings{"batch": 32}

	merged, err := MergeBindings(b1, b2)
	require.NoError(t, err)
	require.Equal(t, AxisBindings{"batch": 32}, merged)
}

func TestMergeBindings_Conflict(t *testing.T) {
	b1 := AxisBindings{"batch": 32}
	b2 := AxisBindings{"batch": 64}

	_, err := MergeBindings(b1, b2)
	require.Error(t, err)
	require.Contains(t, err.Error(), "conflicting")
}

func TestUnifyAxisName(t *testing.T) {
	// Both empty.
	name, err := UnifyAxisName("", "")
	require.NoError(t, err)
	require.Equal(t, "", name)

	// One named.
	name, err = UnifyAxisName("batch", "")
	require.NoError(t, err)
	require.Equal(t, "batch", name)

	name, err = UnifyAxisName("", "batch")
	require.NoError(t, err)
	require.Equal(t, "batch", name)

	// Same name.
	name, err = UnifyAxisName("batch", "batch")
	require.NoError(t, err)
	require.Equal(t, "batch", name)

	// Different names.
	_, err = UnifyAxisName("batch", "time")
	require.Error(t, err)
	require.Contains(t, err.Error(), "incompatible")
}

func TestUnifyAxisNames(t *testing.T) {
	s1 := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
	s2 := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})

	names, err := UnifyAxisNames(s1, s2)
	require.NoError(t, err)
	require.Equal(t, []string{"batch", ""}, names)

	// One unnamed adopts.
	s3 := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
	s4 := Make(dtypes.Float32, 32, 512) // no AxisNames
	names, err = UnifyAxisNames(s3, s4)
	require.NoError(t, err)
	require.Equal(t, []string{"batch", ""}, names)

	// Both nil.
	s5 := Make(dtypes.Float32, 32, 512)
	s6 := Make(dtypes.Float32, 32, 512)
	names, err = UnifyAxisNames(s5, s6)
	require.NoError(t, err)
	require.Nil(t, names)

	// Conflict.
	s7 := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
	s8 := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"time", ""})
	_, err = UnifyAxisNames(s7, s8)
	require.Error(t, err)
}

func TestRoundTrip_ExtractAndResolve(t *testing.T) {
	// Extract bindings from concrete shape, then resolve template with those bindings.
	template := MakeDynamic(dtypes.Float32, []int{-1, -1, 768}, []string{"batch", "seq_len", ""})
	concrete := Make(dtypes.Float32, 16, 64, 768)

	bindings, err := ExtractBindings(template, concrete)
	require.NoError(t, err)

	resolved := template.Resolve(bindings)
	require.Equal(t, concrete.Dimensions, resolved.Dimensions)
	require.False(t, resolved.HasDynamicDims())
}
