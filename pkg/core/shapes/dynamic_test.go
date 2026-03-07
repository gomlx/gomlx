// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package shapes

import (
	"bytes"
	"encoding/gob"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/stretchr/testify/require"
)

func TestMakeDynamic(t *testing.T) {
	s := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
	require.Equal(t, dtypes.Float32, s.DType)
	require.Equal(t, []int{-1, 512}, s.Dimensions)
	require.Equal(t, []string{"batch", ""}, s.AxisNames)
	require.Equal(t, 2, s.Rank())
}

func TestMakeDynamic_MultipleNamedAxes(t *testing.T) {
	s := MakeDynamic(dtypes.Float32, []int{-1, -1, 768}, []string{"batch", "seq_len", ""})
	require.Equal(t, []int{-1, -1, 768}, s.Dimensions)
	require.Equal(t, []string{"batch", "seq_len", ""}, s.AxisNames)
}

func TestMakeDynamic_Panics(t *testing.T) {
	// Mismatched lengths.
	require.Panics(t, func() {
		MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch"})
	})

	// Dynamic dim without name.
	require.Panics(t, func() {
		MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"", ""})
	})

	// Invalid negative dimension (not -1).
	require.Panics(t, func() {
		MakeDynamic(dtypes.Float32, []int{-2, 512}, []string{"batch", ""})
	})
}

func TestHasDynamicDims(t *testing.T) {
	static := Make(dtypes.Float32, 32, 512)
	require.False(t, static.HasDynamicDims())

	dynamic := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
	require.True(t, dynamic.HasDynamicDims())
}

func TestIsDynamicDim(t *testing.T) {
	s := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
	require.True(t, s.IsDynamicDim(0))
	require.False(t, s.IsDynamicDim(1))

	// Negative indexing.
	require.False(t, s.IsDynamicDim(-1))
	require.True(t, s.IsDynamicDim(-2))

	// Out of bounds.
	require.Panics(t, func() { s.IsDynamicDim(2) })
	require.Panics(t, func() { s.IsDynamicDim(-3) })
}

func TestAxisName(t *testing.T) {
	s := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
	require.Equal(t, "batch", s.AxisName(0))
	require.Equal(t, "", s.AxisName(1))
	require.Equal(t, "", s.AxisName(-1))
	require.Equal(t, "batch", s.AxisName(-2))

	// No axis names.
	static := Make(dtypes.Float32, 32, 512)
	require.Equal(t, "", static.AxisName(0))
	require.Equal(t, "", static.AxisName(1))
}

func TestWithAxisNames(t *testing.T) {
	s := Make(dtypes.Float32, 32, 512)
	named := s.WithAxisNames("batch", "features")
	require.Equal(t, []string{"batch", "features"}, named.AxisNames)
	// Original unchanged.
	require.Nil(t, s.AxisNames)

	// Wrong number of names.
	require.Panics(t, func() { s.WithAxisNames("batch") })
}

func TestShape_Equal_WithAxisNames(t *testing.T) {
	s1 := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
	s2 := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
	s3 := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"time", ""})

	// Same axis names → equal.
	require.True(t, s1.Equal(s2))

	// Different axis names → not equal.
	require.False(t, s1.Equal(s3))

	// nil AxisNames equals all-empty AxisNames.
	static := Make(dtypes.Float32, 32, 512)
	staticNamed := Make(dtypes.Float32, 32, 512).WithAxisNames("", "")
	require.True(t, static.Equal(staticNamed))
	require.True(t, staticNamed.Equal(static))
}

func TestShape_EqualDimensions_IgnoresAxisNames(t *testing.T) {
	s1 := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
	s2 := MakeDynamic(dtypes.Float64, []int{-1, 512}, []string{"time", ""})

	// EqualDimensions ignores both DType and AxisNames.
	require.True(t, s1.EqualDimensions(s2))
}

func TestShape_Clone_WithAxisNames(t *testing.T) {
	s := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
	s2 := s.Clone()

	// Equal.
	require.True(t, s.Equal(s2))

	// Independent slices.
	s2.AxisNames[0] = "modified"
	require.Equal(t, "batch", s.AxisNames[0])
}

func TestShape_String_WithAxisNames(t *testing.T) {
	s := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
	str := s.String()
	require.Equal(t, "(Float32)[batch=-1 512]", str)

	s2 := MakeDynamic(dtypes.Float32, []int{-1, -1, 768}, []string{"batch", "seq_len", ""})
	require.Equal(t, "(Float32)[batch=-1 seq_len=-1 768]", s2.String())

	// Named static axes.
	s3 := Make(dtypes.Float32, 32, 512).WithAxisNames("batch", "features")
	require.Equal(t, "(Float32)[batch=32 features=512]", s3.String())
}

func TestShape_Size_PanicsOnDynamic(t *testing.T) {
	s := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
	require.Panics(t, func() { _ = s.Size() })

	// Static shapes still work.
	static := Make(dtypes.Float32, 4, 3, 2)
	require.Equal(t, 24, static.Size())
}

func TestShape_Strides_PanicsOnDynamic(t *testing.T) {
	s := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
	require.Panics(t, func() { _ = s.Strides() })
}

func TestShape_Iter_PanicsOnDynamic(t *testing.T) {
	s := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
	require.Panics(t, func() {
		for range s.Iter() {
		}
	})
}

func TestGobSerialize_WithAxisNames(t *testing.T) {
	original := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})

	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	err := original.GobSerialize(encoder)
	require.NoError(t, err)

	decoder := gob.NewDecoder(&buf)
	decoded, err := GobDeserialize(decoder)
	require.NoError(t, err)
	require.True(t, original.Equal(decoded))
	require.Equal(t, original.AxisNames, decoded.AxisNames)
}

func TestGobSerialize_WithoutAxisNames(t *testing.T) {
	// Static shapes (no axis names) round-trip correctly.
	original := Make(dtypes.Float32, 4, 3, 2)

	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	err := original.GobSerialize(encoder)
	require.NoError(t, err)

	decoder := gob.NewDecoder(&buf)
	decoded, err := GobDeserialize(decoder)
	require.NoError(t, err)
	require.True(t, original.Equal(decoded))
	require.Nil(t, decoded.AxisNames)
}

func TestConcatenateDimensions_WithAxisNames(t *testing.T) {
	s1 := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
	s2 := Make(dtypes.Float32, 3, 4)

	result := ConcatenateDimensions(s1, s2)
	require.Equal(t, []int{-1, 512, 3, 4}, result.Dimensions)
	require.Equal(t, []string{"batch", "", "", ""}, result.AxisNames)
}

func TestCheckDims_WithDynamic(t *testing.T) {
	s := MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})

	// Checking static dim works normally.
	require.NoError(t, s.CheckDims(-1, 512)) // -1 in check args means unchecked

	// Checking dynamic dim against a concrete value fails (as expected, the dim is unknown).
	require.Error(t, s.CheckDims(32, 512))

	// Checking dynamic dim with unchecked works.
	require.NoError(t, s.CheckDims(-1, -1))
}

func TestAxisNamesEqual(t *testing.T) {
	require.True(t, axisNamesEqual(nil, nil))
	require.True(t, axisNamesEqual(nil, []string{"", ""}))
	require.True(t, axisNamesEqual([]string{"", ""}, nil))
	require.True(t, axisNamesEqual([]string{"a", "b"}, []string{"a", "b"}))
	require.False(t, axisNamesEqual([]string{"a", "b"}, []string{"a", "c"}))
	require.False(t, axisNamesEqual(nil, []string{"a", ""}))
	require.False(t, axisNamesEqual([]string{"a"}, []string{"a", "b"}))
}
