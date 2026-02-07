// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package shapes

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/stretchr/testify/require"
)

func TestAxisBindingsKey(t *testing.T) {
	tests := []struct {
		name     string
		bindings AxisBindings
		want     string
	}{
		{
			name:     "empty",
			bindings: AxisBindings{},
			want:     "",
		},
		{
			name:     "nil",
			bindings: nil,
			want:     "",
		},
		{
			name:     "single",
			bindings: AxisBindings{"batch": 32},
			want:     "batch=32",
		},
		{
			name:     "multiple_sorted",
			bindings: AxisBindings{"batch": 32, "seq": 128},
			want:     "batch=32,seq=128",
		},
		{
			name:     "insertion_order_ignored",
			bindings: AxisBindings{"seq": 128, "batch": 32, "hidden": 512},
			want:     "batch=32,hidden=512,seq=128",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.bindings.Key()
			require.Equal(t, tt.want, got)
		})
	}
}

func TestAxisBindingsClone(t *testing.T) {
	original := AxisBindings{"batch": 32, "seq": 128}
	clone := original.Clone()

	require.Equal(t, original, clone)

	// Modifying clone should not affect original
	clone["batch"] = 64
	require.Equal(t, 32, original["batch"])

	// Nil clone
	var nilBindings AxisBindings
	require.Nil(t, nilBindings.Clone())
}

func TestAxisBindingsMerge(t *testing.T) {
	t.Run("non_overlapping", func(t *testing.T) {
		ab := AxisBindings{"batch": 32}
		err := ab.Merge(AxisBindings{"seq": 128})
		require.NoError(t, err)
		require.Equal(t, 32, ab["batch"])
		require.Equal(t, 128, ab["seq"])
	})

	t.Run("same_value", func(t *testing.T) {
		ab := AxisBindings{"batch": 32}
		err := ab.Merge(AxisBindings{"batch": 32})
		require.NoError(t, err)
		require.Equal(t, 32, ab["batch"])
	})

	t.Run("conflict", func(t *testing.T) {
		ab := AxisBindings{"batch": 32}
		err := ab.Merge(AxisBindings{"batch": 64})
		require.Error(t, err)
		require.Contains(t, err.Error(), "conflicting")
	})
}

func TestMakeDynamic(t *testing.T) {
	t.Run("named_dynamic", func(t *testing.T) {
		s := MakeDynamic(dtypes.Float32, "batch", 512)
		require.Equal(t, DimDynamic, s.Dimensions[0])
		require.Equal(t, 512, s.Dimensions[1])
		require.Equal(t, "batch", s.AxisNames[0])
		require.Equal(t, "", s.AxisNames[1])
		require.True(t, s.IsDynamic())
		require.True(t, s.HasNamedAxes())
		require.False(t, s.IsFullyConcrete())
	})

	t.Run("all_concrete", func(t *testing.T) {
		s := MakeDynamic(dtypes.Float32, 32, 512)
		require.Equal(t, 32, s.Dimensions[0])
		require.Equal(t, 512, s.Dimensions[1])
		require.False(t, s.IsDynamic())
		require.False(t, s.HasNamedAxes())
		require.True(t, s.IsFullyConcrete())
	})

	t.Run("multiple_named", func(t *testing.T) {
		s := MakeDynamic(dtypes.Float32, "batch", "seq", 512)
		require.Equal(t, DimDynamic, s.Dimensions[0])
		require.Equal(t, DimDynamic, s.Dimensions[1])
		require.Equal(t, 512, s.Dimensions[2])
		require.Equal(t, "batch", s.AxisNames[0])
		require.Equal(t, "seq", s.AxisNames[1])
		require.Equal(t, "", s.AxisNames[2])
	})

	t.Run("scalar", func(t *testing.T) {
		s := MakeDynamic(dtypes.Float32)
		require.True(t, s.IsScalar())
		require.False(t, s.IsDynamic())
	})
}

func TestShapeResolve(t *testing.T) {
	t.Run("resolve_named_axes", func(t *testing.T) {
		pattern := MakeDynamic(dtypes.Float32, "batch", "seq", 512)
		bindings := AxisBindings{"batch": 32, "seq": 128}

		resolved := pattern.Resolve(bindings)
		require.Equal(t, 32, resolved.Dimensions[0])
		require.Equal(t, 128, resolved.Dimensions[1])
		require.Equal(t, 512, resolved.Dimensions[2])
		require.True(t, resolved.IsFullyConcrete())

		// Original should be unchanged
		require.Equal(t, DimDynamic, pattern.Dimensions[0])
	})

	t.Run("partial_resolve", func(t *testing.T) {
		pattern := MakeDynamic(dtypes.Float32, "batch", "seq", 512)
		bindings := AxisBindings{"batch": 32} // seq not bound

		resolved := pattern.Resolve(bindings)
		require.Equal(t, 32, resolved.Dimensions[0])
		require.Equal(t, DimDynamic, resolved.Dimensions[1]) // Still dynamic
		require.Equal(t, 512, resolved.Dimensions[2])
		require.False(t, resolved.IsFullyConcrete())
	})

	t.Run("static_shape_unchanged", func(t *testing.T) {
		static := Make(dtypes.Float32, 32, 512)
		bindings := AxisBindings{"batch": 64}

		resolved := static.Resolve(bindings)
		require.Equal(t, static, resolved)
	})

	t.Run("nil_bindings", func(t *testing.T) {
		pattern := MakeDynamic(dtypes.Float32, "batch", 512)
		resolved := pattern.Resolve(nil)
		require.Equal(t, pattern, resolved)
	})
}

func TestExtractBindings(t *testing.T) {
	t.Run("extract_single_axis", func(t *testing.T) {
		pattern := MakeDynamic(dtypes.Float32, "batch", 512)
		concrete := Make(dtypes.Float32, 32, 512)

		bindings, err := ExtractBindings(pattern, concrete)
		require.NoError(t, err)
		require.Equal(t, 32, bindings["batch"])
		require.Len(t, bindings, 1)
	})

	t.Run("extract_multiple_axes", func(t *testing.T) {
		pattern := MakeDynamic(dtypes.Float32, "batch", "seq", 512)
		concrete := Make(dtypes.Float32, 32, 128, 512)

		bindings, err := ExtractBindings(pattern, concrete)
		require.NoError(t, err)
		require.Equal(t, 32, bindings["batch"])
		require.Equal(t, 128, bindings["seq"])
	})

	t.Run("same_axis_same_value", func(t *testing.T) {
		// Pattern where same axis name appears twice (both must have same value)
		pattern := Shape{
			DType:      dtypes.Float32,
			Dimensions: []int{DimDynamic, DimDynamic},
			AxisNames:  []string{"batch", "batch"},
		}
		concrete := Make(dtypes.Float32, 32, 32)

		bindings, err := ExtractBindings(pattern, concrete)
		require.NoError(t, err)
		require.Equal(t, 32, bindings["batch"])
	})

	t.Run("same_axis_different_value_error", func(t *testing.T) {
		pattern := Shape{
			DType:      dtypes.Float32,
			Dimensions: []int{DimDynamic, DimDynamic},
			AxisNames:  []string{"batch", "batch"},
		}
		concrete := Make(dtypes.Float32, 32, 64)

		_, err := ExtractBindings(pattern, concrete)
		require.Error(t, err)
		require.Contains(t, err.Error(), "conflicting")
	})

	t.Run("static_dimension_mismatch", func(t *testing.T) {
		pattern := MakeDynamic(dtypes.Float32, "batch", 512)
		concrete := Make(dtypes.Float32, 32, 256)

		_, err := ExtractBindings(pattern, concrete)
		require.Error(t, err)
		require.Contains(t, err.Error(), "mismatch")
	})

	t.Run("rank_mismatch", func(t *testing.T) {
		pattern := MakeDynamic(dtypes.Float32, "batch", 512)
		concrete := Make(dtypes.Float32, 32, 512, 768)

		_, err := ExtractBindings(pattern, concrete)
		require.Error(t, err)
		require.Contains(t, err.Error(), "rank")
	})

	t.Run("dtype_mismatch", func(t *testing.T) {
		pattern := MakeDynamic(dtypes.Float32, "batch", 512)
		concrete := Make(dtypes.Float64, 32, 512)

		_, err := ExtractBindings(pattern, concrete)
		require.Error(t, err)
		require.Contains(t, err.Error(), "dtype")
	})
}

func TestUnifyAxisName(t *testing.T) {
	tests := []struct {
		name    string
		a, b    string
		want    string
		wantErr bool
	}{
		{
			name: "both_empty",
			a:    "", b: "",
			want: "",
		},
		{
			name: "a_empty",
			a:    "", b: "batch",
			want: "batch",
		},
		{
			name: "b_empty",
			a:    "batch", b: "",
			want: "batch",
		},
		{
			name: "same_name",
			a:    "batch", b: "batch",
			want: "batch",
		},
		{
			name:    "different_names",
			a:       "batch", b: "time",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := UnifyAxisName(tt.a, tt.b)
			if tt.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				require.Equal(t, tt.want, got)
			}
		})
	}
}

func TestShapeString(t *testing.T) {
	tests := []struct {
		name  string
		shape Shape
		want  string
	}{
		{
			name:  "static",
			shape: Make(dtypes.Float32, 32, 512),
			want:  "(Float32)[32 512]",
		},
		{
			name:  "dynamic_named",
			shape: MakeDynamic(dtypes.Float32, "batch", 512),
			want:  "(Float32)[batch, 512]",
		},
		{
			name:  "multiple_named",
			shape: MakeDynamic(dtypes.Float32, "batch", "seq", 512),
			want:  "(Float32)[batch, seq, 512]",
		},
		{
			name: "resolved_named",
			shape: func() Shape {
				s := MakeDynamic(dtypes.Float32, "batch", 512)
				return s.Resolve(AxisBindings{"batch": 32})
			}(),
			want: "(Float32)[batch=32, 512]",
		},
		{
			name:  "scalar",
			shape: Make(dtypes.Float32),
			want:  "(Float32)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, tt.shape.String())
		})
	}
}

func TestShapeEqual(t *testing.T) {
	t.Run("static_equal", func(t *testing.T) {
		s1 := Make(dtypes.Float32, 32, 512)
		s2 := Make(dtypes.Float32, 32, 512)
		require.True(t, s1.Equal(s2))
	})

	t.Run("dynamic_equal", func(t *testing.T) {
		s1 := MakeDynamic(dtypes.Float32, "batch", 512)
		s2 := MakeDynamic(dtypes.Float32, "batch", 512)
		require.True(t, s1.Equal(s2))
	})

	t.Run("different_axis_names", func(t *testing.T) {
		s1 := MakeDynamic(dtypes.Float32, "batch", 512)
		s2 := MakeDynamic(dtypes.Float32, "time", 512)
		require.False(t, s1.Equal(s2))
	})

	t.Run("static_vs_dynamic", func(t *testing.T) {
		s1 := Make(dtypes.Float32, 32, 512)
		s2 := MakeDynamic(dtypes.Float32, "batch", 512)
		require.False(t, s1.Equal(s2))
	})

	t.Run("nil_vs_empty_axis_names", func(t *testing.T) {
		s1 := Make(dtypes.Float32, 32, 512)              // nil AxisNames
		s2 := MakeDynamic(dtypes.Float32, 32, 512)       // empty strings in AxisNames
		require.True(t, s1.Equal(s2))                    // Should be equal
	})
}

func TestShapeClone(t *testing.T) {
	original := MakeDynamic(dtypes.Float32, "batch", "seq", 512)
	clone := original.Clone()

	require.True(t, original.Equal(clone))

	// Modifying clone should not affect original
	clone.Dimensions[0] = 32
	require.Equal(t, DimDynamic, original.Dimensions[0])

	clone.AxisNames[0] = "modified"
	require.Equal(t, "batch", original.AxisNames[0])
}
