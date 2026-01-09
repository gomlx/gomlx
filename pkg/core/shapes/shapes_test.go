// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package shapes

import (
	"math"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/stretchr/testify/require"
)

func TestCastAsDType(t *testing.T) {
	value := [][]int{{1, 2}, {3, 4}, {5, 6}}
	{
		want := [][]float32{{1, 2}, {3, 4}, {5, 6}}
		got := CastAsDType(value, dtypes.Float32)
		require.Equal(t, want, got)
	}
	{
		want := [][]complex64{{1, 2}, {3, 4}, {5, 6}}
		got := CastAsDType(value, dtypes.Complex64)
		require.Equal(t, want, got)
	}
}

func TestShape(t *testing.T) {
	invalidShape := Invalid()
	require.False(t, invalidShape.Ok())

	shape0 := Make(dtypes.Float64)
	require.True(t, shape0.Ok())
	require.True(t, shape0.IsScalar())
	require.False(t, shape0.IsTuple())
	require.Equal(t, 0, shape0.Rank())
	require.Len(t, shape0.Dimensions, 0)
	require.Equal(t, 1, shape0.Size())
	require.Equal(t, 8, int(shape0.Memory()))

	shape1 := Make(dtypes.Float32, 4, 3, 2)
	require.True(t, shape1.Ok())
	require.False(t, shape1.IsScalar())
	require.False(t, shape1.IsTuple())
	require.Equal(t, 3, shape1.Rank())
	require.Len(t, shape1.Dimensions, 3)
	require.Equal(t, 4*3*2, shape1.Size())
	require.Equal(t, 4*4*3*2, int(shape1.Memory()))
}

func TestDim(t *testing.T) {
	shape := Make(dtypes.Float32, 4, 3, 2)
	require.Equal(t, 4, shape.Dim(0))
	require.Equal(t, 3, shape.Dim(1))
	require.Equal(t, 2, shape.Dim(2))
	require.Equal(t, 4, shape.Dim(-3))
	require.Equal(t, 3, shape.Dim(-2))
	require.Equal(t, 2, shape.Dim(-1))
	require.Panics(t, func() { _ = shape.Dim(3) })
	require.Panics(t, func() { _ = shape.Dim(-4) })
}

func TestFromAnyValue(t *testing.T) {
	shape, err := FromAnyValue([]int32{1, 2, 3})
	require.NoError(t, err)
	require.NotPanics(t, func() { shape.Assert(dtypes.Int32, 3) })

	shape, err = FromAnyValue([][][]complex64{{{1, 2, -3}, {3, 4 + 2i, -7 - 1i}}})
	require.NoError(t, err)
	require.NotPanics(t, func() { shape.Assert(dtypes.Complex64, 1, 2, 3) })

	// Irregular shape is not accepted:
	shape, err = FromAnyValue([][]float32{{1, 2, 3}, {4, 5}})
	require.Errorf(t, err, "irregular shape should have returned an error, instead got shape %s", shape)
}

func TestCastDType(t *testing.T) {
	t.Run("BFloat16", func(t *testing.T) {
		for _, v := range []float64{math.Inf(-1), -1, 0, 2, math.Inf(1)} {
			vAny := CastAsDType(v, dtypes.BF16)
			if _, ok := vAny.(bfloat16.BFloat16); !ok {
				t.Errorf("Failed CastAsDType from float64(%g) to BFloat16, got %T instead", v, vAny)
			}
			v32 := float32(v)
			vAny = CastAsDType(v32, dtypes.BF16)
			if _, ok := vAny.(bfloat16.BFloat16); !ok {
				t.Errorf("Failed CastAsDType from float32(%g) to BFloat16, got %T instead", v32, vAny)
			}
		}
	})
}
