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

func TestAxisNames(t *testing.T) {
	// Test IsDynamicAxis
	shape := MakeDynamic(dtypes.Float32, -1, 10, -1)
	require.True(t, shape.IsDynamicAxis(0))
	require.False(t, shape.IsDynamicAxis(1))
	require.True(t, shape.IsDynamicAxis(2))
	require.True(t, shape.IsDynamicAxis(-1)) // Last axis

	// Test WithAxisName and AxisName
	named := shape.WithAxisName(0, "batch").WithAxisName(2, "seq")
	require.Equal(t, "batch", named.AxisName(0))
	require.Equal(t, "", named.AxisName(1))
	require.Equal(t, "seq", named.AxisName(2))
	require.Equal(t, "seq", named.AxisName(-1)) // Negative indexing

	// Test WithAxisNames
	allNamed := shape.WithAxisNames("batch", "hidden", "seq")
	require.Equal(t, "batch", allNamed.AxisName(0))
	require.Equal(t, "hidden", allNamed.AxisName(1))
	require.Equal(t, "seq", allNamed.AxisName(2))

	// Test WithDynamicAxis
	staticShape := Make(dtypes.Float32, 32, 128, 64)
	dynamic := staticShape.WithDynamicAxis(0, "batch")
	require.Equal(t, DynamicDim, dynamic.Dimensions[0])
	require.Equal(t, "batch", dynamic.AxisName(0))
	require.Equal(t, 128, dynamic.Dimensions[1])

	// Test String() with axis names
	prettyShape := MakeDynamic(dtypes.Float32, -1, 768).
		WithAxisName(0, "batch")
	str := prettyShape.String()
	require.Contains(t, str, "batch")
	require.Contains(t, str, "768")
}

func TestMakeDynamic(t *testing.T) {
	// Create a shape with dynamic batch dimension
	shape := MakeDynamic(dtypes.Float32, -1, 128, 128, 3)
	require.Equal(t, dtypes.Float32, shape.DType)
	require.Equal(t, 4, shape.Rank())
	require.Equal(t, -1, shape.Dimensions[0])
	require.Equal(t, 128, shape.Dimensions[1])
	require.Equal(t, 128, shape.Dimensions[2])
	require.Equal(t, 3, shape.Dimensions[3])

	// Using DynamicDim constant
	shape2 := MakeDynamic(dtypes.Float32, DynamicDim, DynamicDim, 768)
	require.Equal(t, 3, shape2.Rank())
	require.Equal(t, DynamicDim, shape2.Dimensions[0])
	require.Equal(t, DynamicDim, shape2.Dimensions[1])
	require.Equal(t, 768, shape2.Dimensions[2])
}

func TestMatches(t *testing.T) {
	// Exact match - should work with Equal too
	shape1 := Make(dtypes.Float32, 32, 128, 128, 3)
	shape2 := Make(dtypes.Float32, 32, 128, 128, 3)
	require.True(t, shape1.Matches(shape2))
	require.True(t, shape1.Equal(shape2))

	// Pattern with dynamic batch dimension
	pattern := MakeDynamic(dtypes.Float32, -1, 128, 128, 3)
	shape3 := Make(dtypes.Float32, 16, 128, 128, 3)
	shape4 := Make(dtypes.Float32, 64, 128, 128, 3)
	require.True(t, shape3.Matches(pattern))
	require.True(t, shape4.Matches(pattern))
	require.False(t, shape3.Equal(pattern)) // Equal should fail with dynamic dims

	// Pattern with multiple dynamic dimensions
	pattern2 := MakeDynamic(dtypes.Float32, -1, -2, 768)
	shape5 := Make(dtypes.Float32, 8, 512, 768)
	shape6 := Make(dtypes.Float32, 16, 256, 768)
	require.True(t, shape5.Matches(pattern2))
	require.True(t, shape6.Matches(pattern2))

	// Mismatch on static dimension
	shape7 := Make(dtypes.Float32, 16, 128, 256, 3)
	require.False(t, shape7.Matches(pattern))

	// Mismatch on dtype
	shape8 := Make(dtypes.Float64, 16, 128, 128, 3)
	require.False(t, shape8.Matches(pattern))

	// Mismatch on rank
	shape9 := Make(dtypes.Float32, 16, 128, 128)
	require.False(t, shape9.Matches(pattern))
}

func TestMatchesWithTuples(t *testing.T) {
	// Create tuple shapes
	s1 := Make(dtypes.Float32, 2, 3)
	s2 := Make(dtypes.Int32, 2, 3)
	tuple1 := MakeTuple([]Shape{s1, s2})

	// Exact match
	tuple2 := MakeTuple([]Shape{s1, s2})
	require.True(t, tuple1.Matches(tuple2))

	// Pattern with dynamic dimensions
	s1Dynamic := MakeDynamic(dtypes.Float32, -1, 3)
	s2Dynamic := MakeDynamic(dtypes.Int32, -1, 3)
	patternTuple := MakeTuple([]Shape{s1Dynamic, s2Dynamic})

	s1Concrete := Make(dtypes.Float32, 10, 3)
	s2Concrete := Make(dtypes.Int32, 10, 3)
	concreteTuple := MakeTuple([]Shape{s1Concrete, s2Concrete})

	require.True(t, concreteTuple.Matches(patternTuple))
}

func TestWithDynamicBatch(t *testing.T) {
	// Regular shape
	shape := Make(dtypes.Float32, 32, 128, 128, 3)
	dynamic := shape.WithDynamicBatch()
	require.Equal(t, -1, dynamic.Dimensions[0])
	require.Equal(t, 128, dynamic.Dimensions[1])
	require.Equal(t, 128, dynamic.Dimensions[2])
	require.Equal(t, 3, dynamic.Dimensions[3])

	// Original shape should be unchanged
	require.Equal(t, 32, shape.Dimensions[0])

	// Scalar shape
	scalar := Make(dtypes.Float32)
	scalarDynamic := scalar.WithDynamicBatch()
	require.Equal(t, 0, scalarDynamic.Rank())
}

func TestWithDynamicAxis(t *testing.T) {
	shape := Make(dtypes.Float32, 32, 128, 128, 3)

	// Set first dimension to dynamic with name
	dynamic0 := shape.WithDynamicAxis(0, "batch")
	require.Equal(t, DynamicDim, dynamic0.Dimensions[0])
	require.Equal(t, "batch", dynamic0.AxisName(0))
	require.Equal(t, 128, dynamic0.Dimensions[1])

	// Set second dimension to dynamic with name
	dynamic1 := shape.WithDynamicAxis(1, "seq")
	require.Equal(t, 32, dynamic1.Dimensions[0])
	require.Equal(t, DynamicDim, dynamic1.Dimensions[1])
	require.Equal(t, "seq", dynamic1.AxisName(1))
	require.Equal(t, 128, dynamic1.Dimensions[2])

	// Negative axis
	dynamicLast := shape.WithDynamicAxis(-1, "channels")
	require.Equal(t, 32, dynamicLast.Dimensions[0])
	require.Equal(t, 128, dynamicLast.Dimensions[1])
	require.Equal(t, 128, dynamicLast.Dimensions[2])
	require.Equal(t, DynamicDim, dynamicLast.Dimensions[3])
	require.Equal(t, "channels", dynamicLast.AxisName(3))

	// Out of bounds - should not panic, just return unchanged
	outOfBounds := shape.WithDynamicAxis(10, "oob")
	require.True(t, shape.Equal(outOfBounds))

	// Original shape should be unchanged
	require.Equal(t, 32, shape.Dimensions[0])
	require.Equal(t, 128, shape.Dimensions[1])

	// Empty name should still mark as dynamic
	dynamicNoName := shape.WithDynamicAxis(0, "")
	require.Equal(t, DynamicDim, dynamicNoName.Dimensions[0])
	require.Equal(t, "", dynamicNoName.AxisName(0))
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

func TestConcretize(t *testing.T) {
	// Test basic concretization with symbolic batch dimension
	symbolic := MakeDynamic(dtypes.Float32, -1, 3, 4)
	concrete := Make(dtypes.Float32, 10, 3, 4)
	result := symbolic.Concretize(concrete)

	require.Equal(t, 10, result.Dimensions[0], "First dimension should be concretized to 10")
	require.Equal(t, 3, result.Dimensions[1], "Second dimension should remain 3")
	require.Equal(t, 4, result.Dimensions[2], "Third dimension should remain 4")
	require.Equal(t, dtypes.Float32, result.DType)

	// Test with multiple symbolic dimensions
	symbolic2 := MakeDynamic(dtypes.Float64, -1, -2, 5)
	concrete2 := Make(dtypes.Float64, 8, 16, 5)
	result2 := symbolic2.Concretize(concrete2)

	require.Equal(t, 8, result2.Dimensions[0], "First dimension should be concretized to 8")
	require.Equal(t, 16, result2.Dimensions[1], "Second dimension should be concretized to 16")
	require.Equal(t, 5, result2.Dimensions[2], "Third dimension should remain 5")

	// Test with no symbolic dimensions (should return same shape)
	noSymbolic := Make(dtypes.Int32, 2, 3, 4)
	concrete3 := Make(dtypes.Int32, 2, 3, 4)
	result3 := noSymbolic.Concretize(concrete3)

	require.True(t, noSymbolic.Equal(result3), "Shape without symbolic dims should remain unchanged")

	// Test with mismatched dtypes (should return original)
	symbolic4 := MakeDynamic(dtypes.Float32, -1, 3)
	concrete4 := Make(dtypes.Float64, 10, 3)
	result4 := symbolic4.Concretize(concrete4)

	require.True(t, symbolic4.Equal(result4), "Mismatched dtypes should return original shape")

	// Test with mismatched ranks (should return original)
	symbolic5 := MakeDynamic(dtypes.Float32, -1, 3)
	concrete5 := Make(dtypes.Float32, 10, 3, 4)
	result5 := symbolic5.Concretize(concrete5)

	require.True(t, symbolic5.Equal(result5), "Mismatched ranks should return original shape")
}
