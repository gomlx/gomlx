package graph_test

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/stretchr/testify/require"
)

func TestSort(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("ascending_1d", func(t *testing.T) {
		g := NewGraph(backend, "sort_ascending_1d")
		input := Const(g, []float32{3, 1, 4, 1, 5, 9, 2, 6})
		output := Sort(input, 0)
		g.Compile(output)
		results := g.Run()
		got := results[0]
		want := tensors.FromValue([]float32{1, 1, 2, 3, 4, 5, 6, 9})
		require.True(t, want.InDelta(got, 1e-6), "Sort ascending: want=%v, got=%v", want.GoStr(), got.GoStr())
	})

	t.Run("descending_1d", func(t *testing.T) {
		g := NewGraph(backend, "sort_descending_1d")
		input := Const(g, []float32{3, 1, 4, 1, 5, 9, 2, 6})
		output := SortDescending(input, 0)
		g.Compile(output)
		results := g.Run()
		got := results[0]
		want := tensors.FromValue([]float32{9, 6, 5, 4, 3, 2, 1, 1})
		require.True(t, want.InDelta(got, 1e-6), "Sort descending: want=%v, got=%v", want.GoStr(), got.GoStr())
	})

	t.Run("2d_axis0", func(t *testing.T) {
		g := NewGraph(backend, "sort_2d_axis0")
		input := Const(g, [][]float32{{3, 1}, {1, 4}, {2, 2}})
		output := Sort(input, 0)
		g.Compile(output)
		results := g.Run()
		got := results[0]
		// Sorting along axis 0: each column is sorted independently
		want := tensors.FromValue([][]float32{{1, 1}, {2, 2}, {3, 4}})
		require.True(t, want.InDelta(got, 1e-6), "Sort 2d axis0: want=%v, got=%v", want.GoStr(), got.GoStr())
	})

	t.Run("2d_axis1", func(t *testing.T) {
		g := NewGraph(backend, "sort_2d_axis1")
		input := Const(g, [][]float32{{3, 1}, {4, 2}, {1, 5}})
		output := Sort(input, 1)
		g.Compile(output)
		results := g.Run()
		got := results[0]
		// Sorting along axis 1: each row is sorted independently
		want := tensors.FromValue([][]float32{{1, 3}, {2, 4}, {1, 5}})
		require.True(t, want.InDelta(got, 1e-6), "Sort 2d axis1: want=%v, got=%v", want.GoStr(), got.GoStr())
	})
}

func TestSortWithIndices(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("ascending", func(t *testing.T) {
		g := NewGraph(backend, "sort_with_indices_asc")
		input := Const(g, []float32{3, 1, 4, 1, 5})
		values, indices := SortWithIndices(input, 0, false)
		g.Compile(values, indices)
		results := g.Run()
		gotValues := results[0]
		gotIndices := results[1]

		wantValues := tensors.FromValue([]float32{1, 1, 3, 4, 5})
		wantIndices := tensors.FromValue([]int32{1, 3, 0, 2, 4})

		require.True(t, wantValues.InDelta(gotValues, 1e-6), "Values: want=%v, got=%v", wantValues.GoStr(), gotValues.GoStr())
		require.Equal(t, wantIndices.GoStr(), gotIndices.GoStr(), "Indices mismatch")
	})

	t.Run("descending", func(t *testing.T) {
		g := NewGraph(backend, "sort_with_indices_desc")
		input := Const(g, []float32{3, 1, 4, 1, 5})
		values, indices := SortWithIndices(input, 0, true)
		g.Compile(values, indices)
		results := g.Run()
		gotValues := results[0]
		gotIndices := results[1]

		wantValues := tensors.FromValue([]float32{5, 4, 3, 1, 1})
		wantIndices := tensors.FromValue([]int32{4, 2, 0, 1, 3})

		require.True(t, wantValues.InDelta(gotValues, 1e-6), "Values: want=%v, got=%v", wantValues.GoStr(), gotValues.GoStr())
		require.Equal(t, wantIndices.GoStr(), gotIndices.GoStr(), "Indices mismatch")
	})
}

func TestArgSort(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("ascending", func(t *testing.T) {
		g := NewGraph(backend, "argsort_asc")
		input := Const(g, []float32{3, 1, 4, 1, 5})
		indices := ArgSort(input, 0, false)
		g.Compile(indices)
		results := g.Run()
		gotIndices := results[0]

		wantIndices := tensors.FromValue([]int32{1, 3, 0, 2, 4})
		require.Equal(t, wantIndices.GoStr(), gotIndices.GoStr(), "Indices mismatch")
	})
}

func TestTopK(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("k=3", func(t *testing.T) {
		g := NewGraph(backend, "topk")
		input := Const(g, []float32{3, 1, 4, 1, 5, 9, 2, 6})
		values, indices := TopK(input, 3, 0)
		g.Compile(values, indices)
		results := g.Run()
		gotValues := results[0]
		gotIndices := results[1]

		wantValues := tensors.FromValue([]float32{9, 6, 5})
		wantIndices := tensors.FromValue([]int32{5, 7, 4})

		require.True(t, wantValues.InDelta(gotValues, 1e-6), "Values: want=%v, got=%v", wantValues.GoStr(), gotValues.GoStr())
		require.Equal(t, wantIndices.GoStr(), gotIndices.GoStr(), "Indices mismatch")
	})
}

func TestBottomK(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("k=3", func(t *testing.T) {
		g := NewGraph(backend, "bottomk")
		input := Const(g, []float32{3, 1, 4, 1, 5, 9, 2, 6})
		values, indices := BottomK(input, 3, 0)
		g.Compile(values, indices)
		results := g.Run()
		gotValues := results[0]
		gotIndices := results[1]

		wantValues := tensors.FromValue([]float32{1, 1, 2})
		wantIndices := tensors.FromValue([]int32{1, 3, 6})

		require.True(t, wantValues.InDelta(gotValues, 1e-6), "Values: want=%v, got=%v", wantValues.GoStr(), gotValues.GoStr())
		require.Equal(t, wantIndices.GoStr(), gotIndices.GoStr(), "Indices mismatch")
	})
}

func TestClosureAPI(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("manual_comparator", func(t *testing.T) {
		g := NewGraph(backend, "manual_comparator")

		// Build a custom comparator using the Closure API
		closure := g.NewClosure()
		lhs := closure.AddScalarInput("lhs", shapes.Make(dtypes.Float32))
		rhs := closure.AddScalarInput("rhs", shapes.Make(dtypes.Float32))
		// Compare: lhs < rhs (ascending order)
		result := closure.LessThan(lhs, rhs)
		closure.SetOutput(result)
		comparator := closure.Build()

		// Verify the closure was built successfully
		require.NotNil(t, comparator, "Comparator should be built successfully")
	})

	t.Run("arithmetic_ops", func(t *testing.T) {
		g := NewGraph(backend, "closure_arithmetic")

		closure := g.NewClosure()
		a := closure.AddScalarInput("a", shapes.Make(dtypes.Float32))
		b := closure.AddScalarInput("b", shapes.Make(dtypes.Float32))

		// Test Add, Sub, Mul, Div
		sum := closure.Add(a, b)
		diff := closure.Sub(a, b)
		prod := closure.Mul(a, b)
		quot := closure.Div(a, b)

		// Use one of them as output
		_ = sum
		_ = diff
		_ = prod
		closure.SetOutput(quot)
		result := closure.Build()

		require.NotNil(t, result, "Arithmetic closure should build successfully")
	})

	t.Run("comparison_ops", func(t *testing.T) {
		g := NewGraph(backend, "closure_comparison")

		closure := g.NewClosure()
		a := closure.AddScalarInput("a", shapes.Make(dtypes.Int32))
		b := closure.AddScalarInput("b", shapes.Make(dtypes.Int32))

		// Test all comparison ops
		lt := closure.LessThan(a, b)
		le := closure.LessOrEqual(a, b)
		gt := closure.GreaterThan(a, b)
		ge := closure.GreaterOrEqual(a, b)
		eq := closure.Equal(a, b)
		ne := closure.NotEqual(a, b)

		_ = lt
		_ = le
		_ = gt
		_ = ge
		_ = eq
		closure.SetOutput(ne)
		result := closure.Build()

		require.NotNil(t, result, "Comparison closure should build successfully")
	})

	t.Run("logical_ops", func(t *testing.T) {
		g := NewGraph(backend, "closure_logical")

		closure := g.NewClosure()
		a := closure.AddScalarInput("a", shapes.Make(dtypes.Float32))
		b := closure.AddScalarInput("b", shapes.Make(dtypes.Float32))

		// Build logical expression: (a < b) && !(a == b)
		lt := closure.LessThan(a, b)
		eq := closure.Equal(a, b)
		notEq := closure.LogicalNot(eq)
		result := closure.LogicalAnd(lt, notEq)

		closure.SetOutput(result)
		built := closure.Build()

		require.NotNil(t, built, "Logical closure should build successfully")
	})

	t.Run("constant_and_unary", func(t *testing.T) {
		g := NewGraph(backend, "closure_constant")

		closure := g.NewClosure()
		a := closure.AddScalarInput("a", shapes.Make(dtypes.Float32))

		// Test Neg, Abs, and Constant
		negA := closure.Neg(a)
		absNegA := closure.Abs(negA)
		one := closure.Constant(float32(1.0))
		result := closure.Add(absNegA, one)

		closure.SetOutput(result)
		built := closure.Build()

		require.NotNil(t, built, "Constant/unary closure should build successfully")
	})

	t.Run("min_max", func(t *testing.T) {
		g := NewGraph(backend, "closure_minmax")

		closure := g.NewClosure()
		a := closure.AddScalarInput("a", shapes.Make(dtypes.Float64))
		b := closure.AddScalarInput("b", shapes.Make(dtypes.Float64))

		minVal := closure.Min(a, b)
		maxVal := closure.Max(a, b)
		result := closure.Add(minVal, maxVal)

		closure.SetOutput(result)
		built := closure.Build()

		require.NotNil(t, built, "Min/Max closure should build successfully")
	})
}

func TestSortMultipleDTypes(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("int32", func(t *testing.T) {
		g := NewGraph(backend, "sort_int32")
		input := Const(g, []int32{3, 1, 4, 1, 5})
		output := Sort(input, 0)
		g.Compile(output)
		results := g.Run()
		got := results[0]
		want := tensors.FromValue([]int32{1, 1, 3, 4, 5})
		require.Equal(t, want.GoStr(), got.GoStr(), "Sort int32 mismatch")
	})

	t.Run("int64", func(t *testing.T) {
		g := NewGraph(backend, "sort_int64")
		input := Const(g, []int64{3, 1, 4, 1, 5})
		output := Sort(input, 0)
		g.Compile(output)
		results := g.Run()
		got := results[0]
		want := tensors.FromValue([]int64{1, 1, 3, 4, 5})
		require.Equal(t, want.GoStr(), got.GoStr(), "Sort int64 mismatch")
	})

	t.Run("float64", func(t *testing.T) {
		g := NewGraph(backend, "sort_float64")
		input := Const(g, []float64{3.0, 1.0, 4.0, 1.0, 5.0})
		output := Sort(input, 0)
		g.Compile(output)
		results := g.Run()
		got := results[0]
		want := tensors.FromValue([]float64{1.0, 1.0, 3.0, 4.0, 5.0})
		require.True(t, want.InDelta(got, 1e-10), "Sort float64: want=%v, got=%v", want.GoStr(), got.GoStr())
	})
}

func TestSortNegativeAxis(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("2d_axis_neg1", func(t *testing.T) {
		g := NewGraph(backend, "sort_neg_axis")
		// Shape [2, 3]
		input := Const(g, [][]float32{{3, 1, 2}, {6, 4, 5}})
		output := Sort(input, -1) // Same as axis=1
		g.Compile(output)
		results := g.Run()
		got := results[0]
		want := tensors.FromValue([][]float32{{1, 2, 3}, {4, 5, 6}})
		require.True(t, want.InDelta(got, 1e-6), "Sort negative axis: want=%v, got=%v", want.GoStr(), got.GoStr())
	})

	t.Run("3d_axis_neg2", func(t *testing.T) {
		g := NewGraph(backend, "sort_3d_neg2")
		// Shape [2, 2, 2]
		input := Const(g, [][][]float32{
			{{4, 3}, {2, 1}},
			{{8, 7}, {6, 5}},
		})
		output := Sort(input, -2) // Same as axis=1
		g.Compile(output)
		results := g.Run()
		got := results[0]
		// Sorting along axis 1 (middle dimension)
		want := tensors.FromValue([][][]float32{
			{{2, 1}, {4, 3}},
			{{6, 5}, {8, 7}},
		})
		require.True(t, want.InDelta(got, 1e-6), "Sort 3d negative axis: want=%v, got=%v", want.GoStr(), got.GoStr())
	})
}

func TestTopKBottomK2D(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("topk_2d_axis0", func(t *testing.T) {
		g := NewGraph(backend, "topk_2d_axis0")
		// Shape [3, 2]
		input := Const(g, [][]float32{{1, 2}, {5, 6}, {3, 4}})
		values, indices := TopK(input, 2, 0)
		g.Compile(values, indices)
		results := g.Run()
		gotValues := results[0]
		gotIndices := results[1]

		// Top 2 along axis 0: [[5,6], [3,4]]
		wantValues := tensors.FromValue([][]float32{{5, 6}, {3, 4}})
		wantIndices := tensors.FromValue([][]int32{{1, 1}, {2, 2}})

		require.True(t, wantValues.InDelta(gotValues, 1e-6), "TopK 2D values: want=%v, got=%v", wantValues.GoStr(), gotValues.GoStr())
		require.Equal(t, wantIndices.GoStr(), gotIndices.GoStr(), "TopK 2D indices mismatch")
	})

	t.Run("bottomk_2d_axis1", func(t *testing.T) {
		g := NewGraph(backend, "bottomk_2d_axis1")
		// Shape [2, 4]
		input := Const(g, [][]float32{{4, 1, 3, 2}, {8, 5, 7, 6}})
		values, indices := BottomK(input, 2, 1)
		g.Compile(values, indices)
		results := g.Run()
		gotValues := results[0]
		gotIndices := results[1]

		// Bottom 2 along axis 1
		wantValues := tensors.FromValue([][]float32{{1, 2}, {5, 6}})
		wantIndices := tensors.FromValue([][]int32{{1, 3}, {1, 3}})

		require.True(t, wantValues.InDelta(gotValues, 1e-6), "BottomK 2D values: want=%v, got=%v", wantValues.GoStr(), gotValues.GoStr())
		require.Equal(t, wantIndices.GoStr(), gotIndices.GoStr(), "BottomK 2D indices mismatch")
	})
}

func TestSortWithIndices2D(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("2d_axis0", func(t *testing.T) {
		g := NewGraph(backend, "sort_indices_2d_axis0")
		// Shape [3, 2]
		input := Const(g, [][]float32{{3, 2}, {1, 4}, {2, 1}})
		values, indices := SortWithIndices(input, 0, false)
		g.Compile(values, indices)
		results := g.Run()
		gotValues := results[0]
		gotIndices := results[1]

		// Sorted along axis 0
		wantValues := tensors.FromValue([][]float32{{1, 1}, {2, 2}, {3, 4}})
		wantIndices := tensors.FromValue([][]int32{{1, 2}, {2, 0}, {0, 1}})

		require.True(t, wantValues.InDelta(gotValues, 1e-6), "SortWithIndices 2D values: want=%v, got=%v", wantValues.GoStr(), gotValues.GoStr())
		require.Equal(t, wantIndices.GoStr(), gotIndices.GoStr(), "SortWithIndices 2D indices mismatch")
	})
}
