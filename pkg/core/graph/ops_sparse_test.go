/*
 *	Copyright 2024 Jan Pfeifer
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

package graph_test

import (
	"fmt"
	"math"
	"slices"
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
	"golang.org/x/exp/rand"
)

func TestIndicesForShape(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	g := NewGraph(backend, t.Name())
	shape := MakeShape(F64, 2, 3, 4)
	numbers := IndicesForShape(g, shape)
	g.Compile(numbers)
	got := g.Run()[0]
	fmt.Printf("\tIndicesForShape(%s)=%v\n", shape, got)
	want := [][]int64{
		{0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {0, 0, 3},
		{0, 1, 0}, {0, 1, 1}, {0, 1, 2}, {0, 1, 3},
		{0, 2, 0}, {0, 2, 1}, {0, 2, 2}, {0, 2, 3},
		{1, 0, 0}, {1, 0, 1}, {1, 0, 2}, {1, 0, 3},
		{1, 1, 0}, {1, 1, 1}, {1, 1, 2}, {1, 1, 3},
		{1, 2, 0}, {1, 2, 1}, {1, 2, 2}, {1, 2, 3},
	}
	require.Equalf(t, want, got.Value(), "IndicesForShape(%s): want %v, got %v", shape, want, got)
}

func TestGather(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	// Trivial scalar gather.
	t.Run("scalar", func(t *testing.T) {
		g := NewGraph(backend, t.Name())
		// numbers=(Float64)[5 3]: [[0 1 2] [3 4 5] [6 7 8] [9 10 11] [12 13 14]]
		numbers := IotaFull(g, MakeShape(F64, 5, 3))
		indices := Const(g, 1)
		gather := Gather(numbers, indices, true)
		g.Compile(gather)
		got := g.Run()[0]
		fmt.Printf("\t\tGather=%v\n", got)
		want := []float64{3, 4, 5}
		require.Equalf(t, want, got.Value(), "Gather: want %v, got %v", want, got)
	})

	t.Run("leading indices", func(t *testing.T) {
		g := NewGraph(backend, t.Name())
		// numbers=(Float64)[5 3]: [[0 1 2] [3 4 5] [6 7 8] [9 10 11] [12 13 14]]
		numbers := IotaFull(g, MakeShape(F64, 5, 3))
		indices := Const(g, [][]int{{2}, {0}})
		gather := Gather(numbers, indices)
		g.Compile(gather)
		got := g.Run()[0]
		fmt.Printf("\t\tGather=%v\n", got)
		want := [][]float64{{6, 7, 8}, {0, 1, 2}}
		require.Equalf(t, want, got.Value(), "Gather: want %v, got %v", want, got)
	})

	t.Run("2D leading indices", func(t *testing.T) {
		g := NewGraph(backend, t.Name())
		// numbers=(Float64)[5 3]: [[0 1 2] [3 4 5] [6 7 8] [9 10 11] [12 13 14]]
		numbers := IotaFull(g, MakeShape(F64, 5, 3))
		indices := Const(g, [][][]int{{{2}, {0}}, {{2}, {1}}})
		gather := Gather(numbers, indices, false)
		g.Compile(gather)
		got := g.Run()[0]
		fmt.Printf("\t\tGather=%v\n", got)
		want := [][][]float64{{{6, 7, 8}, {0, 1, 2}}, {{6, 7, 8}, {3, 4, 5}}}
		require.Equalf(t, want, got.Value(), "Gather: want %v, got %v", want, got)
	})

	t.Run("2D tailing params ", func(t *testing.T) {
		g := NewGraph(backend, t.Name())
		// numbers=(Float64)[5 3]: [[0 1 2] [3 4 5] [6 7 8] [9 10 11] [12 13 14]]
		numbers := IotaFull(g, MakeShape(F64, 5, 2, 2))
		indices := Const(g, [][]int{{2}, {0}, {1}, {3}})
		gather := Gather(numbers, indices)
		g.Compile(gather)
		got := g.Run()[0]
		fmt.Printf("\t\tGather=%v\n", got)
		want := [][][]float64{{{8, 9}, {10, 11}}, {{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}, {{12, 13}, {14, 15}}}
		require.Equalf(t, want, got.Value(), "Gather: want %v, got %v", want, got)
	})

	t.Run("negative and out-of-bounds indices", func(t *testing.T) {
		g := NewGraph(backend, t.Name())
		// numbers=(Float64)[5 3]: [[0 1 2] [3 4 5] [6 7 8] [9 10 11] [12 13 14]]
		numbers := IotaFull(g, MakeShape(F64, 5, 3))
		indices := Const(g, [][]int{{-2}, {10}})
		gather := Gather(numbers, indices)
		g.Compile(gather)
		got := g.Run()[0]
		fmt.Printf("\t\tGather=%v\n", got)
		want := [][]float64{
			{0, 1, 2},    // Negative indices should become 0
			{12, 13, 14}, // Out-of-bounds indices should become the last valid index.
		}
		require.Equalf(t, want, got.Value(), "Gather: want %v, got %v", want, got)
	})
}

func TestNormalizeIndices(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("basic negative indices", func(t *testing.T) {
		// Test converting negative indices to positive
		g := NewGraph(backend, t.Name())
		// data shape [5, 3] - dimension 5 on axis 0
		data := IotaFull(g, MakeShape(F64, 5, 3))
		indices := Const(g, []int32{-1, -2, 0, 2, -5})
		normalized := NormalizeIndices(data, indices, 0)
		g.Compile(normalized)
		got := g.Run()[0]
		fmt.Printf("\tNormalizeIndices=%v\n", got)
		// -1 -> 4, -2 -> 3, 0 -> 0, 2 -> 2, -5 -> 0
		want := []int32{4, 3, 0, 2, 0}
		require.Equalf(t, want, got.Value(), "NormalizeIndices: want %v, got %v", want, got)
	})

	t.Run("negative axis", func(t *testing.T) {
		// Test using negative axis parameter
		g := NewGraph(backend, t.Name())
		// data shape [3, 5] - dimension 5 on axis 1 (or -1)
		data := IotaFull(g, MakeShape(F64, 3, 5))
		indices := Const(g, []int32{-1, -3, 2})
		normalized := NormalizeIndices(data, indices, -1) // axis -1 = axis 1
		g.Compile(normalized)
		got := g.Run()[0]
		fmt.Printf("\tNormalizeIndices (negative axis)=%v\n", got)
		// -1 -> 4, -3 -> 2, 2 -> 2
		want := []int32{4, 2, 2}
		require.Equalf(t, want, got.Value(), "NormalizeIndices: want %v, got %v", want, got)
	})

	t.Run("2D indices", func(t *testing.T) {
		// Test with 2D indices array
		g := NewGraph(backend, t.Name())
		data := IotaFull(g, MakeShape(F64, 5, 3))
		indices := Const(g, [][]int32{{-1, 0}, {-2, 1}})
		normalized := NormalizeIndices(data, indices, 0)
		g.Compile(normalized)
		got := g.Run()[0]
		fmt.Printf("\tNormalizeIndices (2D)=%v\n", got)
		// -1 -> 4, 0 -> 0, -2 -> 3, 1 -> 1
		want := [][]int32{{4, 0}, {3, 1}}
		require.Equalf(t, want, got.Value(), "NormalizeIndices: want %v, got %v", want, got)
	})

	t.Run("with Gather - Python-style negative indexing", func(t *testing.T) {
		// Test that NormalizeIndices + Gather gives Python-like behavior
		g := NewGraph(backend, t.Name())
		// numbers=(Float64)[5 3]: [[0 1 2] [3 4 5] [6 7 8] [9 10 11] [12 13 14]]
		numbers := IotaFull(g, MakeShape(F64, 5, 3))
		indices := Const(g, [][]int{{-1}, {-2}}) // Python: -1 = last row, -2 = second to last
		normalizedIndices := NormalizeIndices(numbers, indices, 0)
		gather := Gather(numbers, normalizedIndices)
		g.Compile(gather)
		got := g.Run()[0]
		fmt.Printf("\tGather with NormalizeIndices=%v\n", got)
		// -1 -> row 4 = [12, 13, 14], -2 -> row 3 = [9, 10, 11]
		want := [][]float64{{12, 13, 14}, {9, 10, 11}}
		require.Equalf(t, want, got.Value(), "Gather with NormalizeIndices: want %v, got %v", want, got)
	})

	t.Run("int64 indices", func(t *testing.T) {
		// Test with int64 indices to ensure dtype handling works correctly
		g := NewGraph(backend, t.Name())
		data := IotaFull(g, MakeShape(F64, 10, 3))
		indices := Const(g, []int64{-1, -5, 0, 9})
		normalized := NormalizeIndices(data, indices, 0)
		g.Compile(normalized)
		got := g.Run()[0]
		fmt.Printf("\tNormalizeIndices (int64)=%v\n", got)
		// -1 -> 9, -5 -> 5, 0 -> 0, 9 -> 9
		want := []int64{9, 5, 0, 9}
		require.Equalf(t, want, got.Value(), "NormalizeIndices (int64): want %v, got %v", want, got)
	})

	t.Run("out-of-bounds indices pass through", func(t *testing.T) {
		// Test that out-of-bounds negative indices are converted but not clamped
		// (clamping happens in Gather, not in NormalizeIndices)
		g := NewGraph(backend, t.Name())
		data := IotaFull(g, MakeShape(F64, 5, 3)) // dim=5 on axis 0
		// -6 is out of ONNX valid range [-5, 4], becomes 5+(-6)=-1 after normalization
		// -10 becomes 5+(-10)=-5
		// These will be clamped by Gather to valid range, but NormalizeIndices just converts
		indices := Const(g, []int32{-6, -10, 10})
		normalized := NormalizeIndices(data, indices, 0)
		g.Compile(normalized)
		got := g.Run()[0]
		fmt.Printf("\tNormalizeIndices (out-of-bounds)=%v\n", got)
		// -6 -> -1 (still negative, will be clamped by Gather)
		// -10 -> -5 (still negative, will be clamped by Gather)
		// 10 -> 10 (positive, unchanged, will be clamped by Gather)
		want := []int32{-1, -5, 10}
		require.Equalf(t, want, got.Value(), "NormalizeIndices (out-of-bounds): want %v, got %v", want, got)
	})
}

func TestGatherSlices(t *testing.T) {
	testFuncOneInput(t, "GatherSlices(input, slicedAxes={1}, start={{0}, {1}, {0}}, sizes={1})",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(dtypes.Float32, 4, 5))
			start := Const(g, [][]int32{{0}, {1}, {0}}) // Slice from rows 0, 2 and 0 of each example in the batch.
			sizes := []int{1}                           // Take only one row per start.
			output = GatherSlices(input, []int{0}, start, sizes, true)
			return
		}, [][][]float32{{{0, 1, 2, 3, 4}}, {{5, 6, 7, 8, 9}}, {{0, 1, 2, 3, 4}}})

	testFuncOneInput(t, "GatherSlices(input, slicedAxes={0}, start={{0}, {1}}, sizes={2})",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(dtypes.Float32, 4, 3))
			start := Const(g, [][]int32{{0}, {1}}) // Slice from rows 0 and 1.
			sizes := []int{2}                      // Take two rows per start.
			output = GatherSlices(input, []int{0}, start, sizes, true)
			return
		}, [][][]float32{{{0, 1, 2}, {3, 4, 5}}, {{3, 4, 5}, {6, 7, 8}}})

	testFuncOneInput(t, "GatherSlices(input, slicedAxes={0,1}, start={1, 1}, sizes={2, 3})",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(dtypes.Float32, 4, 10))
			start := Const(g, []int32{1, 1}) // Slice in middle of matrix.
			sizes := []int{2, 3}             // Take a sub-matrix
			output = GatherSlices(input, []int{0, 1}, start, sizes, true)
			return
		}, [][]float32{{11, 12, 13}, {21, 22, 23}})
}

func TestScatter(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	t.Run("Scatter(): trivial scalar scatter", func(t *testing.T) {
		// Trivial scalar scatter.
		g := NewGraph(backend, t.Name())
		// numbers=(Float64)[3]: [2 3 4]
		numbers := Add(IotaFull(g, MakeShape(F64, 3)), Const(g, float64(2)))
		indices := Const(g, 1)
		scatter := Scatter(indices, numbers, MakeShape(F64, 2, 3), true, true)
		g.Compile(scatter)
		got := g.Run()[0]
		fmt.Printf("\t\tscatter=%v\n", got)
		want := [][]float64{{0, 0, 0}, {2, 3, 4}}
		require.Equalf(t, want, got.Value(), "Scatter: want %v, got %v", want, got)
	})

	t.Run("ScatterSum(): leading indices dimension, and deeper slice dimension.", func(t *testing.T) {
		fmt.Println("\tScatterSum(): ")
		g := NewGraph(backend, t.Name())
		// numbers=(Float64)[5 3, 1]: [[[0] [1] [2]] [[3] [4] [5]]]
		numbers := IotaFull(g, MakeShape(F64, 2, 3, 1))
		indices := Const(g, [][]int{{2}, {0}})
		operand := Ones(g, MakeShape(F64, 3, 3, 1))
		scatter := ScatterSum(operand, indices, numbers, false, true)
		g.Compile(scatter)
		got := g.Run()[0]
		fmt.Printf("\t\tscatter=%s\n", got)
		want := [][][]float64{{{4}, {5}, {6}}, {{1}, {1}, {1}}, {{1}, {2}, {3}}}
		require.Equalf(t, want, got.Value(), "Scatter: want %v, got %v", want, got)
	})

	t.Run("ScatterUpdate()", func(t *testing.T) {
		graphtest.RunTestGraphFn(t, t.Name(), func(g *Graph) (inputs, outputs []*Node) {
			operand := OnePlus(IotaFull(g, MakeShape(F32, 4, 3)))
			indices := Const(g, [][]int32{{0, 0}, {0, 2}, {1, 1}, {2, 0}, {3, 2}})
			updates := Const(g, []float32{100, 200, 300, 400, 500})
			inputs = []*Node{operand, indices, updates}
			outputs = []*Node{
				ScatterUpdate(operand, indices, updates, false, true),
			}
			return
		}, []any{
			[][]float32{{100, 2, 200}, {4, 300, 6}, {400, 8, 9}, {10, 11, 500}},
		}, -1)
	})
}

// BenchmarkScatter tests the various scatter combinations: sorted or unique and different dtypes.
// The auto-differentiation of a gather is a scatter: it is used in update of large embedding tables.
func BenchmarkScatter(b *testing.B) {
	backend := graphtest.BuildTestBackend()
	const (
		NumEntries          = 1_000_000
		EmbeddingSize       = 32
		BatchSize           = 100 // Number of indices to scatter
		ConsecutiveScatters = 100
	)
	indices := make([]int32, BatchSize)
	for ii := range indices {
		indices[ii] = int32(rand.Int31n(NumEntries - ConsecutiveScatters))
	}
	slices.Sort(indices)
	indicesT := tensors.FromValue(indices)
	rngStateT := must1(RNGStateFromSeed(42))

	for _, sorted := range []bool{true, false} {
		for _, unique := range []bool{true, false} {
			scatterExec := MustNewExec(backend, func(state, indices, values *Node) *Node {
				g := values.Graph()
				dtype := values.DType()
				zeros := Zeros(g, shapes.Make(dtype, NumEntries, EmbeddingSize))
				indices = ExpandAxes(indices, -1)
				parts := make([]*Node, ConsecutiveScatters)
				for ii := range parts {
					parts[ii] = ExpandAxes(
						ScatterSum(zeros, AddScalar(indices, float64(ii)), values, sorted, unique),
						-1,
					)
				}
				x := ReduceSum(Concatenate(parts, -1), -1)
				return Add(state, x)
			})
			for _, dtype := range []dtypes.DType{dtypes.Float64, dtypes.Float32, dtypes.Float16} { //
				// Create random values tensor shaped [BatchSize, EmbeddingSize] of the given dtype.
				results := MustNewExec(backend, func(rngState *Node) (state, value *Node) {
					_, state = RandomNormal(rngState, shapes.Make(dtype, NumEntries, EmbeddingSize))
					_, value = RandomNormal(rngState, shapes.Make(dtype, BatchSize, EmbeddingSize))
					return
				}).MustExec(rngStateT)
				stateT, valuesT := results[0], results[1]

				// Precompile graph for given inputNodes. It also makes sure the inputNodes are transferred to the accelerator.
				scatterExec.MustExec(stateT, indicesT, valuesT)[0].MustFinalizeAll()
				b.Run(fmt.Sprintf("sorted-%v_unique-%v_dtype-%s", sorted, unique, dtype), func(b *testing.B) {
					for range b.N {
						results := scatterExec.MustExec(stateT, indicesT, valuesT)
						must(stateT.FinalizeAll())
						stateT = results[0]
					}
				})
			}
		}
	}
}

func TestScatterSum(t *testing.T) {
	graphtest.RunTestGraphFn(t, "ScatterSum",
		func(g *Graph) (inputs, outputs []*Node) {
			initialValues := Zeros(g, shapes.Make(F32, 5))
			flat := Const(g, []float32{1, 3, 5, 7, 11, 13})
			indices := ExpandAxes(Const(g, []int32{0, 0, 0, 1, 1, 3}), -1)
			inputs = []*Node{flat, indices}
			outputs = []*Node{ScatterSum(initialValues, indices, flat, true, false)}
			return
		},
		[]any{
			[]float32{1 + 3 + 5, 7 + 11, 0, 13, 0},
		},
		0)
}

func TestScatterSumGradient(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	operand := []float32{0, 0, 0, 0, -1}
	updates := []float32{1, 3, 5, 7, 11, 13}
	indices := []int32{0, 0, 0, 1, 1, 3}
	outputs := MustExecOnceN(backend, func(inputs []*Node) []*Node {
		operand, indices, updates := inputs[0], inputs[1], inputs[2]
		indices = ExpandAxes(indices, -1)
		g := operand.Graph()
		scattered := ScatterSum(operand, indices, updates, true, false)
		mask := Const(g, []bool{true, false, false, true, true})
		loss := MaskedReduceSum(scattered, mask)
		//return []*Node{scattered, loss}
		return append(Gradient(loss, operand, updates), scattered, loss)
	}, operand, indices, updates)

	fmt.Printf("gradOperand=%s\n", outputs[0].GoStr())
	gradOperand := outputs[0].Value().([]float32)
	fmt.Printf("gradFlat=%v\n", outputs[1].GoStr())
	gradFlat := outputs[1].Value().([]float32)
	fmt.Printf("scattered=%s\n", outputs[2].GoStr())
	scattered := outputs[2].Value().([]float32)
	fmt.Printf("loss=%s\n", outputs[3].GoStr())
	loss := outputs[3].Value().(float32)

	require.Equal(t, []float32{1 + 3 + 5, 7 + 11, 0, 13, -1}, scattered)
	require.Equal(t, float32(9+13-1), loss)
	require.Equal(t, []float32{1, 1, 1, 0, 0, 1}, gradFlat)
	require.Equal(t, []float32{1, 0, 0, 1, 1}, gradOperand)
}

func TestScatterMax(t *testing.T) {
	negInf := float32(math.Inf(-1))
	graphtest.RunTestGraphFn(t, "ScatterMax",
		func(g *Graph) (inputs, outputs []*Node) {
			initialValues := BroadcastToDims(Infinity(g, dtypes.F32, -1), 5)
			flat := Const(g, []float32{1, 3, 5, 7, 11, 13})
			indices := ExpandAxes(Const(g, []int32{0, 0, 0, 1, 1, 3}), -1)
			inputs = []*Node{flat, indices}
			outputs = []*Node{ScatterMax(initialValues, indices, flat, true, false)}
			return
		},
		[]any{
			[]float32{5, 11, negInf, 13, negInf},
		},
		0)
}

func TestScatterMaxGradient(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	negInf := float32(math.Inf(-1))
	operand := []float32{negInf, negInf, negInf, negInf, -1}
	updates := []float32{1, 3, 5, 7, 11, 13}
	indices := []int32{0, 0, 0, 1, 1, 3}
	outputs := MustExecOnceN(backend, func(inputs []*Node) []*Node {
		operand, indices, updates := inputs[0], inputs[1], inputs[2]
		indices = ExpandAxes(indices, -1)
		g := operand.Graph()
		scattered := ScatterMax(operand, indices, updates, true, false)
		mask := Const(g, []bool{true, false, false, true, true})
		loss := MaskedReduceSum(scattered, mask)
		//return []*Node{scattered, loss}
		return append(Gradient(loss, operand, updates), scattered, loss)
	}, operand, indices, updates)

	fmt.Printf("gradOperand=%s\n", outputs[0].GoStr())
	gradOperand := outputs[0].Value().([]float32)
	fmt.Printf("gradFlat=%v\n", outputs[1].GoStr())
	gradFlat := outputs[1].Value().([]float32)
	fmt.Printf("scattered=%s\n", outputs[2].GoStr())
	scattered := outputs[2].Value().([]float32)
	fmt.Printf("loss=%s\n", outputs[3].GoStr())
	loss := outputs[3].Value().(float32)

	require.Equal(t, []float32{5, 11, negInf, 13, -1}, scattered)
	require.Equal(t, float32(5+13-1), loss)
	require.Equal(t, []float32{0, 0, 1, 0, 0, 1}, gradFlat)
	require.Equal(t, []float32{0, 0, 0, 0, 1}, gradOperand)
}

func TestScatterMin(t *testing.T) {
	posInf := float32(math.Inf(1))
	graphtest.RunTestGraphFn(t, "ScatterMin",
		func(g *Graph) (inputs, outputs []*Node) {
			initialValues := BroadcastToDims(Infinity(g, dtypes.F32, 1), 5)
			flat := Const(g, []float32{1, 3, 5, 7, 11, 13})
			indices := ExpandAxes(Const(g, []int32{0, 0, 0, 1, 1, 3}), -1)
			inputs = []*Node{flat, indices}
			outputs = []*Node{ScatterMin(initialValues, indices, flat, true, false)}
			return
		},
		[]any{
			[]float32{1, 7, posInf, 13, posInf},
		},
		0)
}

func TestScatterMinGradient(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	posInf := float32(math.Inf(1))
	operand := []float32{posInf, posInf, posInf, posInf, 100}
	updates := []float32{1, 3, 5, 7, 11, 13}
	indices := []int32{0, 0, 0, 1, 1, 3}
	outputs := MustExecOnceN(backend, func(inputs []*Node) []*Node {
		operand, indices, updates := inputs[0], inputs[1], inputs[2]
		indices = ExpandAxes(indices, -1)
		g := operand.Graph()
		scattered := ScatterMin(operand, indices, updates, true, false)
		mask := Const(g, []bool{true, false, false, true, true})
		loss := MaskedReduceSum(scattered, mask)
		//return []*Node{scattered, loss}
		return append(Gradient(loss, operand, updates), scattered, loss)
	}, operand, indices, updates)

	fmt.Printf("gradOperand=%s\n", outputs[0].GoStr())
	gradOperand := outputs[0].Value().([]float32)
	fmt.Printf("gradFlat=%v\n", outputs[1].GoStr())
	gradFlat := outputs[1].Value().([]float32)
	fmt.Printf("scattered=%s\n", outputs[2].GoStr())
	scattered := outputs[2].Value().([]float32)
	fmt.Printf("loss=%s\n", outputs[3].GoStr())
	loss := outputs[3].Value().(float32)

	require.Equal(t, []float32{1, 7, posInf, 13, 100}, scattered)
	require.Equal(t, float32(1+13+100), loss)
	require.Equal(t, []float32{1, 0, 0, 0, 0, 1}, gradFlat)
	require.Equal(t, []float32{0, 0, 0, 0, 1}, gradOperand)
}
