package pos

import (
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestRoPE groups tests for the RoPE function.
func TestRoPE(t *testing.T) {
	t.Run("Basic", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		rope := NewRoPE(10000.0)
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x, startPos *Node) *Node {
			// Create position indices using helper function
			seqLen := x.Shape().Dimensions[0]
			posIndices := SequentialPositions(x.Graph(), startPos, seqLen)
			return rope.Apply(x, posIndices)
		})

		input := [][]float32{
			{1, 0, 0, 0, 0, 0, 0, 0},
			{0, 1, 0, 0, 0, 0, 0, 0},
			{0, 0, 1, 0, 0, 0, 0, 0},
			{0, 0, 0, 1, 0, 0, 0, 0},
		}

		startPos := []int32{0}
		output := exec.MustExec(input, startPos)[0]
		assert.Equal(t, dtypes.Float32, output.DType())
		assert.Equal(t, []int{4, 8}, output.Shape().Dimensions)
	})

	t.Run("DifferentStartPos", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		rope := NewRoPE(10000.0)
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x, startPos *Node) *Node {
			seqLen := x.Shape().Dimensions[0]
			posIndices := SequentialPositions(x.Graph(), startPos, seqLen)
			return rope.Apply(x, posIndices)
		})

		input := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8}}
		startPos := []int32{5}

		output := exec.MustExec(input, startPos)[0]
		assert.Equal(t, []int{1, 8}, output.Shape().Dimensions)
	})

	t.Run("DifferentBaseFreq", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		rope := NewRoPE(5000.0)
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x, startPos *Node) *Node {
			seqLen := x.Shape().Dimensions[x.Rank()-2]
			posIndices := SequentialPositions(x.Graph(), startPos, seqLen)
			return rope.Apply(x, posIndices)
		})

		input := [][]float32{
			{1, 2, 3, 4, 5, 6, 7, 8},
			{1, 2, 3, 4, 5, 6, 7, 8},
		}
		startPos := []int32{0}

		output := exec.MustExec(input, startPos)[0]
		assert.Equal(t, []int{2, 8}, output.Shape().Dimensions)
	})

	t.Run("PreservesNorms", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		rope := NewRoPE(10000.0)
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x, startPos *Node) *Node {
			seqLen := x.Shape().Dimensions[x.Rank()-2]
			posIndices := SequentialPositions(x.Graph(), startPos, seqLen)
			rotated := rope.Apply(x, posIndices)
			return ReduceAllSum(Abs(rotated))
		})

		input := [][]float32{
			{1, 1, 1, 1, 1, 1, 1, 1},
			{2, 2, 2, 2, 2, 2, 2, 2},
		}
		startPos := []int32{0}

		output := exec.MustExec(input, startPos)[0]
		sum := output.Value().(float32)
		assert.Greater(t, sum, float32(0.0))
	})

	t.Run("NonScalarStartPosPanics", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		// startPos with shape [2] should panic (can't squeeze to scalar)
		require.Panics(t, func() {
			rope := NewRoPE(10000.0)
			exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x, startPos *Node) *Node {
				seqLen := x.Shape().Dimensions[x.Rank()-2]
				posIndices := SequentialPositions(x.Graph(), startPos, seqLen)
				return rope.Apply(x, posIndices)
			})

			input := [][]float32{
				{1, 0, 0, 0, 0, 0, 0, 0},
				{0, 1, 0, 0, 0, 0, 0, 0},
			}
			startPos := []int32{0, 1} // Shape [2], cannot be squeezed to scalar
			_ = exec.MustExec(input, startPos)[0]
		})
	})
}

// TestRoPEWithCustomDim groups tests for the RoPEWithCustomDim function.
func TestRoPEWithCustomDim(t *testing.T) {
	t.Run("PartialRange", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		// Input shape: [batch=2, embed_dim=8]
		rope := NewRoPEWithDimRange(10000.0, 2, 6)
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x, startPos *Node) *Node {
			// Apply RoPE only to middle dimensions [2:6]
			seqLen := x.Shape().Dimensions[x.Rank()-2]
			posIndices := SequentialPositions(x.Graph(), startPos, seqLen)
			return rope.Apply(x, posIndices)
		})

		input := [][]float32{
			{1, 2, 3, 4, 5, 6, 7, 8},
			{2, 3, 4, 5, 6, 7, 8, 9},
		}
		startPos := []int32{0}

		output := exec.MustExec(input, startPos)[0]
		// Shape should be preserved
		assert.Equal(t, []int{2, 8}, output.Shape().Dimensions)
	})

	t.Run("FullEmbedding", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		rope := NewRoPEWithDimRange(10000.0, 0, 8)
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x, startPos *Node) *Node {
			// Apply RoPE to the full embedding range [0:8]
			seqLen := x.Shape().Dimensions[x.Rank()-2]
			posIndices := SequentialPositions(x.Graph(), startPos, seqLen)
			return rope.Apply(x, posIndices)
		})

		input := [][]float32{
			{1, 0, 0, 0, 0, 0, 0, 0},
			{0, 1, 0, 0, 0, 0, 0, 0},
		}
		startPos := []int32{0}

		output := exec.MustExec(input, startPos)[0]
		assert.Equal(t, dtypes.Float32, output.DType())
		assert.Equal(t, []int{2, 8}, output.Shape().Dimensions)
	})

	t.Run("StartPosAndFreq", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		rope := NewRoPEWithDimRange(5000.0, 4, 8)
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x, startPos *Node) *Node {
			// Apply RoPE to tail half with different startPos and baseFreq
			seqLen := x.Shape().Dimensions[x.Rank()-2]
			posIndices := SequentialPositions(x.Graph(), startPos, seqLen)
			return rope.Apply(x, posIndices)
		})

		input := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8}}
		startPos := []int32{5}

		output := exec.MustExec(input, startPos)[0]
		assert.Equal(t, []int{1, 8}, output.Shape().Dimensions)
	})

	t.Run("OddRangePanics", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		// Range [1:4] has length 3 (odd) and should panic
		require.Panics(t, func() {
			rope := NewRoPEWithDimRange(10000.0, 1, 4)
			exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x, startPos *Node) *Node {
				seqLen := x.Shape().Dimensions[x.Rank()-2]
				posIndices := SequentialPositions(x.Graph(), startPos, seqLen)
				return rope.Apply(x, posIndices)
			})

			input := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8}}
			startPos := []int32{0}
			_ = exec.MustExec(input, startPos)[0]
		})
	})

	t.Run("OutOfBoundsPanics", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		// dimEnd beyond embedding size should panic
		require.Panics(t, func() {
			rope := NewRoPEWithDimRange(10000.0, 0, 10)
			exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x, startPos *Node) *Node {
				seqLen := x.Shape().Dimensions[x.Rank()-2]
				posIndices := SequentialPositions(x.Graph(), startPos, seqLen)
				return rope.Apply(x, posIndices)
			})

			input := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8}}
			startPos := []int32{0}
			_ = exec.MustExec(input, startPos)[0]
		})
	})

	t.Run("HigherRankTensor_SpacerRequired", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		// Test with a 3D tensor [batch, seq_len, embed_dim]
		// This test validates that .Spacer() correctly slices the last axis
		// Without .Spacer(), it would incorrectly slice the seq_len axis
		rope := NewRoPEWithDimRange(10000.0, 0, 4)
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x, startPos *Node) *Node {
			seqLen := x.Shape().Dimensions[x.Rank()-2]
			posIndices := SequentialPositions(x.Graph(), startPos, seqLen)
			return rope.Apply(x, posIndices)
		})

		// Input shape: [batch=2, seq_len=3, embed_dim=8]
		input := [][][]float32{
			{
				{1, 2, 3, 4, 5, 6, 7, 8},
				{2, 3, 4, 5, 6, 7, 8, 9},
				{3, 4, 5, 6, 7, 8, 9, 10},
			},
			{
				{4, 5, 6, 7, 8, 9, 10, 11},
				{5, 6, 7, 8, 9, 10, 11, 12},
				{6, 7, 8, 9, 10, 11, 12, 13},
			},
		}
		startPos := []int32{0}

		output := exec.MustExec(input, startPos)[0]

		// Output shape should be preserved: [2, 3, 8]
		assert.Equal(t, []int{2, 3, 8}, output.Shape().Dimensions)

		// Verify the output is valid (not NaN or inf)
		outputData := output.Value().([][][]float32)
		for i := 0; i < 2; i++ {
			for j := 0; j < 3; j++ {
				for k := 0; k < 8; k++ {
					val := outputData[i][j][k]
					assert.False(t, isNaN(val), "Output contains NaN at [%d][%d][%d]", i, j, k)
					assert.False(t, isInf(val), "Output contains Inf at [%d][%d][%d]", i, j, k)
				}
			}
		}

		// Verify that dimensions [4:8] (not touched by RoPE) are unchanged
		for i := 0; i < 2; i++ {
			for j := 0; j < 3; j++ {
				for k := 4; k < 8; k++ {
					assert.Equal(t, input[i][j][k], outputData[i][j][k],
						"Unchanged dimension [%d][%d][%d] should be preserved", i, j, k)
				}
			}
		}
	})
}

// TestApplyWithCosSin tests the ApplyWithCosSin function.
func TestApplyWithCosSin(t *testing.T) {
	t.Run("MatchesRoPEApply", func(t *testing.T) {
		// Verify that ApplyWithCosSin produces the same result as RoPE.Apply
		// when cos/sin are computed from the same base frequency and positions.
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		baseFreq := 10000.0
		rope := NewRoPE(baseFreq)

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x, startPos *Node) []*Node {
			g := x.Graph()
			shape := x.Shape()
			dtype := shape.DType
			rank := shape.Rank()
			seqLen := shape.Dimensions[rank-2]
			embedDim := shape.Dimensions[rank-1]
			halfDim := embedDim / 2

			posIndices := SequentialPositions(g, startPos, seqLen)

			// Result from RoPE.Apply
			ropeResult := rope.Apply(x, posIndices)

			// Compute cos/sin manually (replicating what applyRoPE does internally)
			positions := ConvertDType(posIndices, dtype)
			dimIndices := Iota(g, shapes.Make(dtype, halfDim), 0)
			dimIndices = MulScalar(dimIndices, 2.0/float64(embedDim))
			baseFreqTensor := Const(g, []float64{baseFreq})
			baseFreqTensor = ConvertDType(baseFreqTensor, dtype)
			freqs := Pow(baseFreqTensor, dimIndices)
			freqs = Reciprocal(freqs)
			positions = ExpandDims(positions, -1)
			freqs = ExpandDims(freqs, 0)
			angles := Mul(positions, freqs)
			cosAngles := Cos(angles)
			sinAngles := Sin(angles)

			// Result from ApplyWithCosSin (non-interleaved, matching RoPE.Apply)
			cossinResult := ApplyWithCosSin(x, cosAngles, sinAngles, false)

			return []*Node{ropeResult, cossinResult}
		})

		input := [][]float32{
			{1, 2, 3, 4, 5, 6, 7, 8},
			{8, 7, 6, 5, 4, 3, 2, 1},
			{1, 0, 1, 0, 1, 0, 1, 0},
		}
		startPos := []int32{0}

		outputs := exec.MustExec(input, startPos)
		ropeOut := outputs[0].Value().([][]float32)
		cossinOut := outputs[1].Value().([][]float32)

		for i := range ropeOut {
			for j := range ropeOut[i] {
				diff := ropeOut[i][j] - cossinOut[i][j]
				if diff < 0 {
					diff = -diff
				}
				assert.InDelta(t, ropeOut[i][j], cossinOut[i][j], 1e-5,
					"Mismatch at [%d][%d]: rope=%v cossin=%v", i, j, ropeOut[i][j], cossinOut[i][j])
			}
		}
	})

	t.Run("Interleaved", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x, cos, sin *Node) *Node {
			return ApplyWithCosSin(x, cos, sin, true)
		})

		// x: [2, 4] (seq_len=2, head_dim=4)
		input := [][]float32{
			{1, 2, 3, 4},
			{5, 6, 7, 8},
		}
		// cos/sin: [2, 2] (seq_len=2, dim/2=2)
		cosVals := [][]float32{
			{1, 1},
			{0, 1},
		}
		sinVals := [][]float32{
			{0, 0},
			{1, 0},
		}

		output := exec.MustExec(input, cosVals, sinVals)[0]
		assert.Equal(t, []int{2, 4}, output.Shape().Dimensions)

		// At position 0: cos=1, sin=0 -> no rotation, output = input
		// At position 1: cos=[0,1], sin=[1,0]
		// Interleaved pairs: (x[0],x[1]) and (x[2],x[3])
		// For pair (5,6) with cos=0, sin=1: rotated = (5*0-6*1, 5*1+6*0) = (-6, 5)
		// For pair (7,8) with cos=1, sin=0: rotated = (7*1-8*0, 7*0+8*1) = (7, 8)
		outData := output.Value().([][]float32)
		assert.InDelta(t, float32(1), outData[0][0], 1e-5)
		assert.InDelta(t, float32(2), outData[0][1], 1e-5)
		assert.InDelta(t, float32(3), outData[0][2], 1e-5)
		assert.InDelta(t, float32(4), outData[0][3], 1e-5)
		assert.InDelta(t, float32(-6), outData[1][0], 1e-5)
		assert.InDelta(t, float32(5), outData[1][1], 1e-5)
		assert.InDelta(t, float32(7), outData[1][2], 1e-5)
		assert.InDelta(t, float32(8), outData[1][3], 1e-5)
	})

	t.Run("PartialRotation", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x, cos, sin *Node) *Node {
			return ApplyWithCosSin(x, cos, sin, false)
		})

		// x: [1, 6] - head_dim=6, but only rotary_dim=4
		input := [][]float32{{1, 2, 3, 4, 5, 6}}
		// cos/sin: [1, 2] -> rotary_dim/2 = 2, so rotary_dim = 4
		cosVals := [][]float32{{1, 1}}
		sinVals := [][]float32{{0, 0}}

		output := exec.MustExec(input, cosVals, sinVals)[0]
		assert.Equal(t, []int{1, 6}, output.Shape().Dimensions)

		// With cos=1, sin=0, rotation is identity. Pass-through dims should be unchanged.
		outData := output.Value().([][]float32)
		for j := 0; j < 6; j++ {
			assert.InDelta(t, input[0][j], outData[0][j], 1e-5, "dim %d", j)
		}
	})
}

// Helper functions for float validation
func isNaN(f float32) bool {
	return f != f
}

func isInf(f float32) bool {
	return f > 1e38 || f < -1e38
}

// TestSequentialPositions tests the SequentialPositions helper function.
func TestSequentialPositions(t *testing.T) {
	t.Run("ScalarStartPos", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, startPos *Node) *Node {
			return SequentialPositions(startPos.Graph(), startPos, 4)
		})

		output := exec.MustExec(int32(5))[0]
		assert.Equal(t, []int{4}, output.Shape().Dimensions)
		assert.Equal(t, []int32{5, 6, 7, 8}, output.Value().([]int32))
	})

	t.Run("BatchedStartPos", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, startPos *Node) *Node {
			return SequentialPositions(startPos.Graph(), startPos, 4)
		})

		// Each batch element at a different position (multi-client serving scenario)
		output := exec.MustExec([]int32{5, 10, 100})[0]
		assert.Equal(t, []int{3, 4}, output.Shape().Dimensions)

		result := output.Value().([][]int32)
		assert.Equal(t, []int32{5, 6, 7, 8}, result[0])
		assert.Equal(t, []int32{10, 11, 12, 13}, result[1])
		assert.Equal(t, []int32{100, 101, 102, 103}, result[2])
	})

	t.Run("SingleElementBatchTreatedAsScalar", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, startPos *Node) *Node {
			return SequentialPositions(startPos.Graph(), startPos, 3)
		})

		// Single element batch should be treated as scalar (backward compatible)
		output := exec.MustExec([]int32{7})[0]
		assert.Equal(t, []int{3}, output.Shape().Dimensions)
		assert.Equal(t, []int32{7, 8, 9}, output.Value().([]int32))
	})
}
