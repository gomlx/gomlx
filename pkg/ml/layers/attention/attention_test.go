// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"math"
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/assert"
)

func TestAxesLayoutEquivalence(t *testing.T) {
	// Verify that BSHD and BHSD layouts produce the same results when
	// given transposed versions of the same inputs.
	backend := graphtest.BuildTestBackend()
	ctx := context.New()

	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, q, k, v *Node) []*Node {
		headDim := q.Shape().Dimensions[3]
		scale := 1.0 / math.Sqrt(float64(headDim))

		// Compute with BHSD layout (default): q/k/v are [batch, heads, seq, dim]
		bhsdOut, _ := Core(ctx, q, k, v, scale, nil, 0, LayoutBHSD)

		// Transpose to BSHD: [batch, seq, heads, dim]
		qBSHD := TransposeAllDims(q, 0, 2, 1, 3)
		kBSHD := TransposeAllDims(k, 0, 2, 1, 3)
		vBSHD := TransposeAllDims(v, 0, 2, 1, 3)

		// Compute with BSHD layout
		bshdOut, _ := Core(ctx, qBSHD, kBSHD, vBSHD, scale, nil, 0, LayoutBSHD)

		// Transpose BSHD output back to BHSD for comparison
		bshdOutTransposed := TransposeAllDims(bshdOut, 0, 2, 1, 3)

		return []*Node{bhsdOut, bshdOutTransposed}
	})

	// [batch=2, heads=2, seq=3, dim=4]
	query := [][][][]float32{
		{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
			{{0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}, {0.9, 1.0, 1.1, 1.2}}},
		{{{-1, -2, -3, -4}, {-5, -6, -7, -8}, {-9, -10, -11, -12}},
			{{2, 3, 4, 5}, {6, 7, 8, 9}, {10, 11, 12, 13}}},
	}
	key := query   // self-attention
	value := query // self-attention

	outputs := exec.MustExec(query, key, value)
	bhsdData := outputs[0].Value().([][][][]float32)
	bshdData := outputs[1].Value().([][][][]float32)

	for i := range bhsdData {
		for j := range bhsdData[i] {
			for k := range bhsdData[i][j] {
				for l := range bhsdData[i][j][k] {
					assert.InDelta(t, bhsdData[i][j][k][l], bshdData[i][j][k][l], 1e-5,
						"mismatch at [%d][%d][%d][%d]", i, j, k, l)
				}
			}
		}
	}
}

func TestCore(t *testing.T) {
	t.Run("BasicIdentity", func(t *testing.T) {
		// With identity-like inputs, verify output shape and basic computation.
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, q, k, v *Node) *Node {
			headDim := q.Shape().Dimensions[3]
			scale := 1.0 / math.Sqrt(float64(headDim))
			output, _ := Core(ctx, q, k, v, scale, nil, 0, LayoutBHSD)
			return output
		})

		// [batch=1, heads=1, seq=2, dim=2]
		query := [][][][]float32{{{{1, 0}, {0, 1}}}}
		key := [][][][]float32{{{{1, 0}, {0, 1}}}}
		value := [][][][]float32{{{{1, 2}, {3, 4}}}}

		output := exec.MustExec(query, key, value)[0]
		assert.Equal(t, []int{1, 1, 2, 2}, output.Shape().Dimensions)
	})

	t.Run("CustomScale", func(t *testing.T) {
		// Verify that providing the default scale explicitly produces the same result.
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, q, k, v *Node) []*Node {
			headDim := q.Shape().Dimensions[3]
			defaultScale := 1.0 / math.Sqrt(float64(headDim))
			// Both use the same scale — should produce identical results
			defaultOut, _ := Core(ctx, q, k, v, defaultScale, nil, 0, LayoutBHSD)
			explicitOut, _ := Core(ctx, q, k, v, 1.0/math.Sqrt(2.0), nil, 0, LayoutBHSD)
			return []*Node{defaultOut, explicitOut}
		})

		query := [][][][]float32{{{{1, 2}, {3, 4}}}}
		key := [][][][]float32{{{{5, 6}, {7, 8}}}}
		value := [][][][]float32{{{{9, 10}, {11, 12}}}}

		outputs := exec.MustExec(query, key, value)
		defaultData := outputs[0].Value().([][][][]float32)
		explicitData := outputs[1].Value().([][][][]float32)

		for i := range defaultData {
			for j := range defaultData[i] {
				for k := range defaultData[i][j] {
					for l := range defaultData[i][j][k] {
						assert.InDelta(t, defaultData[i][j][k][l], explicitData[i][j][k][l], 1e-5)
					}
				}
			}
		}
	})

	t.Run("WithAdditiveMask", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, q, k, v, mask *Node) *Node {
			headDim := q.Shape().Dimensions[3]
			scale := 1.0 / math.Sqrt(float64(headDim))
			output, _ := Core(ctx, q, k, v, scale, mask, 0, LayoutBHSD)
			return output
		})

		// [batch=1, heads=1, seq=2, dim=2]
		query := [][][][]float32{{{{1, 0}, {0, 1}}}}
		key := [][][][]float32{{{{1, 0}, {0, 1}}}}
		value := [][][][]float32{{{{10, 20}, {30, 40}}}}

		// Causal mask: prevent position 0 from attending to position 1
		// [batch=1, heads=1, q_seq=2, kv_seq=2]
		mask := [][][][]float32{{{{0, -1e9}, {0, 0}}}}

		output := exec.MustExec(query, key, value, mask)[0]
		outData := output.Value().([][][][]float32)

		// Position 0 can only attend to position 0, so output[0] ~ value[0] = [10, 20]
		assert.InDelta(t, float32(10), outData[0][0][0][0], 0.1)
		assert.InDelta(t, float32(20), outData[0][0][0][1], 0.1)
	})

	t.Run("WithBooleanMask", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, q, k, v, boolMask *Node) *Node {
			headDim := q.Shape().Dimensions[3]
			scale := 1.0 / math.Sqrt(float64(headDim))
			// Pass boolean mask directly — Core auto-detects and uses MaskedSoftmax.
			output, _ := Core(ctx, q, k, v, scale, boolMask, 0, LayoutBHSD)
			return output
		})

		// [batch=1, heads=1, seq=2, dim=2]
		query := [][][][]float32{{{{1, 0}, {0, 1}}}}
		key := [][][][]float32{{{{1, 0}, {0, 1}}}}
		value := [][][][]float32{{{{10, 20}, {30, 40}}}}

		// Boolean causal mask: position 0 attends only to position 0
		mask := [][][][]bool{{{{true, false}, {true, true}}}}

		output := exec.MustExec(query, key, value, mask)[0]
		outData := output.Value().([][][][]float32)

		// Position 0 can only attend to position 0, so output[0] ~ value[0] = [10, 20]
		assert.InDelta(t, float32(10), outData[0][0][0][0], 0.1)
		assert.InDelta(t, float32(20), outData[0][0][0][1], 0.1)
	})

	t.Run("ReturnWeights", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, q, k, v *Node) []*Node {
			headDim := q.Shape().Dimensions[3]
			scale := 1.0 / math.Sqrt(float64(headDim))
			output, coefficients := Core(ctx, q, k, v, scale, nil, 0, LayoutBHSD)
			return []*Node{output, coefficients}
		})

		query := [][][][]float32{{{{1, 0}, {0, 1}}}}
		key := [][][][]float32{{{{1, 0}, {0, 1}}}}
		value := [][][][]float32{{{{1, 2}, {3, 4}}}}

		outputs := exec.MustExec(query, key, value)
		output := outputs[0]
		coefficients := outputs[1]

		assert.Equal(t, []int{1, 1, 2, 2}, output.Shape().Dimensions)
		assert.Equal(t, []int{1, 1, 2, 2}, coefficients.Shape().Dimensions)

		// Verify coefficients sum to 1 along last axis (softmax property)
		wData := coefficients.Value().([][][][]float32)
		for i := range wData {
			for j := range wData[i] {
				for k := range wData[i][j] {
					sum := float32(0)
					for _, w := range wData[i][j][k] {
						sum += w
					}
					assert.InDelta(t, float32(1.0), sum, 1e-5)
				}
			}
		}
	})
}
