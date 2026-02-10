package attention

import (
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/assert"
)

// TestKVCacheFunctions tests the KV cache functions.
func TestKVCacheFunctions(t *testing.T) {
	t.Run("InitAndGet", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		batchSize := 1
		numHeads := 2
		maxSeqLen := 16
		headDim := 8
		cacheShape := shapes.Make(dtypes.Float32, batchSize, numHeads, maxSeqLen, headDim)

		exec := context.MustNewExec(backend, ctx, func(testCtx *context.Context, input *Node) *Node {
			g := input.Graph()
			cacheCtx := testCtx.In("cache").Reuse().Checked(false)

			keys, _ := getKVCache(cacheCtx, g, cacheShape)
			return keys
		})

		input := [][]float32{{1.0}}
		result := exec.MustExec(input)[0]
		assert.Equal(t, []int{batchSize, numHeads, maxSeqLen, headDim}, result.Shape().Dimensions)
	})

	t.Run("SingleUpdateWritesToCache", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		batchSize := 1
		numHeads := 2
		maxSeqLen := 16
		headDim := 4
		cacheShape := shapes.Make(dtypes.Float32, batchSize, numHeads, maxSeqLen, headDim)

		exec := context.MustNewExec(backend, ctx, func(testCtx *context.Context, position, keys, values *Node) *Node {
			g := keys.Graph()
			cacheCtx := testCtx.In("cache").Reuse().Checked(false)

			KVCacheUpdate(cacheCtx, g, cacheShape, position, keys, values)
			cachedKeys, _ := getKVCache(cacheCtx, g, cacheShape)
			return cachedKeys
		})

		keys := [][][][]float32{{
			{{1.0, 2.0, 3.0, 4.0}},
			{{5.0, 6.0, 7.0, 8.0}},
		}}
		values := [][][][]float32{{
			{{9.0, 10.0, 11.0, 12.0}},
			{{13.0, 14.0, 15.0, 16.0}},
		}}

		result := exec.MustExec(int32(0), keys, values)[0]
		// Verify the keys were written at position 0
		cachedValues := result.Value().([][][][]float32)
		assert.InDelta(t, 1.0, cachedValues[0][0][0][0], 0.01) // batch 0, head 0, pos 0, dim 0
		assert.InDelta(t, 5.0, cachedValues[0][1][0][0], 0.01) // batch 0, head 1, pos 0, dim 0
	})

	t.Run("MultipleUpdatesAtDifferentPositions", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		batchSize := 1
		numHeads := 2
		maxSeqLen := 16
		headDim := 4
		cacheShape := shapes.Make(dtypes.Float32, batchSize, numHeads, maxSeqLen, headDim)

		// First update at position 0
		exec1 := context.MustNewExec(backend, ctx, func(testCtx *context.Context, position, keys, values *Node) *Node {
			g := keys.Graph()
			cacheCtx := testCtx.In("cache").Reuse().Checked(false)

			KVCacheUpdate(cacheCtx, g, cacheShape, position, keys, values)
			cachedKeys, _ := getKVCache(cacheCtx, g, cacheShape)
			return cachedKeys
		})

		keys1 := [][][][]float32{{{{1.0, 2.0, 3.0, 4.0}}, {{5.0, 6.0, 7.0, 8.0}}}}
		values1 := [][][][]float32{{{{9.0, 10.0, 11.0, 12.0}}, {{13.0, 14.0, 15.0, 16.0}}}}
		result1 := exec1.MustExec(int32(0), keys1, values1)[0]
		cached1 := result1.Value().([][][][]float32)
		assert.InDelta(t, 1.0, cached1[0][0][0][0], 0.01)

		// Second update at position 1
		ctx2 := ctx.Reuse()
		exec2 := context.MustNewExec(backend, ctx2, func(testCtx *context.Context, position, keys, values *Node) *Node {
			g := keys.Graph()
			cacheCtx := testCtx.In("cache").Reuse().Checked(false)
			KVCacheUpdate(cacheCtx, g, cacheShape, position, keys, values)
			cachedKeys, _ := getKVCache(cacheCtx, g, cacheShape)
			return cachedKeys
		})

		keys2 := [][][][]float32{{{{2.0, 3.0, 4.0, 5.0}}, {{6.0, 7.0, 8.0, 9.0}}}}
		values2 := [][][][]float32{{{{10.0, 11.0, 12.0, 13.0}}, {{14.0, 15.0, 16.0, 17.0}}}}
		result2 := exec2.MustExec(int32(1), keys2, values2)[0]
		cached2 := result2.Value().([][][][]float32)
		// Position 0 should still have first update
		assert.InDelta(t, 1.0, cached2[0][0][0][0], 0.01)
		// Position 1 should have second update
		assert.InDelta(t, 2.0, cached2[0][0][1][0], 0.01)
	})

	t.Run("BatchProcessing", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		batchSize := 3
		numHeads := 2
		maxSeqLen := 16
		headDim := 4
		cacheShape := shapes.Make(dtypes.Float32, batchSize, numHeads, maxSeqLen, headDim)

		exec := context.MustNewExec(backend, ctx, func(testCtx *context.Context, position, keys, values *Node) *Node {
			g := keys.Graph()
			cacheCtx := testCtx.In("cache").Reuse().Checked(false)

			KVCacheUpdate(cacheCtx, g, cacheShape, position, keys, values)
			cachedKeys, _ := getKVCache(cacheCtx, g, cacheShape)
			return cachedKeys
		})

		keys := [][][][]float32{
			{{{1.0, 2.0, 3.0, 4.0}}, {{5.0, 6.0, 7.0, 8.0}}},
			{{{2.0, 3.0, 4.0, 5.0}}, {{6.0, 7.0, 8.0, 9.0}}},
			{{{3.0, 4.0, 5.0, 6.0}}, {{7.0, 8.0, 9.0, 10.0}}},
		}
		values := [][][][]float32{
			{{{9.0, 10.0, 11.0, 12.0}}, {{13.0, 14.0, 15.0, 16.0}}},
			{{{10.0, 11.0, 12.0, 13.0}}, {{14.0, 15.0, 16.0, 17.0}}},
			{{{11.0, 12.0, 13.0, 14.0}}, {{15.0, 16.0, 17.0, 18.0}}},
		}

		result := exec.MustExec(int32(0), keys, values)[0]
		cachedValues := result.Value().([][][][]float32)
		// Check first element of each batch at position 0
		assert.InDelta(t, 1.0, cachedValues[0][0][0][0], 0.01) // batch 0
		assert.InDelta(t, 2.0, cachedValues[1][0][0][0], 0.01) // batch 1
		assert.InDelta(t, 3.0, cachedValues[2][0][0][0], 0.01) // batch 2
	})

	t.Run("GetAfterUpdateShape", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		batchSize := 1
		numHeads := 2
		maxSeqLen := 16
		headDim := 4
		cacheShape := shapes.Make(dtypes.Float32, batchSize, numHeads, maxSeqLen, headDim)

		exec := context.MustNewExec(backend, ctx, func(testCtx *context.Context, position, keys, values *Node) *Node {
			g := keys.Graph()
			cacheCtx := testCtx.In("cache").Reuse().Checked(false)

			KVCacheUpdate(cacheCtx, g, cacheShape, position, keys, values)
			cachedKeys, _ := getKVCache(cacheCtx, g, cacheShape)
			return cachedKeys
		})

		keys := [][][][]float32{{
			{{1.0, 2.0, 3.0, 4.0}},
			{{5.0, 6.0, 7.0, 8.0}},
		}}
		values := [][][][]float32{{
			{{9.0, 10.0, 11.0, 12.0}},
			{{13.0, 14.0, 15.0, 16.0}},
		}}

		result := exec.MustExec(int32(0), keys, values)[0]
		assert.Equal(t, []int{batchSize, numHeads, maxSeqLen, headDim}, result.Shape().Dimensions)
	})

	t.Run("ResetClearsCache", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		batchSize := 2
		numHeads := 2
		maxSeqLen := 16
		headDim := 4
		cacheShape := shapes.Make(dtypes.Float32, batchSize, numHeads, maxSeqLen, headDim)

		// First: update the cache with some values
		updateExec := context.MustNewExec(backend, ctx, func(testCtx *context.Context, position, keys, values *Node) *Node {
			g := keys.Graph()
			cacheCtx := testCtx.In("cache").Reuse().Checked(false)

			KVCacheUpdate(cacheCtx, g, cacheShape, position, keys, values)
			cachedKeys, _ := getKVCache(cacheCtx, g, cacheShape)
			return cachedKeys
		})

		keys := [][][][]float32{
			{{{1.0, 2.0, 3.0, 4.0}}, {{5.0, 6.0, 7.0, 8.0}}},
			{{{1.0, 2.0, 3.0, 4.0}}, {{5.0, 6.0, 7.0, 8.0}}},
		}
		values := [][][][]float32{
			{{{9.0, 10.0, 11.0, 12.0}}, {{13.0, 14.0, 15.0, 16.0}}},
			{{{9.0, 10.0, 11.0, 12.0}}, {{13.0, 14.0, 15.0, 16.0}}},
		}

		// Update cache
		updateResult := updateExec.MustExec(int32(0), keys, values)[0]
		cached := updateResult.Value().([][][][]float32)
		assert.InDelta(t, 1.0, cached[0][0][0][0], 0.01)

		// Reset cache outside of graph execution
		cacheCtx := ctx.In("cache").Reuse().Checked(false)
		KVCacheReset(cacheCtx)

		// Verify cache is cleared by reading key values
		getExec := context.MustNewExec(backend, ctx.Reuse(), func(testCtx *context.Context, dummy *Node) *Node {
			g := dummy.Graph()
			cacheCtx := testCtx.In("cache").Reuse().Checked(false)
			cachedKeys, _ := getKVCache(cacheCtx, g, cacheShape)
			return cachedKeys
		})

		result := getExec.MustExec(int32(0))[0]
		cachedAfterReset := result.Value().([][][][]float32)
		// After reset, the value should be 0 (zero-initialized)
		assert.InDelta(t, 0.0, cachedAfterReset[0][0][0][0], 0.01)
	})

	t.Run("CreateAttentionMaskShape", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		batchSize := 1
		numHeads := 2
		maxSeqLen := 8
		headDim := 4
		cacheShape := shapes.Make(dtypes.Float32, batchSize, numHeads, maxSeqLen, headDim)

		exec := context.MustNewExec(backend, ctx, func(testCtx *context.Context, position, keys, values *Node) *Node {
			g := keys.Graph()
			cacheCtx := testCtx.In("cache").Reuse().Checked(false)

			KVCacheUpdate(cacheCtx, g, cacheShape, position, keys, values)
			mask := createKVCacheAttentionMask(g, cacheShape, position, 1, 1)
			return mask
		})

		keys := [][][][]float32{{{{1.0, 2.0, 3.0, 4.0}}, {{5.0, 6.0, 7.0, 8.0}}}}
		values := [][][][]float32{{{{9.0, 10.0, 11.0, 12.0}}, {{13.0, 14.0, 15.0, 16.0}}}}

		result := exec.MustExec(int32(0), keys, values)[0]
		assert.Equal(t, []int{batchSize, 1, 1, 1}, result.Shape().Dimensions)
	})
}

// TestKVCachePersistence tests that cache variables persist across multiple graph executions
func TestKVCachePersistence(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New()

	batchSize := 1
	numHeads := 2
	maxSeqLen := 10
	headDim := 4
	cacheShape := shapes.Make(dtypes.Float32, batchSize, numHeads, maxSeqLen, headDim)

	// First execution: Initialize and update with 3 keys/values at position 0
	exec1 := context.MustNewExec(backend, ctx.Reuse(), func(testCtx *context.Context, position *Node) *Node {
		g := position.Graph()
		cacheCtx := testCtx.In("cache").Reuse().Checked(false)

		// Create dummy keys/values: [batch=1, heads=2, seq=3, dim=4]
		// Use specific values we can track
		keys := AddScalar(IotaFull(g, shapes.Make(dtypes.Float32, 1, 2, 3, 4)), 10)
		values := Mul(keys, Const(g, float32(2.0))) // values = 2 * keys

		KVCacheUpdate(cacheCtx, g, cacheShape, position, keys, values)
		cachedKeys, _ := getKVCache(cacheCtx, g, cacheShape)
		return cachedKeys
	})

	outputs1 := exec1.MustExec(int32(0))
	cached1 := outputs1[0].Value().([][][][]float32)
	assert.True(t, cached1[0][0][0][0] >= 10.0, "Expected non-zero value at position 0 after first update")
	assert.True(t, cached1[0][0][2][0] >= 10.0, "Expected non-zero value at position 2 after first update")

	// Second execution: Update with 1 more key/value at position 3
	exec2 := context.MustNewExec(backend, ctx.Reuse(), func(testCtx *context.Context, position *Node) *Node {
		g := position.Graph()
		cacheCtx := testCtx.In("cache").Reuse().Checked(false)

		// Create dummy keys/values: [batch=1, heads=2, seq=1, dim=4]
		keys := AddScalar(IotaFull(g, shapes.Make(dtypes.Float32, 1, 2, 1, 4)), 100)
		values := Mul(keys, Const(g, float32(2.0)))

		KVCacheUpdate(cacheCtx, g, cacheShape, position, keys, values)
		cachedKeys, _ := getKVCache(cacheCtx, g, cacheShape)
		return cachedKeys
	})

	outputs2 := exec2.MustExec(int32(3))
	cached2 := outputs2[0].Value().([][][][]float32)
	value2 := cached2[0][0][3][0] // batch=0, head=0, pos=3, dim=0
	assert.InDelta(t, 100.0, value2, 0.01, "Expected value at position 3 after second update")
	// Previous positions should still have first update's values
	assert.True(t, cached2[0][0][0][0] >= 10.0, "Position 0 should still have first update value")
}

// TestKVCacheCircular tests the circular/rotating cache functionality.
func TestKVCacheCircular(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("SingleTokenWrapAround", func(t *testing.T) {
		ctx := context.New()
		batchSize := 1
		numHeads := 1
		maxSeqLen := 4 // Small cache to test wrapping
		headDim := 2
		cacheShape := shapes.Make(dtypes.Float32, batchSize, numHeads, maxSeqLen, headDim)

		exec := context.MustNewExec(backend, ctx, func(testCtx *context.Context, position, keys, values *Node) *Node {
			g := position.Graph()
			cacheCtx := testCtx.In("cache").Reuse().Checked(false)

			KVCacheUpdate(cacheCtx, g, cacheShape, position, keys, values)
			cachedKeys, _ := getKVCache(cacheCtx, g, cacheShape)

			return cachedKeys
		})

		// Helper to create keys with identifiable values
		makeKeys := func(value float32) [][][][]float32 {
			return [][][][]float32{{{{value, value}}}}
		}

		// Fill cache with positions 0-3
		exec.MustExec(int32(0), makeKeys(1.0), makeKeys(1.0))
		exec.MustExec(int32(1), makeKeys(2.0), makeKeys(2.0))
		exec.MustExec(int32(2), makeKeys(3.0), makeKeys(3.0))
		exec.MustExec(int32(3), makeKeys(4.0), makeKeys(4.0))

		// Position 4 should wrap to slot 0
		results := exec.MustExec(int32(4), makeKeys(5.0), makeKeys(5.0))

		cachedKeys := results[0].Value().([][][][]float32)
		assert.InDelta(t, 5.0, cachedKeys[0][0][0][0], 0.01, "Slot 0 should have 5.0 (wrapped)")
		assert.InDelta(t, 2.0, cachedKeys[0][0][1][0], 0.01, "Slot 1 should still have 2.0")
		assert.InDelta(t, 3.0, cachedKeys[0][0][2][0], 0.01, "Slot 2 should still have 3.0")
		assert.InDelta(t, 4.0, cachedKeys[0][0][3][0], 0.01, "Slot 3 should still have 4.0")
	})

	t.Run("MultiTokenWrapAround", func(t *testing.T) {
		ctx := context.New()
		batchSize := 1
		numHeads := 1
		maxSeqLen := 4 // Small cache to test wrapping
		headDim := 2
		cacheShape := shapes.Make(dtypes.Float32, batchSize, numHeads, maxSeqLen, headDim)

		exec := context.MustNewExec(backend, ctx, func(testCtx *context.Context, position, keys, values *Node) *Node {
			g := position.Graph()
			cacheCtx := testCtx.In("cache").Reuse().Checked(false)

			KVCacheUpdate(cacheCtx, g, cacheShape, position, keys, values)
			cachedKeys, _ := getKVCache(cacheCtx, g, cacheShape)

			return cachedKeys
		})

		// Create 2-token update with identifiable values: token0=10.0, token1=20.0
		// Shape: [batch=1, heads=1, seq=2, dim=2]
		twoTokenKeys := [][][][]float32{{{{10.0, 10.0}, {20.0, 20.0}}}}
		twoTokenValues := [][][][]float32{{{{10.0, 10.0}, {20.0, 20.0}}}}

		// Update at position 3 with 2 tokens:
		//   - Token 0 (10.0) should go to slot 3
		//   - Token 1 (20.0) should wrap to slot 0
		results := exec.MustExec(int32(3), twoTokenKeys, twoTokenValues)

		cachedKeys := results[0].Value().([][][][]float32)
		assert.InDelta(t, 20.0, cachedKeys[0][0][0][0], 0.01, "Slot 0 should have 20.0 (second token, wrapped)")
		assert.InDelta(t, 0.0, cachedKeys[0][0][1][0], 0.01, "Slot 1 should be zero (untouched)")
		assert.InDelta(t, 0.0, cachedKeys[0][0][2][0], 0.01, "Slot 2 should be zero (untouched)")
		assert.InDelta(t, 10.0, cachedKeys[0][0][3][0], 0.01, "Slot 3 should have 10.0 (first token)")
	})
}
