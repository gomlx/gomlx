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

			_, _, pos := getKVCache(cacheCtx, g, cacheShape)
			return pos
		})

		input := [][]float32{{1.0}}
		result := exec.MustExec(input)[0]
		assert.Equal(t, []int{batchSize}, result.Shape().Dimensions)
	})

	t.Run("SingleUpdateIncrementsPosition", func(t *testing.T) {
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

			return KVCacheUpdate(cacheCtx, g, cacheShape, position, keys, values)
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
		position := result.Value().(int32)
		assert.Equal(t, int32(1), position)
	})

	t.Run("MultipleUpdatesWithExplicitPosition", func(t *testing.T) {
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

			return KVCacheUpdate(cacheCtx, g, cacheShape, position, keys, values)
		})

		keys1 := [][][][]float32{{{{1.0, 2.0, 3.0, 4.0}}, {{5.0, 6.0, 7.0, 8.0}}}}
		values1 := [][][][]float32{{{{9.0, 10.0, 11.0, 12.0}}, {{13.0, 14.0, 15.0, 16.0}}}}
		result1 := exec1.MustExec(int32(0), keys1, values1)[0]
		pos1 := result1.Value().(int32)
		assert.Equal(t, int32(1), pos1)

		// Second update at position 1
		ctx2 := ctx.Reuse()
		exec2 := context.MustNewExec(backend, ctx2, func(testCtx *context.Context, position, keys, values *Node) *Node {
			g := keys.Graph()
			cacheCtx := testCtx.In("cache").Reuse().Checked(false)
			return KVCacheUpdate(cacheCtx, g, cacheShape, position, keys, values)
		})

		keys2 := [][][][]float32{{{{2.0, 3.0, 4.0, 5.0}}, {{6.0, 7.0, 8.0, 9.0}}}}
		values2 := [][][][]float32{{{{10.0, 11.0, 12.0, 13.0}}, {{14.0, 15.0, 16.0, 17.0}}}}
		result2 := exec2.MustExec(int32(1), keys2, values2)[0]
		pos2 := result2.Value().(int32)
		assert.Equal(t, int32(2), pos2)
	})

	t.Run("BatchProcessingPositions", func(t *testing.T) {
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

			updatedPos := KVCacheUpdate(cacheCtx, g, cacheShape, position, keys, values)
			return BroadcastToShape(ExpandDims(updatedPos, 0), shapes.Make(dtypes.Int32, batchSize))
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
		positions := result.Value().([]int32)
		assert.Equal(t, int32(1), positions[0])
		assert.Equal(t, int32(1), positions[1])
		assert.Equal(t, int32(1), positions[2])
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
			cachedKeys, _, _ := getKVCache(cacheCtx, g, cacheShape)
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

	t.Run("ResetPositionToZero", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		batchSize := 2
		numHeads := 2
		maxSeqLen := 16
		headDim := 4
		cacheShape := shapes.Make(dtypes.Float32, batchSize, numHeads, maxSeqLen, headDim)

		// First: update the cache
		updateExec := context.MustNewExec(backend, ctx, func(testCtx *context.Context, position, keys, values *Node) *Node {
			g := keys.Graph()
			cacheCtx := testCtx.In("cache").Reuse().Checked(false)

			return KVCacheUpdate(cacheCtx, g, cacheShape, position, keys, values)
		})

		keys := [][][][]float32{
			{{{1.0, 2.0, 3.0, 4.0}}, {{5.0, 6.0, 7.0, 8.0}}},
			{{{1.0, 2.0, 3.0, 4.0}}, {{5.0, 6.0, 7.0, 8.0}}},
		}
		values := [][][][]float32{
			{{{9.0, 10.0, 11.0, 12.0}}, {{13.0, 14.0, 15.0, 16.0}}},
			{{{9.0, 10.0, 11.0, 12.0}}, {{13.0, 14.0, 15.0, 16.0}}},
		}

		// Update cache - position should be 1 after this
		updateResult := updateExec.MustExec(int32(0), keys, values)[0]
		assert.Equal(t, int32(1), updateResult.Value().(int32))

		// Reset cache outside of graph execution
		cacheCtx := ctx.In("cache").Reuse().Checked(false)
		KVCacheReset(cacheCtx)

		// Verify position is now 0 by reading cache
		getExec := context.MustNewExec(backend, ctx.Reuse(), func(testCtx *context.Context, dummy *Node) *Node {
			g := dummy.Graph()
			cacheCtx := testCtx.In("cache").Reuse().Checked(false)
			_, _, pos := getKVCache(cacheCtx, g, cacheShape)
			return pos
		})

		result := getExec.MustExec(int32(0))[0]
		positions := result.Value().([]int32)
		assert.Equal(t, int32(0), positions[0])
		assert.Equal(t, int32(0), positions[1])
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
			mask := createKVCacheAttentionMask(cacheCtx, g, cacheShape, 1, 1)
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

		// Initialize cache

		// Create dummy keys/values: [batch=1, heads=2, seq=3, dim=4]
		keys := IotaFull(g, shapes.Make(dtypes.Float32, 1, 2, 3, 4))
		values := Mul(keys, Const(g, float32(2.0))) // values = 2 * keys

		// Update cache and return the updated position
		return KVCacheUpdate(cacheCtx, g, cacheShape, position, keys, values)
	})

	outputs1 := exec1.MustExec(int32(0))
	position1 := outputs1[0].Value().(int32)
	t.Logf("After first update: position = %d", position1)

	assert.Equal(t, int32(3), position1, "Expected position=3 after first update")

	// Second execution: Update with 1 more key/value at position 3
	exec2 := context.MustNewExec(backend, ctx.Reuse(), func(testCtx *context.Context, position *Node) *Node {
		g := position.Graph()
		cacheCtx := testCtx.In("cache").Reuse().Checked(false)

		// Create dummy keys/values: [batch=1, heads=2, seq=1, dim=4]
		keys := IotaFull(g, shapes.Make(dtypes.Float32, 1, 2, 1, 4))
		values := Mul(keys, Const(g, float32(2.0)))

		return KVCacheUpdate(cacheCtx, g, cacheShape, position, keys, values)
	})

	outputs2 := exec2.MustExec(int32(3))
	position2 := outputs2[0].Value().(int32)
	t.Logf("After second update: position = %d", position2)

	assert.Equal(t, int32(4), position2, "Expected position=4 after second update")
}

// TestKVCacheCircular tests the circular/rotating cache functionality.
// When position exceeds maxSeqLen, the cache should wrap around.
func TestKVCacheCircular(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New()

	batchSize := 1
	numHeads := 1
	maxSeqLen := 4 // Small cache to test wrapping
	headDim := 2
	cacheShape := shapes.Make(dtypes.Float32, batchSize, numHeads, maxSeqLen, headDim)

	exec := context.MustNewExec(backend, ctx, func(testCtx *context.Context, position, keys, values *Node) []*Node {
		g := position.Graph()
		cacheCtx := testCtx.In("cache").Reuse().Checked(false)

		newPos := KVCacheUpdate(cacheCtx, g, cacheShape, position, keys, values)
		cachedKeys, _, _ := getKVCache(cacheCtx, g, cacheShape)

		return []*Node{newPos, cachedKeys}
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
	newPos := results[0].Value().(int32)
	assert.Equal(t, int32(5), newPos, "Absolute position should be 5")

	cachedKeys := results[1].Value().([][][][]float32)
	assert.InDelta(t, 5.0, cachedKeys[0][0][0][0], 0.01, "Slot 0 should have 5.0 (wrapped)")
	assert.InDelta(t, 2.0, cachedKeys[0][0][1][0], 0.01, "Slot 1 should still have 2.0")
	assert.InDelta(t, 3.0, cachedKeys[0][0][2][0], 0.01, "Slot 2 should still have 3.0")
	assert.InDelta(t, 4.0, cachedKeys[0][0][3][0], 0.01, "Slot 3 should still have 4.0")
}
