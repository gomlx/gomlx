package attention

import (
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestNewKVCache groups tests for the NewKVCache constructor.
func TestNewKVCache(t *testing.T) {
	t.Run("CreationAndInitialize", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		batchSize := 2
		numHeads := 4
		maxSeqLen := 128
		headDim := 64

		cache := NewKVCache(ctx, "creation_cache", batchSize, numHeads, maxSeqLen, headDim, dtypes.Float32)
		require.NotNil(t, cache)

		exec := context.MustNewExec(backend, ctx, func(testCtx *context.Context, input *Node) *Node {
			cache.Initialize(input.Graph())
			return input
		})

		input := [][]float32{{1.0, 2.0, 3.0, 4.0}}
		result := exec.MustExec(input)[0]
		assert.NotNil(t, result)
	})
}

// TestKVCacheInitialize groups tests for KVCache.Initialize.
func TestKVCacheInitialize(t *testing.T) {
	t.Run("CreatesPositionVariable", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		batchSize := 1
		numHeads := 2
		maxSeqLen := 16
		headDim := 8

		cache := NewKVCache(ctx, "initialization_cache", batchSize, numHeads, maxSeqLen, headDim, dtypes.Float32)

		exec := context.MustNewExec(backend, ctx, func(testCtx *context.Context, input *Node) *Node {
			g := input.Graph()
			cache.Initialize(g)
			_, _, pos := cache.Get(g)
			return pos
		})

		input := [][]float32{{1.0}}
		result := exec.MustExec(input)[0]
		assert.Equal(t, []int{batchSize}, result.Shape().Dimensions)
	})
}

// TestKVCacheUpdate groups tests for KVCache.Update.
func TestKVCacheUpdate(t *testing.T) {
	t.Run("SingleUpdateIncrementsPosition", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		batchSize := 1
		numHeads := 2
		maxSeqLen := 16
		headDim := 4

		cache := NewKVCache(ctx, "update_cache", batchSize, numHeads, maxSeqLen, headDim, dtypes.Float32)

		exec := context.MustNewExec(backend, ctx, func(testCtx *context.Context, keys, values *Node) *Node {
			g := keys.Graph()
			cache.Initialize(g)
			return cache.Update(g, keys, values)
		})

		keys := [][][][]float32{{
			{{1.0, 2.0, 3.0, 4.0}},
			{{5.0, 6.0, 7.0, 8.0}},
		}}
		values := [][][][]float32{{
			{{9.0, 10.0, 11.0, 12.0}},
			{{13.0, 14.0, 15.0, 16.0}},
		}}

		result := exec.MustExec(keys, values)[0]
		positions := result.Value().([]int32)
		assert.Equal(t, int32(1), positions[0])
	})

	t.Run("MultipleUpdatesTrackPosition", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		batchSize := 1
		numHeads := 2
		maxSeqLen := 16
		headDim := 4

		cache := NewKVCache(ctx, "multi_update_cache", batchSize, numHeads, maxSeqLen, headDim, dtypes.Float32)

		exec1 := context.MustNewExec(backend, ctx, func(testCtx *context.Context, keys, values *Node) *Node {
			g := keys.Graph()
			cache.Initialize(g)
			return cache.Update(g, keys, values)
		})

		keys1 := [][][][]float32{{{{1.0, 2.0, 3.0, 4.0}}, {{5.0, 6.0, 7.0, 8.0}}}}
		values1 := [][][][]float32{{{{9.0, 10.0, 11.0, 12.0}}, {{13.0, 14.0, 15.0, 16.0}}}}
		result1 := exec1.MustExec(keys1, values1)[0]
		pos1 := result1.Value().([]int32)
		assert.Equal(t, int32(1), pos1[0])

		ctx2 := ctx.Reuse()
		exec2 := context.MustNewExec(backend, ctx2, func(testCtx *context.Context, keys, values *Node) *Node {
			g := keys.Graph()
			return cache.Update(g, keys, values)
		})

		keys2 := [][][][]float32{{{{2.0, 3.0, 4.0, 5.0}}, {{6.0, 7.0, 8.0, 9.0}}}}
		values2 := [][][][]float32{{{{10.0, 11.0, 12.0, 13.0}}, {{14.0, 15.0, 16.0, 17.0}}}}
		result2 := exec2.MustExec(keys2, values2)[0]
		pos2 := result2.Value().([]int32)
		assert.Equal(t, int32(2), pos2[0])
	})

	t.Run("BatchProcessingPositions", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		batchSize := 3
		numHeads := 2
		maxSeqLen := 16
		headDim := 4

		cache := NewKVCache(ctx, "batch_cache", batchSize, numHeads, maxSeqLen, headDim, dtypes.Float32)

		exec := context.MustNewExec(backend, ctx, func(testCtx *context.Context, keys, values *Node) *Node {
			g := keys.Graph()
			cache.Initialize(g)
			return cache.Update(g, keys, values)
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

		result := exec.MustExec(keys, values)[0]
		positions := result.Value().([]int32)
		assert.Equal(t, int32(1), positions[0])
		assert.Equal(t, int32(1), positions[1])
		assert.Equal(t, int32(1), positions[2])
	})
}

// TestKVCacheGet groups tests for KVCache.Get.
func TestKVCacheGet(t *testing.T) {
	t.Run("GetAfterUpdateShape", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		batchSize := 1
		numHeads := 2
		maxSeqLen := 16
		headDim := 4

		cache := NewKVCache(ctx, "get_cache", batchSize, numHeads, maxSeqLen, headDim, dtypes.Float32)

		exec := context.MustNewExec(backend, ctx, func(testCtx *context.Context, keys, values *Node) *Node {
			g := keys.Graph()
			cache.Initialize(g)
			cache.Update(g, keys, values)
			cachedKeys, _, _ := cache.Get(g)
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

		result := exec.MustExec(keys, values)[0]
		assert.Equal(t, []int{batchSize, numHeads, maxSeqLen, headDim}, result.Shape().Dimensions)
	})
}

// TestKVCacheReset groups tests for KVCache.Reset.
func TestKVCacheReset(t *testing.T) {
	t.Run("ResetPositionToZero", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		batchSize := 2
		numHeads := 2
		maxSeqLen := 16
		headDim := 4

		cache := NewKVCache(ctx, "reset_cache", batchSize, numHeads, maxSeqLen, headDim, dtypes.Float32)

		exec := context.MustNewExec(backend, ctx, func(testCtx *context.Context, keys, values *Node) *Node {
			g := keys.Graph()
			cache.Initialize(g)
			cache.Update(g, keys, values)
			cache.Reset(g)
			_, _, pos := cache.Get(g)
			return pos
		})

		keys := [][][][]float32{
			{{{1.0, 2.0, 3.0, 4.0}}, {{5.0, 6.0, 7.0, 8.0}}},
			{{{1.0, 2.0, 3.0, 4.0}}, {{5.0, 6.0, 7.0, 8.0}}},
		}
		values := [][][][]float32{
			{{{9.0, 10.0, 11.0, 12.0}}, {{13.0, 14.0, 15.0, 16.0}}},
			{{{9.0, 10.0, 11.0, 12.0}}, {{13.0, 14.0, 15.0, 16.0}}},
		}

		result := exec.MustExec(keys, values)[0]
		positions := result.Value().([]int32)
		assert.Equal(t, int32(0), positions[0])
		assert.Equal(t, int32(0), positions[1])
	})
}

// TestKVCacheCreateAttentionMask groups tests for KVCache.CreateAttentionMask.
func TestKVCacheCreateAttentionMask(t *testing.T) {
	t.Run("MaskShape", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		batchSize := 1
		numHeads := 2
		maxSeqLen := 8
		headDim := 4

		cache := NewKVCache(ctx, "mask_cache", batchSize, numHeads, maxSeqLen, headDim, dtypes.Float32)

		exec := context.MustNewExec(backend, ctx, func(testCtx *context.Context, keys, values *Node) *Node {
			g := keys.Graph()
			cache.Initialize(g)
			cache.Update(g, keys, values)
			mask := cache.CreateAttentionMask(g, 1, 1)
			return mask
		})

		keys := [][][][]float32{{{{1.0, 2.0, 3.0, 4.0}}, {{5.0, 6.0, 7.0, 8.0}}}}
		values := [][][][]float32{{{{9.0, 10.0, 11.0, 12.0}}, {{13.0, 14.0, 15.0, 16.0}}}}

		result := exec.MustExec(keys, values)[0]
		assert.Equal(t, []int{batchSize, 1, 1, 1}, result.Shape().Dimensions)
	})
}

// TestKVCachePersistence tests that cache variables persist across multiple graph executions
func TestKVCachePersistence(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New()

	// Create a cache
	cache := NewKVCache(ctx, "test", 1, 2, 10, 4, dtypes.Float32)

	// First execution: Initialize and update with 3 keys/values
	exec1 := context.MustNewExec(backend, ctx.Reuse(), func(testCtx *context.Context, input *Node) *Node {
		g := input.Graph()

		// Initialize cache
		cache.Initialize(g)

		// Create dummy keys/values: [batch=1, heads=2, seq=3, dim=4]
		keys := IotaFull(g, shapes.Make(dtypes.Float32, 1, 2, 3, 4))
		values := Mul(keys, Const(g, float32(2.0))) // values = 2 * keys

		// Update cache
		_ = cache.Update(g, keys, values)

		// Get cache position
		_, _, pos := cache.Get(g)
		return pos
	})

	outputs1 := exec1.MustExec(int32(0))
	position1 := outputs1[0].Value().([]int32)[0]
	t.Logf("After first update: position = %d", position1)

	assert.Equal(t, int32(3), position1, "Expected position=3 after first update")

	// Second execution: Update with 1 more key/value
	exec2 := context.MustNewExec(backend, ctx.Reuse(), func(testCtx *context.Context, input *Node) *Node {
		g := input.Graph()

		// Create dummy keys/values: [batch=1, heads=2, seq=1, dim=4]
		keys := IotaFull(g, shapes.Make(dtypes.Float32, 1, 2, 1, 4))
		values := Mul(keys, Const(g, float32(2.0)))

		// Update cache (should append to existing)
		_ = cache.Update(g, keys, values)

		// Get cache position
		_, _, pos := cache.Get(g)
		return pos
	})

	outputs2 := exec2.MustExec(int32(0))
	position2 := outputs2[0].Value().([]int32)[0]
	t.Logf("After second update: position = %d", position2)

	assert.Equal(t, int32(4), position2, "Expected position=4 after second update")
}
