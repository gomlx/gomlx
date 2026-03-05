package attention

import (
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/assert"
)

func TestBlockManager(t *testing.T) {
	t.Run("AllocateAndFree", func(t *testing.T) {
		config := PagedKVCacheConfig{
			NumBlocks:  8,
			BlockSize:  4,
			NumKVHeads: 2,
			HeadDim:    4,
			DType:      dtypes.Float32,
		}
		bm := NewBlockManager(config)

		assert.Equal(t, 8, bm.NumFreeBlocks())

		// Allocate 3 blocks for request 1.
		blocks1, err := bm.AllocateBlocks(1, 3)
		assert.NoError(t, err)
		assert.Len(t, blocks1, 3)
		assert.Equal(t, 5, bm.NumFreeBlocks())

		// Allocate 2 blocks for request 2.
		blocks2, err := bm.AllocateBlocks(2, 2)
		assert.NoError(t, err)
		assert.Len(t, blocks2, 2)
		assert.Equal(t, 3, bm.NumFreeBlocks())

		// Free request 1.
		bm.ReleaseRequest(1)
		assert.Equal(t, 6, bm.NumFreeBlocks())

		// Page table for request 2 should still have its blocks.
		pt2 := bm.GetPageTable(2)
		assert.Equal(t, blocks2, pt2)

		// Free request 2.
		bm.ReleaseRequest(2)
		assert.Equal(t, 8, bm.NumFreeBlocks())
	})

	t.Run("OutOfBlocks", func(t *testing.T) {
		config := PagedKVCacheConfig{NumBlocks: 4}
		bm := NewBlockManager(config)

		_, err := bm.AllocateBlocks(1, 4)
		assert.NoError(t, err)

		_, err = bm.AllocateBlocks(2, 1)
		assert.ErrorIs(t, err, ErrNoFreeBlocks)
	})

	t.Run("BlocksNeeded", func(t *testing.T) {
		config := PagedKVCacheConfig{BlockSize: 16}
		bm := NewBlockManager(config)

		assert.Equal(t, 1, bm.BlocksNeeded(1))
		assert.Equal(t, 1, bm.BlocksNeeded(16))
		assert.Equal(t, 2, bm.BlocksNeeded(17))
		assert.Equal(t, 4, bm.BlocksNeeded(64))
	})

	t.Run("EnsureBlocks", func(t *testing.T) {
		config := PagedKVCacheConfig{NumBlocks: 8, BlockSize: 4}
		bm := NewBlockManager(config)

		// Need 1 block for 3 tokens.
		err := bm.EnsureBlocks(1, 3)
		assert.NoError(t, err)
		assert.Len(t, bm.GetPageTable(1), 1)

		// Need 2 blocks for 5 tokens — should allocate 1 more.
		err = bm.EnsureBlocks(1, 5)
		assert.NoError(t, err)
		assert.Len(t, bm.GetPageTable(1), 2)

		// Still 2 blocks needed for 8 tokens — no new allocation.
		err = bm.EnsureBlocks(1, 8)
		assert.NoError(t, err)
		assert.Len(t, bm.GetPageTable(1), 2)
	})
}

func TestPagedKVCacheWriteAndRead(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("SingleTokenWrite", func(t *testing.T) {
		ctx := context.New()
		config := PagedKVCacheConfig{
			NumBlocks:  4,
			BlockSize:  4,
			NumKVHeads: 1,
			HeadDim:    2,
			DType:      dtypes.Float32,
		}

		// Page table: request uses physical block 2 for logical block 0.
		pageTable := []int32{2, 0, 0, 0} // max 4 blocks

		exec := context.MustNewExec(backend, ctx,
			func(testCtx *context.Context, pageTableNode, position, newKey, newValue *Node) *Node {
				g := position.Graph()
				cacheCtx := testCtx.In("cache").Reuse().Checked(false)

				PagedKVCacheWrite(cacheCtx, g, config, pageTableNode, position, newKey, newValue)

				// Read back the raw physical cache.
				keyVar, _ := PagedKVCacheGetVars(cacheCtx, config)
				return keyVar.ValueGraph(g)
			},
		)

		// Write key [5.0, 6.0] at logical position 1 → block 2, offset 1.
		// newKey shape: [1, numKVHeads=1, 1, headDim=2]
		newKey := [][][][]float32{{{{5.0, 6.0}}}}
		newValue := [][][][]float32{{{{7.0, 8.0}}}}

		result := exec.MustExec(pageTable, int32(1), newKey, newValue)
		cached := result[0].Value().([][][][]float32)

		// Physical block 2, head 0, offset 1 should have [5.0, 6.0].
		assert.InDelta(t, 5.0, cached[2][0][1][0], 0.01, "Block 2, offset 1, dim 0")
		assert.InDelta(t, 6.0, cached[2][0][1][1], 0.01, "Block 2, offset 1, dim 1")

		// Other positions in block 2 should be zero.
		assert.InDelta(t, 0.0, cached[2][0][0][0], 0.01, "Block 2, offset 0 should be zero")
		// Block 0 should be untouched.
		assert.InDelta(t, 0.0, cached[0][0][0][0], 0.01, "Block 0 should be zero")
	})

	t.Run("ReadGatheredBlocks", func(t *testing.T) {
		ctx := context.New()
		config := PagedKVCacheConfig{
			NumBlocks:  4,
			BlockSize:  2,
			NumKVHeads: 1,
			HeadDim:    2,
			DType:      dtypes.Float32,
		}

		// First, write some data to physical blocks.
		writeExec := context.MustNewExec(backend, ctx,
			func(testCtx *context.Context, pageTableNode, position, newKey, newValue *Node) *Node {
				g := position.Graph()
				cacheCtx := testCtx.In("cache").Reuse().Checked(false)
				PagedKVCacheWrite(cacheCtx, g, config, pageTableNode, position, newKey, newValue)
				keyVar, _ := PagedKVCacheGetVars(cacheCtx, config)
				return keyVar.ValueGraph(g) // dummy return
			},
		)

		// Request uses blocks [3, 1]: logical block 0 → physical block 3, logical block 1 → physical block 1.
		pageTable := []int32{3, 1, 0, 0}

		// Write at logical position 0 → block 3, offset 0.
		writeExec.MustExec(pageTable, int32(0), [][][][]float32{{{{1.0, 2.0}}}}, [][][][]float32{{{{1.0, 2.0}}}})
		// Write at logical position 1 → block 3, offset 1.
		writeExec.MustExec(pageTable, int32(1), [][][][]float32{{{{3.0, 4.0}}}}, [][][][]float32{{{{3.0, 4.0}}}})
		// Write at logical position 2 → block 1, offset 0.
		writeExec.MustExec(pageTable, int32(2), [][][][]float32{{{{5.0, 6.0}}}}, [][][][]float32{{{{5.0, 6.0}}}})

		// Now read using PagedKVCacheRead.
		readExec := context.MustNewExec(backend, ctx.Reuse(),
			func(testCtx *context.Context, pageTableNode *Node) *Node {
				g := pageTableNode.Graph()
				cacheCtx := testCtx.In("cache").Reuse().Checked(false)
				keys, _ := PagedKVCacheRead(cacheCtx, g, config, pageTableNode, 2)
				return keys // [1, numKVHeads, seqLen=4, headDim=2]
			},
		)

		// Read with 2-block page table [3, 1].
		readPageTable := []int32{3, 1}
		result := readExec.MustExec(readPageTable)
		keys := result[0].Value().([][][][]float32)

		// keys shape: [1, 1, 4, 2] (batch=1, heads=1, seqLen=2*2=4, headDim=2)
		// Logical order: block 3 slots [0,1] then block 1 slots [0,1].
		assert.InDelta(t, 1.0, keys[0][0][0][0], 0.01, "Logical pos 0 (block 3, offset 0)")
		assert.InDelta(t, 3.0, keys[0][0][1][0], 0.01, "Logical pos 1 (block 3, offset 1)")
		assert.InDelta(t, 5.0, keys[0][0][2][0], 0.01, "Logical pos 2 (block 1, offset 0)")
		assert.InDelta(t, 0.0, keys[0][0][3][0], 0.01, "Logical pos 3 (block 1, offset 1) — not written")
	})
}

func TestBuildPageTableTensor(t *testing.T) {
	pageTable := []int{3, 1, 5}
	tensor := BuildPageTableTensor(pageTable, 6)
	vals := tensor.Value().([]int32)
	assert.Equal(t, []int32{3, 1, 5, 0, 0, 0}, vals)
}
