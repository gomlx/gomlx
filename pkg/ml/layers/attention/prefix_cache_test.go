package attention

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestPrefixCache(t *testing.T) {
	t.Run("StoreAndLookup", func(t *testing.T) {
		pc := NewPrefixCache(0)

		tokens := []int32{1, 2, 3, 4, 5}
		hash := HashTokens(tokens)
		blocks := []int{0, 1, 2}

		// Not found initially.
		_, _, ok := pc.Lookup(hash)
		assert.False(t, ok)

		// Store.
		pc.Store(hash, blocks, len(tokens))
		assert.Equal(t, 1, pc.NumEntries())

		// Lookup succeeds.
		foundBlocks, numTokens, ok := pc.Lookup(hash)
		assert.True(t, ok)
		assert.Equal(t, blocks, foundBlocks)
		assert.Equal(t, 5, numTokens)
	})

	t.Run("RefAndUnref", func(t *testing.T) {
		pc := NewPrefixCache(0)

		hash := HashTokens([]int32{10, 20, 30})
		blocks := []int{5, 6}
		pc.Store(hash, blocks, 3)

		// Blocks should be referenced (refcount=1 from Store).
		assert.True(t, pc.IsBlockReferenced(5))
		assert.True(t, pc.IsBlockReferenced(6))

		// Add another reference (second request reuses prefix).
		pc.Ref(blocks)

		// Unref once — still referenced.
		freed := pc.Unref(blocks)
		assert.Empty(t, freed)
		assert.True(t, pc.IsBlockReferenced(5))

		// Unref again — now freed.
		freed = pc.Unref(blocks)
		assert.ElementsMatch(t, blocks, freed)
		assert.False(t, pc.IsBlockReferenced(5))
		assert.False(t, pc.IsBlockReferenced(6))
	})

	t.Run("Evict", func(t *testing.T) {
		pc := NewPrefixCache(0)

		hash := HashTokens([]int32{1, 2})
		blocks := []int{3, 4}
		pc.Store(hash, blocks, 2)

		assert.Equal(t, 1, pc.NumEntries())

		freed := pc.Evict(hash)
		assert.ElementsMatch(t, blocks, freed)
		assert.Equal(t, 0, pc.NumEntries())

		// Lookup should fail now.
		_, _, ok := pc.Lookup(hash)
		assert.False(t, ok)
	})

	t.Run("DuplicateStore", func(t *testing.T) {
		pc := NewPrefixCache(0)

		hash := HashTokens([]int32{1})
		pc.Store(hash, []int{0}, 1)
		pc.Store(hash, []int{1}, 1) // should be ignored

		blocks, _, ok := pc.Lookup(hash)
		assert.True(t, ok)
		assert.Equal(t, []int{0}, blocks) // first store wins
	})

	t.Run("HashDeterminism", func(t *testing.T) {
		tokens := []int32{100, 200, 300}
		h1 := HashTokens(tokens)
		h2 := HashTokens(tokens)
		assert.Equal(t, h1, h2)

		different := HashTokens([]int32{100, 200, 301})
		assert.NotEqual(t, h1, different)
	})

	t.Run("LRUEviction", func(t *testing.T) {
		pc := NewPrefixCache(2) // max 2 entries

		hash1 := HashTokens([]int32{1, 2})
		hash2 := HashTokens([]int32{3, 4})
		hash3 := HashTokens([]int32{5, 6})

		pc.Store(hash1, []int{0, 1}, 2)
		time.Sleep(time.Millisecond) // ensure distinct timestamps
		pc.Store(hash2, []int{2, 3}, 2)

		assert.Equal(t, 2, pc.NumEntries())

		// Access hash1 to make hash2 the LRU entry.
		time.Sleep(time.Millisecond)
		_, _, ok := pc.Lookup(hash1)
		assert.True(t, ok)

		// Store a third entry — should evict hash2 (LRU).
		time.Sleep(time.Millisecond)
		freed := pc.Store(hash3, []int{4, 5}, 2)
		assert.ElementsMatch(t, []int{2, 3}, freed)
		assert.Equal(t, 2, pc.NumEntries())

		// hash2 should be gone, hash1 and hash3 should remain.
		_, _, ok = pc.Lookup(hash2)
		assert.False(t, ok)
		_, _, ok = pc.Lookup(hash1)
		assert.True(t, ok)
		_, _, ok = pc.Lookup(hash3)
		assert.True(t, ok)
	})

	t.Run("LRUEvictionSkipsActiveUsers", func(t *testing.T) {
		pc := NewPrefixCache(2)

		hash1 := HashTokens([]int32{10, 20})
		hash2 := HashTokens([]int32{30, 40})
		hash3 := HashTokens([]int32{50, 60})

		pc.Store(hash1, []int{0, 1}, 2)
		time.Sleep(time.Millisecond)
		pc.Store(hash2, []int{2, 3}, 2)

		// Simulate an active request using hash1's blocks (refcount becomes 2).
		blocks1, _, _ := pc.Lookup(hash1)
		pc.Ref(blocks1)

		// hash1 is LRU but has active users, so hash2 should be evicted.
		// Access hash2 to make hash1 the LRU. But hash1 has active users.
		time.Sleep(time.Millisecond)
		freed := pc.Store(hash3, []int{4, 5}, 2)
		assert.ElementsMatch(t, []int{2, 3}, freed) // hash2 evicted, not hash1

		// hash1 should still exist.
		_, _, ok := pc.Lookup(hash1)
		assert.True(t, ok)

		// Clean up: unref hash1's blocks.
		pc.Unref(blocks1)
	})

	t.Run("EvictLRU", func(t *testing.T) {
		pc := NewPrefixCache(0) // unlimited

		hash1 := HashTokens([]int32{1})
		hash2 := HashTokens([]int32{2})

		pc.Store(hash1, []int{10}, 1)
		time.Sleep(time.Millisecond)
		pc.Store(hash2, []int{20}, 1)

		// Explicit eviction should remove hash1 (oldest).
		freed := pc.EvictLRU()
		assert.Equal(t, []int{10}, freed)
		assert.Equal(t, 1, pc.NumEntries())

		_, _, ok := pc.Lookup(hash1)
		assert.False(t, ok)
		_, _, ok = pc.Lookup(hash2)
		assert.True(t, ok)

		// Evict again — removes hash2.
		freed = pc.EvictLRU()
		assert.Equal(t, []int{20}, freed)
		assert.Equal(t, 0, pc.NumEntries())

		// Nothing left to evict.
		freed = pc.EvictLRU()
		assert.Nil(t, freed)
	})
}
