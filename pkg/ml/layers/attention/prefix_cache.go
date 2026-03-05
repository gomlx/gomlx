package attention

import (
	"crypto/sha256"
	"encoding/binary"
	"sync"
	"time"
)

// PrefixCache enables sharing KV cache blocks across requests with identical
// token sequences. Blocks are reference-counted and freed only when all
// referencing requests complete.
//
// Note: the current implementation uses a hash of the full token sequence as
// the cache key (via HashTokens). This means only requests with exactly the
// same input tokens get cache hits. True prefix sharing (e.g., a common system
// prompt followed by different user messages) would require trie-based lookup
// and is not yet implemented.
//
// Usage:
//  1. Hash the token sequence to get a key.
//  2. Call Lookup to check if the sequence is cached.
//  3. If cached: call Ref to increment reference counts on the shared blocks.
//  4. If not cached: compute normally, then call Store.
//  5. When a request completes: call Unref to decrement reference counts.
type PrefixCache struct {
	mu sync.Mutex

	// prefixes maps content hash → cached prefix entry.
	prefixes map[[32]byte]*prefixEntry

	// refCounts tracks reference counts per physical block index.
	// A block with refCount > 0 must not be freed by the BlockManager.
	refCounts map[int]int

	// maxEntries is the maximum number of cached prefixes. When a new entry
	// is stored and the cache is at capacity, the least-recently-accessed
	// entry with no active users is evicted. 0 means unlimited.
	maxEntries int
}

type prefixEntry struct {
	blocks     []int     // physical block indices
	numTokens  int       // number of tokens in the prefix
	lastAccess time.Time // last time this entry was accessed (Store or Lookup)
}

// NewPrefixCache creates an empty prefix cache.
// maxEntries limits the number of cached prefixes (0 = unlimited).
func NewPrefixCache(maxEntries int) *PrefixCache {
	return &PrefixCache{
		prefixes:   make(map[[32]byte]*prefixEntry),
		refCounts:  make(map[int]int),
		maxEntries: maxEntries,
	}
}

// HashTokens produces a content hash for a token sequence.
func HashTokens(tokens []int32) [32]byte {
	h := sha256.New()
	var buf [4]byte
	for _, t := range tokens {
		binary.LittleEndian.PutUint32(buf[:], uint32(t))
		h.Write(buf[:])
	}
	var result [32]byte
	h.Sum(result[:0])
	return result
}

// Lookup checks if the given prefix hash has cached blocks.
// Returns the physical block indices and number of tokens if found.
func (pc *PrefixCache) Lookup(hash [32]byte) (blocks []int, numTokens int, ok bool) {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	entry, found := pc.prefixes[hash]
	if !found {
		return nil, 0, false
	}

	entry.lastAccess = time.Now()

	blocks = make([]int, len(entry.blocks))
	copy(blocks, entry.blocks)
	return blocks, entry.numTokens, true
}

// Store registers a computed prefix in the cache. The caller should have
// already written the KV entries to the given physical blocks.
// Initial reference count for each block is set to 1.
//
// If the cache is at capacity, the least-recently-accessed entry with no
// active users is evicted. Returns the freed blocks (if any) so the caller
// can recycle them via BlockManager.RecycleBlocks.
func (pc *PrefixCache) Store(hash [32]byte, blocks []int, numTokens int) []int {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	if _, exists := pc.prefixes[hash]; exists {
		return nil // already stored
	}

	// Evict if at capacity.
	var freed []int
	if pc.maxEntries > 0 && len(pc.prefixes) >= pc.maxEntries {
		freed = pc.evictLRULocked()
	}

	blocksCopy := make([]int, len(blocks))
	copy(blocksCopy, blocks)

	pc.prefixes[hash] = &prefixEntry{
		blocks:     blocksCopy,
		numTokens:  numTokens,
		lastAccess: time.Now(),
	}

	for _, b := range blocksCopy {
		pc.refCounts[b]++
	}
	return freed
}

// LookupAndRef atomically looks up a prefix and increments reference counts
// on its blocks. This avoids a TOCTOU race between Lookup and Ref where a
// concurrent Store could evict the entry between the two calls.
func (pc *PrefixCache) LookupAndRef(hash [32]byte) (blocks []int, numTokens int, ok bool) {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	entry, found := pc.prefixes[hash]
	if !found {
		return nil, 0, false
	}

	entry.lastAccess = time.Now()

	blocks = make([]int, len(entry.blocks))
	copy(blocks, entry.blocks)
	for _, b := range blocks {
		pc.refCounts[b]++
	}
	return blocks, entry.numTokens, true
}

// Ref increments reference counts on the given blocks (called when a new
// request reuses a cached prefix).
func (pc *PrefixCache) Ref(blocks []int) {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	for _, b := range blocks {
		pc.refCounts[b]++
	}
}

// Unref decrements reference counts. Returns any blocks whose count dropped
// to zero (the caller should free them via the BlockManager).
func (pc *PrefixCache) Unref(blocks []int) []int {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	return pc.decrementBlocksLocked(blocks)
}

// decrementBlocksLocked decrements reference counts for the given blocks.
// Returns blocks whose count dropped to zero. Caller must hold pc.mu.
func (pc *PrefixCache) decrementBlocksLocked(blocks []int) []int {
	var freed []int
	for _, b := range blocks {
		count, ok := pc.refCounts[b]
		if !ok || count <= 0 {
			continue // already freed or never tracked — skip to avoid underflow
		}
		pc.refCounts[b] = count - 1
		if count-1 <= 0 {
			delete(pc.refCounts, b)
			freed = append(freed, b)
		}
	}
	return freed
}

// IsBlockReferenced returns true if the block has a positive reference count.
func (pc *PrefixCache) IsBlockReferenced(block int) bool {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	return pc.refCounts[block] > 0
}

// Evict removes a prefix entry from the cache and decrements reference counts.
// Returns blocks whose count dropped to zero.
func (pc *PrefixCache) Evict(hash [32]byte) []int {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	entry, exists := pc.prefixes[hash]
	if !exists {
		return nil
	}

	delete(pc.prefixes, hash)
	return pc.decrementBlocksLocked(entry.blocks)
}

// NumEntries returns the number of cached prefixes.
func (pc *PrefixCache) NumEntries() int {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	return len(pc.prefixes)
}

// EvictLRU evicts the least-recently-accessed prefix entry that has no active
// users (all blocks have refcount == 1, meaning only the cache's own reference).
// Returns freed blocks so the caller can recycle them via BlockManager.RecycleBlocks.
// Returns nil if no evictable entry exists.
func (pc *PrefixCache) EvictLRU() []int {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	return pc.evictLRULocked()
}

// evictLRULocked finds the least-recently-accessed entry where all blocks have
// refcount == 1 (no active request users), removes it, and returns freed blocks.
// Caller must hold pc.mu.
func (pc *PrefixCache) evictLRULocked() []int {
	var oldestHash [32]byte
	var oldestEntry *prefixEntry
	found := false

	for hash, entry := range pc.prefixes {
		if pc.hasActiveUsersLocked(entry.blocks) {
			continue
		}
		if !found || entry.lastAccess.Before(oldestEntry.lastAccess) {
			oldestHash = hash
			oldestEntry = entry
			found = true
		}
	}

	if !found {
		return nil
	}

	delete(pc.prefixes, oldestHash)
	return pc.decrementBlocksLocked(oldestEntry.blocks)
}

// hasActiveUsersLocked returns true if any block has refcount > 1,
// meaning an active request is using the block beyond the cache's own reference.
// Caller must hold pc.mu.
func (pc *PrefixCache) hasActiveUsersLocked(blocks []int) bool {
	for _, b := range blocks {
		if pc.refCounts[b] > 1 {
			return true
		}
	}
	return false
}
