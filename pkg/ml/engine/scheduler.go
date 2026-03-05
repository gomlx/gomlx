package engine

import (
	"errors"
	"math/bits"
	"sync"
)

var (
	// ErrNoFreeSlots is returned when all KV cache batch slots are occupied.
	ErrNoFreeSlots = errors.New("no free KV cache slots")
)

// slotManager manages KV cache batch slots. The KV cache has shape
// [maxBatchSize, numKVHeads, maxSeqLen, headDim]; each request gets
// a slot index into the batch dimension. When the request completes
// the slot is freed for reuse.
type slotManager struct {
	mu        sync.Mutex
	maxSlots  int
	freeSlots []int // stack of free slot indices
}

func newSlotManager(maxBatchSize int) *slotManager {
	free := make([]int, maxBatchSize)
	for i := range maxBatchSize {
		free[i] = maxBatchSize - 1 - i // stack: pop gives lowest indices first
	}
	return &slotManager{
		maxSlots:  maxBatchSize,
		freeSlots: free,
	}
}

// Allocate assigns a slot to the given request ID. Returns the slot index.
func (sm *slotManager) Allocate(reqID uint64) (int, error) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if len(sm.freeSlots) == 0 {
		return -1, ErrNoFreeSlots
	}

	// Pop from stack.
	slot := sm.freeSlots[len(sm.freeSlots)-1]
	sm.freeSlots = sm.freeSlots[:len(sm.freeSlots)-1]
	return slot, nil
}

// Free returns a slot to the pool.
func (sm *slotManager) Free(slot int) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.freeSlots = append(sm.freeSlots, slot)
}

// NumFree returns the number of available slots.
func (sm *slotManager) NumFree() int {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	return len(sm.freeSlots)
}

// batch represents a group of requests to process in one forward pass.
type batch struct {
	requests  []*engineRequest
	positions []int32 // per-element positions, len == len(requests)
}

// paddedBatchSize rounds n up to the next power of 2 (minimum 1).
// This limits the number of unique compiled graphs to O(log(maxBatch)).
func paddedBatchSize(n int) int {
	if n <= 1 {
		return 1
	}
	// Round up to next power of 2.
	return 1 << bits.Len(uint(n-1))
}

// scheduler forms batches from active requests. It separates prefill
// (position == 0) and decode (position > 0) requests into different
// batches since they have different sequence lengths.
type scheduler struct {
	maxBatchSize int
}

func newScheduler(maxBatchSize int) *scheduler {
	return &scheduler{maxBatchSize: maxBatchSize}
}

// FormDecodeBatch selects up to maxBatchSize decode-phase requests
// (those with position > 0 and not finished) and returns them as a batch.
func (s *scheduler) FormDecodeBatch(requests []*engineRequest) *batch {
	var b batch
	for _, req := range requests {
		if req.eosReached || req.position == 0 {
			continue
		}
		if len(b.requests) >= s.maxBatchSize {
			break
		}
		b.requests = append(b.requests, req)
		b.positions = append(b.positions, int32(req.position))
	}
	if len(b.requests) == 0 {
		return nil
	}
	return &b
}

// NextPrefillRequest returns the first request that needs prefill
// (position == 0, not finished). Prefills are done one at a time
// because prompt lengths vary, requiring different graph shapes.
func (s *scheduler) NextPrefillRequest(requests []*engineRequest) *engineRequest {
	for _, req := range requests {
		if !req.eosReached && req.position == 0 {
			return req
		}
	}
	return nil
}
