package engine

import (
	"sync"

	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// PreemptionPolicy defines how to handle memory pressure.
type PreemptionPolicy int

const (
	// PreemptSwap moves KV blocks to CPU memory for later restoration.
	PreemptSwap PreemptionPolicy = iota

	// PreemptRecompute discards KV blocks entirely; the request will
	// recompute them when resumed (more expensive but uses no CPU memory).
	PreemptRecompute
)

// preemptionManager handles swapping KV cache blocks to/from CPU memory
// when GPU memory is under pressure (no free blocks in the BlockManager).
type preemptionManager struct {
	mu     sync.Mutex
	policy PreemptionPolicy

	// swapped stores CPU-side copies of KV blocks, keyed by request ID.
	// Each entry contains the key and value tensors for the request's blocks.
	swapped map[uint64]*swappedEntry
}

type swappedEntry struct {
	// req is the full engine request, preserved for restoration.
	// Channels remain open while preempted; the client is blocked waiting.
	req *engineRequest

	// keyBlocks and valueBlocks are CPU-side copies of the KV cache blocks.
	// Each slice element corresponds to one physical block.
	keyBlocks   []*tensors.Tensor
	valueBlocks []*tensors.Tensor

	// pageTable records which physical blocks were used.
	pageTable []int

	// generatedTokens is the full generated token history.
	generatedTokens []int32
}

func newPreemptionManager(policy PreemptionPolicy) *preemptionManager {
	return &preemptionManager{
		policy:  policy,
		swapped: make(map[uint64]*swappedEntry),
	}
}

// Preempt saves the request's state and frees its GPU blocks.
// In swap mode, KV blocks are copied to CPU. In recompute mode,
// only the token history is saved (KV will be recomputed on resume).
func (pm *preemptionManager) Preempt(reqID uint64, req *engineRequest, pageTable []int) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	entry := &swappedEntry{
		req:             req,
		pageTable:       make([]int, len(pageTable)),
		generatedTokens: make([]int32, len(req.generatedTokens)),
	}
	copy(entry.pageTable, pageTable)
	copy(entry.generatedTokens, req.generatedTokens)

	// In swap mode, the caller is responsible for copying the actual
	// tensor data before freeing the blocks. This struct just records
	// the metadata for restoration.
	pm.swapped[reqID] = entry
}

// IsPreempted returns true if the request has been preempted.
func (pm *preemptionManager) IsPreempted(reqID uint64) bool {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	_, ok := pm.swapped[reqID]
	return ok
}

// Restore returns the saved state for a preempted request and removes it
// from the preemption manager. Returns nil if not found.
func (pm *preemptionManager) Restore(reqID uint64) *swappedEntry {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	entry, ok := pm.swapped[reqID]
	if !ok {
		return nil
	}
	delete(pm.swapped, reqID)
	return entry
}

// NumPreempted returns the number of currently preempted requests.
func (pm *preemptionManager) NumPreempted() int {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	return len(pm.swapped)
}

// PreemptedIDs returns the IDs of all currently preempted requests.
func (pm *preemptionManager) PreemptedIDs() []uint64 {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	ids := make([]uint64, 0, len(pm.swapped))
	for id := range pm.swapped {
		ids = append(ids, id)
	}
	return ids
}

// preemptLowestPriority selects the request with the lowest priority
// (currently: most recently submitted, i.e., highest ID) and preempts it.
// The victim is removed from the active request map and its slot is freed.
// Returns the preempted request ID, or 0 if no requests can be preempted.
//
// For recompute mode, the victim will re-prefill its full token history
// (original prompt + generated tokens) when restored.
func (e *Engine) preemptLowestPriority() uint64 {
	if e.preemptMgr == nil {
		return 0
	}

	victim, victimID := e.removeLowestPriorityRequest()
	if victim == nil {
		return 0
	}

	// Free the victim's KV cache slot.
	if e.batchedMode {
		e.slotMgr.Free(victim.slot)
	}

	// Save state and free blocks.
	if e.pagedMode {
		pageTable := e.blockMgr.GetPageTable(victimID)
		e.preemptMgr.Preempt(victimID, victim, pageTable)
		e.blockMgr.ReleaseRequest(victimID)
	}

	return victimID
}

// removeLowestPriorityRequest finds the active request with the highest ID
// (lowest priority), removes it from the request map, and returns it.
// Returns nil if no eligible request exists.
func (e *Engine) removeLowestPriorityRequest() (*engineRequest, uint64) {
	e.mu.Lock()
	defer e.mu.Unlock()

	var victim *engineRequest
	var victimID uint64
	for id, req := range e.requests {
		if req.eosReached || req.position == 0 {
			continue
		}
		if victim == nil || id > victimID {
			victim = req
			victimID = id
		}
	}
	if victim != nil {
		delete(e.requests, victimID)
	}
	return victim, victimID
}
