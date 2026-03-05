package serving

import (
	"context"
	"testing"
	"time"
)

func TestPreemptionManager(t *testing.T) {
	t.Run("PreemptAndRestore", func(t *testing.T) {
		pm := newPreemptionManager(PreemptSwap)

		req := &engineRequest{
			id:              1,
			position:        5,
			generatedTokens: []int32{10, 20, 30},
			ctx:             context.Background(),
			startTime:       time.Now(),
		}
		pageTable := []int{2, 4, 6}

		pm.Preempt(1, req, pageTable)
		if !pm.IsPreempted(1) {
			t.Error("Expected request 1 to be preempted")
		}
		if pm.NumPreempted() != 1 {
			t.Errorf("Expected 1 preempted, got %d", pm.NumPreempted())
		}

		entry := pm.Restore(1)
		if entry == nil {
			t.Fatal("Expected non-nil restored entry")
		}
		if len(entry.generatedTokens) != 3 {
			t.Errorf("Expected 3 tokens, got %d", len(entry.generatedTokens))
		}
		if len(entry.pageTable) != 3 {
			t.Errorf("Expected 3 page table entries, got %d", len(entry.pageTable))
		}

		// After restore, should no longer be preempted.
		if pm.IsPreempted(1) {
			t.Error("Should not be preempted after restore")
		}
	})

	t.Run("RestoreNonExistent", func(t *testing.T) {
		pm := newPreemptionManager(PreemptSwap)
		entry := pm.Restore(999)
		if entry != nil {
			t.Error("Expected nil for non-existent request")
		}
	})
}
