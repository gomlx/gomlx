package packgemm

import (
	"fmt"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestFeedWorkItems(t *testing.T) {
	// collectWorkItems runs feedWorkItems and collects the output channel into a slice.
	collectWorkItems := func(batchSize, lhsCrossSize, rhsCrossSize int, params *CacheParams, maxWorkers int) []workItem {
		ch := make(chan workItem, 100)
		feedWorkItems(batchSize, lhsCrossSize, rhsCrossSize, params, maxWorkers, ch)
		var got []workItem
		for w := range ch {
			got = append(got, w)
		}
		// Sort for deterministic comparison, although feedWorkItems should be deterministic.
		// It produces items in order, but let's just rely on that order.
		return got
	}

	params := &CacheParams{
		LHSPanelCrossSize: 4,
		RHSPanelCrossSize: 4,
	}

	tests := []struct {
		name                                  string
		batchSize, lhsCrossSize, rhsCrossSize int
		maxWorkers                            int
		want                                  []workItem
	}{
		{
			name:         "Only Batch Splitting",
			batchSize:    10,
			lhsCrossSize: 10, rhsCrossSize: 10,
			maxWorkers: 2,
			want: []workItem{
				{0, 5, 0, 10, 0, 10},
				{5, 10, 0, 10, 0, 10},
			},
		},
		{
			name:         "Mixed Splitting - Batch then LHS",
			batchSize:    2,
			lhsCrossSize: 16, rhsCrossSize: 4,
			maxWorkers: 2 + 2, // 2 for batch, remaining 2 for split
			// Logic:
			// batchSize (2) < 2*maxWorkers (8) -> condition false?
			// Wait: batchSize >= 2*maxWorkers check: 2 >= 8 is false.
			//
			// 1. First maxWorkers batches (here batchSize=2 < maxWorkers=4).
			//    So it emits batch 0 and batch 1 fully first?
			//    Let's re-read feedWorkItems logic.
			//    if batchSize >= maxWorkers:
			//       loops batchIdx from 0 to maxWorkers-1.
			//       BUT here batchSize=2, maxWorkers=4. So this loop doesn't run?
			//       Wait, `if batchSize >= maxWorkers` is false.
			//       So batchIdx stays 0.
			//
			//    remaining = 2 - 0 = 2.
			//    splitFactor = (4 + 2 - 1) / 2 = 5/2 = 2.
			//
			//    lhsCrossSize (16) > rhsCrossSize (4) -> Split LHS.
			//    lhsSplitSize = (16 + 2 - 1) / 2 = 17/2 = 8.
			//    Aligned to params.LHSPanelCrossSize (4): max(1, 8/4)*4 = 8.
			//
			//    Loop batchIdx 0 to 2:
			//      batch 0: lhs 0..8, 8..16
			//      batch 1: lhs 0..8, 8..16
			want: []workItem{
				{0, 1, 0, 8, 0, 4},
				{1, 2, 0, 8, 0, 4},
				{0, 1, 8, 16, 0, 4},
				{1, 2, 8, 16, 0, 4},
			},
		},
		{
			name:         "Mixed Splitting - Batch then RHS",
			batchSize:    2,
			lhsCrossSize: 4, rhsCrossSize: 16,
			maxWorkers: 4,
			// Same logic but RHS split.
			want: []workItem{
				{0, 1, 0, 4, 0, 8},
				{1, 2, 0, 4, 0, 8},
				{0, 1, 0, 4, 8, 16},
				{1, 2, 0, 4, 8, 16},
			},
		},
		{
			name:         "Exact maxWorkers match batchSize",
			batchSize:    4,
			lhsCrossSize: 10, rhsCrossSize: 10,
			maxWorkers: 4,
			// batchSize >= maxWorkers (4>=4) -> True.
			// Loop batchIdx 0..4 emits 4 items.
			// remaining = 0. Returns.
			want: []workItem{
				{0, 1, 0, 10, 0, 10},
				{1, 2, 0, 10, 0, 10},
				{2, 3, 0, 10, 0, 10},
				{3, 4, 0, 10, 0, 10},
			},
		},
		{
			name:         "LHS Splitting small batch",
			batchSize:    1,
			lhsCrossSize: 16, rhsCrossSize: 4,
			maxWorkers: 4,
			// batchIdx=0.
			// remaining=1.
			// splitFactor = (4+0)/1 = 4.
			// lhsSplitSize = (16+3)/4 = 4. Aligned 4.
			want: []workItem{
				{0, 1, 0, 4, 0, 4},
				{0, 1, 4, 8, 0, 4},
				{0, 1, 8, 12, 0, 4},
				{0, 1, 12, 16, 0, 4},
			},
		},
		{
			name:         "Uneven LHS Splitting",
			batchSize:    1,
			lhsCrossSize: 14, rhsCrossSize: 4,
			maxWorkers: 2,
			// splitFactor = 2.
			// lhsSplitSize = (14+1)/2 = 7.
			// params.LHSPanelCrossSize = 4.
			// max(1, 7/4) * 4 = 1*4 = 4.  Wait, 7/4 = 1.
			// So split size is 4.
			want: []workItem{
				{0, 1, 0, 4, 0, 4},
				{0, 1, 4, 8, 0, 4},
				{0, 1, 8, 12, 0, 4},
				{0, 1, 12, 14, 0, 4},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := collectWorkItems(tc.batchSize, tc.lhsCrossSize, tc.rhsCrossSize, params, tc.maxWorkers)
			if diff := cmp.Diff(tc.want, got, cmp.AllowUnexported(workItem{})); diff != "" {
				fmt.Printf("- Got: %+v\n", got)
				fmt.Printf("- Want: %+v\n", tc.want)
				t.Errorf("feedWorkItems() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// Ensure slices package is used (it might be already imported in packgemm.go but this file is separate compilation unit if internal test? No, same package).
var _ = slices.Clone[[]int]
