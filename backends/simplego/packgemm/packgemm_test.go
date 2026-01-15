// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package packgemm_test

import (
	"fmt"
	"testing"

	"github.com/gomlx/gomlx/backends/simplego/packgemm"
	"github.com/gomlx/gomlx/pkg/support/xslices"
)

var (
	// Test closures used for allocating buffers and starting goroutines.
	float32PerSizeBufferPool    = make(map[int][]float32, 10)
	sequentialFloat32BufAllocFn = func(size int) (ref any, data []float32) {
		var found bool
		data, found = float32PerSizeBufferPool[size]
		if found {
			delete(float32PerSizeBufferPool, size)
			return data, data
		}
		data = make([]float32, size)
		return data, data
	}
	sequentialFloat32BufReleaseFn = func(ref any) {
		data := ref.([]float32)
		float32PerSizeBufferPool[len(data)] = data
	}
	sequentialWorkerPool = func(work func()) bool { return false }
)

func TestPackGemmFloat32(t *testing.T) {
	if packgemm.Float32 == nil {
		t.Skip("packgemm.Float32 not implemented for this architecture")
	}

	t.Run("large-contracting-size", func(t *testing.T) {
		contractingSize := packgemm.Float32Params.ContractingPanelSize + 1 // Make it larger than contracting panel size.
		batchSize, lhsCrossSize, rhsCrossSize := 1, 1, 1
		fmt.Printf("- C=AxB, shapes [1, 1, %d] x [1, %d, 1] -> [1, 1, 1]\n", contractingSize, contractingSize)

		// C = alpha * (A x B) + beta * C
		alpha := float32(1)
		beta := float32(3)
		Adata := xslices.Iota(float32(0), contractingSize)
		Bdata := xslices.SliceWithValue(contractingSize, float32(1))
		Cdata := []float32{1_000} // With beta==0, the 1_000 should be discarded.
		packgemm.Float32(alpha, beta, Adata, Bdata, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, Cdata,
			sequentialFloat32BufAllocFn, sequentialFloat32BufReleaseFn, sequentialWorkerPool)
		want := 3*1_000 + float32(contractingSize*(contractingSize-1))/2
		if Cdata[0] != want {
			t.Errorf("Cdata[0] = %g, want %g", Cdata[0], want)
		}
	})
}
