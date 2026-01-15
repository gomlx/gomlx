// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package packgemm_test

import (
	"fmt"
	"slices"
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

	t.Run("kernel-rows-p1", func(t *testing.T) {
		contractingSize := packgemm.Float32Params.ContractingPanelSize + 1 // Make it larger than contracting panel size.
		lhsCrossSize := packgemm.Float32Params.LHSL1KernelRows + 1
		rhsCrossSize := 1
		batchSize := 1
		fmt.Printf("- C=AxB, shapes [1, %d, %d] x [1, %d, 1] -> [1, %d, 1]\n", lhsCrossSize, contractingSize, contractingSize, lhsCrossSize)

		// C = alpha * (A x B) + beta * C
		alpha := float32(1)
		beta := float32(3)
		Adata := xslices.Iota(float32(0), lhsCrossSize*contractingSize)
		Bdata := xslices.SliceWithValue(contractingSize, float32(1))
		Cdata := xslices.Iota(float32(1000), lhsCrossSize)
		want := slices.Clone(Cdata)
		base := float32(contractingSize*(contractingSize-1)) / 2
		rowIncrement := float32(contractingSize * contractingSize)
		for ii := range want {
			want[ii] *= beta
			want[ii] += alpha * (base + rowIncrement*float32(ii))
		}

		packgemm.Float32(alpha, beta, Adata, Bdata, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, Cdata,
			sequentialFloat32BufAllocFn, sequentialFloat32BufReleaseFn, sequentialWorkerPool)

		if slices.Compare(Cdata, want) != 0 {
			t.Errorf("Cdata = %v, want %v", Cdata, want)
		}
	})

	t.Run("kernel-cols-p1", func(t *testing.T) {
		contractingSize := packgemm.Float32Params.ContractingPanelSize + 1 // Make it larger than contracting panel size.
		lhsCrossSize := packgemm.Float32Params.LHSL1KernelRows + 1
		rhsCrossSize := packgemm.Float32Params.RHSL1KernelCols + 1
		batchSize := 1
		fmt.Printf("- C=AxB, shapes [1, %d, %d] x [1, %d, %d] -> [1, %d, %d]\n", lhsCrossSize, contractingSize, contractingSize, rhsCrossSize, lhsCrossSize, rhsCrossSize)

		// C = alpha * (A x B) + beta * C
		alpha := float32(1)
		beta := float32(3)
		Adata := xslices.Iota(float32(0), lhsCrossSize*contractingSize)
		Bdata := xslices.SliceWithValue(contractingSize*rhsCrossSize, float32(1))
		Cdata := xslices.Iota(float32(1000), lhsCrossSize*rhsCrossSize)
		want := slices.Clone(Cdata)
		base := float32(contractingSize*(contractingSize-1)) / 2
		rowIncrement := float32(contractingSize * contractingSize)
		for row := range lhsCrossSize {
			for col := range rhsCrossSize {
				idx := col + row*rhsCrossSize
				want[idx] *= beta
				want[idx] += alpha * (base + rowIncrement*float32(row))
			}
		}
		packgemm.Float32(alpha, beta, Adata, Bdata, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, Cdata,
			sequentialFloat32BufAllocFn, sequentialFloat32BufReleaseFn, sequentialWorkerPool)

		if slices.Compare(Cdata, want) != 0 {
			t.Errorf("Cdata = %v, want %v", Cdata, want)
		}
	})
}
