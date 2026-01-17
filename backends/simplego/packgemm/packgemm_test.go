// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package packgemm_test

import (
	"fmt"
	"slices"
	"testing"

	"github.com/gomlx/gomlx/backends/simplego/packgemm"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
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

func TestPackGemm(t *testing.T) {
	t.Run("Float32", func(t *testing.T) {
		gemmRegs := packgemm.DTypeToGEMM[packgemm.DTypePair{dtypes.Float32, dtypes.Float32}]
		if len(gemmRegs) == 0 {
			t.Fatal("No implmentation for Float32!?")
		}
		for _, reg := range gemmRegs {
			t.Run(reg.Name, func(t *testing.T) {
				gemmFn, ok := reg.GEMMFn.(func(alpha, beta float32, lhsFlat, rhsFlat []float32, batchSize,
					lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat []float32,
					bufAllocFn packgemm.BufAllocFn[float32], bufReleaseFn packgemm.BufReleaseFn, starter packgemm.GoroutineStarter) error)
				if !ok {
					t.Fatalf("Registered GEMM function invalid for Float32!? This is a bug, we got"+
						"instead %T as the registered function as %q", reg.GEMMFn, reg.Name)
				}
				params := reg.Params

				t.Run("large-contracting-size", func(t *testing.T) {
					contractingSize := params.PanelContractingSize + 1 // Make it larger than contracting panel size.
					batchSize, lhsCrossSize, rhsCrossSize := 1, 1, 1
					fmt.Printf("- C=AxB, shapes [1, 1, %d] x [1, %d, 1] -> [1, 1, 1]\n", contractingSize, contractingSize)

					// C = alpha * (A x B) + beta * C
					alpha := float32(1)
					beta := float32(3)
					Adata := xslices.Iota(float32(0), contractingSize)
					Bdata := xslices.SliceWithValue(contractingSize, float32(1))
					Cdata := []float32{1_000} // With beta==0, the 1_000 should be discarded.
					gemmFn(alpha, beta, Adata, Bdata, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, Cdata,
						sequentialFloat32BufAllocFn, sequentialFloat32BufReleaseFn, sequentialWorkerPool)
					want := 3*1_000 + float32(contractingSize*(contractingSize-1))/2
					if Cdata[0] != want {
						t.Errorf("Cdata[0] = %g, want %g", Cdata[0], want)
					}
				})

				t.Run("kernel-rows-p1", func(t *testing.T) {
					contractingSize := params.PanelContractingSize + 1 // Make it larger than contracting panel size.
					lhsCrossSize := params.LHSL1KernelRows + 1
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

					gemmFn(alpha, beta, Adata, Bdata, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, Cdata,
						sequentialFloat32BufAllocFn, sequentialFloat32BufReleaseFn, sequentialWorkerPool)

					if err := xslices.MustSlicesInRelData(Cdata, want, 1e-3); err != nil {
						t.Errorf("Cdata = %v, want %v, error: %+v", Cdata, want, err)
					}
				})

				t.Run("kernel-cols-p1", func(t *testing.T) {
					contractingSize := params.PanelContractingSize + 1 // Make it larger than contracting panel size.
					lhsCrossSize := params.LHSL1KernelRows + 1
					rhsCrossSize := params.RHSL1KernelCols + 1
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
					gemmFn(alpha, beta, Adata, Bdata, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, Cdata,
						sequentialFloat32BufAllocFn, sequentialFloat32BufReleaseFn, sequentialWorkerPool)

					if err := xslices.MustSlicesInRelData(Cdata, want, 1e-3); err != nil {
						t.Errorf("Cdata = %v, want %v, error: %+v", Cdata, want, err)
					}
				})
			})
		}
	})
}
