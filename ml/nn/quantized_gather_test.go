// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn_test

import (
	"testing"

	"github.com/gomlx/compute"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/graphtest"
	"github.com/gomlx/gomlx/ml/nn"
	"github.com/gomlx/gomlx/support/testutil"
)

// TestQuantizedGather_GGML_Q8_0 tests QuantizedGather with Q8_0 block format.
// Q8_0 blocks: 2-byte fp16 scale + 32 int8 quants.
func TestQuantizedGather_GGML_Q8_0(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		// vocabSize=3, K=32 (one block per row).
		// Row 0: scale=2.0, quants=[1, 0, ...] → [2.0, 0, ...]
		// Row 1: scale=3.0, quants=[0, 1, ...] → [0, 3.0, ...]
		// Row 2: scale=1.0, quants=[1, 1, ...] → [1.0, 1.0, ...]
		const bytesPerRow = 34

		row0 := make([]uint8, bytesPerRow)
		row0[0], row0[1] = 0x00, 0x40 // fp16 LE for 2.0
		row0[2] = 1                   // quant[0] = 1

		row1 := make([]uint8, bytesPerRow)
		row1[0], row1[1] = 0x00, 0x42 // fp16 LE for 3.0
		row1[3] = 1                   // quant[1] = 1

		row2 := make([]uint8, bytesPerRow)
		row2[0], row2[1] = 0x00, 0x3C // fp16 LE for 1.0
		row2[2] = 1                   // quant[0] = 1
		row2[3] = 1                   // quant[1] = 1

		tableData := [][]uint8{row0, row1, row2}

		// Gather rows [2, 0]: expect [row2, row0].
		// row2 = [1.0, 1.0, 0, ...], row0 = [2.0, 0, ...]
		expected0 := make([]float32, 32)
		expected0[0] = 1.0
		expected0[1] = 1.0
		expected1 := make([]float32, 32)
		expected1[0] = 2.0

		graphtest.RunTestGraphFnWithBackend(t, "basic", backend, func(g *Graph) (inputs, outputs []*Node) {
			table := Const(g, tableData)
			indices := Const(g, [][]int32{{2}, {0}})
			quant := &Quantization{
				Scheme:   compute.QuantGGML,
				GGMLType: compute.GGMLQ8_0,
			}
			y := nn.QuantizedGather(table, indices, quant)
			return []*Node{table}, []*Node{y}
		}, []any{[][]float32{expected0, expected1}}, 1e-4)

		// Gather single row [1]: expect [row1] = [0, 3.0, 0, ...].
		expected2 := make([]float32, 32)
		expected2[1] = 3.0

		graphtest.RunTestGraphFnWithBackend(t, "single", backend, func(g *Graph) (inputs, outputs []*Node) {
			table := Const(g, tableData)
			indices := Const(g, [][]int32{{1}})
			quant := &Quantization{
				Scheme:   compute.QuantGGML,
				GGMLType: compute.GGMLQ8_0,
			}
			y := nn.QuantizedGather(table, indices, quant)
			return []*Node{table}, []*Node{y}
		}, []any{[][]float32{expected2}}, 1e-4)
	})
}

// TestQuantizedGather_GGML_Q4_0 tests QuantizedGather with Q4_0 block format.
// Q4_0 blocks: 2-byte fp16 scale + 16 nibble bytes.
func TestQuantizedGather_GGML_Q4_0(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		// vocabSize=2, K=32 (one block per row).
		// Row 0: scale=2.0, byte0 = 0x89 (lo=9→+1, hi=8→0), rest = 0x88
		//   dequant[0]=2*(9-8)=2.0, rest=0
		// Row 1: scale=3.0, byte0 = 0x88, byte1 = 0x89
		//   dequant[1]=3*(9-8)=3.0, rest=0
		const bytesPerRow = 18

		row0 := make([]uint8, bytesPerRow)
		row0[0], row0[1] = 0x00, 0x40 // fp16 LE for 2.0
		row0[2] = 0x89
		for i := 3; i < bytesPerRow; i++ {
			row0[i] = 0x88
		}

		row1 := make([]uint8, bytesPerRow)
		row1[0], row1[1] = 0x00, 0x42 // fp16 LE for 3.0
		row1[2] = 0x88
		row1[3] = 0x89
		for i := 4; i < bytesPerRow; i++ {
			row1[i] = 0x88
		}

		tableData := [][]uint8{row0, row1}

		// Gather rows [1, 0]: expect [row1, row0].
		expected0 := make([]float32, 32)
		expected0[1] = 3.0
		expected1 := make([]float32, 32)
		expected1[0] = 2.0

		graphtest.RunTestGraphFnWithBackend(t, "basic", backend, func(g *Graph) (inputs, outputs []*Node) {
			table := Const(g, tableData)
			indices := Const(g, [][]int32{{1}, {0}})
			quant := &Quantization{
				Scheme:   compute.QuantGGML,
				GGMLType: compute.GGMLQ4_0,
			}
			y := nn.QuantizedGather(table, indices, quant)
			return []*Node{table}, []*Node{y}
		}, []any{[][]float32{expected0, expected1}}, 1e-4)
	})
}

// TestQuantizedGather_GGML_IQ4NL tests QuantizedGather with IQ4_NL block format.
// Same layout as Q4_0, but nibbles are LUT indices. LUT[8]=1, LUT[9]=13.
func TestQuantizedGather_GGML_IQ4NL(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		// vocabSize=2, K=32 (one block per row).
		// Row 0: scale=1.0, byte0 = 0x89 (lo=9→LUT[9]=13, hi=8→LUT[8]=1), rest = 0x88
		//   dequant[0]=1*13=13, rest=1*1=1
		// Row 1: scale=2.0, byte0 = 0x88, byte1 = 0x89
		//   dequant[0]=2*1=2, dequant[1]=2*13=26, rest=2*1=2
		const bytesPerRow = 18

		row0 := make([]uint8, bytesPerRow)
		row0[0], row0[1] = 0x00, 0x3C // fp16 LE for 1.0
		row0[2] = 0x89
		for i := 3; i < bytesPerRow; i++ {
			row0[i] = 0x88
		}

		row1 := make([]uint8, bytesPerRow)
		row1[0], row1[1] = 0x00, 0x40 // fp16 LE for 2.0
		row1[2] = 0x88
		row1[3] = 0x89
		for i := 4; i < bytesPerRow; i++ {
			row1[i] = 0x88
		}

		tableData := [][]uint8{row0, row1}

		// Gather row [0]: dequant[0]=13, dequant[1..15]=1, dequant[16]=1, rest=1
		expected0 := make([]float32, 32)
		for i := range expected0 {
			expected0[i] = 1.0
		}
		expected0[0] = 13.0

		graphtest.RunTestGraphFnWithBackend(t, "basic", backend, func(g *Graph) (inputs, outputs []*Node) {
			table := Const(g, tableData)
			indices := Const(g, [][]int32{{0}})
			quant := &Quantization{
				Scheme:   compute.QuantGGML,
				GGMLType: compute.GGMLIQ4NL,
			}
			y := nn.QuantizedGather(table, indices, quant)
			return []*Node{table}, []*Node{y}
		}, []any{[][]float32{expected0}}, 1e-4)
	})
}
