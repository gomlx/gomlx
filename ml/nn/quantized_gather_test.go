// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn_test

import (
	"encoding/binary"
	"testing"

	"github.com/gomlx/compute"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/graphtest"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/nn"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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

// TestQuantizedGather_Q5_0_And_Dynamic verifies Q5_0 quantized lookup and dynamic batch/seq shapes
// (regression test for InternalFusedOpCaller eager decomposed evaluation on dynamic shapes).
func TestQuantizedGather_Q5_0_And_Dynamic(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		const bytesPerRow = 22
		row0 := make([]uint8, bytesPerRow)
		row0[0], row0[1] = 0x00, 0x3C // d = 1.0
		// q0[0] = 17 (low nibble 1, high bit 1 -> +1.0), rest = 16 (low nibble 0, high bit 1 -> 0.0)
		binary.LittleEndian.PutUint32(row0[2:6], 0xFFFFFFFF)
		row0[6] = 0x01

		row1 := make([]uint8, bytesPerRow)
		row1[0], row1[1] = 0x00, 0x40 // d = 2.0
		// q0[16] = 18 (high nibble of byte 0 = 2, high bit 16 = 1 -> 2*(18-16) = 4.0), rest = 16 (-> 0.0)
		binary.LittleEndian.PutUint32(row1[2:6], 0xFFFFFFFF)
		row1[6] = 0x20

		tableData := [][]uint8{row0, row1}
		wantRow0 := make([]float32, 32)
		wantRow0[0] = 1.0
		wantRow1 := make([]float32, 32)
		wantRow1[16] = 4.0

		graphtest.RunTestGraphFnWithBackend(t, "Q5_0_static", backend, func(g *Graph) ([]*Node, []*Node) {
			table := Const(g, tableData)
			indices := Const(g, [][]int32{{1}, {0}})
			y := nn.QuantizedGather(table, indices, &Quantization{
				Scheme:   compute.QuantGGML,
				GGMLType: compute.GGMLQ5_0,
			})
			return []*Node{table}, []*Node{y}
		}, []any{[][]float32{wantRow1, wantRow0}}, 1e-4)

		if !backend.Capabilities().Operations[compute.OpTypeQuantizedEmbeddingLookup] {
			return
		}

		// Dynamic [batch, seq, 1] execution on backends with fused QuantizedEmbeddingLookup.
		var sawDynamic bool
		exec := MustNewExec(backend, func(indices *Node) *Node {
			sawDynamic = indices.Shape().IsDynamic()
			g := indices.Graph()
			table := Const(g, tableData)
			return nn.QuantizedGather(table, indices, &Quantization{
				Scheme:   compute.QuantGGML,
				GGMLType: compute.GGMLQ5_0,
			})
		}).WithDynamicAxes([]string{"batch", "seq", ""})

		inTensor := tensors.FromValue([][][]int32{{{0}, {1}}, {{1}, {0}}})
		outs, err := exec.Exec(inTensor)
		require.NoError(t, err)
		assert.True(t, sawDynamic, "expected indices parameter to have dynamic shape")
		require.Len(t, outs, 1)
		assert.Equal(t, []int{2, 2, 32}, outs[0].Shape().Dimensions)
		gotFlat, err := tensors.CopyFlatData[float32](outs[0])
		require.NoError(t, err)
		assert.InDeltaSlice(t, wantRow0, gotFlat[0:32], 1e-4)
		assert.InDeltaSlice(t, wantRow1, gotFlat[32:64], 1e-4)
		assert.InDeltaSlice(t, wantRow1, gotFlat[64:96], 1e-4)
		assert.InDeltaSlice(t, wantRow0, gotFlat[96:128], 1e-4)
	})
}
