// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn_test

import (
	"encoding/binary"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/float16"
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

		if !backend.Capabilities().Operations[compute.OpTypeQuantizedGather] {
			return
		}

		// Dynamic [batch, seq, 1] execution on backends with fused QuantizedGather.
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

func TestQuantizedGather_Linear_And_NF4(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		t.Run("NilQuant_Int8", func(t *testing.T) {
			tableData := [][]int8{{1, -2, 3}, {-4, 5, -6}}
			for _, fusion := range []string{"true", "false"} {
				t.Run("fusion="+fusion, func(t *testing.T) {
					t.Setenv(FusionEnv, fusion)
					graphtest.RunTestGraphFnWithBackend(t, "int8", backend, func(g *Graph) ([]*Node, []*Node) {
						table := Const(g, tableData)
						y := nn.QuantizedGather(table, Const(g, [][]int32{{1}, {0}}), nil)
						return []*Node{table}, []*Node{y}
					}, []any{[][]int8{{-4, 5, -6}, {1, -2, 3}}}, 0)
				})
			}
		})

		// Uint8 / Int8 block-linear tests run on every official backend (including XLA/ONNX via decomposed fallback).
		t.Run("Int8_NonDivisibleBlock_Dynamic", func(t *testing.T) {
			tableData := [][]int8{
				{2, 3, 4, 5, 6, 7},
				{3, 4, 5, 6, 7, 8},
			}
			scalesData := [][]float32{
				{0.5, 2.0},
				{1.5, 0.25},
			}
			zpData := [][]int8{
				{1, 2},
				{3, 4},
			}
			want0 := []float32{0.5, 1.0, 1.5, 2.0, 8.0, 10.0}
			want1 := []float32{0.0, 1.5, 3.0, 4.5, 0.75, 1.0}

			for _, fusion := range []string{"true", "false"} {
				t.Run("fusion="+fusion, func(t *testing.T) {
					t.Setenv(FusionEnv, fusion)
					graphtest.RunTestGraphFnWithBackend(t, "static", backend, func(g *Graph) ([]*Node, []*Node) {
						table := Const(g, tableData)
						scales := Const(g, scalesData)
						zp := Const(g, zpData)
						indices := Const(g, [][][]int32{{{1}, {0}}})
						y := nn.QuantizedGather(table, indices, &Quantization{
							Scheme:    compute.QuantLinear,
							Scale:     scales,
							ZeroPoint: zp,
							BlockAxis: 1,
							BlockSize: 4,
						})
						return []*Node{table}, []*Node{y}
					}, []any{[][][]float32{{want1, want0}}}, 1e-5)

					if !backend.Capabilities().DynamicAxes {
						return
					}
					exec := MustNewExec(backend, func(indices *Node) *Node {
						g := indices.Graph()
						return nn.QuantizedGather(Const(g, tableData), indices, &Quantization{
							Scheme:    compute.QuantLinear,
							Scale:     Const(g, scalesData),
							ZeroPoint: Const(g, zpData),
							BlockAxis: 1,
							BlockSize: 4,
						})
					}).WithDynamicAxes([]string{"batch", "seq", ""})
					defer exec.Finalize()

					got := exec.MustExec1([][][]int32{{{1}, {0}}})
					defer got.FinalizeAll()
					vals := got.Value().([][][]float32)
					assert.InDeltaSlice(t, want1, vals[0][0], 1e-5)
					assert.InDeltaSlice(t, want0, vals[0][1], 1e-5)
				})
			}
		})

		t.Run("PackedUint8_QuantNone_Int2_Dynamic", func(t *testing.T) {
			// Signed 2-bit values [-2, -1, 0, 1] packed in Uint8[2, 1]:
			// Row 0: {0, 1, -2, -1} -> 0xE4
			// Row 1: {-1, 0, 1, -2} -> 0x93
			packedW := [][]uint8{{0xE4}, {0x93}}
			want0 := []int8{0, 1, -2, -1}
			want1 := []int8{-1, 0, 1, -2}

			for _, fusion := range []string{"true", "false"} {
				t.Run("fusion="+fusion, func(t *testing.T) {
					t.Setenv(FusionEnv, fusion)
					graphtest.RunTestGraphFnWithBackend(t, "static", backend, func(g *Graph) ([]*Node, []*Node) {
						w := Const(g, packedW)
						y := nn.QuantizedGather(w, Const(g, [][]int32{{1}, {0}}), &Quantization{
							Scheme:      compute.QuantNone,
							PackedDType: dtypes.Int2,
						})
						return []*Node{w}, []*Node{y}
					}, []any{[][]int8{want1, want0}}, 0)

					if !backend.Capabilities().DynamicAxes {
						return
					}
					exec := MustNewExec(backend, func(indices *Node) *Node {
						g := indices.Graph()
						return nn.QuantizedGather(Const(g, packedW), indices, &Quantization{
							Scheme:      compute.QuantNone,
							PackedDType: dtypes.Int2,
						})
					}).WithDynamicAxes([]string{"batch", "seq", ""})
					defer exec.Finalize()

					got := exec.MustExec1([][][]int32{{{1}, {0}}})
					defer got.FinalizeAll()
					require.Equal(t, dtypes.Int8, got.DType())
					vals := got.Value().([][][]int8)
					assert.Equal(t, want1, vals[0][0])
					assert.Equal(t, want0, vals[0][1])
				})
			}
		})

		t.Run("PackedUint8_QuantLinear_Uint4", func(t *testing.T) {
			packedW := [][]uint8{
				{0x32, 0x54, 0x76, 0x98}, // {2,3,4,5,6,7,8,9}
				{0x43, 0x65, 0x87, 0xA9}, // {3,4,5,6,7,8,9,10}
			}
			zp := [][]uint8{{1, 2}, {3, 4}}
			scales := [][]float32{{0.5, 2.0}, {1.5, 0.25}}
			want0 := []float32{0.5, 1, 1.5, 2, 8, 10, 12, 14}
			want1 := []float32{0, 1.5, 3, 4.5, 0.75, 1, 1.25, 1.5}

			for _, fusion := range []string{"true", "false"} {
				t.Run("fusion="+fusion, func(t *testing.T) {
					t.Setenv(FusionEnv, fusion)
					graphtest.RunTestGraphFnWithBackend(t, "packed_linear", backend, func(g *Graph) ([]*Node, []*Node) {
						w := Const(g, packedW)
						y := nn.QuantizedGather(w, Const(g, [][]int32{{1}, {0}}), &Quantization{
							Scheme:      compute.QuantLinear,
							Scale:       Const(g, scales),
							ZeroPoint:   Const(g, zp),
							BlockAxis:   1,
							BlockSize:   4,
							PackedDType: dtypes.Uint4,
						})
						return []*Node{w}, []*Node{y}
					}, []any{[][]float32{want1, want0}}, 1e-5)
				})
			}
		})

		t.Run("PackedUint8_QuantNF4", func(t *testing.T) {
			packedW := [][]uint8{
				{0x70, 0x3F}, // {0, 7, 15, 3}
				{0xA2, 0xE5}, // {2, 10, 5, 14}
			}
			scales := [][]float32{{0.5, 2.0}, {1.0, 0.25}}
			want0 := []float32{
				compute.NF4LookupTable[0] * 0.5,
				compute.NF4LookupTable[7] * 0.5,
				compute.NF4LookupTable[15] * 2.0,
				compute.NF4LookupTable[3] * 2.0,
			}
			want1 := []float32{
				compute.NF4LookupTable[2] * 1.0,
				compute.NF4LookupTable[10] * 1.0,
				compute.NF4LookupTable[5] * 0.25,
				compute.NF4LookupTable[14] * 0.25,
			}
			for _, fusion := range []string{"true", "false"} {
				t.Run("fusion="+fusion, func(t *testing.T) {
					t.Setenv(FusionEnv, fusion)
					graphtest.RunTestGraphFnWithBackend(t, "packed_nf4", backend, func(g *Graph) ([]*Node, []*Node) {
						w := Const(g, packedW)
						y := nn.QuantizedGather(w, Const(g, [][]int32{{1}, {0}}), &Quantization{
							Scheme:      compute.QuantNF4,
							Scale:       Const(g, scales),
							BlockAxis:   1,
							BlockSize:   2,
							PackedDType: dtypes.Uint4,
						})
						return []*Node{w}, []*Node{y}
					}, []any{[][]float32{want1, want0}}, 1e-5)
				})
			}
		})

		if !backend.Capabilities().DTypes[dtypes.Uint4] {
			return
		}

		// Sub-byte Uint4 / Int2 / Uint2 tests (both fused and GOMLX_FUSION=false decomposed).
		t.Run("NilQuant_Int2_Dynamic", func(t *testing.T) {
			// Signed 2-bit values [-2, -1, 0, 1] packed 4 per byte.
			// Row 0: {0, 1, -2, -1} -> 0b11_10_01_00 = 0xE4
			// Row 1: {-1, 0, 1, -2} -> 0b10_01_00_11 = 0x93
			packedW := [][]uint8{{0xE4}, {0x93}}
			want0 := []int8{0, 1, -2, -1}
			want1 := []int8{-1, 0, 1, -2}

			for _, fusion := range []string{"true", "false"} {
				t.Run("fusion="+fusion, func(t *testing.T) {
					t.Setenv(FusionEnv, fusion)
					exec := MustNewExec(backend, func(indices *Node) *Node {
						g := indices.Graph()
						w := Reshape(Bitcast(Const(g, packedW), dtypes.Int2), 2, 4)
						return nn.QuantizedGather(w, indices, nil)
					}).WithDynamicAxes([]string{"batch", "seq", ""})
					defer exec.Finalize()

					got := exec.MustExec1([][][]int32{{{1}, {0}}})
					defer got.FinalizeAll()
					require.Equal(t, dtypes.Int8, got.DType())
					vals := got.Value().([][][]int8)
					assert.Equal(t, want1, vals[0][0])
					assert.Equal(t, want0, vals[0][1])
				})
			}
		})

		t.Run("NilQuant_Uint4_NonByteAligned", func(t *testing.T) {
			// K=3 (not divisible by 2): [2, 3] reshaped from packed Uint8 [2, 2] via Slice after ConvertDType.
			packedW := [][]uint8{
				{0x32, 0x05}, // {2, 3, 5, 0}
				{0x76, 0x09}, // {6, 7, 9, 0}
			}
			for _, fusion := range []string{"true", "false"} {
				t.Run("fusion="+fusion, func(t *testing.T) {
					t.Setenv(FusionEnv, fusion)
					graphtest.RunTestGraphFnWithBackend(t, "non_aligned", backend, func(g *Graph) ([]*Node, []*Node) {
						w := Reshape(Bitcast(Const(g, packedW), dtypes.Uint4), 2, 4)
						y := nn.QuantizedGather(w, Const(g, [][]int32{{1}, {0}}), nil)
						return []*Node{w}, []*Node{y}
					}, []any{[][]uint8{{6, 7, 9, 0}, {2, 3, 5, 0}}}, 0)
				})
			}
		})

		t.Run("Uint4_BlockLinear_Float16Scale", func(t *testing.T) {
			packedW := [][]uint8{
				{0x32, 0x54, 0x76, 0x98}, // {2,3,4,5,6,7,8,9}
				{0x43, 0x65, 0x87, 0xA9}, // {3,4,5,6,7,8,9,10}
			}
			packedZP := [][]uint8{
				{0x21}, // {1, 2}
				{0x43}, // {3, 4}
			}
			scalesData := [][]float16.Float16{
				{float16.FromFloat32(0.5), float16.FromFloat32(2.0)},
				{float16.FromFloat32(1.5), float16.FromFloat32(0.25)},
			}
			want0 := []float32{0.5, 1, 1.5, 2, 8, 10, 12, 14}
			want1 := []float32{0, 1.5, 3, 4.5, 0.75, 1, 1.25, 1.5}

			for _, fusion := range []string{"true", "false"} {
				t.Run("fusion="+fusion, func(t *testing.T) {
					t.Setenv(FusionEnv, fusion)
					exec := MustNewExec(backend, func(indices *Node) *Node {
						g := indices.Graph()
						w := Reshape(Bitcast(Const(g, packedW), dtypes.Uint4), 2, 8)
						zp := Reshape(Bitcast(Const(g, packedZP), dtypes.Uint4), 2, 2)
						return nn.QuantizedGather(w, indices, &Quantization{
							Scheme:    compute.QuantLinear,
							Scale:     Const(g, scalesData),
							ZeroPoint: zp,
							BlockAxis: 1,
							BlockSize: 4,
						})
					}).WithDynamicAxes([]string{"batch", "seq", ""})
					defer exec.Finalize()

					got := exec.MustExec1([][][]int64{{{1}, {0}}})
					defer got.FinalizeAll()
					require.Equal(t, dtypes.Float16, got.DType())
					raw := got.Value().([][][]float16.Float16)
					got0 := make([]float32, 8)
					got1 := make([]float32, 8)
					for i := range 8 {
						got1[i] = raw[0][0][i].Float32()
						got0[i] = raw[0][1][i].Float32()
					}
					assert.InDeltaSlice(t, want1, got1, 1e-3)
					assert.InDeltaSlice(t, want0, got0, 1e-3)
				})
			}
		})

		t.Run("NF4_Uint4", func(t *testing.T) {
			packedW := [][]uint8{
				{0x70, 0x3F}, // {0, 7, 15, 3}
				{0xA2, 0xE5}, // {2, 10, 5, 14}
			}
			scalesData := [][]float32{{0.5, 2.0}, {1.0, 0.25}}
			want0 := []float32{
				compute.NF4LookupTable[0] * 0.5,
				compute.NF4LookupTable[7] * 0.5,
				compute.NF4LookupTable[15] * 2.0,
				compute.NF4LookupTable[3] * 2.0,
			}
			want1 := []float32{
				compute.NF4LookupTable[2] * 1.0,
				compute.NF4LookupTable[10] * 1.0,
				compute.NF4LookupTable[5] * 0.25,
				compute.NF4LookupTable[14] * 0.25,
			}
			for _, fusion := range []string{"true", "false"} {
				t.Run("fusion="+fusion, func(t *testing.T) {
					t.Setenv(FusionEnv, fusion)
					graphtest.RunTestGraphFnWithBackend(t, "nf4", backend, func(g *Graph) ([]*Node, []*Node) {
						w := Reshape(Bitcast(Const(g, packedW), dtypes.Uint4), 2, 4)
						y := nn.QuantizedGather(w, Const(g, [][]int32{{1}, {0}}), &Quantization{
							Scheme:    compute.QuantNF4,
							Scale:     Const(g, scalesData),
							BlockAxis: 1,
							BlockSize: 2,
						})
						return []*Node{w}, []*Node{y}
					}, []any{[][]float32{want1, want0}}, 1e-5)
				})
			}
		})
	})
}
