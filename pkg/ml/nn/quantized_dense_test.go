// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn_test

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	_ "github.com/gomlx/gomlx/backends/simplego"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/nn"
	"github.com/gomlx/gomlx/pkg/support/testutil"
)

func TestQuantizedDense_Int8(t *testing.T) {
	// Test Int8 QuantLinear across all official backends (both fused simplego and decomposed XLA paths).
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		// M=2, K=4, N=3, blockSize=3 → numBlocks=1, scales [4, 1].
		K, N, blockSize := 4, 3, 3

		xData := [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}}
		weightsData := [][]int8{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}}
		numBlocks := (N + blockSize - 1) / blockSize
		scalesData := make([][]float32, K)
		for k := range K {
			scalesData[k] = make([]float32, numBlocks)
			for g := range numBlocks {
				scalesData[k][g] = 1.0
			}
		}
		biasData := []float32{0.1, 0.2, 0.3}

		// Expected: x @ float32(weights) + bias
		// x[0]=[1,0,0,0] → y[0]=[1,2,3]+[0.1,0.2,0.3]=[1.1,2.2,3.3]
		// x[1]=[0,1,0,0] → y[1]=[4,5,6]+[0.1,0.2,0.3]=[4.1,5.2,6.3]
		graphtest.RunTestGraphFnWithBackend(t, "with_bias", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, xData)
			w := Const(g, weightsData)
			s := Const(g, scalesData)
			b := Const(g, biasData)
			quant := &Quantization{
				Scheme:    compute.QuantLinear,
				Scale:     s,
				BlockAxis: 1,
				BlockSize: blockSize,
			}
			y := nn.QuantizedDense(x, w, quant, b)
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{1.1, 2.2, 3.3}, {4.1, 5.2, 6.3}}}, 1e-4)

		graphtest.RunTestGraphFnWithBackend(t, "no_bias", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, xData)
			w := Const(g, weightsData)
			s := Const(g, scalesData)
			quant := &Quantization{
				Scheme:    compute.QuantLinear,
				Scale:     s,
				BlockAxis: 1,
				BlockSize: blockSize,
			}
			y := nn.QuantizedDense(x, w, quant, nil)
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{1, 2, 3}, {4, 5, 6}}}, 1e-4)

		// All output values are positive, so ReLU acts as identity.
		graphtest.RunTestGraphFnWithBackend(t, "with_relu", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, xData)
			w := Const(g, weightsData)
			s := Const(g, scalesData)
			b := Const(g, biasData)
			quant := &Quantization{
				Scheme:    compute.QuantLinear,
				Scale:     s,
				BlockAxis: 1,
				BlockSize: blockSize,
			}
			y := nn.QuantizedDense(x, w, quant, b, activations.TypeRelu)
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{1.1, 2.2, 3.3}, {4.1, 5.2, 6.3}}}, 1e-4)
	})
}

func TestQuantizedDense_NF4(t *testing.T) {
	// Sub-byte dtypes (Uint4) are only supported by the simplego backend; exclude XLA.
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		// M=1, K=2, N=4, blockSize=4 → numBlocks=1.
		//
		// Nibble indices (low nibble = even col, high nibble = odd col):
		//   row 0: [0, 15, 7, 7] → NF4: [-1.0, 1.0, 0.0, 0.0]
		//   row 1: [15, 0, 7, 7] → NF4: [ 1.0, -1.0, 0.0, 0.0]
		//
		// Packed bytes (low nibble first):
		//   row 0: byte 0: low=0,high=15 → 0xF0; byte 1: low=7,high=7 → 0x77
		//   row 1: byte 0: low=15,high=0 → 0x0F; byte 1: low=7,high=7 → 0x77
		N, blockSize := 4, 4
		packedData := [][]uint8{{0xF0, 0x77}, {0x0F, 0x77}}
		scalesData := [][]float32{{1.0}, {1.0}}
		xData := [][]float32{{1.0, 2.0}}

		// Expected: x @ dequant_weights
		// = 1*[-1, 1, 0, 0] + 2*[1, -1, 0, 0] = [1, -1, 0, 0]
		graphtest.RunTestGraphFnWithBackend(t, "basic", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, xData)
			packed := Const(g, packedData)
			s := Const(g, scalesData)
			// Bitcast uint8 [2, 2] → Uint4 [2, 2, 2] (packed: 2 nibbles per byte).
			// FusedQuantizedDense handles packed sub-byte weights internally.
			weights := Bitcast(packed, dtypes.Uint4) // [2, 2, 2]
			weights = Reshape(weights, 2, N)         // [2, 4]
			quant := &Quantization{
				Scheme:    compute.QuantNF4,
				Scale:     s,
				BlockAxis: 1,
				BlockSize: blockSize,
			}
			y := nn.QuantizedDense(x, weights, quant, nil)
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{1.0, -1.0, 0.0, 0.0}}}, 1e-4)
	}, "xla:cpu", "xla:cuda")
}

// TestQuantizedDense_MultiBlock exercises numBlocks > 1 with distinct per-row and per-block
// scale values. This verifies the block-index arithmetic in both the fused executor and the
// decomposed graph-level fallback.
func TestQuantizedDense_MultiBlock(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		// M=1, K=2, N=4, blockSize=2 → numBlocks=2, scales [2, 2].
		//
		// weights [K=2, N=4]:
		//   row 0: [1, 2, 3, 4]
		//   row 1: [5, 6, 7, 8]
		//
		// scales [K=2, numBlocks=2] — distinct per-row AND per-block:
		//   row 0: [2.0, 3.0]  (block 0 cols 0-1 → scale=2.0, block 1 cols 2-3 → scale=3.0)
		//   row 1: [4.0, 5.0]  (block 0 cols 0-1 → scale=4.0, block 1 cols 2-3 → scale=5.0)
		//
		// Dequantized weights: float = int * scale[k][n/blockSize]
		//   [0][0]=1*2=2   [0][1]=2*2=4   [0][2]=3*3=9    [0][3]=4*3=12
		//   [1][0]=5*4=20  [1][1]=6*4=24  [1][2]=7*5=35   [1][3]=8*5=40
		//
		// y = x @ dequant(weights), x=[1,1]:
		//   y = [2+20, 4+24, 9+35, 12+40] = [22, 28, 44, 52]
		K, N, blockSize := 2, 4, 2

		xData := [][]float32{{1.0, 1.0}}
		weightsData := [][]int8{{1, 2, 3, 4}, {5, 6, 7, 8}}
		scalesData := [][]float32{{2.0, 3.0}, {4.0, 5.0}}

		graphtest.RunTestGraphFnWithBackend(t, "int8_multi_block", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, xData)
			w := Const(g, weightsData)
			s := Const(g, scalesData)
			quant := &Quantization{
				Scheme:    compute.QuantLinear,
				Scale:     s,
				BlockAxis: 1,
				BlockSize: blockSize,
			}
			y := nn.QuantizedDense(x, w, quant, nil)
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{22.0, 28.0, 44.0, 52.0}}}, 1e-4)

		// Also test with bias to verify bias is added after the blocked matmul.
		biasData := []float32{0.1, 0.2, 0.3, 0.4}
		graphtest.RunTestGraphFnWithBackend(t, "int8_multi_block_with_bias", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, xData)
			w := Const(g, weightsData)
			s := Const(g, scalesData)
			b := Const(g, biasData)
			quant := &Quantization{
				Scheme:    compute.QuantLinear,
				Scale:     s,
				BlockAxis: 1,
				BlockSize: blockSize,
			}
			y := nn.QuantizedDense(x, w, quant, b)
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{22.1, 28.2, 44.3, 52.4}}}, 1e-4)

		// Test with multiple batch rows to verify M > 1 with multi-block.
		// x = [[1,0], [0,1]] → y[0] = [2,4,9,12], y[1] = [20,24,35,40]
		xData2 := [][]float32{{1.0, 0.0}, {0.0, 1.0}}
		graphtest.RunTestGraphFnWithBackend(t, "int8_multi_block_batch", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, xData2)
			w := Const(g, weightsData)
			s := Const(g, scalesData)
			quant := &Quantization{
				Scheme:    compute.QuantLinear,
				Scale:     s,
				BlockAxis: 1,
				BlockSize: blockSize,
			}
			y := nn.QuantizedDense(x, w, quant, nil)
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{2.0, 4.0, 9.0, 12.0}, {20.0, 24.0, 35.0, 40.0}}}, 1e-4)
		_ = K
		_ = N
	})
}

// TestQuantizedDense_ZeroPoint exercises asymmetric quantization with non-nil ZeroPoint,
// testing both the fused executor and decomposed graph-level fallback.
func TestQuantizedDense_ZeroPoint(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		// M=1, K=2, N=2, blockSize=2 → numBlocks=1, scales [2,1], zeroPoints [2,1].
		//
		// weights [2, 2] int8:
		//   row 0: [1, 2]
		//   row 1: [3, 4]
		//
		// scales [2, 1]:  row 0: [2.0], row 1: [3.0]
		// zeroPoints [2, 1]: row 0: [0.5], row 1: [-0.5]
		//
		// Dequantized: float = int * scale + zeroPoint
		//   [0][0] = 1*2 + 0.5 = 2.5    [0][1] = 2*2 + 0.5 = 4.5
		//   [1][0] = 3*3 + (-0.5) = 8.5  [1][1] = 4*3 + (-0.5) = 11.5
		//
		// x = [1, 1] → y = [2.5+8.5, 4.5+11.5] = [11, 16]
		blockSize := 2
		xData := [][]float32{{1.0, 1.0}}
		weightsData := [][]int8{{1, 2}, {3, 4}}
		scalesData := [][]float32{{2.0}, {3.0}}
		zpData := [][]float32{{0.5}, {-0.5}}

		graphtest.RunTestGraphFnWithBackend(t, "with_zero_point", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, xData)
			w := Const(g, weightsData)
			s := Const(g, scalesData)
			zp := Const(g, zpData)
			quant := &Quantization{
				Scheme:    compute.QuantLinear,
				Scale:     s,
				ZeroPoint: zp,
				BlockAxis: 1,
				BlockSize: blockSize,
			}
			y := nn.QuantizedDense(x, w, quant, nil)
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{11.0, 16.0}}}, 1e-4)

		// Same test with bias.
		biasData := []float32{0.1, 0.2}
		graphtest.RunTestGraphFnWithBackend(t, "with_zero_point_and_bias", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, xData)
			w := Const(g, weightsData)
			s := Const(g, scalesData)
			zp := Const(g, zpData)
			b := Const(g, biasData)
			quant := &Quantization{
				Scheme:    compute.QuantLinear,
				Scale:     s,
				ZeroPoint: zp,
				BlockAxis: 1,
				BlockSize: blockSize,
			}
			y := nn.QuantizedDense(x, w, quant, b)
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{11.1, 16.2}}}, 1e-4)
	})
}

// TestQuantizedDense_GGML_Q8_0 tests QuantGGML with Q8_0 block format.
// Q8_0 blocks: 2-byte fp16 scale + 32 int8 quants. dequant: output[i] = scale * int8(qs[i]).
// GGML is only supported by the simplego backend.
func TestQuantizedDense_GGML_Q8_0(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		// K=32 (one block), N=2.
		// Row 0: scale=2.0 (fp16 0x4000), quants=[1, 0, 0, ...] → dequant=[2.0, 0, 0, ...]
		// Row 1: scale=3.0 (fp16 0x4200), quants=[0, 1, 0, ...] → dequant=[0, 3.0, 0, ...]
		const bytesPerRow = 34 // one Q8_0 block

		row0 := make([]uint8, bytesPerRow)
		row0[0], row0[1] = 0x00, 0x40 // fp16 LE for 2.0
		row0[2] = 1                   // quant[0] = 1

		row1 := make([]uint8, bytesPerRow)
		row1[0], row1[1] = 0x00, 0x42 // fp16 LE for 3.0
		row1[3] = 1                   // quant[1] = 1

		weightData := [][]uint8{row0, row1}

		// x = [1, 1, 0, ..., 0] (K=32)
		xData := make([]float32, 32)
		xData[0] = 1.0
		xData[1] = 1.0

		// y[0] = 1*2.0 + 1*0 = 2.0
		// y[1] = 1*0 + 1*3.0 = 3.0
		graphtest.RunTestGraphFnWithBackend(t, "basic", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, [][]float32{xData})
			w := Const(g, weightData)
			quant := &Quantization{
				Scheme:   compute.QuantGGML,
				GGMLType: compute.GGMLQ8_0,
			}
			y := nn.QuantizedDense(x, w, quant, nil)
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{2.0, 3.0}}}, 1e-4)

		// Test with bias.
		graphtest.RunTestGraphFnWithBackend(t, "with_bias", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, [][]float32{xData})
			w := Const(g, weightData)
			b := Const(g, []float32{0.5, -0.5})
			quant := &Quantization{
				Scheme:   compute.QuantGGML,
				GGMLType: compute.GGMLQ8_0,
			}
			y := nn.QuantizedDense(x, w, quant, b)
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{2.5, 2.5}}}, 1e-4)

		// Test with batch (M=2).
		graphtest.RunTestGraphFnWithBackend(t, "batch", backend, func(g *Graph) (inputs, outputs []*Node) {
			xData2 := make([]float32, 32)
			xData2[0] = 0.5
			x := Const(g, [][]float32{xData, xData2})
			w := Const(g, weightData)
			quant := &Quantization{
				Scheme:   compute.QuantGGML,
				GGMLType: compute.GGMLQ8_0,
			}
			y := nn.QuantizedDense(x, w, quant, nil)
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{2.0, 3.0}, {1.0, 0.0}}}, 1e-4)
	})
}

// TestQuantizedDense_GGML_Q4_0 tests QuantGGML with Q4_0 block format.
// Q4_0 blocks: 2-byte fp16 scale + 16 nibble bytes.
// Split layout: low nibbles → first 16 values, high nibbles → last 16 values.
// dequant: output[j] = scale * (lo_nibble - 8), output[j+16] = scale * (hi_nibble - 8).
func TestQuantizedDense_GGML_Q4_0(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		// K=32 (one block), N=2.
		// Row 0: scale=2.0, nibble byte0 = lo=9,hi=8 → 0x89, rest = 0x88
		//   dequant[0] = 2*(9-8) = 2.0, all others = 2*(8-8) = 0
		// Row 1: scale=3.0, nibble byte0 = 0x88, byte1 = lo=9,hi=8 → 0x89, rest = 0x88
		//   dequant[1] = 3*(9-8) = 3.0, all others = 0
		const bytesPerRow = 18 // one Q4_0 block

		row0 := make([]uint8, bytesPerRow)
		row0[0], row0[1] = 0x00, 0x40 // fp16 LE for 2.0
		row0[2] = 0x89                // byte0: lo=9 (val=+1*scale), hi=8 (val=0)
		for i := 3; i < bytesPerRow; i++ {
			row0[i] = 0x88 // nibble 8 → (8-8)=0
		}

		row1 := make([]uint8, bytesPerRow)
		row1[0], row1[1] = 0x00, 0x42 // fp16 LE for 3.0
		row1[2] = 0x88                // byte0: both nibbles = 8 → 0
		row1[3] = 0x89                // byte1: lo=9 (val=+1*scale), hi=8
		for i := 4; i < bytesPerRow; i++ {
			row1[i] = 0x88
		}

		weightData := [][]uint8{row0, row1}

		// x = [1, 1, 0, ..., 0] (K=32)
		xData := make([]float32, 32)
		xData[0] = 1.0
		xData[1] = 1.0

		// y[0] = 1*2.0 + 1*0 = 2.0
		// y[1] = 1*0 + 1*3.0 = 3.0
		graphtest.RunTestGraphFnWithBackend(t, "basic", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, [][]float32{xData})
			w := Const(g, weightData)
			quant := &Quantization{
				Scheme:   compute.QuantGGML,
				GGMLType: compute.GGMLQ4_0,
			}
			y := nn.QuantizedDense(x, w, quant, nil)
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{2.0, 3.0}}}, 1e-4)

		// Test with activation (SiLU).
		graphtest.RunTestGraphFnWithBackend(t, "with_silu", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, [][]float32{xData})
			w := Const(g, weightData)
			quant := &Quantization{
				Scheme:   compute.QuantGGML,
				GGMLType: compute.GGMLQ4_0,
			}
			y := nn.QuantizedDense(x, w, quant, nil, activations.TypeSilu)
			return []*Node{x}, []*Node{y}
		}, []any{
			// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
			// SiLU(2.0) = 2 / (1 + exp(-2)) ≈ 1.7616
			// SiLU(3.0) = 3 / (1 + exp(-3)) ≈ 2.8577
			[][]float32{{1.7616, 2.8577}},
		}, 1e-3)
	})
}

// TestQuantizedDense_GGML_IQ4NL tests QuantGGML with IQ4_NL block format.
// Same layout as Q4_0 (2-byte fp16 scale + 16 nibble bytes), but nibbles
// are indices into a non-linear lookup table instead of linear (nibble - 8).
func TestQuantizedDense_GGML_IQ4NL(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		// K=32 (one block), N=2.
		// LUT[8]=1, LUT[9]=13 (from IQ4NLLookupTable).
		//
		// Row 0: scale=1.0, byte0 = 0x89 (lo=9→LUT[9]=13, hi=8→LUT[8]=1), rest = 0x88
		//   dequant[0]=13, dequant[1..15]=1, dequant[16]=1, dequant[17..31]=1
		// Row 1: scale=2.0, byte0 = 0x88, byte1 = 0x89 (lo=9→LUT[9]=13), rest = 0x88
		//   dequant[0]=2*1=2, dequant[1]=2*13=26, rest=2*1=2
		const bytesPerRow = 18

		row0 := make([]uint8, bytesPerRow)
		row0[0], row0[1] = 0x00, 0x3C // fp16 LE for 1.0
		row0[2] = 0x89                // lo=9 (LUT[9]=13), hi=8 (LUT[8]=1)
		for i := 3; i < bytesPerRow; i++ {
			row0[i] = 0x88 // nibble 8 → LUT[8]=1
		}

		row1 := make([]uint8, bytesPerRow)
		row1[0], row1[1] = 0x00, 0x40 // fp16 LE for 2.0
		row1[2] = 0x88
		row1[3] = 0x89 // byte1: lo=9 (LUT[9]=13), hi=8 (LUT[8]=1)
		for i := 4; i < bytesPerRow; i++ {
			row1[i] = 0x88
		}

		weightData := [][]uint8{row0, row1}

		// x = [1, 1, 0, ..., 0] (K=32)
		xData := make([]float32, 32)
		xData[0] = 1.0
		xData[1] = 1.0

		// y[0] = 1*13 + 1*1 = 14.0
		// y[1] = 1*2 + 1*26 = 28.0
		graphtest.RunTestGraphFnWithBackend(t, "basic", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, [][]float32{xData})
			w := Const(g, weightData)
			quant := &Quantization{
				Scheme:   compute.QuantGGML,
				GGMLType: compute.GGMLIQ4NL,
			}
			y := nn.QuantizedDense(x, w, quant, nil)
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{14.0, 28.0}}}, 1e-4)
	})
}

func TestQuantizedDense_Int4(t *testing.T) {
	// Sub-byte dtypes (Int4) are only supported by the simplego backend; exclude XLA.
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		// Same packed layout as the NF4 test, but QuantLinear dequant with Int4 weights.
		//
		// Packed bytes (low nibble first):
		//   row 0: byte 0: low=0,high=15 → 0xF0; byte 1: low=7,high=7 → 0x77
		//   row 1: byte 0: low=15,high=0 → 0x0F; byte 1: low=7,high=7 → 0x77
		//
		// After Bitcast to Int4 (sign-extended):
		//   row 0: [0, -1, 7, 7]  (nibble 0→0, nibble 15→-1, nibble 7→7)
		//   row 1: [-1, 0, 7, 7]  (nibble 15→-1, nibble 0→0, nibble 7→7)
		//
		// Note: Int4 sign-extends nibbles: 0-7 stay as-is, 8-15 map to -8 to -1.
		N, blockSize := 4, 4
		packedData := [][]uint8{{0xF0, 0x77}, {0x0F, 0x77}}
		scalesData := [][]float32{{1.0}, {1.0}}
		xData := [][]float32{{1.0, 2.0}}

		// Expected: x @ float32(int4_weights)
		// = 1*[0, -1, 7, 7] + 2*[-1, 0, 7, 7] = [-2, -1, 21, 21]
		graphtest.RunTestGraphFnWithBackend(t, "basic", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, xData)
			packed := Const(g, packedData)
			s := Const(g, scalesData)
			// Bitcast uint8 [2, 2] → Int4 [2, 2, 2], then reshape to [2, 4].
			weights := Bitcast(packed, dtypes.Int4) // [2, 2, 2]
			weights = Reshape(weights, 2, N)        // [2, 4]
			quant := &Quantization{
				Scheme:    compute.QuantLinear,
				Scale:     s,
				BlockAxis: 1,
				BlockSize: blockSize,
			}
			y := nn.QuantizedDense(x, weights, quant, nil)
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{-2.0, -1.0, 21.0, 21.0}}}, 1e-4)
	}, "xla:cpu", "xla:cuda")
}
