// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn_test

import (
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/nn"
)

func TestQuantizedDense_Int8(t *testing.T) {
	// Test Int8 QuantLinear across all official backends (both fused simplego and decomposed XLA paths).
	graphtest.TestOfficialBackends(t, func(t *testing.T, backend backends.Backend) {
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
				Scheme:    backends.QuantLinear,
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
				Scheme:    backends.QuantLinear,
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
				Scheme:    backends.QuantLinear,
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
	graphtest.TestOfficialBackends(t, func(t *testing.T, backend backends.Backend) {
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
			// Bitcast uint8 [2, 2] → Uint4 [2, 2, 2], then reshape to [2, 4].
			weights := Bitcast(packed, dtypes.Uint4)     // [2, 2, 2]
			weights = Reshape(weights, 2, N)              // [2, 4]
			quant := &Quantization{
				Scheme:    backends.QuantNF4,
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
	graphtest.TestOfficialBackends(t, func(t *testing.T, backend backends.Backend) {
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
				Scheme:    backends.QuantLinear,
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
				Scheme:    backends.QuantLinear,
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
				Scheme:    backends.QuantLinear,
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
	graphtest.TestOfficialBackends(t, func(t *testing.T, backend backends.Backend) {
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
				Scheme:    backends.QuantLinear,
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
				Scheme:    backends.QuantLinear,
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

func TestQuantizedDense_Int4(t *testing.T) {
	// Sub-byte dtypes (Int4) are only supported by the simplego backend; exclude XLA.
	graphtest.TestOfficialBackends(t, func(t *testing.T, backend backends.Backend) {
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
			weights := Bitcast(packed, dtypes.Int4)      // [2, 2, 2]
			weights = Reshape(weights, 2, N)              // [2, 4]
			quant := &Quantization{
				Scheme:    backends.QuantLinear,
				Scale:     s,
				BlockAxis: 1,
				BlockSize: blockSize,
			}
			y := nn.QuantizedDense(x, weights, quant, nil)
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{-2.0, -1.0, 21.0, 21.0}}}, 1e-4)
	}, "xla:cpu", "xla:cuda")
}
