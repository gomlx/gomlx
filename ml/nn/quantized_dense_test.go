// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/graphtest"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/ggml"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/nn"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestQuantizedDense_Int8(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
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

		graphtest.RunTestGraphFnWithBackend(t, "with_bias", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, xData)
			w := Const(g, weightsData)
			s := Const(g, scalesData)
			b := Const(g, biasData)
			cfg := QuantizedDenseConfig{
				WeightQuantization: &Quantization{
					Scheme:    compute.QuantLinear,
					Scale:     s,
					BlockAxis: 1,
					BlockSize: blockSize,
				},
			}
			y := nn.QuantizedDense(x, w, b, cfg)
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{1.1, 2.2, 3.3}, {4.1, 5.2, 6.3}}}, 1e-4)

		graphtest.RunTestGraphFnWithBackend(t, "no_bias", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, xData)
			w := Const(g, weightsData)
			s := Const(g, scalesData)
			cfg := QuantizedDenseConfig{
				WeightQuantization: &Quantization{
					Scheme:    compute.QuantLinear,
					Scale:     s,
					BlockAxis: 1,
					BlockSize: blockSize,
				},
			}
			y := nn.QuantizedDense(x, w, nil, cfg)
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{1, 2, 3}, {4, 5, 6}}}, 1e-4)

		graphtest.RunTestGraphFnWithBackend(t, "with_relu", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, xData)
			w := Const(g, weightsData)
			s := Const(g, scalesData)
			b := Const(g, biasData)
			cfg := QuantizedDenseConfig{
				WeightQuantization: &Quantization{
					Scheme:    compute.QuantLinear,
					Scale:     s,
					BlockAxis: 1,
					BlockSize: blockSize,
				},
			}
			y := nn.QuantizedDense(x, w, b, cfg, activation.TypeRelu)
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{1.1, 2.2, 3.3}, {4.1, 5.2, 6.3}}}, 1e-4)
	})
}

func TestQuantizedDense_NF4(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		if backend.Name() == "xla" {
			t.Skip("Skipping: XLA seems to have some bugs when supporting some sub-byte dtypes operations.")
		}

		N, blockSize := 4, 4
		packedData := [][]uint8{{0xF0, 0x77}, {0x0F, 0x77}}
		scalesData := [][]float32{{1.0}, {1.0}}
		xData := [][]float32{{1.0, 2.0}}

		graphtest.RunTestGraphFnWithBackend(t, "basic", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, xData)
			packed := Const(g, packedData)
			s := Const(g, scalesData)
			weights := Reshape(Bitcast(packed, dtypes.Uint4), 2, N)
			cfg := QuantizedDenseConfig{
				WeightQuantization: &Quantization{
					Scheme:    compute.QuantNF4,
					Scale:     s,
					BlockAxis: 1,
					BlockSize: blockSize,
				},
			}
			y := nn.QuantizedDense(x, weights, nil, cfg)
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{1.0, -1.0, 0.0, 0.0}}}, 1e-4)
	}, "xla:cpu", "xla:cuda")
}

func TestQuantizedDense_MultiBlock_And_BlockAxis0(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		blockSize := 2
		xData := [][]float32{{1.0, 1.0}}
		weightsData := [][]int8{{1, 2, 3, 4}, {5, 6, 7, 8}}
		scalesData := [][]float32{{2.0, 3.0}, {4.0, 5.0}}

		graphtest.RunTestGraphFnWithBackend(t, "int8_multi_block_axis1", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, xData)
			w := Const(g, weightsData)
			s := Const(g, scalesData)
			y := nn.QuantizedDense(x, w, nil, QuantizedDenseConfig{
				WeightQuantization: &Quantization{
					Scheme:    compute.QuantLinear,
					Scale:     s,
					BlockAxis: 1,
					BlockSize: blockSize,
				},
			})
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{22.0, 28.0, 44.0, 52.0}}}, 1e-4)

		biasData := []float32{0.1, 0.2, 0.3, 0.4}
		graphtest.RunTestGraphFnWithBackend(t, "int8_multi_block_with_bias", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, xData)
			w := Const(g, weightsData)
			s := Const(g, scalesData)
			b := Const(g, biasData)
			y := nn.QuantizedDense(x, w, b, QuantizedDenseConfig{
				WeightQuantization: &Quantization{
					Scheme:    compute.QuantLinear,
					Scale:     s,
					BlockAxis: 1,
					BlockSize: blockSize,
				},
			})
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{22.1, 28.2, 44.3, 52.4}}}, 1e-4)

		// GPTQ/AWQ-style BlockAxis=0 (blocking along K: K=4, N=2, BlockSize=2 -> scales [2, 2])
		// plus OutputsInput layout [N, K] with 1D per-channel scale [N].
		graphtest.RunTestGraphFnWithBackend(t, "block_axis0_gptq_and_outputs_input", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, [][]float32{{1, 2, 3, 4}})
			wIO := Const(g, [][]int8{{1, 2}, {3, 4}, {5, 6}, {7, 8}}) // [4, 2]
			// Block 0 (k=0,1): scale [0.5, 1.0]; Block 1 (k=2,3): scale [2.0, 0.25]
			// col 0: (1*1 + 2*3)*0.5 + (3*5 + 4*7)*2.0 = 3.5 + 86.0 = 89.5
			// col 1: (1*2 + 2*4)*1.0 + (3*6 + 4*8)*0.25 = 10.0 + 12.5 = 22.5
			sAxis0 := Const(g, [][]float32{{0.5, 1.0}, {2.0, 0.25}})
			y0 := nn.QuantizedDense(x, wIO, nil, QuantizedDenseConfig{
				WeightQuantization: &Quantization{
					Scheme:    compute.QuantLinear,
					Scale:     sAxis0,
					BlockAxis: 0,
					BlockSize: 2,
				},
				WeightLayout: compute.DenseLayoutInputOutputs,
			})

			// OutputsInput [N=2, K=4] with 1D per-channel scale [N=2]
			wOI := Const(g, [][]int8{{1, 3, 5, 7}, {2, 4, 6, 8}})
			s1D := Const(g, []float32{0.5, 0.25})
			// col 0: (1*1 + 2*3 + 3*5 + 4*7)*0.5 = 50 * 0.5 = 25.0
			// col 1: (1*2 + 2*4 + 3*6 + 4*8)*0.25 = 60 * 0.25 = 15.0
			y1 := nn.QuantizedDense(x, wOI, nil, QuantizedDenseConfig{
				WeightQuantization: &Quantization{
					Scheme: compute.QuantLinear,
					Scale:  s1D,
				},
				WeightLayout: compute.DenseLayoutOutputsInput,
			})
			return []*Node{x}, []*Node{y0, y1}
		}, []any{
			[][]float32{{89.5, 22.5}},
			[][]float32{{25.0, 15.0}},
		}, 1e-4)
	})
}

func TestQuantizedDense_ZeroPoint(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		// Standard affine dequantization: real = (float(w) - float(zeroPoint)) * scale.
		//   row 0: w=[1, 2], scale=2.0, zp=0.5  -> [(1-0.5)*2, (2-0.5)*2] = [1.0, 3.0]
		//   row 1: w=[3, 4], scale=3.0, zp=-0.5 -> [(3+0.5)*3, (4+0.5)*3] = [10.5, 13.5]
		//   x = [1, 1] -> y = [11.5, 16.5]
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
			y := nn.QuantizedDense(x, w, nil, QuantizedDenseConfig{
				WeightQuantization: &Quantization{
					Scheme:    compute.QuantLinear,
					Scale:     s,
					ZeroPoint: zp,
					BlockAxis: 1,
					BlockSize: blockSize,
				},
			})
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{11.5, 16.5}}}, 1e-4)

		biasData := []float32{0.1, 0.2}
		graphtest.RunTestGraphFnWithBackend(t, "with_zero_point_and_bias", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, xData)
			w := Const(g, weightsData)
			s := Const(g, scalesData)
			zp := Const(g, zpData)
			b := Const(g, biasData)
			y := nn.QuantizedDense(x, w, b, QuantizedDenseConfig{
				WeightQuantization: &Quantization{
					Scheme:    compute.QuantLinear,
					Scale:     s,
					ZeroPoint: zp,
					BlockAxis: 1,
					BlockSize: blockSize,
				},
			})
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{11.6, 16.7}}}, 1e-4)
	})
}

func TestQuantizedDense_GGML_Q8_0(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		const bytesPerRow = 34

		row0 := make([]uint8, bytesPerRow)
		row0[0], row0[1] = 0x00, 0x40 // fp16 LE for 2.0
		row0[2] = 1

		row1 := make([]uint8, bytesPerRow)
		row1[0], row1[1] = 0x00, 0x42 // fp16 LE for 3.0
		row1[3] = 1

		weightData := [][]uint8{row0, row1}

		xData := make([]float32, 32)
		xData[0] = 1.0
		xData[1] = 1.0

		cfg := func() QuantizedDenseConfig {
			return QuantizedDenseConfig{
				WeightQuantization: &Quantization{
					Scheme:   compute.QuantGGML,
					GGMLType: compute.GGMLQ8_0,
				},
				WeightLayout: compute.DenseLayoutOutputsInput,
			}
		}

		graphtest.RunTestGraphFnWithBackend(t, "basic", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, [][]float32{xData})
			w := Const(g, weightData)
			y := nn.QuantizedDense(x, w, nil, cfg())
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{2.0, 3.0}}}, 1e-4)

		graphtest.RunTestGraphFnWithBackend(t, "with_bias", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, [][]float32{xData})
			w := Const(g, weightData)
			b := Const(g, []float32{0.5, -0.5})
			y := nn.QuantizedDense(x, w, b, cfg())
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{2.5, 2.5}}}, 1e-4)

		graphtest.RunTestGraphFnWithBackend(t, "batch", backend, func(g *Graph) (inputs, outputs []*Node) {
			xData2 := make([]float32, 32)
			xData2[0] = 0.5
			x := Const(g, [][]float32{xData, xData2})
			w := Const(g, weightData)
			y := nn.QuantizedDense(x, w, nil, cfg())
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{2.0, 3.0}, {1.0, 0.0}}}, 1e-4)
	})
}

func TestQuantizedDense_GGML_Q4_0(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		const bytesPerRow = 18

		row0 := make([]uint8, bytesPerRow)
		row0[0], row0[1] = 0x00, 0x40
		row0[2] = 0x89
		for i := 3; i < bytesPerRow; i++ {
			row0[i] = 0x88
		}

		row1 := make([]uint8, bytesPerRow)
		row1[0], row1[1] = 0x00, 0x42
		row1[2] = 0x88
		row1[3] = 0x89
		for i := 4; i < bytesPerRow; i++ {
			row1[i] = 0x88
		}

		weightData := [][]uint8{row0, row1}
		xData := make([]float32, 32)
		xData[0] = 1.0
		xData[1] = 1.0

		cfg := func() QuantizedDenseConfig {
			return QuantizedDenseConfig{
				WeightQuantization: &Quantization{
					Scheme:   compute.QuantGGML,
					GGMLType: compute.GGMLQ4_0,
				},
				WeightLayout: compute.DenseLayoutOutputsInput,
			}
		}

		graphtest.RunTestGraphFnWithBackend(t, "basic", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, [][]float32{xData})
			w := Const(g, weightData)
			y := nn.QuantizedDense(x, w, nil, cfg())
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{2.0, 3.0}}}, 1e-4)

		graphtest.RunTestGraphFnWithBackend(t, "with_silu", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, [][]float32{xData})
			w := Const(g, weightData)
			y := nn.QuantizedDense(x, w, nil, cfg(), activation.TypeSilu)
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{1.7616, 2.8577}}}, 1e-3)
	})
}

func TestQuantizedDense_GGML_IQ4NL(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		const bytesPerRow = 18

		row0 := make([]uint8, bytesPerRow)
		row0[0], row0[1] = 0x00, 0x3C
		row0[2] = 0x89
		for i := 3; i < bytesPerRow; i++ {
			row0[i] = 0x88
		}

		row1 := make([]uint8, bytesPerRow)
		row1[0], row1[1] = 0x00, 0x40
		row1[2] = 0x88
		row1[3] = 0x89
		for i := 4; i < bytesPerRow; i++ {
			row1[i] = 0x88
		}

		weightData := [][]uint8{row0, row1}
		xData := make([]float32, 32)
		xData[0] = 1.0
		xData[1] = 1.0

		graphtest.RunTestGraphFnWithBackend(t, "basic", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, [][]float32{xData})
			w := Const(g, weightData)
			y := nn.QuantizedDense(x, w, nil, QuantizedDenseConfig{
				WeightQuantization: &Quantization{
					Scheme:   compute.QuantGGML,
					GGMLType: compute.GGMLIQ4NL,
				},
				WeightLayout: compute.DenseLayoutOutputsInput,
			})
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{14.0, 28.0}}}, 1e-4)
	})
}

func TestQuantizedDense_Int4(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		if backend.Name() == "xla" {
			t.Skip("Skipping: XLA seems to have some bugs when supporting sub-byte dtypes operations.")
		}

		N, blockSize := 4, 4
		packedData := [][]uint8{{0xF0, 0x77}, {0x0F, 0x77}}
		scalesData := [][]float32{{1.0}, {1.0}}
		xData := [][]float32{{1.0, 2.0}}

		graphtest.RunTestGraphFnWithBackend(t, "basic", backend, func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, xData)
			packed := Const(g, packedData)
			s := Const(g, scalesData)
			weights := Reshape(Bitcast(packed, dtypes.Int4), 2, N)
			y := nn.QuantizedDense(x, weights, nil, QuantizedDenseConfig{
				WeightQuantization: &Quantization{
					Scheme:    compute.QuantLinear,
					Scale:     s,
					BlockAxis: 1,
					BlockSize: blockSize,
				},
			})
			return []*Node{x}, []*Node{y}
		}, []any{[][]float32{{-2.0, -1.0, 21.0, 21.0}}}, 1e-4)
	}, "xla:cpu", "xla:cuda")
}

// TestQuantizedDense_PreQuantizedLinear_And_OutputQuantization verifies exact parity between
// fused execution and GOMLX_FUSION=0 decomposed execution for ONNX/SmoothQuant-style
// pre-quantized integer x + weight matmuls and QLinearMatMul output quantization.
func TestQuantizedDense_PreQuantizedLinear_And_OutputQuantization(t *testing.T) {
	backend := gobackend.GetBackend()

	xU8 := [][]uint8{{130, 120, 140, 110}, {100, 150, 125, 135}} // [2, 4]
	xScale := []float32{0.05, 0.1}                               // per-row [M]
	xZP := []uint8{125, 125}
	wI8 := [][]int8{{10, -5, 8}, {-4, 12, -6}, {6, -3, 9}, {-8, 7, -2}} // [4, 3]
	wScale := []float32{0.2, 0.1, 0.25}                                 // per-channel [N]
	wZP := []int8{2, -1, 0}
	bias := []float32{0.5, -0.25, 0.1}
	yScale := float32(0.05)
	yZP := uint8(128)

	buildFloatOut := func(g *Graph) *Node {
		return nn.QuantizedDense(Const(g, xU8), Const(g, wI8), Const(g, bias), QuantizedDenseConfig{
			WeightQuantization: &Quantization{
				Scheme:    compute.QuantLinear,
				Scale:     Const(g, wScale),
				ZeroPoint: Const(g, wZP),
			},
			XQuantization: &Quantization{
				Scheme:    compute.QuantLinear,
				Scale:     Const(g, xScale),
				ZeroPoint: Const(g, xZP),
			},
			Activation:   compute.ActivationConfig{Type: compute.ActivationRelu},
			WeightLayout: compute.DenseLayoutInputOutputs,
		})
	}

	buildQuantOut := func(g *Graph) *Node {
		return nn.QuantizedDense(Const(g, xU8), Const(g, wI8), Const(g, bias), QuantizedDenseConfig{
			WeightQuantization: &Quantization{
				Scheme:    compute.QuantLinear,
				Scale:     Const(g, wScale),
				ZeroPoint: Const(g, wZP),
			},
			XQuantization: &Quantization{
				Scheme:    compute.QuantLinear,
				Scale:     Const(g, xScale),
				ZeroPoint: Const(g, xZP),
			},
			OutputQuantization: &Quantization{
				Scheme:    compute.QuantLinear,
				Scale:     Const(g, yScale),
				ZeroPoint: Const(g, yZP),
			},
			Activation:   compute.ActivationConfig{Type: compute.ActivationRelu},
			WeightLayout: compute.DenseLayoutInputOutputs,
		})
	}

	t.Setenv(FusionEnv, "1")
	fusedFloat := MustExecOnce(backend, buildFloatOut)
	fusedQuant := MustExecOnce(backend, buildQuantOut)

	t.Setenv(FusionEnv, "0")
	decompFloat := MustExecOnce(backend, buildFloatOut)
	decompQuant := MustExecOnce(backend, buildQuantOut)

	gotFusedF, err := tensors.CopyFlatData[float32](fusedFloat)
	require.NoError(t, err)
	gotDecompF, err := tensors.CopyFlatData[float32](decompFloat)
	require.NoError(t, err)
	assert.InDeltaSlice(t, gotDecompF, gotFusedF, 1e-5)

	gotFusedQ, err := tensors.CopyFlatData[uint8](fusedQuant)
	require.NoError(t, err)
	gotDecompQ, err := tensors.CopyFlatData[uint8](decompQuant)
	require.NoError(t, err)
	assert.Equal(t, gotDecompQ, gotFusedQ)
}

// TestQuantizedDense_GGML_SharedPreQuantizedX tests ggml.Quantize (for both GGMLQ8_0 and GGMLQ8_K)
// shared across multiple nn.QuantizedDense projections (Q/K-style) in both fused and decomposed modes.
func TestQuantizedDense_GGML_SharedPreQuantizedX(t *testing.T) {
	backend := gobackend.GetBackend()

	t.Run("Q8_0_SharedAcrossTwoQ4_0Projections", func(t *testing.T) {
		const K = 32
		xRow0 := make([]float32, K)
		xRow1 := make([]float32, K)
		for i := range K {
			xRow0[i] = float32((i%7)-3) * 0.25
			xRow1[i] = float32(((i*3)%9)-4) * 0.2
		}

		makeQ4Weights := func(scale0, scale1 float32) [][]uint8 {
			r0 := make([]uint8, 18)
			r1 := make([]uint8, 18)
			binary.LittleEndian.PutUint16(r0[0:2], float16.FromFloat32(scale0).Bits())
			binary.LittleEndian.PutUint16(r1[0:2], float16.FromFloat32(scale1).Bits())
			for j := range 16 {
				r0[2+j] = uint8(((j % 15) + 1) | (((15 - (j % 15)) & 0x0F) << 4))
				r1[2+j] = uint8((((j * 3) % 15) + 1) | ((((j * 5) % 15) & 0x0F) << 4))
			}
			return [][]uint8{r0, r1}
		}
		wQ := makeQ4Weights(0.125, 0.25)
		wK := makeQ4Weights(0.2, 0.15)

		buildFn := func(g *Graph) []*Node {
			x := Const(g, [][]float32{xRow0, xRow1})
			xQ8 := ggml.Quantize(x, compute.GGMLQ8_0)
			xq := &Quantization{Scheme: compute.QuantGGML, GGMLType: compute.GGMLQ8_0}
			qOut := nn.QuantizedDense(xQ8, Const(g, wQ), nil, QuantizedDenseConfig{
				WeightQuantization: &Quantization{Scheme: compute.QuantGGML, GGMLType: compute.GGMLQ4_0},
				XQuantization:      xq,
				WeightLayout:       compute.DenseLayoutOutputsInput,
			})
			kOut := nn.QuantizedDense(xQ8, Const(g, wK), nil, QuantizedDenseConfig{
				WeightQuantization: &Quantization{Scheme: compute.QuantGGML, GGMLType: compute.GGMLQ4_0},
				XQuantization:      xq,
				WeightLayout:       compute.DenseLayoutOutputsInput,
			})
			return []*Node{qOut, kOut}
		}

		t.Setenv(FusionEnv, "1")
		fusedOuts := MustExecOnceN(backend, buildFn)
		t.Setenv(FusionEnv, "0")
		decompOuts := MustExecOnceN(backend, buildFn)

		for idx := range 2 {
			fVals, err := tensors.CopyFlatData[float32](fusedOuts[idx])
			require.NoError(t, err)
			dVals, err := tensors.CopyFlatData[float32](decompOuts[idx])
			require.NoError(t, err)
			assert.InDeltaSlice(t, dVals, fVals, 1e-4)
		}
	})

	t.Run("Q8_K_QuantizeAndDequantize_Parity", func(t *testing.T) {
		const K = 256
		xRow := make([]float32, K)
		wBytes := make([]uint8, 292)
		binary.LittleEndian.PutUint32(wBytes[0:4], math.Float32bits(0.05))
		var bsums [16]int16
		for i := range K {
			xRow[i] = float32(((i*5)%17)-8) * 0.15
			q := int8(((i * 7) % 21) - 10)
			wBytes[4+i] = uint8(q)
			bsums[i/16] += int16(q)
		}
		for g := range 16 {
			binary.LittleEndian.PutUint16(wBytes[260+2*g:262+2*g], uint16(bsums[g]))
		}
		wData := [][]uint8{wBytes}

		buildFn := func(g *Graph) *Node {
			x := Const(g, [][]float32{xRow})
			xQ8K := ggml.Quantize(x, compute.GGMLQ8_K)
			return nn.QuantizedDense(xQ8K, Const(g, wData), nil, QuantizedDenseConfig{
				WeightQuantization: &Quantization{Scheme: compute.QuantGGML, GGMLType: compute.GGMLQ8_K},
				XQuantization:      &Quantization{Scheme: compute.QuantGGML, GGMLType: compute.GGMLQ8_K},
				WeightLayout:       compute.DenseLayoutOutputsInput,
			})
		}

		t.Setenv(FusionEnv, "1")
		fusedOut := MustExecOnce(backend, buildFn)
		t.Setenv(FusionEnv, "0")
		decompOut := MustExecOnce(backend, buildFn)

		fVals, err := tensors.CopyFlatData[float32](fusedOut)
		require.NoError(t, err)
		dVals, err := tensors.CopyFlatData[float32](decompOut)
		require.NoError(t, err)
		assert.InDeltaSlice(t, dVals, fVals, 1e-3)
	})
}

// TestQuantizedDense_FusedVJP verifies reverse-mode autodiff (FusedQuantizedDenseVJP)
// against decomposed autodiff (GOMLX_FUSION=0) for QuantLinear, QuantNF4, and QuantGGML
// across ActivationNone, Relu, Silu (split activation), and SwiGLU.
func TestQuantizedDense_FusedVJP(t *testing.T) {
	backend := gobackend.GetBackend()

	t.Run("QuantLinear_BlockAxis0_WithBias_Activations", func(t *testing.T) {
		xData := [][]float32{{0.5, -1.2, 0.8, 1.5}, {-0.7, 0.4, -0.9, 0.6}}
		wData := [][]int8{{3, -4, 5, -2}, {-6, 2, -1, 7}, {4, 5, -3, 2}, {-2, 6, 4, -5}}
		sData := [][]float32{{0.25, 0.5, 0.125, 0.2}, {0.5, 0.25, 0.3, 0.15}}
		zpData := [][]int8{{1, -1, 0, 2}, {0, 1, -2, -1}}
		bData := []float32{0.1, -0.2, 0.3, -0.1}

		for _, act := range []activation.Type{activation.TypeNone, activation.TypeRelu, activation.TypeSilu, activation.TypeSwiGLU} {
			t.Run(act.String(), func(t *testing.T) {
				gradFn := func(g *Graph) []*Node {
					x := Const(g, xData)
					b := Const(g, bData)
					y := nn.QuantizedDense(x, Const(g, wData), b, QuantizedDenseConfig{
						WeightQuantization: &Quantization{
							Scheme:    compute.QuantLinear,
							Scale:     Const(g, sData),
							ZeroPoint: Const(g, zpData),
							BlockAxis: 0,
							BlockSize: 2,
						},
						WeightLayout: compute.DenseLayoutInputOutputs,
					}, act)
					// Weighted sum so every output element has a distinct upstream gradient.
					outDim := y.Shape().Dimensions[1]
					weights := AddScalar(Iota(g, y.Shape(), 1), 1.0)
					_ = outDim
					loss := ReduceAllSum(Mul(y, weights))
					return Gradient(loss, x, b)
				}

				t.Setenv(FusionEnv, "1")
				fusedGrads := MustExecOnceN(backend, gradFn)
				t.Setenv(FusionEnv, "0")
				decompGrads := MustExecOnceN(backend, gradFn)

				for idx, name := range []string{"dx", "dBias"} {
					fg, err := tensors.CopyFlatData[float32](fusedGrads[idx])
					require.NoError(t, err)
					dg, err := tensors.CopyFlatData[float32](decompGrads[idx])
					require.NoError(t, err)
					assert.InDeltaSlice(t, dg, fg, 1e-4, "%s mismatch for %s", name, act)
				}
			})
		}
	})

	t.Run("QuantGGML_Q4_K_FusedBackward", func(t *testing.T) {
		// Q4_K is a 256-value K-quant with NO decomposed graph fallback; FusedQuantizedDenseVJP
		// enables backpropagation through frozen Q4_K GGUF layers directly!
		const K, N = 256, 2
		wQ4K := make([][]uint8, N)
		for n := range N {
			blk := make([]uint8, 144)
			binary.LittleEndian.PutUint16(blk[0:2], float16.FromFloat32(0.01*float32(n+1)).Bits()) // d
			binary.LittleEndian.PutUint16(blk[2:4], float16.FromFloat32(0.005).Bits())             // dmin
			for j := range 12 {
				blk[4+j] = 0x12
			}
			for j := range 128 {
				blk[16+j] = uint8(((j + n) % 16) | (((j*3 + n) % 16) << 4))
			}
			wQ4K[n] = blk
		}
		xRow := make([]float32, K)
		for i := range K {
			xRow[i] = float32((i%11)-5) * 0.1
		}
		bVals := []float32{0.2, -0.3}

		evalLoss := func(xIn []float32, bIn []float32) float32 {
			out := MustExecOnce(backend, func(g *Graph) *Node {
				x := Const(g, [][]float32{xIn})
				b := Const(g, bIn)
				y := nn.QuantizedDense(x, Const(g, wQ4K), b, QuantizedDenseConfig{
					WeightQuantization: &Quantization{
						Scheme:   compute.QuantGGML,
						GGMLType: compute.GGMLQ4_K,
					},
					WeightLayout: compute.DenseLayoutOutputsInput,
				}, activation.TypeSilu)
				return ReduceAllSum(Mul(y, Const(g, [][]float32{{1.5, -0.75}})))
			})
			vals, err := tensors.CopyFlatData[float32](out)
			require.NoError(t, err)
			return vals[0]
		}

		t.Setenv(FusionEnv, "1")
		grads := MustExecOnceN(backend, func(g *Graph) []*Node {
			x := Const(g, [][]float32{xRow})
			b := Const(g, bVals)
			y := nn.QuantizedDense(x, Const(g, wQ4K), b, QuantizedDenseConfig{
				WeightQuantization: &Quantization{
					Scheme:   compute.QuantGGML,
					GGMLType: compute.GGMLQ4_K,
				},
				WeightLayout: compute.DenseLayoutOutputsInput,
			}, activation.TypeSilu)
			loss := ReduceAllSum(Mul(y, Const(g, [][]float32{{1.5, -0.75}})))
			return Gradient(loss, x, b)
		})
		dx, err := tensors.CopyFlatData[float32](grads[0])
		require.NoError(t, err)
		dBias, err := tensors.CopyFlatData[float32](grads[1])
		require.NoError(t, err)
		require.Len(t, dx, K)
		require.Len(t, dBias, N)
		var norm float64
		for _, v := range dx {
			norm += float64(v * v)
		}
		assert.Greater(t, norm, 0.0, "expected non-zero gradient dx through Q4_K FusedQuantizedDenseVJP")

		// Verify numerical agreement against central finite differences on sampled coordinates.
		const eps = float32(1e-3)
		for _, k := range []int{0, 17, 63, 128, 255} {
			xPlus := append([]float32(nil), xRow...)
			xMinus := append([]float32(nil), xRow...)
			xPlus[k] += eps
			xMinus[k] -= eps
			fd := (evalLoss(xPlus, bVals) - evalLoss(xMinus, bVals)) / (2 * eps)
			assert.InDelta(t, fd, dx[k], 1e-3, "dx[%d] finite-difference mismatch", k)
		}
		for n := range N {
			bPlus := append([]float32(nil), bVals...)
			bMinus := append([]float32(nil), bVals...)
			bPlus[n] += eps
			bMinus[n] -= eps
			fd := (evalLoss(xRow, bPlus) - evalLoss(xRow, bMinus)) / (2 * eps)
			assert.InDelta(t, fd, dBias[n], 1e-3, "dBias[%d] finite-difference mismatch", n)
		}
	})
}
