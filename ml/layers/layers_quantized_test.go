// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package layers_test

import (
	"encoding/binary"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/graphtest"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/layers/fnn"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

func TestDenseAndEmbeddingWithQuantizedVariables(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	// Build a tiny vocab=2, K=32 Q8_0 embedding table (34 bytes/row):
	// row 0: d=1.0, q=[1, 0, ...] -> [1, 0, ...]
	// row 1: d=2.0, q=[0, 1, ...] -> [0, 2, ...]
	embRow0 := make([]uint8, 34)
	embRow0[0], embRow0[1] = 0x00, 0x3C // 1.0
	embRow0[2] = 1
	embRow1 := make([]uint8, 34)
	embRow1[0], embRow1[1] = 0x00, 0x40 // 2.0
	embRow1[3] = 1

	// Build an outDim=2, inDim=32 Q5_0 dense weight matrix (22 bytes/row):
	// out 0: d=1.0, q5[0]=18 (+2), rest=16 (0) -> dot([x0, x1, ...]) = 2*x0
	// out 1: d=1.0, q5[1]=15 (-1), rest=16 (0) -> dot([x0, x1, ...]) = -1*x1
	wRow0 := make([]uint8, 22)
	wRow0[0], wRow0[1] = 0x00, 0x3C // 1.0
	binary.LittleEndian.PutUint32(wRow0[2:6], 0xFFFFFFFF)
	wRow0[6] = 0x02 // q5[0] = 16 + 2 = 18 -> +2

	wRow1 := make([]uint8, 22)
	wRow1[0], wRow1[1] = 0x00, 0x3C // 1.0
	binary.LittleEndian.PutUint32(wRow1[2:6], 0xFFFFFFFD) // bit 1 = 0, others = 1
	wRow1[7] = 0x0F                                       // q5[1] = 0 + 15 = 15 -> -1

	store := model.NewStore()
	store.RootScope().In("token_embed").
		VariableWithValue("embeddings", [][]uint8{embRow0, embRow1}).
		WithQuantization(&model.Quantization{
			Scheme:   compute.QuantGGML,
			GGMLType: compute.GGMLQ8_0,
		})
	denseScope := store.RootScope().In("proj").In("dense")
	denseScope.VariableWithValue("weights", [][]uint8{wRow0, wRow1}).
		WithQuantization(&model.Quantization{
			Scheme:   compute.QuantGGML,
			GGMLType: compute.GGMLQ5_0,
		})
	denseScope.VariableWithValue("biases", []float32{0.5, 1.0})

	for _, quantMatmul := range []bool{true, false} {
		store.SetParam(layers.ParamQuantizedMatmuls, quantMatmul)
		exec := model.MustNewExec(backend, store, func(scope *model.Scope, tokens *Node) *Node {
			emb := layers.Embedding(scope.In("token_embed"), tokens, dtypes.Float32, 2, 32)
			return layers.NewDense(scope.In("proj"), emb, 2).
				WithBias(true).
				WithLayout(compute.DenseLayoutOutputsInput).
				WithActivation(activation.TypeRelu).
				Done()
		})

		outs := exec.MustExec([][]int32{{0, 1}})
		require.Len(t, outs, 1)
		got, err := tensors.CopyFlatData[float32](outs[0])
		require.NoError(t, err)
		// token 0 -> emb=[1, 0, ...]: pre-relu=[2*1+0.5, -1*0+1.0]=[2.5, 1.0] -> relu=[2.5, 1.0]
		// token 1 -> emb=[0, 2, ...]: pre-relu=[2*0+0.5, -1*2+1.0]=[0.5, -1.0] -> relu=[0.5, 0.0]
		assert.InDeltaSlice(t, []float32{2.5, 1.0, 0.5, 0.0}, got, 1e-2, "quantMatmul=%v", quantMatmul)
	}
}

func TestFNNWithQuantizedVariables(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	// Hidden layer: inDim=32 -> hidden=32, Q8_0 (34 bytes/row), identity on first 2 dims with scale 1.0:
	hiddenRows := make([][]uint8, 32)
	for r := range 32 {
		row := make([]uint8, 34)
		row[0], row[1] = 0x00, 0x3C // d = 1.0
		if r < 2 {
			row[2+r] = 1
		}
		hiddenRows[r] = row
	}

	// Output layer: inDim=32 -> outDim=2, Q8_0:
	// row 0: [3, -1, 0, ...], row 1: [0, 2, 0, ...]
	outRow0 := make([]uint8, 34)
	outRow0[0], outRow0[1] = 0x00, 0x3C
	outRow0[2] = 3
	outRow0[3] = 0xFF // -1
	outRow1 := make([]uint8, 34)
	outRow1[0], outRow1[1] = 0x00, 0x3C
	outRow1[3] = 2

	store := model.NewStore()
	fnnScope := store.RootScope().In("mlp")
	fnnScope.In("fnn_hidden_layer_0").VariableWithValue("weights", hiddenRows).
		WithQuantization(&model.Quantization{Scheme: compute.QuantGGML, GGMLType: compute.GGMLQ8_0})
	fnnScope.In("fnn_output_layer").VariableWithValue("weights", [][]uint8{outRow0, outRow1}).
		WithQuantization(&model.Quantization{Scheme: compute.QuantGGML, GGMLType: compute.GGMLQ8_0})

	exec := model.MustNewExec(backend, store, func(scope *model.Scope, x *Node) *Node {
		return fnn.New(scope.In("mlp"), x, 2).
			NumHiddenLayers(1, 32).
			UseBias(false).
			WithWeightsLayout(compute.DenseLayoutOutputsInput).
			Activation(activation.TypeRelu).
			Done()
	})

	in := make([][]float32, 1)
	in[0] = make([]float32, 32)
	in[0][0] = 2.0
	in[0][1] = -3.0 // clipped to 0 by hidden ReLU

	outs := exec.MustExec(in)
	require.Len(t, outs, 1)
	got, err := tensors.CopyFlatData[float32](outs[0])
	require.NoError(t, err)
	// After hidden + ReLU: [2, 0, 0, ...] -> output: [3*2 + (-1)*0, 0*2 + 2*0] = [6, 0]
	assert.InDeltaSlice(t, []float32{6.0, 0.0}, got, 1e-2)
}
