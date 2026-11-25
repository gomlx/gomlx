// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package transformer

import (
	"strings"
	"testing"

	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/layers/attention/pos"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/initializer"
	"github.com/gomlx/gomlx/ml/zoo/transformer/generate"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestQwen35HybridModelPrefillAndDecodeParity(t *testing.T) {
	backend := testutil.BuildTestBackend()
	store := model.NewStore()
	scope := store.RootScope()
	initFn := initializer.RandomNormalFn(scope, 0.05)

	buildModel := func(s *model.Scope) *Model {
		return New(s.WithInitializer(initFn)).
			WithArchitecture(ArchitectureQwen35).
			WithVocabSize(64).
			WithEmbedDim(32).
			WithNumLayers(4).
			WithNumHeads(4).
			WithNumKVHeads(2).
			WithHeadDim(16).
			WithFFNDim(64).
			WithMaxPosEmbed(256).
			WithNormalization(layers.NormalizationRMSNorm).
			WithFinalNormalization(layers.NormalizationRMSNorm).
			WithNormEpsilon(1e-6).
			WithRMSNormOffset(1.0).
			WithActivation(activation.TypeSilu).
			WithBias(false).
			WithCausalMask(true).
			WithTransposedWeights(true).
			WithAttnOutputGate(true).
			WithLayerTypes([]LayerType{LinearLayer, LinearLayer, LinearLayer, GlobalLayer}).
			WithLinearConvKernelDim(4).
			WithLinearKeyHeadDim(8).
			WithLinearValueHeadDim(8).
			WithLinearNumKeyHeads(2).
			WithLinearNumValueHeads(4).
			WithLinearStateDType(dtypes.Float32).
			WithPositionalEncoder(pos.NewRoPEWithDimRange(10000.0, 0, 4))
	}

	// 1. Run full-sequence uncached forward pass on 7 tokens (initializes all model variables).
	fullTokens := [][]int32{{3, 11, 7, 21, 5, 17, 2}}
	fullExec, err := model.NewExec(backend, store, func(s *model.Scope, tokens *Node) *Node {
		m := buildModel(s)
		return m.Logits(tokens, CallOptions{})
	})
	require.NoError(t, err)
	defer fullExec.Finalize()

	fullRes, err := fullExec.Call(tensors.FromValue(fullTokens))
	require.NoError(t, err)
	defer fullRes[0].FinalizeAll()

	var fullLogits []float32
	fullRes[0].MustConstFlatData(func(flat any) { fullLogits = flat.([]float32) })
	require.Equal(t, []int{1, 7, 64}, fullRes[0].Shape().Dimensions)

	// 2. Run cached prefill on first 4 tokens, then 3 single-token decode steps, and verify exact match!
	mCfg := buildModel(scope)
	mCfg.PopulateKVCacheConfigs()
	kvCfg := mCfg.KVCache
	cacheTensors := kvCfg.InitializeTensors(1, mCfg.NumKVHeads, mCfg.HeadDim, mCfg.DType, 16)

	prefillExec, err := model.NewExec(backend, store, func(s *model.Scope, inputs []*Node) []*Node {
		tokens := inputs[0]
		cache := kvCfg.DeserializeNodes(inputs[1:])
		m := buildModel(s)
		m.KVCache = kvCfg
		logits, updated := m.Forward(tokens, CallOptions{
			PositionIds: Const(tokens.Graph(), int32(0)),
			Cache:       cache,
		})
		ser, err := kvCfg.SerializeNodes(updated)
		if err != nil {
			panic(err)
		}
		return append([]*Node{logits}, ser...)
	})
	require.NoError(t, err)
	defer prefillExec.Finalize()
	if backend.Capabilities().HasDynamicShapes() {
		prefillExec.WithDynamicAxes([]string{"", "seq_len"})
	}

	decodeExec, err := model.NewExec(backend, store, func(s *model.Scope, inputs []*Node) []*Node {
		tokens := inputs[0]
		posNode := inputs[1]
		cache := kvCfg.DeserializeNodes(inputs[2:])
		m := buildModel(s)
		m.KVCache = kvCfg
		logits, updated := m.Forward(tokens, CallOptions{
			PositionIds: posNode,
			Cache:       cache,
		})
		ser, err := kvCfg.SerializeNodes(updated)
		if err != nil {
			panic(err)
		}
		return append([]*Node{logits}, ser...)
	})
	require.NoError(t, err)
	defer decodeExec.Finalize()

	// Prefill tokens[0:4]
	serCache, err := kvCfg.SerializeTensors(cacheTensors)
	require.NoError(t, err)
	prefillArgs := make([]any, 1+len(serCache))
	prefillArgs[0] = tensors.FromValue([][]int32{{3, 11, 7, 21}})
	for i, ct := range serCache {
		prefillArgs[i+1] = ct
	}
	prefillOut, err := prefillExec.Call(prefillArgs...)
	require.NoError(t, err)

	var prefillLogits []float32
	prefillOut[0].MustConstFlatData(func(flat any) { prefillLogits = flat.([]float32) })
	logitTol := 1e-4
	if strings.Contains(backend.Description(), "cuda") {
		logitTol = 1e-2 // CUDA Tensor Cores use TF32 for Float32 matmuls
	}
	for i := range 4 * 64 {
		assert.InDelta(t, fullLogits[i], prefillLogits[i], logitTol, "prefill logit mismatch at %d", i)
	}
	prefillOut[0].FinalizeAll()
	cacheTensors = kvCfg.DeserializeTensors(prefillOut[1:])

	// Decode tokens[4], tokens[5], tokens[6] one by one
	for stepIdx, tok := range []int32{5, 17, 2} {
		posVal := int32(4 + stepIdx)
		serCache, err = kvCfg.SerializeTensors(cacheTensors)
		require.NoError(t, err)
		decodeArgs := make([]any, 2+len(serCache))
		decodeArgs[0] = tensors.FromValue([][]int32{{tok}})
		decodeArgs[1] = tensors.FromValue(posVal)
		for i, ct := range serCache {
			decodeArgs[i+2] = ct
		}
		decOut, err := decodeExec.Call(decodeArgs...)
		require.NoError(t, err)

		var stepLogits []float32
		decOut[0].MustConstFlatData(func(flat any) { stepLogits = flat.([]float32) })
		base := (4 + stepIdx) * 64
		for v := range 64 {
			assert.InDelta(t, fullLogits[base+v], stepLogits[v], logitTol, "decode step %d vocab %d", stepIdx, v)
		}
		decOut[0].FinalizeAll()
		cacheTensors = kvCfg.DeserializeTensors(decOut[1:])
	}
	for _, ct := range cacheTensors {
		ct.FinalizeAll()
	}

	// 3. End-to-end generate.Generator test with hybrid KVCache!
	genModel := buildModel(scope)
	genModel.PopulateKVCacheConfigs()
	gen := generate.New(MakeKVCacheModelFn(genModel)).
		WithKVCache(genModel.KVCache, genModel.NumKVHeads, genModel.HeadDim, genModel.DType).
		WithMaxLength(10)
	gen.Temperature = 0
	gen.TopK = 0
	gen.TopP = 0

	generated, err := gen.Decode(backend, scope, []int32{3, 11, 7, 21})
	require.NoError(t, err)
	defer generated.FinalizeAll()
	assert.Equal(t, []int{1, 10}, generated.Shape().Dimensions)
}
