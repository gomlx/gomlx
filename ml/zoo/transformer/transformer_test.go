// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package transformer

import (
	"fmt"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/attention/kvcache"
	"github.com/gomlx/gomlx/ml/layers/attention/pos"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestModel groups transformer model tests.
func TestModel(t *testing.T) {
	t.Run("Defaults", func(t *testing.T) {
		cfg := New(nil).
			WithVocabSize(1000).
			WithEmbedDim(128).
			WithNumLayers(4).
			WithNumHeads(8).
			WithHeadDim(16)
		assert.Equal(t, 1000, cfg.VocabSize)
		assert.Equal(t, 128, cfg.EmbedDim)
		assert.Equal(t, 4, cfg.NumLayers)
		assert.Equal(t, 8, cfg.NumHeads)
		assert.Equal(t, 16, cfg.HeadDim)
		assert.Equal(t, 512, cfg.FFNDim)
		assert.Equal(t, 512, cfg.MaxPosEmbed)
		assert.Equal(t, dtypes.Float32, cfg.DType)
		assert.Equal(t, 0.0, cfg.Dropout)
		assert.Nil(t, cfg.posEncoder)
		assert.Equal(t, cfg.Normalization, layers.NormalizationLayerNorm)
		assert.True(t, cfg.UseBias)
	})

	t.Run("Builders", func(t *testing.T) {
		cfg := New(nil).
			WithVocabSize(1000).
			WithEmbedDim(128).
			WithNumLayers(4).
			WithNumHeads(8).
			WithHeadDim(16).
			WithFFNDim(256).
			WithMaxPosEmbed(1024).
			WithDType(dtypes.Float16).
			WithDropout(0.1).
			WithRoPE(5000.0).
			WithNormalization("").
			WithBias(false)
		assert.Equal(t, 256, cfg.FFNDim)
		assert.Equal(t, 1024, cfg.MaxPosEmbed)
		assert.Equal(t, dtypes.Float16, cfg.DType)
		assert.Equal(t, 0.1, cfg.Dropout)
		require.NotNil(t, cfg.posEncoder)
		rope, ok := cfg.posEncoder.(*pos.RoPE)
		require.True(t, ok)
		assert.Equal(t, 5000.0, rope.BaseFreq)
		assert.Equal(t, cfg.Normalization, layers.NormalizationNone)
		assert.False(t, cfg.UseBias)
	})

	t.Run("NewFromScope", func(t *testing.T) {
		store := model.NewStore()
		scope := store.RootScope()
		scope.SetParams(map[string]any{
			ParamVocabSize:     1000,
			ParamEmbedDim:      128,
			ParamNumLayers:     4,
			ParamNumHeads:      8,
			ParamHeadDim:       16,
			ParamFFNDim:        256,
			ParamMaxPosEmbed:   1024,
			ParamDType:         "float16",
			ParamDropout:       0.1,
			ParamUseRoPE:       true,
			ParamRoPEBaseFreq:  5000.0,
			ParamUseBias:       false,
			ParamNormalization: "none", // layers.NormalizationNone or "".
		})

		cfg := New(scope)
		assert.Equal(t, 1000, cfg.VocabSize)
		assert.Equal(t, 128, cfg.EmbedDim)
		assert.Equal(t, 4, cfg.NumLayers)
		assert.Equal(t, 8, cfg.NumHeads)
		assert.Equal(t, 16, cfg.HeadDim)
		assert.Equal(t, 256, cfg.FFNDim)
		assert.Equal(t, 1024, cfg.MaxPosEmbed)
		assert.Equal(t, dtypes.Float16, cfg.DType)
		assert.Equal(t, 0.1, cfg.Dropout)
		require.NotNil(t, cfg.posEncoder)
		rope, ok := cfg.posEncoder.(*pos.RoPE)
		require.True(t, ok)
		assert.Equal(t, 5000.0, rope.BaseFreq)
		assert.Equal(t, cfg.Normalization, layers.NormalizationNone)
		assert.False(t, cfg.UseBias)
	})

	t.Run("ValidationRejectsMissingConfig", func(t *testing.T) {
		assert.Panics(t, func() {
			New(nil).validate(CallOptions{})
		})
	})

	t.Run("ValidationRejectsBothMaskAndSeqLen", func(t *testing.T) {
		assert.Panics(t, func() {
			backend := testutil.BuildTestBackend()
			g := NewGraph(backend, "test")
			mask := ScalarZero(g, dtypes.Int32)
			seqLen := ScalarZero(g, dtypes.Int32)
			m := New(nil).WithVocabSize(10).WithEmbedDim(16).WithNumLayers(1).WithNumHeads(1).WithHeadDim(16)
			m.validate(CallOptions{AttentionMask: mask, SeqLen: seqLen})
		})
	})
}

// TestTransformerBuilder groups transformer builder/exposed functions tests.
func TestTransformerBuilder(t *testing.T) {
	t.Run("BuildGraph", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		store := model.NewStore()
		scope := store.RootScope()
		g := NewGraph(backend, "BuildGraph")
		tokens := IotaFull(g, shapes.Make(dtypes.Int32, 2, 8))
		cfg := New(scope).
			WithVocabSize(100).
			WithEmbedDim(64).
			WithNumLayers(2).
			WithNumHeads(4).
			WithHeadDim(16).
			WithFFNDim(128).
			WithMaxPosEmbed(128)
		tokens = Mod(tokens, Const(g, int32(cfg.VocabSize)))
		logits := cfg.Logits(tokens, CallOptions{})
		require.NotNil(t, logits)
		assert.Equal(t, []int{2, 8, 100}, logits.Shape().Dimensions)
		assert.Equal(t, dtypes.Float32, logits.DType())
	})

	t.Run("BuildGraphWithKVCache", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		store := model.NewStore()
		scope := store.RootScope()
		g := NewGraph(backend, "BuildGraphWithKVCache")
		prompt := IotaFull(g, shapes.Make(dtypes.Int32, 2, 5))
		m := New(scope).
			WithVocabSize(100).
			WithEmbedDim(64).
			WithNumLayers(2).
			WithNumHeads(4).
			WithHeadDim(16).
			WithFFNDim(128).
			WithMaxPosEmbed(128)
		prompt = Mod(prompt, Const(g, int32(m.VocabSize)))
		logits, _ := m.LogitsWithKVCache(prompt, nil, nil)
		require.NotNil(t, logits)
		assert.Equal(t, []int{2, 5, 100}, logits.Shape().Dimensions)
	})
}

// TestTransformerVariants groups variant configurations.
func TestTransformerVariants(t *testing.T) {
	t.Run("WithRoPE", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		store := model.NewStore()
		scope := store.RootScope()
		g := NewGraph(backend, "WithRoPE")
		tokens := IotaFull(g, shapes.Make(dtypes.Int32, 1, 4))
		m := New(scope).
			WithVocabSize(100).
			WithEmbedDim(64).
			WithNumLayers(2).
			WithNumHeads(4).
			WithHeadDim(16).
			WithFFNDim(128).
			WithMaxPosEmbed(128).
			WithRoPE(10000.0)
		tokens = Mod(tokens, Const(g, int32(m.VocabSize)))
		logits, _ := m.LogitsWithKVCache(tokens, nil, nil)
		require.NotNil(t, logits)
		assert.Equal(t, []int{1, 4, 100}, logits.Shape().Dimensions)
	})

	t.Run("WithSeqLen", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		store := model.NewStore()
		scope := store.RootScope()
		g := NewGraph(backend, "WithSeqLen")
		tokens := IotaFull(g, shapes.Make(dtypes.Int32, 2, 8))
		seqLen := Const(g, []int32{5, 3}) // non-padded sequence lengths
		m := New(scope).
			WithVocabSize(100).
			WithEmbedDim(64).
			WithNumLayers(2).
			WithNumHeads(4).
			WithHeadDim(16).
			WithFFNDim(128).
			WithMaxPosEmbed(128)
		tokens = Mod(tokens, Const(g, int32(m.VocabSize)))
		logits := m.Logits(tokens, CallOptions{SeqLen: seqLen})
		require.NotNil(t, logits)
		assert.Equal(t, []int{2, 8, 100}, logits.Shape().Dimensions)
	})

	t.Run("ExplicitKVCacheExecution", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		store := model.NewStore()
		scope := store.RootScope()
		m := New(scope).
			WithVocabSize(100).
			WithEmbedDim(64).
			WithNumLayers(2).
			WithNumHeads(4).
			WithHeadDim(16).
			WithFFNDim(128).
			WithMaxPosEmbed(128)

		// 1. Initialize Cache
		m.populateOrderedScopes()
		m.populateLayerTypes()
		cacheTensors := m.KVCache.InitializeTensors(2, 4, 16, dtypes.Float32, 5) // batch=2, numKVHeads=4, headDim=16, seqLen=5

		// 2. Build graph for prompt execution
		exec, err := model.NewExec(backend, store, func(scope *model.Scope, inputs []*Node) []*Node {
			tokens := inputs[0]
			cacheNodes := inputs[1:]
			cache := m.KVCache.DeserializeNodes(cacheNodes)
			logits, updatedCache := m.Forward(tokens, CallOptions{Cache: cache})
			serializedCache, err := m.KVCache.SerializeNodes(updatedCache)
			if err != nil {
				panic(err)
			}
			res := make([]*Node, 1+len(serializedCache))
			res[0] = logits
			copy(res[1:], serializedCache)
			return res
		})
		require.NoError(t, err)
		defer exec.Finalize()

		// 3. Prepare inputs
		promptTensor := tensors.FromValue([][]int32{
			{1, 2, 3, 4, 5},
			{5, 4, 3, 2, 1},
		})

		serializedCacheTensors, err := m.KVCache.SerializeTensors(cacheTensors)
		require.NoError(t, err)

		args := make([]any, 0, 1+len(serializedCacheTensors))
		args = append(args, promptTensor)
		for _, tensor := range serializedCacheTensors {
			args = append(args, tensor)
		}

		// 4. Run Execution
		results, err := exec.Call(args...)
		require.NoError(t, err)

		logits := results[0]
		assert.Equal(t, []int{2, 5, 100}, logits.Shape().Dimensions)

		updatedCacheTensors := m.KVCache.DeserializeTensors(results[1:])
		assert.Equal(t, 2*len(m.KVCache.OrderedScopes), len(results)-1)

		// Verify shapes of updated cache
		for _, scopePath := range m.KVCache.OrderedScopes {
			kTensor := updatedCacheTensors[scopePath+kvcache.KeySuffix]
			assert.Equal(t, []int{2, 4, 32, 16}, kTensor.Shape().Dimensions) // padded from 5 to 32
		}
	})

	t.Run("WithoutLayerNorm", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		store := model.NewStore()
		scope := store.RootScope()
		m := New(scope).
			WithVocabSize(100).
			WithEmbedDim(64).
			WithNumLayers(2).
			WithNumHeads(4).
			WithHeadDim(16).
			WithFFNDim(128).
			WithMaxPosEmbed(128).
			WithNormalization("none")
		g := NewGraph(backend, "WithoutLayerNorm")
		tokens := IotaFull(g, shapes.Make(dtypes.Int32, 2, 8))
		tokens = Mod(tokens, Const(g, int32(m.VocabSize)))
		logits := m.Logits(tokens, CallOptions{})
		require.NotNil(t, logits)
		assert.Equal(t, []int{2, 8, 100}, logits.Shape().Dimensions)
	})

	t.Run("WithDyTNormalization", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		store := model.NewStore()
		scope := store.RootScope()
		m := New(scope).
			WithVocabSize(100).
			WithEmbedDim(64).
			WithNumLayers(2).
			WithNumHeads(4).
			WithHeadDim(16).
			WithFFNDim(128).
			WithMaxPosEmbed(128).
			WithNormalization(layers.NormalizationDyT).
			WithDyTAlpha(0.75)
		g := NewGraph(backend, "WithDyTNormalization")
		tokens := IotaFull(g, shapes.Make(dtypes.Int32, 2, 8))
		tokens = Mod(tokens, Const(g, int32(m.VocabSize)))
		logits := m.Logits(tokens, CallOptions{})
		require.NotNil(t, logits)
		assert.Equal(t, []int{2, 8, 100}, logits.Shape().Dimensions)
	})

	t.Run("WithoutBias", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		store := model.NewStore()
		scope := store.RootScope()
		m := New(scope).
			WithVocabSize(100).
			WithEmbedDim(64).
			WithNumLayers(2).
			WithNumHeads(4).
			WithHeadDim(16).
			WithFFNDim(128).
			WithMaxPosEmbed(128).
			WithBias(false)
		g := NewGraph(backend, "WithoutBias")
		tokens := IotaFull(g, shapes.Make(dtypes.Int32, 2, 8))
		tokens = Mod(tokens, Const(g, int32(m.VocabSize)))
		logits := m.Logits(tokens, CallOptions{})
		require.NotNil(t, logits)
		assert.Equal(t, []int{2, 8, 100}, logits.Shape().Dimensions)
	})

	t.Run("WithDropout", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		store := model.NewStore()
		scope := store.RootScope()
		m := New(scope).
			WithVocabSize(100).
			WithEmbedDim(64).
			WithNumLayers(2).
			WithNumHeads(4).
			WithHeadDim(16).
			WithFFNDim(128).
			WithMaxPosEmbed(128).
			WithDropout(0.1)
		g := NewGraph(backend, "WithDropout")
		tokens := IotaFull(g, shapes.Make(dtypes.Int32, 2, 8))
		tokens = Mod(tokens, Const(g, int32(m.VocabSize)))
		logits := m.Logits(tokens, CallOptions{})
		require.NotNil(t, logits)
		assert.Equal(t, []int{2, 8, 100}, logits.Shape().Dimensions)
	})
}

// TestTransformerBatchSizes groups batch size variations.
func TestTransformerBatchSizes(t *testing.T) {
	batchSizes := []int{1, 2, 4}
	backend := testutil.BuildTestBackend()

	for _, batchSize := range batchSizes {
		t.Run(fmt.Sprintf("BatchSize%d", batchSize), func(t *testing.T) {
			store := model.NewStore()
			scope := store.RootScope()

			m := New(scope).
				WithVocabSize(100).
				WithEmbedDim(64).
				WithNumLayers(2).
				WithNumHeads(4).
				WithHeadDim(16).
				WithFFNDim(128).
				WithMaxPosEmbed(128)

			seqLen := 8

			manager := backend
			g := NewGraph(manager, "TestTransformerBatchSize")

			tokens := IotaFull(g, shapes.Make(dtypes.Int32, batchSize, seqLen))
			tokens = Mod(tokens, Const(g, int32(m.VocabSize)))

			logits := m.Logits(tokens, CallOptions{})
			require.NotNil(t, logits)
			assert.Equal(t, []int{batchSize, seqLen, 100}, logits.Shape().Dimensions)
		})
	}
}

func TestHalfPrecisionMixedVariables(t *testing.T) {
	backend := testutil.BuildTestBackend()

	t.Run("UnexpectedNormDTypeMismatchPanics", func(t *testing.T) {
		store := model.NewStore()
		// Pre-populate final LayerNorm variables in Float32 while model expects Float16 norms by default.
		finalLNScope := store.RootScope().In("final_norm").In("layer_normalization")
		finalLNScope.VariableWithShape("gain", shapes.Make(dtypes.Float32, 32))
		finalLNScope.VariableWithShape("offset", shapes.Make(dtypes.Float32, 32))

		exec := model.MustNewExec(backend, store, func(scope *model.Scope, tokens *Node) *Node {
			m := New(scope).
				WithVocabSize(64).
				WithEmbedDim(32).
				WithNumLayers(1).
				WithNumHeads(2).
				WithHeadDim(16).
				WithFFNDim(64).
				WithMaxPosEmbed(16).
				WithDType(dtypes.Float16).
				WithNormalization(layers.NormalizationLayerNorm).
				WithFinalNormalization(layers.NormalizationLayerNorm)
			return m.Logits(tokens, CallOptions{})
		})
		require.Panics(t, func() { exec.MustCall([][]int32{{1, 2, 3, 4}}) })
	})

	t.Run("UnexpectedLayerScalarDTypeMismatchPanics", func(t *testing.T) {
		store := model.NewStore()
		store.RootScope().In("layer_0").VariableWithShape("layer_scalar", shapes.Make(dtypes.Float32, 1))

		exec := model.MustNewExec(backend, store, func(scope *model.Scope, tokens *Node) *Node {
			m := New(scope).
				WithArchitecture(ArchitectureGemma4).
				WithVocabSize(64).
				WithEmbedDim(32).
				WithNumLayers(1).
				WithNumHeads(2).
				WithHeadDim(16).
				WithFFNDim(64).
				WithRoPE(10000).
				WithDType(dtypes.Float16).
				WithNormDType(dtypes.Float32).
				WithHiddenSizePerLayerInput(16).
				WithVocabSizePerLayerInput(64)
			return m.Logits(tokens, CallOptions{})
		})
		require.Panics(t, func() { exec.MustCall([][]int32{{1, 2, 3, 4}}) })
	})

	t.Run("Float16WithExplicitNormDTypeFloat32LayerNorm", func(t *testing.T) {
		store := model.NewStore()
		finalLNScope := store.RootScope().In("final_norm").In("layer_normalization")
		finalLNScope.VariableWithShape("gain", shapes.Make(dtypes.Float32, 32))
		finalLNScope.VariableWithShape("offset", shapes.Make(dtypes.Float32, 32))

		exec := model.MustNewExec(backend, store, func(scope *model.Scope, tokens *Node) *Node {
			m := New(scope).
				WithVocabSize(64).
				WithEmbedDim(32).
				WithNumLayers(1).
				WithNumHeads(2).
				WithHeadDim(16).
				WithFFNDim(64).
				WithMaxPosEmbed(16).
				WithDType(dtypes.Float16).
				WithNormDType(dtypes.Float32).
				WithNormalization(layers.NormalizationLayerNorm).
				WithFinalNormalization(layers.NormalizationLayerNorm)
			return m.Logits(tokens, CallOptions{})
		})
		out := exec.MustCall([][]int32{{1, 2, 3, 4}})[0]
		require.Equal(t, dtypes.Float16, out.DType())
		require.Equal(t, []int{1, 4, 64}, out.Shape().Dimensions)
		require.Equal(t, dtypes.Float32, store.GetVariable("/final_norm/layer_normalization/gain").DType())
		require.Equal(t, dtypes.Float32, store.GetVariable("/layer_0/norm1/layer_normalization/gain").DType())
	})

	t.Run("Float16WithExplicitNormDTypeParamRMSNorm", func(t *testing.T) {
		store := model.NewStore()
		store.SetParam(model.ScopeSeparator+ParamNormDType, "float32")
		store.RootScope().In("final_norm").In("rms_norm").VariableWithShape("scale", shapes.Make(dtypes.Float32, 32))

		exec := model.MustNewExec(backend, store, func(scope *model.Scope, tokens *Node) *Node {
			m := New(scope).
				WithVocabSize(64).
				WithEmbedDim(32).
				WithNumLayers(1).
				WithNumHeads(2).
				WithHeadDim(16).
				WithFFNDim(64).
				WithRoPE(10000).
				WithDType(dtypes.Float16).
				WithNormalization(layers.NormalizationRMSNorm).
				WithFinalNormalization(layers.NormalizationRMSNorm)
			return m.Logits(tokens, CallOptions{})
		})
		out := exec.MustCall([][]int32{{1, 2, 3, 4}})[0]
		require.Equal(t, dtypes.Float16, out.DType())
		require.Equal(t, []int{1, 4, 64}, out.Shape().Dimensions)
		require.Equal(t, dtypes.Float32, store.GetVariable("/final_norm/rms_norm/scale").DType())
		require.Equal(t, dtypes.Float32, store.GetVariable("/layer_0/norm1/rms_norm/scale").DType())
	})
}

func TestGemma4DoubleWideMLP(t *testing.T) {
	backend := testutil.BuildTestBackend()
	store := model.NewStore()
	exec := model.MustNewExec(backend, store, func(scope *model.Scope, tokens *Node) *Node {
		m := New(scope).
			WithArchitecture(ArchitectureGemma4).
			WithVocabSize(64).
			WithEmbedDim(32).
			WithNumLayers(3).
			WithNumHeads(2).
			WithNumKVHeads(1).
			WithHeadDim(16).
			WithFFNDim(64).
			WithRoPE(10000).
			WithNumKVSharedLayers(1).
			WithUseDoubleWideMLP(true).
			WithTransposedWeights(true)
		return m.Logits(tokens, CallOptions{})
	})
	out := exec.MustCall([][]int32{{1, 2, 3, 4}})[0]
	require.Equal(t, []int{1, 4, 64}, out.Shape().Dimensions)

	// Layers 0 and 1 are non-shared: gate_proj/up_proj have FFNDim (64).
	require.Equal(t, []int{64, 32}, store.GetVariable("/layer_0/mlp/gate_proj/dense/weights").Shape().Dimensions)
	require.Equal(t, []int{64, 32}, store.GetVariable("/layer_1/mlp/gate_proj/dense/weights").Shape().Dimensions)
	// Layer 2 is KV-shared: gate_proj/up_proj have 2*FFNDim (128), down_proj maps 128 -> 32.
	require.Equal(t, []int{128, 32}, store.GetVariable("/layer_2/mlp/gate_proj/dense/weights").Shape().Dimensions)
	require.Equal(t, []int{128, 32}, store.GetVariable("/layer_2/mlp/up_proj/dense/weights").Shape().Dimensions)
	require.Equal(t, []int{32, 128}, store.GetVariable("/layer_2/mlp/down_proj/dense/weights").Shape().Dimensions)
}

func TestOnlyLastTokenLogits(t *testing.T) {
	backend := testutil.BuildTestBackend()
	store := model.NewStore()

	exec := model.MustNewExec(backend, store, func(scope *model.Scope, tokens *Node) (*Node, *Node) {
		m := New(scope).
			WithVocabSize(64).
			WithEmbedDim(32).
			WithNumLayers(2).
			WithNumHeads(2).
			WithHeadDim(16).
			WithFFNDim(64).
			WithRoPE(10000).
			WithCausalMask(true)
		fullLogits, _ := m.Forward(tokens, CallOptions{})
		m.Scope = m.Scope.ResetVisited()
		lastOnlyLogits, _ := m.Forward(tokens, CallOptions{OnlyLastTokenLogits: true})
		return fullLogits, lastOnlyLogits
	})

	outs := exec.MustCall([][]int32{{1, 2, 3, 4, 5}})
	full := outs[0].Value().([][][]float32)
	lastOnly := outs[1].Value().([][][]float32)

	require.Equal(t, []int{1, 5, 64}, outs[0].Shape().Dimensions)
	require.Equal(t, []int{1, 1, 64}, outs[1].Shape().Dimensions)
	assert.InDeltaSlice(t, full[0][4], lastOnly[0][0], 1e-5)
}
