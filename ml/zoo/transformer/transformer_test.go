// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package transformer

import (
	"fmt"
	"math"
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
		cfg := New(1000, 128, 4, 8, 16)
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
		cfg := New(1000, 128, 4, 8, 16).
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

		cfg := NewFromScope(scope)
		assert.Equal(t, 1000, cfg.VocabSize)
		assert.Equal(t, 128, cfg.EmbedDim)
		assert.Equal(t, 4, cfg.NumLayers)
		assert.Equal(t, 8, cfg.NumHeads)
		assert.Equal(t, 16, cfg.HeadDim)
		assert.Equal(t, 256, cfg.FFNDim)
		assert.Equal(t, 1024, cfg.MaxPosEmbed)
		assert.Equal(t, dtypes.Float16, cfg.DType)
		assert.Equal(t, 0.1, cfg.Dropout)
		assert.Equal(t, dtypes.Float16, cfg.DType)
		assert.Equal(t, 0.1, cfg.Dropout)
		require.NotNil(t, cfg.posEncoder)
		rope, ok := cfg.posEncoder.(*pos.RoPE)
		require.True(t, ok)
		assert.Equal(t, 5000.0, rope.BaseFreq)
		assert.Equal(t, cfg.Normalization, layers.NormalizationNone)
		assert.False(t, cfg.UseBias)
	})

	t.Run("FromScope", func(t *testing.T) {
		store := model.NewStore()
		scope := store.RootScope()
		scope.SetParams(map[string]any{
			ParamFFNDim:        256,
			ParamMaxPosEmbed:   1024,
			ParamDType:         "float16",
			ParamDropout:       0.1,
			ParamUseRoPE:       true,
			ParamRoPEBaseFreq:  5000.0,
			ParamUseBias:       false,
			ParamNormalization: "", // or layers.NormalizationNone.
		})

		cfg := New(1000, 128, 4, 8, 16).FromScope(scope)
		assert.Equal(t, 256, cfg.FFNDim)
		assert.Equal(t, 1024, cfg.MaxPosEmbed)
		assert.Equal(t, dtypes.Float16, cfg.DType)
		assert.Equal(t, 0.1, cfg.Dropout)
		assert.Equal(t, dtypes.Float16, cfg.DType)
		assert.Equal(t, 0.1, cfg.Dropout)
		require.NotNil(t, cfg.posEncoder)
		rope, ok := cfg.posEncoder.(*pos.RoPE)
		require.True(t, ok)
		assert.Equal(t, 5000.0, rope.BaseFreq)
		assert.Equal(t, layers.NormalizationNone, cfg.Normalization)
		assert.False(t, cfg.UseBias)
	})
}

// TestTransformerBuilder groups transformer builder/exposed functions tests.
func TestTransformerBuilder(t *testing.T) {
	t.Run("BuildGraph", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		store := model.NewStore()
		scope := store.RootScope()
		cfg := New(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128)
		g := NewGraph(backend, "BuildGraph")
		tokens := IotaFull(g, shapes.Make(dtypes.Int32, 2, 8))
		tokens = Mod(tokens, Const(g, int32(cfg.VocabSize)))
		logits := cfg.Logits(scope, tokens, nil)
		require.NotNil(t, logits)
		assert.Equal(t, []int{2, 8, 100}, logits.Shape().Dimensions)
		assert.Equal(t, dtypes.Float32, logits.DType())
	})

	t.Run("BuildGraphWithKVCache", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		store := model.NewStore()
		scope := store.RootScope()
		model := New(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128)
		g := NewGraph(backend, "BuildGraphWithKVCache")
		prompt := IotaFull(g, shapes.Make(dtypes.Int32, 2, 5))
		prompt = Mod(prompt, Const(g, int32(model.VocabSize)))
		logits, _ := model.LogitsWithKVCache(scope, prompt, nil, nil)
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
		model := New(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128).WithRoPE(10000.0)
		g := NewGraph(backend, "WithRoPE")
		tokens := IotaFull(g, shapes.Make(dtypes.Int32, 1, 4))
		tokens = Mod(tokens, Const(g, int32(model.VocabSize)))
		logits, _ := model.LogitsWithKVCache(scope, tokens, nil, nil)
		require.NotNil(t, logits)
		assert.Equal(t, []int{1, 4, 100}, logits.Shape().Dimensions)
	})

	t.Run("ExplicitKVCacheExecution", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		store := model.NewStore()
		scope := store.RootScope()
		m := New(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128)
		
		// 1. Initialize Cache
		m.populateOrderedScopes(scope)
		m.populateLayerTypes(scope)
		cacheTensors := m.KVCache.InitializeTensors(2, 4, 16, dtypes.Float32, 5) // batch=2, numKVHeads=4, headDim=16, seqLen=5
		
		// 2. Build graph for prompt execution
		exec, err := model.NewExec(backend, store, func(scope *model.Scope, inputs []*Node) []*Node {
			tokens := inputs[0]
			cacheNodes := inputs[1:]
			cache := m.KVCache.DeserializeNodes(cacheNodes)
			logits, updatedCache := m.Forward(scope, tokens, nil, nil, cache)
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
		model := New(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128).WithNormalization("none")
		g := NewGraph(backend, "WithoutLayerNorm")
		tokens := IotaFull(g, shapes.Make(dtypes.Int32, 2, 8))
		tokens = Mod(tokens, Const(g, int32(model.VocabSize)))
		logits := model.Logits(scope, tokens, nil)
		require.NotNil(t, logits)
		assert.Equal(t, []int{2, 8, 100}, logits.Shape().Dimensions)
	})

	t.Run("WithoutBias", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		store := model.NewStore()
		scope := store.RootScope()
		model := New(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128).WithBias(false)
		g := NewGraph(backend, "WithoutBias")
		tokens := IotaFull(g, shapes.Make(dtypes.Int32, 2, 8))
		tokens = Mod(tokens, Const(g, int32(model.VocabSize)))
		logits := model.Logits(scope, tokens, nil)
		require.NotNil(t, logits)
		assert.Equal(t, []int{2, 8, 100}, logits.Shape().Dimensions)
	})

	t.Run("WithDropout", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		store := model.NewStore()
		scope := store.RootScope()
		model := New(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128).WithDropout(0.1)
		g := NewGraph(backend, "WithDropout")
		tokens := IotaFull(g, shapes.Make(dtypes.Int32, 2, 8))
		tokens = Mod(tokens, Const(g, int32(model.VocabSize)))
		logits := model.Logits(scope, tokens, nil)
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

			model := New(100, 64, 2, 4, 16).
				WithFFNDim(128).
				WithMaxPosEmbed(128)

			seqLen := 8

			manager := backend
			g := NewGraph(manager, "TestTransformerBatchSize")

			tokens := IotaFull(g, shapes.Make(dtypes.Int32, batchSize, seqLen))
			tokens = Mod(tokens, Const(g, int32(model.VocabSize)))

			logits := model.Logits(scope, tokens, nil)
			require.NotNil(t, logits)
			assert.Equal(t, []int{batchSize, seqLen, 100}, logits.Shape().Dimensions)
		})
	}
}

// TestSeqLensFromMask verifies seqLensFromMask converts a contiguous right-padded [B, Skv]
// boolean mask to correct int32 [B] sequence lengths, and that the full model path using
// WithSeqLens produces numerically equivalent output to the WithMask path.
func TestSeqLensFromMask(t *testing.T) {
	backend := testutil.BuildTestBackend()

	t.Run("HelperValues", func(t *testing.T) {
		// mask[0] has 3 valid tokens, mask[1] has 5 valid tokens (contiguous right-pad).
		maskData := [][]bool{
			{true, true, true, false, false},
			{true, true, true, true, true},
		}
		maskTensor := tensors.FromValue(maskData)

		result := MustExecOnce(backend, func(m *Node) *Node {
			q, _ := seqLensFromMask(m)
			return q
		}, maskTensor)

		tensors.MustConstFlatData[int32](result, func(flat []int32) {
			require.Len(t, flat, 2)
			assert.Equal(t, int32(3), flat[0], "batch[0] valid tokens")
			assert.Equal(t, int32(5), flat[1], "batch[1] valid tokens")
		})
	})

	t.Run("ModelParityMaskVsSeqLens", func(t *testing.T) {
		// Run two model instances with shared weights: one sees nil mask (no masking),
		// the other sees a full-length mask ([B, S] all-ones) which is rank-2 and triggers
		// the WithSeqLens path with seqlens == S.  Both paths must produce the same logits
		// because a full-length seqlen disables masking just as nil does.
		const (
			vocabSize = 100
			embedDim  = 32
			numLayers = 1
			numHeads  = 2
			headDim   = 8
			seqLen    = 4
			batchSize = 2
		)
		store := model.NewStore()
		m := New(vocabSize, embedDim, numLayers, numHeads, headDim).
			WithFFNDim(64).
			WithMaxPosEmbed(32)

		tokenData := [][]int32{
			{1, 2, 3, 4},
			{5, 6, 7, 8},
		}
		tokenTensor := tensors.FromValue(tokenData)

		// Full mask: all positions valid — seqlen path returns S for each batch element.
		fullMaskData := [][]bool{
			{true, true, true, true},
			{true, true, true, true},
		}
		fullMaskTensor := tensors.FromValue(fullMaskData)

		// nil-mask run.
		nilMaskLogits := model.MustExecOnce(backend, store, func(scope *model.Scope, tokens *Node) *Node {
			return m.Logits(scope, tokens, nil)
		}, tokenTensor)

		// full-mask run (rank-2, triggers WithSeqLens path).
		fullMaskLogits := model.MustExecOnce(backend, store, func(scope *model.Scope, tokens, mask *Node) *Node {
			return m.Logits(scope, tokens, mask)
		}, tokenTensor, fullMaskTensor)

		// Both should produce identical logits.
		tensors.MustConstFlatData[float32](nilMaskLogits, func(nilFlat []float32) {
			tensors.MustConstFlatData[float32](fullMaskLogits, func(fullFlat []float32) {
				require.Equal(t, len(nilFlat), len(fullFlat))
				for i, v := range nilFlat {
					if math.IsNaN(float64(v)) {
						continue
					}
					assert.InDelta(t, v, fullFlat[i], 1e-4, "logit[%d] mismatch: nil-mask=%v seqlens=%v", i, v, fullFlat[i])
				}
			})
		})
	})
}
