// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package transformer

import (
	"math"
	"math/rand/v2"
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

// referenceGatedDeltaRule computes the exact token-by-token recurrent gated delta rule in float64
// matching HuggingFace's torch_recurrent_gated_delta_rule (with L2-normalized q/k already applied
// and q scaled by 1/sqrt(Dk)).
func referenceGatedDeltaRule(
	q, k, v []float32, // [B, S, H, D]
	gDecay, beta []float32, // [B, S, H]
	initState []float32, // nil or [B, H, Dk, Dv]
	B, S, H, Dk, Dv int,
) (out []float32, finalState []float32) {
	out = make([]float32, B*S*H*Dv)
	state := make([]float64, B*H*Dk*Dv)
	if initState != nil {
		for i, x := range initState {
			state[i] = float64(x)
		}
	}

	for b := range B {
		for h := range H {
			sBase := (b*H + h) * Dk * Dv
			for t := range S {
				gb := float64(gDecay[(b*S+t)*H+h])
				bt := float64(beta[(b*S+t)*H+h])
				decay := math.Exp(gb)
				qBase := ((b*S+t)*H + h) * Dk
				vBase := ((b*S+t)*H + h) * Dv

				// Decay state
				for i := range Dk * Dv {
					state[sBase+i] *= decay
				}
				// kv_mem = k_t @ S
				kvMem := make([]float64, Dv)
				for di := range Dk {
					kv := float64(k[qBase+di])
					row := state[sBase+di*Dv : sBase+(di+1)*Dv]
					for dj := range Dv {
						kvMem[dj] += kv * row[dj]
					}
				}
				// delta = (v_t - kv_mem) * beta_t
				delta := make([]float64, Dv)
				for dj := range Dv {
					delta[dj] = (float64(v[vBase+dj]) - kvMem[dj]) * bt
				}
				// S += outer(k_t, delta)
				for di := range Dk {
					kv := float64(k[qBase+di])
					row := state[sBase+di*Dv : sBase+(di+1)*Dv]
					for dj := range Dv {
						row[dj] += kv * delta[dj]
					}
				}
				// out_t = q_t @ S
				for dj := range Dv {
					var sum float64
					for di := range Dk {
						sum += float64(q[qBase+di]) * state[sBase+di*Dv+dj]
					}
					out[vBase+dj] = float32(sum)
				}
			}
		}
	}

	finalState = make([]float32, len(state))
	for i, x := range state {
		finalState[i] = float32(x)
	}
	return out, finalState
}

func TestInvertUnitLowerTriangular64(t *testing.T) {
	backend := testutil.BuildTestBackend()
	rng := rand.New(rand.NewPCG(42, 7))
	const N = 64
	lData := make([]float32, 2*N*N)
	for batch := range 2 {
		for i := range N {
			for j := range i {
				// Realistic bounded strictly-lower-triangular values (like k_beta @ k^T * decay).
				lData[batch*N*N+i*N+j] = float32((rng.Float64()*2-1)*0.15) * float32(math.Exp(-0.05*float64(i-j)))
			}
		}
	}

	lTensor := tensors.FromFlatDataAndDimensions(lData, 2, N, N)
	exec, err := NewExec(backend, func(L *Node) *Node {
		inv := invertUnitLowerTriangular64(L)
		// Compute (I + L) @ inv; should equal I.
		eye := makeIdentityMatrix(L.Graph(), N, 3)
		m := Add(L, BroadcastLike(eye, L))
		return DotGeneral(m, []int{2}, []int{0}, inv, []int{1}, []int{0})
	})
	require.NoError(t, err)
	defer exec.Finalize()

	res, err := exec.Call(lTensor)
	require.NoError(t, err)
	defer res[0].FinalizeAll()

	var prod []float32
	res[0].MustConstFlatData(func(flat any) { prod = flat.([]float32) })
	maxErr := 0.0
	for batch := range 2 {
		for i := range N {
			for j := range N {
				want := 0.0
				if i == j {
					want = 1.0
				}
				diff := math.Abs(float64(prod[batch*N*N+i*N+j]) - want)
				if diff > maxErr {
					maxErr = diff
				}
			}
		}
	}
	tol := 1e-5
	if strings.Contains(backend.Name(), "xla") {
		tol = 5e-4 // CUDA Tensor Cores use TF32 for Float32 matmuls
	}
	assert.Less(t, maxErr, tol, "max |(I+L)@inv - I|")
}

func TestGatedDeltaRuleChunkedMatchesReference(t *testing.T) {
	backend := testutil.BuildTestBackend()
	rng := rand.New(rand.NewPCG(123, 456))
	const (
		B  = 2
		H  = 4
		Dk = 8
		Dv = 8
	)

	makeInputs := func(S int) (q, k, v, gDecay, beta, initState []float32) {
		q = make([]float32, B*S*H*Dk)
		k = make([]float32, B*S*H*Dk)
		v = make([]float32, B*S*H*Dv)
		gDecay = make([]float32, B*S*H)
		beta = make([]float32, B*S*H)
		initState = make([]float32, B*H*Dk*Dv)

		// Generate L2-normalized q and k, scaled q by 1/sqrt(Dk).
		for idx := range B * S * H {
			var qNorm, kNorm float64
			for d := range Dk {
				qv := rng.NormFloat64()
				kv := rng.NormFloat64()
				q[idx*Dk+d] = float32(qv)
				k[idx*Dk+d] = float32(kv)
				qNorm += qv * qv
				kNorm += kv * kv
			}
			qInv := 1.0 / (math.Sqrt(qNorm+1e-6) * math.Sqrt(float64(Dk)))
			kInv := 1.0 / math.Sqrt(kNorm+1e-6)
			for d := range Dk {
				q[idx*Dk+d] = float32(float64(q[idx*Dk+d]) * qInv)
				k[idx*Dk+d] = float32(float64(k[idx*Dk+d]) * kInv)
			}
			for d := range Dv {
				v[idx*Dv+d] = float32(rng.NormFloat64() * 0.5)
			}
			gDecay[idx] = float32(-0.01 - rng.Float64()*0.3)
			beta[idx] = float32(0.05 + rng.Float64()*0.8)
		}
		for i := range initState {
			initState[i] = float32(rng.NormFloat64() * 0.1)
		}
		return
	}

	// Test both static and dynamic sequence lengths across single-chunk, exact-chunk, and multi-chunk lengths.
	for _, dynamic := range []bool{false, true} {
		if dynamic && !backend.Capabilities().HasDynamicShapes() {
			continue
		}
		name := "Static"
		if dynamic {
			name = "Dynamic"
		}
		t.Run(name, func(t *testing.T) {
			exec, err := NewExec(backend, func(q, k, v, gDecay, beta, initState *Node) (*Node, *Node) {
				return gatedDeltaRuleChunked(q, k, v, gDecay, beta, initState, gatedDeltaRuleChunkSize)
			})
			require.NoError(t, err)
			defer exec.Finalize()
			if dynamic {
				exec.WithDynamicAxes(
					[]string{"batch", "seq_len", "", ""},
					[]string{"batch", "seq_len", "", ""},
					[]string{"batch", "seq_len", "", ""},
					[]string{"batch", "seq_len", ""},
					[]string{"batch", "seq_len", ""},
					[]string{"batch", "", "", ""},
				)
			}

			for _, S := range []int{1, 7, 64, 65, 130} {
				q, k, v, gDecay, beta, initState := makeInputs(S)
				wantOut, wantState := referenceGatedDeltaRule(q, k, v, gDecay, beta, initState, B, S, H, Dk, Dv)

				qT := tensors.FromFlatDataAndDimensions(q, B, S, H, Dk)
				kT := tensors.FromFlatDataAndDimensions(k, B, S, H, Dk)
				vT := tensors.FromFlatDataAndDimensions(v, B, S, H, Dv)
				gT := tensors.FromFlatDataAndDimensions(gDecay, B, S, H)
				bT := tensors.FromFlatDataAndDimensions(beta, B, S, H)
				sT := tensors.FromFlatDataAndDimensions(initState, B, H, Dk, Dv)

				res, err := exec.Call(qT, kT, vT, gT, bT, sT)
				require.NoError(t, err, "S=%d", S)

				var gotOut, gotState []float32
				res[0].MustConstFlatData(func(flat any) { gotOut = flat.([]float32) })
				res[1].MustConstFlatData(func(flat any) { gotState = flat.([]float32) })

				assert.Equal(t, []int{B, S, H, Dv}, res[0].Shape().Dimensions, "S=%d", S)
				assert.Equal(t, []int{B, H, Dk, Dv}, res[1].Shape().Dimensions, "S=%d", S)

				maxOutDiff := 0.0
				for i := range wantOut {
					d := math.Abs(float64(gotOut[i]) - float64(wantOut[i]))
					if d > maxOutDiff {
						maxOutDiff = d
					}
				}
				maxStateDiff := 0.0
				for i := range wantState {
					d := math.Abs(float64(gotState[i]) - float64(wantState[i]))
					if d > maxStateDiff {
						maxStateDiff = d
					}
				}
				deltaTol := 1e-4
				if strings.Contains(backend.Name(), "xla") {
					deltaTol = 1e-3 // CUDA Tensor Cores use TF32 for Float32 matmuls
				}
				assert.Less(t, maxOutDiff, deltaTol, "S=%d output diff", S)
				assert.Less(t, maxStateDiff, deltaTol, "S=%d state diff", S)

				qT.FinalizeAll()
				kT.FinalizeAll()
				vT.FinalizeAll()
				gT.FinalizeAll()
				bT.FinalizeAll()
				sT.FinalizeAll()
				res[0].FinalizeAll()
				res[1].FinalizeAll()
			}
		})
	}
}

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
