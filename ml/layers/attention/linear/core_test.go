// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package linear

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/gomlx/compute/dtypes"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// referenceLinearAttention computes the exact token-by-token recurrence in float64 for all 6 variants:
//   - logDecay == nil (no decay), len == B*S*H (scalar decay), or len == B*S*H*Dk (channel-wise decay)
//   - beta == nil (pure associative write) or len == B*S*H (delta rule)
//   - numKHeads <= H (GVA repeat_interleave if numKHeads != H)
//   - optional Q/K L2 normalization and query scaling.
func referenceLinearAttention(
	qIn, kIn, vIn []float32,
	logDecay []float32, channelDecay bool,
	beta []float32,
	initState []float32,
	B, S, numKHeads, H, Dk, Dv int,
	useQKL2Norm bool,
	queryScale float64,
) (out, finalState []float32) {
	if queryScale == 0 {
		queryScale = 1.0 / math.Sqrt(float64(Dk))
	}
	if queryScale < 0 {
		queryScale = 1.0
	}
	rep := H / numKHeads

	out = make([]float32, B*S*H*Dv)
	state := make([]float64, B*H*Dk*Dv)
	if initState != nil {
		for i, x := range initState {
			state[i] = float64(x)
		}
	}

	qVec := make([]float64, Dk)
	kVec := make([]float64, Dk)
	kvMem := make([]float64, Dv)
	delta := make([]float64, Dv)

	for b := range B {
		for h := range H {
			kh := h / rep
			sBase := (b*H + h) * Dk * Dv
			for t := range S {
				qkBase := ((b*S+t)*numKHeads + kh) * Dk
				vBase := ((b*S+t)*H + h) * Dv

				var qSq, kSq float64
				for d := range Dk {
					qVec[d] = float64(qIn[qkBase+d])
					kVec[d] = float64(kIn[qkBase+d])
					qSq += qVec[d] * qVec[d]
					kSq += kVec[d] * kVec[d]
				}
				if useQKL2Norm {
					qInv := 1.0 / math.Sqrt(qSq+1e-6)
					kInv := 1.0 / math.Sqrt(kSq+1e-6)
					for d := range Dk {
						qVec[d] *= qInv
						kVec[d] *= kInv
					}
				}
				for d := range Dk {
					qVec[d] *= queryScale
				}

				// 1. Decay state.
				if logDecay != nil {
					if !channelDecay {
						decay := math.Exp(float64(logDecay[(b*S+t)*H+h]))
						for i := range Dk * Dv {
							state[sBase+i] *= decay
						}
					} else {
						gBase := ((b*S+t)*H + h) * Dk
						for di := range Dk {
							decay := math.Exp(float64(logDecay[gBase+di]))
							row := state[sBase+di*Dv : sBase+(di+1)*Dv]
							for dj := range Dv {
								row[dj] *= decay
							}
						}
					}
				}

				// 2. State update (delta rule or associative write).
				if beta != nil {
					bt := float64(beta[(b*S+t)*H+h])
					for dj := range Dv {
						kvMem[dj] = 0
					}
					for di := range Dk {
						row := state[sBase+di*Dv : sBase+(di+1)*Dv]
						for dj := range Dv {
							kvMem[dj] += kVec[di] * row[dj]
						}
					}
					for dj := range Dv {
						delta[dj] = (float64(vIn[vBase+dj]) - kvMem[dj]) * bt
					}
					for di := range Dk {
						row := state[sBase+di*Dv : sBase+(di+1)*Dv]
						for dj := range Dv {
							row[dj] += kVec[di] * delta[dj]
						}
					}
				} else {
					for di := range Dk {
						row := state[sBase+di*Dv : sBase+(di+1)*Dv]
						for dj := range Dv {
							row[dj] += kVec[di] * float64(vIn[vBase+dj])
						}
					}
				}

				// 3. Read output along q.
				for dj := range Dv {
					var sum float64
					for di := range Dk {
						sum += qVec[di] * state[sBase+di*Dv+dj]
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

func TestInvertUnitLowerTriangular(t *testing.T) {
	backend := testutil.BuildTestBackend()
	rng := rand.New(rand.NewPCG(42, 7))
	tol := 1e-5
	if strings.Contains(backend.Description(), "cuda") {
		tol = 5e-4 // CUDA Tensor Cores use TF32 for Float32 matmuls
	}

	for _, c := range []int{2, 4, 8, 16, 32, 64} {
		t.Run(fmt.Sprintf("C=%d", c), func(t *testing.T) {
			lData := make([]float32, 2*c*c)
			for batch := range 2 {
				for i := range c {
					for j := range i {
						lData[batch*c*c+i*c+j] = float32((rng.Float64()*2-1)*0.15) * float32(math.Exp(-0.05*float64(i-j)))
					}
				}
			}
			lTensor := tensors.FromFlatDataAndDimensions(lData, 2, c, c)
			defer lTensor.FinalizeAll()

			exec, err := NewExec(backend, func(L *Node) *Node {
				inv := invertUnitLowerTriangular(L)
				eye := makeIdentityMatrix(L.Graph(), c, 3)
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
				for i := range c {
					for j := range c {
						want := 0.0
						if i == j {
							want = 1.0
						}
						if diff := math.Abs(float64(prod[batch*c*c+i*c+j]) - want); diff > maxErr {
							maxErr = diff
						}
					}
				}
			}
			assert.Less(t, maxErr, tol, "C=%d max |(I+L)@inv - I|", c)
		})
	}
}

func TestCoreAllModesMatchReference(t *testing.T) {
	backend := testutil.BuildTestBackend()
	rng := rand.New(rand.NewPCG(123, 456))
	const (
		B         = 2
		numKHeads = 2
		H         = 4
		Dk        = 8
		Dv        = 6
	)

	type modeCase struct {
		name         string
		hasDecay     bool
		channelDecay bool
		useDelta     bool
		useQKL2Norm  bool
		chunkSize    int
	}

	modes := []modeCase{
		{name: "Simple", hasDecay: false, channelDecay: false, useDelta: false, useQKL2Norm: false, chunkSize: 16},
		{name: "Mamba2ScalarDecay", hasDecay: true, channelDecay: false, useDelta: false, useQKL2Norm: false, chunkSize: 16},
		{name: "GLAChannelDecay", hasDecay: true, channelDecay: true, useDelta: false, useQKL2Norm: false, chunkSize: 16},
		{name: "DeltaNet", hasDecay: false, channelDecay: false, useDelta: true, useQKL2Norm: true, chunkSize: 16},
		{name: "GatedDeltaNet", hasDecay: true, channelDecay: false, useDelta: true, useQKL2Norm: true, chunkSize: 64},
		{name: "KimiDeltaAttention", hasDecay: true, channelDecay: true, useDelta: true, useQKL2Norm: true, chunkSize: 16},
	}

	tol := 1e-4
	if strings.Contains(backend.Description(), "cuda") {
		tol = 1.5e-3 // CUDA Tensor Cores use TF32 for Float32 matmuls
	}

	for _, mc := range modes {
		t.Run(mc.name, func(t *testing.T) {
			type execVariant struct {
				name    string
				dynamic bool
				unroll  bool
			}
			variants := []execVariant{
				{name: "StaticWhile", dynamic: false, unroll: false},
				{name: "StaticUnroll", dynamic: false, unroll: true},
			}
			if backend.Capabilities().HasDynamicShapes() {
				variants = append(variants, execVariant{name: "DynamicWhile", dynamic: true, unroll: false})
			}

			for _, ev := range variants {
				t.Run(ev.name, func(t *testing.T) {
					exec, err := NewExec(backend, func(inputs []*Node) []*Node {
						q, k, v, initState := inputs[0], inputs[1], inputs[2], inputs[3]
						idx := 4
						var logDecay, beta *Node
						if mc.hasDecay {
							logDecay = inputs[idx]
							idx++
						}
						if mc.useDelta {
							beta = inputs[idx]
							idx++
						}
						out, state := Core(q, k, v, CoreOptions{
							LogDecay:     logDecay,
							Beta:         beta,
							InitialState: initState,
							UseQKL2Norm:  mc.useQKL2Norm,
							ChunkSize:    mc.chunkSize,
							UnrollChunks: ev.unroll,
						})
						return []*Node{out, state}
					})
					require.NoError(t, err)
					defer exec.Finalize()

					if ev.dynamic {
						dynSpecs := [][]string{
							{"batch", "seq_len", "", ""},
							{"batch", "seq_len", "", ""},
							{"batch", "seq_len", "", ""},
							{"batch", "", "", ""},
						}
						if mc.hasDecay {
							if mc.channelDecay {
								dynSpecs = append(dynSpecs, []string{"batch", "seq_len", "", ""})
							} else {
								dynSpecs = append(dynSpecs, []string{"batch", "seq_len", ""})
							}
						}
						if mc.useDelta {
							dynSpecs = append(dynSpecs, []string{"batch", "seq_len", ""})
						}
						exec.WithDynamicAxes(dynSpecs...)
					}

					seqLens := []int{1, 7, 16, 17, 35}
					if mc.chunkSize == 64 {
						seqLens = []int{1, 7, 64, 65, 130}
					}

					for _, S := range seqLens {
						qData := make([]float32, B*S*numKHeads*Dk)
						kData := make([]float32, B*S*numKHeads*Dk)
						vData := make([]float32, B*S*H*Dv)
						initData := make([]float32, B*H*Dk*Dv)
						for i := range qData {
							qData[i] = float32(rng.NormFloat64() * 0.4)
							kData[i] = float32(rng.NormFloat64() * 0.4)
						}
						for i := range vData {
							vData[i] = float32(rng.NormFloat64() * 0.4)
						}
						for i := range initData {
							initData[i] = float32(rng.NormFloat64() * 0.1)
						}

						var logDecayData, betaData []float32
						if mc.hasDecay {
							if mc.channelDecay {
								logDecayData = make([]float32, B*S*H*Dk)
							} else {
								logDecayData = make([]float32, B*S*H)
							}
							for i := range logDecayData {
								logDecayData[i] = float32(-0.02 - rng.Float64()*0.25)
							}
						}
						if mc.useDelta {
							betaData = make([]float32, B*S*H)
							for i := range betaData {
								betaData[i] = float32(0.05 + rng.Float64()*0.7)
							}
						}

						wantOut, wantState := referenceLinearAttention(
							qData, kData, vData,
							logDecayData, mc.channelDecay,
							betaData, initData,
							B, S, numKHeads, H, Dk, Dv,
							mc.useQKL2Norm, 0,
						)

						qT := tensors.FromFlatDataAndDimensions(qData, B, S, numKHeads, Dk)
						kT := tensors.FromFlatDataAndDimensions(kData, B, S, numKHeads, Dk)
						vT := tensors.FromFlatDataAndDimensions(vData, B, S, H, Dv)
						sT := tensors.FromFlatDataAndDimensions(initData, B, H, Dk, Dv)
						args := []any{qT, kT, vT, sT}
						var cleanup []*tensors.Tensor
						cleanup = append(cleanup, qT, kT, vT, sT)

						if mc.hasDecay {
							var gT *tensors.Tensor
							if mc.channelDecay {
								gT = tensors.FromFlatDataAndDimensions(logDecayData, B, S, H, Dk)
							} else {
								gT = tensors.FromFlatDataAndDimensions(logDecayData, B, S, H)
							}
							args = append(args, gT)
							cleanup = append(cleanup, gT)
						}
						if mc.useDelta {
							bT := tensors.FromFlatDataAndDimensions(betaData, B, S, H)
							args = append(args, bT)
							cleanup = append(cleanup, bT)
						}

						res, err := exec.Call(args...)
						require.NoError(t, err, "mode=%s variant=%s S=%d", mc.name, ev.name, S)

						var gotOut, gotState []float32
						res[0].MustConstFlatData(func(flat any) { gotOut = flat.([]float32) })
						res[1].MustConstFlatData(func(flat any) { gotState = flat.([]float32) })
						assert.Equal(t, []int{B, S, H, Dv}, res[0].Shape().Dimensions)
						assert.Equal(t, []int{B, H, Dk, Dv}, res[1].Shape().Dimensions)
						assert.Equal(t, dtypes.Float32, res[1].DType())

						maxOutDiff := 0.0
						for i := range wantOut {
							if d := math.Abs(float64(gotOut[i]) - float64(wantOut[i])); d > maxOutDiff {
								maxOutDiff = d
							}
						}
						maxStateDiff := 0.0
						for i := range wantState {
							if d := math.Abs(float64(gotState[i]) - float64(wantState[i])); d > maxStateDiff {
								maxStateDiff = d
							}
						}
						assert.Less(t, maxOutDiff, tol, "mode=%s variant=%s S=%d out diff", mc.name, ev.name, S)
						assert.Less(t, maxStateDiff, tol, "mode=%s variant=%s S=%d state diff", mc.name, ev.name, S)

						res[0].FinalizeAll()
						res[1].FinalizeAll()
						for _, ct := range cleanup {
							ct.FinalizeAll()
						}
					}
				})
			}
		})
	}
}
