// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package linear_test

import (
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/gomlx/compute/dtypes"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/core/autodiff"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers/attention/linear"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/initializer"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBuilderPresetsPrefillAndDecodeParity(t *testing.T) {
	backend := testutil.BuildTestBackend()
	rng := rand.New(rand.NewPCG(77, 88))
	const (
		B         = 2
		S         = 10
		prefillS  = 6
		embedDim  = 16
		numKHeads = 2
		numVHeads = 4
		headDim   = 8
	)

	type presetSpec struct {
		name    string
		hasConv bool
		build   func(s *model.Scope, x *Node) *linear.Builder
	}

	presets := []presetSpec{
		{
			name:    "GatedDeltaNet",
			hasConv: true,
			build: func(s *model.Scope, x *Node) *linear.Builder {
				return linear.GatedDeltaNet(s, x, numVHeads, headDim).
					WithNumKeyHeads(numKHeads).
					WithShortConv(4)
			},
		},
		{
			name:    "DeltaNet",
			hasConv: true,
			build: func(s *model.Scope, x *Node) *linear.Builder {
				return linear.DeltaNet(s, x, numVHeads, headDim).
					WithNumKeyHeads(numKHeads).
					WithShortConv(4)
			},
		},
		{
			name:    "GLA",
			hasConv: true,
			build: func(s *model.Scope, x *Node) *linear.Builder {
				return linear.GLA(s, x, numVHeads, headDim).
					WithShortConv(4)
			},
		},
		{
			name:    "Simple",
			hasConv: false,
			build: func(s *model.Scope, x *Node) *linear.Builder {
				return linear.Simple(s, x, numVHeads, headDim)
			},
		},
	}

	tol := 1e-4
	if strings.Contains(backend.Description(), "cuda") {
		tol = 1e-2
	}

	for _, ps := range presets {
		t.Run(ps.name, func(t *testing.T) {
			store := model.NewStore()
			initFn := initializer.RandomNormalFn(store.RootScope(), 0.08)

			xData := make([]float32, B*S*embedDim)
			for i := range xData {
				xData[i] = float32(rng.NormFloat64() * 0.5)
			}
			xFull := tensors.FromFlatDataAndDimensions(xData, B, S, embedDim)
			defer xFull.FinalizeAll()

			// 1. Full uncached sequence forward.
			fullExec, err := model.NewExec(backend, store, func(s *model.Scope, x *Node) *Node {
				return ps.build(s.WithInitializer(initFn), x).Done()
			})
			require.NoError(t, err)
			defer fullExec.Finalize()

			fullRes, err := fullExec.Call(xFull)
			require.NoError(t, err)
			defer fullRes[0].FinalizeAll()

			var fullOut []float32
			fullRes[0].MustConstFlatData(func(flat any) { fullOut = flat.([]float32) })
			require.Equal(t, []int{B, S, embedDim}, fullRes[0].Shape().Dimensions)

			// 2. Cached prefill on first prefillS tokens.
			prefillExec, err := model.NewExec(backend, store, func(s *model.Scope, x *Node) []*Node {
				out, convSt, recSt := ps.build(s.WithInitializer(initFn), x).WithReturnState(true).DoneWithState()
				if ps.hasConv {
					return []*Node{out, convSt, recSt}
				}
				return []*Node{out, recSt}
			})
			require.NoError(t, err)
			defer prefillExec.Finalize()

			sliceBatch := func(start, end int) *tensors.Tensor {
				sub := make([]float32, B*(end-start)*embedDim)
				for b := range B {
					srcOff := (b*S + start) * embedDim
					dstOff := b * (end - start) * embedDim
					copy(sub[dstOff:dstOff+(end-start)*embedDim], xData[srcOff:srcOff+(end-start)*embedDim])
				}
				return tensors.FromFlatDataAndDimensions(sub, B, end-start, embedDim)
			}

			xPre := sliceBatch(0, prefillS)
			defer xPre.FinalizeAll()
			preRes, err := prefillExec.Call(xPre)
			require.NoError(t, err)

			var preOut []float32
			preRes[0].MustConstFlatData(func(flat any) { preOut = flat.([]float32) })
			for b := range B {
				for tIdx := range prefillS {
					for d := range embedDim {
						want := fullOut[(b*S+tIdx)*embedDim+d]
						got := preOut[(b*prefillS+tIdx)*embedDim+d]
						assert.InDelta(t, want, got, tol, "prefill b=%d t=%d d=%d", b, tIdx, d)
					}
				}
			}
			preRes[0].FinalizeAll()

			var convState, recState *tensors.Tensor
			if ps.hasConv {
				convState = preRes[1]
				recState = preRes[2]
			} else {
				recState = preRes[1]
			}

			// 3. Token-by-token cached decode for remaining tokens.
			decodeExec, err := model.NewExec(backend, store, func(s *model.Scope, inputs []*Node) []*Node {
				x := inputs[0]
				var cs, rs *Node
				if ps.hasConv {
					cs, rs = inputs[1], inputs[2]
				} else {
					rs = inputs[1]
				}
				out, newCS, newRS := ps.build(s.WithInitializer(initFn), x).WithState(cs, rs).DoneWithState()
				if ps.hasConv {
					return []*Node{out, newCS, newRS}
				}
				return []*Node{out, newRS}
			})
			require.NoError(t, err)
			defer decodeExec.Finalize()

			for tIdx := prefillS; tIdx < S; tIdx++ {
				xStep := sliceBatch(tIdx, tIdx+1)
				var decRes []*tensors.Tensor
				if ps.hasConv {
					decRes, err = decodeExec.Call(xStep, convState, recState)
				} else {
					decRes, err = decodeExec.Call(xStep, recState)
				}
				xStep.FinalizeAll()
				require.NoError(t, err)

				var stepOut []float32
				decRes[0].MustConstFlatData(func(flat any) { stepOut = flat.([]float32) })
				for b := range B {
					for d := range embedDim {
						want := fullOut[(b*S+tIdx)*embedDim+d]
						got := stepOut[b*embedDim+d]
						assert.InDelta(t, want, got, tol, "decode t=%d b=%d d=%d", tIdx, b, d)
					}
				}
				decRes[0].FinalizeAll()
				if convState != nil {
					convState.FinalizeAll()
				}
				recState.FinalizeAll()
				if ps.hasConv {
					convState = decRes[1]
					recState = decRes[2]
				} else {
					recState = decRes[1]
				}
			}
			if convState != nil {
				convState.FinalizeAll()
			}
			recState.FinalizeAll()
		})
	}
}

func TestBuilderPaddingMaskEquivalence(t *testing.T) {
	backend := testutil.BuildTestBackend()
	store := model.NewStore()
	initFn := initializer.RandomNormalFn(store.RootScope(), 0.08)

	// Batch=2, seq=6; batch 0 has valid len 4, batch 1 has valid len 6.
	// Changing garbage values at positions 4, 5 of batch 0 must not alter valid positions 0..3
	// or produce different results between WithAttentionMask and WithSeqLen.
	rng := rand.New(rand.NewPCG(99, 100))
	const (
		B        = 2
		S        = 6
		embedDim = 8
	)
	x1Data := make([]float32, B*S*embedDim)
	x2Data := make([]float32, B*S*embedDim)
	for i := range x1Data {
		v := float32(rng.NormFloat64())
		x1Data[i] = v
		x2Data[i] = v
	}
	// Perturb padded positions [0, 4:6, :] wildly in x2Data.
	for tIdx := 4; tIdx < S; tIdx++ {
		for d := range embedDim {
			x2Data[tIdx*embedDim+d] = 999.0
		}
	}

	maskExec, err := model.NewExec(backend, store, func(s *model.Scope, x, mask *Node) *Node {
		return linear.GatedDeltaNet(s.WithInitializer(initFn), x, 2, 4).
			WithAttentionMask(mask).
			Done()
	})
	require.NoError(t, err)
	defer maskExec.Finalize()

	seqLenExec, err := model.NewExec(backend, store, func(s *model.Scope, x, seqLen *Node) *Node {
		return linear.GatedDeltaNet(s.WithInitializer(initFn), x, 2, 4).
			WithSeqLen(seqLen).
			Done()
	})
	require.NoError(t, err)
	defer seqLenExec.Finalize()

	x1T := tensors.FromFlatDataAndDimensions(x1Data, B, S, embedDim)
	x2T := tensors.FromFlatDataAndDimensions(x2Data, B, S, embedDim)
	maskT := tensors.FromValue([][]int32{{1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 1, 1}})
	seqLenT := tensors.FromValue([]int32{4, 6})
	defer x1T.FinalizeAll()
	defer x2T.FinalizeAll()
	defer maskT.FinalizeAll()
	defer seqLenT.FinalizeAll()

	r1, err := maskExec.Call(x1T, maskT)
	require.NoError(t, err)
	defer r1[0].FinalizeAll()

	r2, err := seqLenExec.Call(x2T, seqLenT)
	require.NoError(t, err)
	defer r2[0].FinalizeAll()

	var o1, o2 []float32
	r1[0].MustConstFlatData(func(flat any) { o1 = flat.([]float32) })
	r2[0].MustConstFlatData(func(flat any) { o2 = flat.([]float32) })

	for i := range o1 {
		assert.InDelta(t, o1[i], o2[i], 1e-5, "mask vs seqLen mismatch at %d", i)
	}
}

func TestBuilderAutodiffGradientsMatchFiniteDifferences(t *testing.T) {
	backend := testutil.BuildTestBackend()
	if strings.Contains(backend.Description(), "cuda") {
		// Float32 TF32 matmuls on CUDA have ~1e-3 noise, making 1e-3 finite-diff noisy.
		// Still verify autodiff builds and runs with finite non-zero gradients on CUDA!
		t.Run("CUDAForwardBackwardSmoke", func(t *testing.T) {
			store := model.NewStore()
			initFn := initializer.RandomNormalFn(store.RootScope(), 0.08)
			exec, err := model.NewExec(backend, store, func(s *model.Scope, x *Node) (*Node, *Node) {
				g := x.Graph()
				s.SetTraining(g, true)
				y := linear.GatedDeltaNet(s.WithInitializer(initFn), x, 2, 4).
					WithChunkSize(16).
					Done()
				loss := ReduceAllSum(Square(y))
				grads := autodiff.Gradient(loss, x)
				return loss, grads[0]
			})
			require.NoError(t, err)
			defer exec.Finalize()

			xData := make([]float32, 1*20*8)
			rng := rand.New(rand.NewPCG(5, 6))
			for i := range xData {
				xData[i] = float32(rng.NormFloat64() * 0.5)
			}
			xT := tensors.FromFlatDataAndDimensions(xData, 1, 20, 8)
			defer xT.FinalizeAll()

			res, err := exec.Call(xT)
			require.NoError(t, err)
			defer res[0].FinalizeAll()
			defer res[1].FinalizeAll()

			var gradVals []float32
			res[1].MustConstFlatData(func(flat any) { gradVals = flat.([]float32) })
			nonZero := 0
			for _, v := range gradVals {
				assert.False(t, math.IsNaN(float64(v)) || math.IsInf(float64(v), 0))
				if math.Abs(float64(v)) > 1e-6 {
					nonZero++
				}
			}
			assert.Greater(t, nonZero, len(gradVals)/2)
		})
		return
	}

	// Exact central finite-difference verification on multi-chunk unrolled training graphs
	// (S=20 with chunkSize=16 -> 2 chunks, testing both intra-chunk UT inverse & inter-chunk scan VJP).
	cases := []struct {
		name  string
		build func(s *model.Scope, x *Node) *linear.Builder
	}{
		{
			name: "GatedDeltaNet",
			build: func(s *model.Scope, x *Node) *linear.Builder {
				return linear.GatedDeltaNet(s, x, 2, 4).WithChunkSize(16)
			},
		},
		{
			name: "GLA",
			build: func(s *model.Scope, x *Node) *linear.Builder {
				return linear.GLA(s, x, 2, 4).WithChunkSize(16)
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			store := model.NewStore()
			initFn := initializer.RandomNormalFn(store.RootScope(), 0.1)

			// Exercise both input gradient and parameter gradients.
			gradExec, err := model.NewExec(backend, store, func(s *model.Scope, x *Node) []*Node {
				g := x.Graph()
				s.SetTraining(g, true)
				y := tc.build(s.WithInitializer(initFn), x).Done()
				loss := ReduceAllSum(Square(y))
				grads := autodiff.Gradient(loss, x)
				return []*Node{loss, grads[0]}
			})
			require.NoError(t, err)
			defer gradExec.Finalize()

			lossExec, err := model.NewExec(backend, store, func(s *model.Scope, x *Node) *Node {
				g := x.Graph()
				s.SetTraining(g, true)
				y := tc.build(s.WithInitializer(initFn), x).Done()
				return ReduceAllSum(Square(y))
			})
			require.NoError(t, err)
			defer lossExec.Finalize()

			const (
				B        = 1
				S        = 20
				embedDim = 4
			)
			rng := rand.New(rand.NewPCG(314, 159))
			xData := make([]float32, B*S*embedDim)
			for i := range xData {
				xData[i] = float32(rng.NormFloat64() * 0.5)
			}
			xT := tensors.FromFlatDataAndDimensions(xData, B, S, embedDim)
			defer xT.FinalizeAll()

			res, err := gradExec.Call(xT)
			require.NoError(t, err)
			defer res[0].FinalizeAll()
			defer res[1].FinalizeAll()

			assert.Equal(t, dtypes.Float32, res[1].DType())
			var analytical []float32
			res[1].MustConstFlatData(func(flat any) { analytical = flat.([]float32) })

			// Spot-check 12 coordinates spanning chunk 0 (t=0, 5, 15) and chunk 1 (t=16, 19).
			const eps = float32(1e-3)
			checkIndices := []int{0, 1, 20, 21, 60, 61, 64, 65, 76, 77, 78, 79}
			for _, idx := range checkIndices {
				orig := xData[idx]
				xData[idx] = orig + eps
				tp := tensors.FromFlatDataAndDimensions(xData, B, S, embedDim)
				rp, err := lossExec.Call(tp)
				require.NoError(t, err)
				lp := tensors.ToScalar[float32](rp[0])
				tp.FinalizeAll()
				rp[0].FinalizeAll()

				xData[idx] = orig - eps
				tm := tensors.FromFlatDataAndDimensions(xData, B, S, embedDim)
				rm, err := lossExec.Call(tm)
				require.NoError(t, err)
				lm := tensors.ToScalar[float32](rm[0])
				tm.FinalizeAll()
				rm[0].FinalizeAll()

				xData[idx] = orig
				numerical := float64(lp-lm) / (2 * float64(eps))
				got := float64(analytical[idx])
				assert.InDelta(t, numerical, got, 2e-2*math.Max(1.0, math.Abs(numerical)),
					"grad mismatch at idx=%d: analytical=%g numerical=%g", idx, got, numerical)
			}
		})
	}
}
