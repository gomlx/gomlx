// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"math"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/support/exceptions"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/require"
)

// backendSupportsBiasedFusion reports whether the backend implements fused SDPA with a [B,H,S,S]
// bias, using the same probe-then-detect pattern as backendSupportsFusion. Dims match the real test
// (S=64, H=2, D=64) so "probe succeeds" reliably implies the real test's fused arm will fuse.
func backendSupportsBiasedFusion(backend compute.Backend) bool {
	const B, S, H, D = 1, 64, 2, 64
	scale := 1.0 / math.Sqrt(float64(D))
	q := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 20), B, S, H, D)
	biasTensor := tensors.FromFlatDataAndDimensions(make([]float32, B*H*S*S), B, H, S, S)
	supported := false
	probeErr := exceptions.TryCatch[error](func() {
		exec := MustNewExec(backend, func(qn, bn *Node) *Node {
			round := func(n *Node) *Node { return ConvertDType(n, dtypes.BFloat16) }
			cfg := &compute.ScaledDotProductAttentionConfig{
				Bias:   InternalBackendOutputs(round(bn))[0],
				Scale:  scale,
				Causal: false,
			}
			fused, _ := BackendFusedScaledDotProductAttention(
				round(qn), round(qn), round(qn),
				compute.AxesLayoutBSHD, cfg)
			return ConvertDType(ReduceAllSum(fused), dtypes.Float32)
		})
		_ = exec.MustCall(q, biasTensor)
		supported = true
	})
	if probeErr != nil {
		if compute.IsNotImplemented(probeErr) {
			return false
		}
		panic(probeErr)
	}
	return supported
}

// TestCoreAttentionBiasDecomposed checks that a non-nil AttentionBias yields output matching the
// inline reference softmax(scale*QK^T + bias)*V on the CPU backend (decomposed). The bias strongly
// favors key position 1, so the output diverges measurably from no-bias attention; the test fails
// if bias is silently dropped.
func TestCoreAttentionBiasDecomposed(t *testing.T) {
	backend := testutil.BuildTestBackend()
	// [B=1, S=2, H=1, D=4] — BSHD layout, scores are [B,Sq,H,Skv] = [1,2,1,2].
	const B, S, H, D = 1, 2, 1, 4
	scale := 1.0 / math.Sqrt(float64(D))

	q := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 1), B, S, H, D)
	k := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 2), B, S, H, D)
	v := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 3), B, S, H, D)
	// Bias [B,H,Sq,Skv] = [1,1,2,2]: strongly favor key position 1 for both query positions.
	biasData := []float32{-100, 0, -100, 0}
	bias := tensors.FromFlatDataAndDimensions(biasData, B, H, S, S)

	store := model.NewStore()
	exec := model.MustNewExec(backend, store, func(scope *model.Scope, qn, kn, vn, bn *Node) []*Node {
		// Decomposed forced via WantCoefficients.
		biasedOut, _ := Core(qn, kn, vn, LayoutBSHD, CoreOptions{Scale: scale, WantCoefficients: true, AttentionBias: bn})

		// Inline reference: softmax(scale*QK^T + bias)*V. bn is [B,H,Sq,Skv]; BSHD scores are
		// [B,Sq,H,Skv] so permute before Add.
		scores := MulScalar(Einsum("bqhd,bkhd->bqhk", qn, kn), scale)
		scores = Add(scores, TransposeAllAxes(bn, 0, 2, 1, 3))
		attn := Softmax(scores, -1)
		refOut := Einsum("bqhk,bkhd->bqhd", attn, vn)

		noBiasOut, _ := Core(qn, kn, vn, LayoutBSHD, CoreOptions{Scale: scale, WantCoefficients: true})
		return []*Node{biasedOut, refOut, noBiasOut}
	})

	out := exec.MustCall(q, k, v, bias)
	biasedData := out[0].Value().([][][][]float32)
	refData := out[1].Value().([][][][]float32)
	noBiasData := out[2].Value().([][][][]float32)

	for b := range biasedData {
		for s := range biasedData[b] {
			for h := range biasedData[b][s] {
				for d := range biasedData[b][s][h] {
					require.InDeltaf(t, refData[b][s][h][d], biasedData[b][s][h][d], 1e-5,
						"biased Core output != inline reference at [%d][%d][%d][%d]", b, s, h, d)
				}
			}
		}
	}

	maxDiff := float32(0)
	for b := range biasedData {
		for s := range biasedData[b] {
			for h := range biasedData[b][s] {
				for d := range biasedData[b][s][h] {
					if diff := abs32(biasedData[b][s][h][d] - noBiasData[b][s][h][d]); diff > maxDiff {
						maxDiff = diff
					}
				}
			}
		}
	}
	require.Greater(t, maxDiff, float32(0.01),
		"biased and unbiased outputs are too similar; bias appears to be ignored (maxDiff=%v)", maxDiff)
}

// TestCoreAttentionBiasGQA verifies AttentionBias is correctly applied under GQA (numQueryHeads >
// numKVHeads). On the decomposed path the score tensor is 5D ([B,H,G,Sq,Skv] for BHSD), so the 4D
// bias must be reshaped before Add; without the reshape the Add broadcasts along the wrong axis.
func TestCoreAttentionBiasGQA(t *testing.T) {
	backend := testutil.BuildTestBackend()
	// 4 query heads, 2 kv heads (groupSize=2), BHSD layout.
	const B, QH, KVH, Sq, Skv, D = 1, 4, 2, 3, 3, 4
	scale := 1.0 / math.Sqrt(float64(D))

	q := tensors.FromFlatDataAndDimensions(randFlat(B*QH*Sq*D, 7), B, QH, Sq, D)
	k := tensors.FromFlatDataAndDimensions(randFlat(B*KVH*Skv*D, 8), B, KVH, Skv, D)
	v := tensors.FromFlatDataAndDimensions(randFlat(B*KVH*Skv*D, 9), B, KVH, Skv, D)
	// Bias [B,QH,Sq,Skv]: strongly favor key position 0 for every query position.
	biasFlat := make([]float32, B*QH*Sq*Skv)
	for i := range biasFlat {
		if i%Skv != 0 {
			biasFlat[i] = -100
		}
	}
	biasTensor := tensors.FromFlatDataAndDimensions(biasFlat, B, QH, Sq, Skv)

	store := model.NewStore()
	exec := model.MustNewExec(backend, store, func(scope *model.Scope, qn, kn, vn, bn *Node) []*Node {
		biasedOut, _ := Core(qn, kn, vn, LayoutBHSD, CoreOptions{Scale: scale, WantCoefficients: true, AttentionBias: bn})

		// Inline reference: explicit GQA grouping, bias reshaped to 5D per group.
		group := QH / KVH
		qg := Reshape(qn, B, KVH, group, Sq, D)
		scores := MulScalar(Einsum("bhgqd,bhkd->bhgqk", qg, kn), scale)
		biasReshaped := Reshape(bn, B, KVH, group, Sq, Skv)
		scores = Add(scores, biasReshaped)
		attn := Softmax(scores, -1)
		out5d := Einsum("bhgqk,bhkd->bhgqd", attn, vn)
		refOut := Reshape(out5d, B, QH, Sq, D)

		noBiasOut, _ := Core(qn, kn, vn, LayoutBHSD, CoreOptions{Scale: scale, WantCoefficients: true})
		return []*Node{biasedOut, refOut, noBiasOut}
	})

	out := exec.MustCall(q, k, v, biasTensor)
	biasedData := out[0].Value().([][][][]float32)
	refData := out[1].Value().([][][][]float32)
	noBiasData := out[2].Value().([][][][]float32)

	for b := range biasedData {
		for h := range biasedData[b] {
			for s := range biasedData[b][h] {
				for d := range biasedData[b][h][s] {
					require.InDeltaf(t, refData[b][h][s][d], biasedData[b][h][s][d], 1e-5,
						"GQA+bias Core output != inline reference at [%d][%d][%d][%d]", b, h, s, d)
				}
			}
		}
	}

	maxDiff := float32(0)
	for b := range biasedData {
		for h := range biasedData[b] {
			for s := range biasedData[b][h] {
				for d := range biasedData[b][h][s] {
					if diff := abs32(biasedData[b][h][s][d] - noBiasData[b][h][s][d]); diff > maxDiff {
						maxDiff = diff
					}
				}
			}
		}
	}
	require.Greater(t, maxDiff, float32(0.01),
		"GQA biased and unbiased outputs too similar; bias appears ignored (maxDiff=%v)", maxDiff)
}

// TestAttentionBiasFusedParity verifies the fused bias path (cfg.Bias set) matches the decomposed
// reference, forward and gradient (dQ), within bf16 tolerance. The fused arm uses the backend's
// biased fusion (cuDNN ScaleBias on xla:cuda); the reference arm handles bias in software. Runs on
// every official backend; those without biased fusion skip.
func TestAttentionBiasFusedParity(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		if !backendSupportsFusionForBFloat16(backend) {
			t.Skipf("%s: fused attention backend not available", backend)
		}
		if !backendSupportsBiasedFusion(backend) {
			t.Skipf("%s: supports plain fused SDPA but not biased fusion", backend)
		}

		const B, S, H, D = 2, 64, 2, 64
		scale := 1.0 / math.Sqrt(float64(D))

		// bf16 inputs and bias: the backend does no auto-conversion, so f32 would make the fused arm
		// fall back to decomposed and the parity would pass trivially. HasFusedSDPA below guards that.
		q := tensors.FromFlatDataAndDimensions(randFlatBF16(B*S*H*D, 10), B, S, H*D)
		k := tensors.FromFlatDataAndDimensions(randFlatBF16(B*S*H*D, 11), B, S, H*D)
		v := tensors.FromFlatDataAndDimensions(randFlatBF16(B*S*H*D, 12), B, S, H*D)
		biasTensor := tensors.FromFlatDataAndDimensions(randFlatBF16(B*H*S*S, 99), B, H, S, S)

		store := model.NewStore()
		exec := model.MustNewExec(backend, store, func(scope *model.Scope, qIn, kIn, vIn, biasIn *Node) []*Node {
			qProj := Reshape(qIn, B, S, H, D)
			kProj := Reshape(kIn, B, S, H, D)
			vProj := Reshape(vIn, B, S, H, D)

			// Backward arms first: dQ through the fused bias path vs the decomposed one. These take the
			// gradient, so they put both the fused forward and the fused VJP into the graph.
			fusedBwdOut, _ := Core(qProj, kProj, vProj, LayoutBSHD, CoreOptions{Scale: scale, AttentionBias: biasIn})
			refBwdOut, _ := Core(qProj, kProj, vProj, LayoutBSHD, CoreOptions{Scale: scale, AttentionBias: biasIn, DisableFusion: true})
			// Gradients are w.r.t. bf16 qProj, so cast to f32 before the metric (ToScalar wants f32).
			dqFused := ConvertDType(Gradient(ReduceAllSum(ConvertDType(fusedBwdOut, dtypes.Float32)), qProj)[0], dtypes.Float32)
			dqRef := ConvertDType(Gradient(ReduceAllSum(ConvertDType(refBwdOut, dtypes.Float32)), qProj)[0], dtypes.Float32)
			relBwd := Div(ReduceAllMax(Abs(Sub(dqFused, dqRef))), AddScalar(ReduceAllMax(Abs(dqRef)), 1e-6))

			// Forward parity through the builder, sharing projection weights.
			fusedOut := MultiHeadAttention(scope.In("shared"), qIn, kIn, vIn, H, D).
				WithAttentionBias(biasIn).WithPreProjected(true).Done()
			refOut := MultiHeadAttention(scope.Shared("shared"), qIn, kIn, vIn, H, D).
				WithAttentionBias(biasIn).WithFusion(false).WithPreProjected(true).Done()
			fusedF32 := ConvertDType(fusedOut, dtypes.Float32)
			refF32 := ConvertDType(refOut, dtypes.Float32)
			relFwd := Div(ReduceAllMax(Abs(Sub(fusedF32, refF32))), AddScalar(ReduceAllMax(Abs(refF32)), 1e-6))

			return []*Node{fusedF32, relFwd, relBwd}
		})

		out, g, err := exec.CallWithGraph(q, k, v, biasTensor)
		require.NoError(t, err)

		// Verify the fused bias op actually engaged, so the parity is meaningful and not a silent fallback.
		hasFwd, hasBwd := HasFusedSDPA(g)
		require.True(t, hasFwd, "fused bias arm must use FusedScaledDotProductAttention (not fall back to decomposed)")
		require.True(t, hasBwd, "fused bias arm must use FusedScaledDotProductAttentionVJP in the backward")

		require.Equal(t, []int{B, S, H * D}, out[0].Shape().Dimensions, "fused bias output shape must be [B, S, H*D]")
		require.LessOrEqual(t, float64(tensors.ToScalar[float32](out[1])), 0.06,
			"fused and decomposed bias outputs diverge (forward rel err)")
		require.LessOrEqual(t, float64(tensors.ToScalar[float32](out[2])), 0.06,
			"fused and decomposed bias gradients diverge (dQ rel err)")
	})
}

// TestAttentionBiasFusedParityMatrix extends the bias fused-parity check across dtype
// (Float16) and layout (BHSD), mirroring the plain-path BF16/Float16 coverage. Each case asserts the
// fused bias op actually engaged (HasFusedSDPA) so a silent fallback cannot pass trivially. The base
// bf16/BSHD case lives in TestAttentionBiasFusedParity. Runs on every official backend; those
// without biased fusion skip.
func TestAttentionBiasFusedParityMatrix(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		if !backendSupportsFusionForBFloat16(backend) {
			t.Skipf("%s: fused attention backend not available", backend)
		}
		if !backendSupportsBiasedFusion(backend) {
			t.Skipf("%s: supports plain fused SDPA but not biased fusion", backend)
		}

		const B, S, H, D = 2, 64, 2, 64
		scale := 1.0 / math.Sqrt(float64(D))
		cases := []struct {
			name   string
			dtype  dtypes.DType
			layout AxesLayout
			// fwdOnly skips the backward parity, kept as a knob for variants whose fused backward is
			// unsupported. All current cases run full fwd+bwd.
			fwdOnly bool
		}{
			{"f16_bshd", dtypes.Float16, LayoutBSHD, false},
			{"bf16_bhsd", dtypes.BFloat16, LayoutBHSD, false},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				// q/k/v shaped per layout; bias is [B,H,Sq,Skv] regardless. f32 seed data cast to the
				// target dtype in-graph so the fused path engages (no auto-conversion at the backend).
				qkvDims := []int{B, S, H, D}
				if tc.layout == LayoutBHSD {
					qkvDims = []int{B, H, S, D}
				}
				q := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 10), qkvDims...)
				k := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 11), qkvDims...)
				v := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 12), qkvDims...)
				bias := tensors.FromFlatDataAndDimensions(randFlat(B*H*S*S, 99), B, H, S, S)

				store := model.NewStore()
				exec := model.MustNewExec(backend, store, func(scope *model.Scope, qn, kn, vn, bn *Node) []*Node {
					cast := func(n *Node) *Node { return ConvertDType(n, tc.dtype) }
					qc, kc, vc, bc := cast(qn), cast(kn), cast(vn), cast(bn)
					fusedOut, _ := Core(qc, kc, vc, tc.layout, CoreOptions{Scale: scale, AttentionBias: bc})
					refOut, _ := Core(qc, kc, vc, tc.layout, CoreOptions{Scale: scale, AttentionBias: bc, DisableFusion: true})
					fF32 := ConvertDType(fusedOut, dtypes.Float32)
					rF32 := ConvertDType(refOut, dtypes.Float32)
					relFwd := Div(ReduceAllMax(Abs(Sub(fF32, rF32))), AddScalar(ReduceAllMax(Abs(rF32)), 1e-6))
					if tc.fwdOnly {
						return []*Node{relFwd}
					}
					dqF := ConvertDType(Gradient(ReduceAllSum(fF32), qc)[0], dtypes.Float32)
					dqR := ConvertDType(Gradient(ReduceAllSum(rF32), qc)[0], dtypes.Float32)
					relBwd := Div(ReduceAllMax(Abs(Sub(dqF, dqR))), AddScalar(ReduceAllMax(Abs(dqR)), 1e-6))
					return []*Node{relFwd, relBwd}
				})

				out, g, err := exec.CallWithGraph(q, k, v, bias)
				require.NoError(t, err)
				hasFwd, hasBwd := HasFusedSDPA(g)
				require.True(t, hasFwd, "fused bias must engage in forward for %s", tc.name)
				require.LessOrEqual(t, float64(tensors.ToScalar[float32](out[0])), 0.06, "forward rel err (%s)", tc.name)
				if !tc.fwdOnly {
					require.True(t, hasBwd, "fused bias must engage in backward for %s", tc.name)
					require.LessOrEqual(t, float64(tensors.ToScalar[float32](out[1])), 0.06, "backward rel err (%s)", tc.name)
				}
			})
		}
	})
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
