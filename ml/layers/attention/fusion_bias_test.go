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
			cfg := &compute.ScaledDotProductAttentionConfig{Bias: InternalBackendOutputs(round(bn))[0]}
			fused, _ := BackendFusedScaledDotProductAttention(
				round(qn), round(qn), round(qn), nil, H, H,
				compute.AxesLayoutBSHD, scale, false, cfg)
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

// TestAttentionBiasFusedParity_cuda [cuda] verifies the fused bias path (cfg.Bias set) matches the
// decomposed reference, forward and gradient (dQ), within bf16 tolerance. On xla:cuda the fused arm
// uses cuDNN ScaleBias; the reference arm handles bias in software. Skipped unless xla:cuda supports
// fused attention.
func TestAttentionBiasFusedParity_cuda(t *testing.T) {
	backend := testutil.GetOfficialBackend("xla:cuda")
	if backend == nil || !backendSupportsFusionForBFloat16(backend) {
		t.Skip("xla:cuda fused attention backend not available")
	}
	require.True(t, backendSupportsBiasedFusion(backend),
		"xla:cuda supports plain fused SDPA but not biased fusion; bias layout rejected by validateBias")

	const B, S, H, D = 2, 64, 2, 64
	scale := 1.0 / math.Sqrt(float64(D))

	q := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 10), B, S, H*D)
	k := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 11), B, S, H*D)
	v := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 12), B, S, H*D)
	biasTensor := tensors.FromFlatDataAndDimensions(randFlat(B*H*S*S, 99), B, H, S, S)

	store := model.NewStore()
	exec := model.MustNewExec(backend, store, func(scope *model.Scope, qIn, kIn, vIn, biasIn *Node) []*Node {
		qProj := Reshape(qIn, B, S, H, D)
		kProj := Reshape(kIn, B, S, H, D)
		vProj := Reshape(vIn, B, S, H, D)

		// Fused vs decomposed arms through the builder, sharing projection weights.
		fusedOut := MultiHeadAttention(scope.In("shared"), qIn, kIn, vIn, H, D).
			WithAttentionBias(biasIn).WithPreProjected(true).Done()
		refOut := MultiHeadAttention(scope.Shared("shared"), qIn, kIn, vIn, H, D).
			WithAttentionBias(biasIn).WithFusion(false).WithPreProjected(true).Done()

		fusedF32 := ConvertDType(fusedOut, dtypes.Float32)
		refF32 := ConvertDType(refOut, dtypes.Float32)
		relFwd := Div(ReduceAllMax(Abs(Sub(fusedF32, refF32))), AddScalar(ReduceAllMax(Abs(refF32)), 1e-6))

		// Backward: dQ through the fused bias path vs the decomposed one.
		fusedBwdOut, _ := Core(qProj, kProj, vProj, LayoutBSHD, CoreOptions{Scale: scale, AttentionBias: biasIn})
		refBwdOut, _ := Core(qProj, kProj, vProj, LayoutBSHD, CoreOptions{Scale: scale, AttentionBias: biasIn, DisableFusion: true})
		dqFused := Gradient(ReduceAllSum(ConvertDType(fusedBwdOut, dtypes.Float32)), qProj)[0]
		dqRef := Gradient(ReduceAllSum(ConvertDType(refBwdOut, dtypes.Float32)), qProj)[0]
		relBwd := Div(ReduceAllMax(Abs(Sub(dqFused, dqRef))), AddScalar(ReduceAllMax(Abs(dqRef)), 1e-6))

		return []*Node{fusedF32, relFwd, relBwd}
	})

	out := exec.MustCall(q, k, v, biasTensor)
	require.Equal(t, []int{B, S, H * D}, out[0].Shape().Dimensions, "fused bias output shape must be [B, S, H*D]")
	require.LessOrEqual(t, float64(tensors.ToScalar[float32](out[1])), 0.06,
		"fused and decomposed bias outputs diverge (forward rel err)")
	require.LessOrEqual(t, float64(tensors.ToScalar[float32](out[2])), 0.06,
		"fused and decomposed bias gradients diverge (dQ rel err)")
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
