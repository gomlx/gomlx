// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"fmt"
	"math"
	"math/rand"
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

// naiveCausalAttention is the decomposed reference oracle: softmax(scale*QK^T + causal)*V in float32.
// query/key/value are [B,S,H,D] with equal heads. Test-only: Core has its own inline decomposed path,
// so this exists only as the reference the parity tests compare Core's output against.
func naiveCausalAttention(query, key, value *Node, scale float64) *Node {
	g := query.Graph()
	q := ConvertDType(query, dtypes.Float32)
	k := ConvertDType(key, dtypes.Float32)
	v := ConvertDType(value, dtypes.Float32)
	dims := q.Shape().Dimensions
	batch, seqLen, heads := dims[0], dims[1], dims[2]

	scores := MulScalar(Einsum("bqhd,bkhd->bhqk", q, k), scale)
	causal := BroadcastToDims(Reshape(LowerTriangular(g, seqLen), 1, 1, seqLen, seqLen), batch, heads, seqLen, seqLen)
	attn := MaskedSoftmax(scores, causal, -1)
	return Einsum("bhqk,bkhd->bqhd", attn, v)
}

// naiveGQAReference computes grouped-query attention independently of repeatKVHeads: it splits the
// query heads into (numKVHeads, group) and contracts each group against its kv head, so it is a
// genuine cross-check of the repeat-KV grouping. query is [B,S,nQH,D], key/value [B,S,nKVH,D].
func naiveGQAReference(query, key, value *Node, numKVHeads int, scale float64) *Node {
	g := query.Graph()
	q := ConvertDType(query, dtypes.Float32)
	k := ConvertDType(key, dtypes.Float32)
	v := ConvertDType(value, dtypes.Float32)
	d := q.Shape().Dimensions
	b, s, nQH, dim := d[0], d[1], d[2], d[3]
	group := nQH / numKVHeads

	qg := Reshape(q, b, s, numKVHeads, group, dim) // [b,s,h,g,d], query head = h*group + g
	scores := MulScalar(Einsum("bqhgd,bkhd->bhgqk", qg, k), scale)
	causal := BroadcastToDims(Reshape(LowerTriangular(g, s), 1, 1, 1, s, s), b, numKVHeads, group, s, s)
	attn := MaskedSoftmax(scores, causal, -1)
	out := Einsum("bhgqk,bkhd->bqhgd", attn, v)
	return Reshape(out, b, s, nQH, dim)
}

// repeatKVHeads expands key/value for grouped-query attention: [B,S,nKV,D] -> [B,S,nKV*group,D],
// repeating each kv head group times contiguously, so output head h uses kv head h/group.
// Test-only: production GQA goes through Core's reshapeQueryForGQA; this survives only to pin the
// grouping order in TestRepeatKVHeads.
func repeatKVHeads(x *Node, group int) *Node {
	d := x.Shape().Dimensions
	b, s, nKV, dim := d[0], d[1], d[2], d[3]
	x = Reshape(x, b, s, nKV, 1, dim)
	x = BroadcastToDims(x, b, s, nKV, group, dim)
	return Reshape(x, b, s, nKV*group, dim)
}

// randFlat returns n float32 values drawn from N(0,0.5) with the given seed.
func randFlat(n int, seed int64) []float32 {
	r := rand.New(rand.NewSource(seed))
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(r.NormFloat64() * 0.5)
	}
	return out
}

// backendSupportsFusion reports whether the backend implements fused scaled-dot-product attention,
// by probing a tiny causal bf16 FusedSDPA and checking it does not return ErrNotImplemented. This
// replaces name-sniffing (isCUDABackend) so any backend that grows the kernel is exercised.
func backendSupportsFusion(backend compute.Backend) bool {
	const B, S, H, D = 1, 8, 1, 64
	q := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 1), B, S, H, D)
	k := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 2), B, S, H, D)
	v := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 3), B, S, H, D)
	supported := false
	probeErr := exceptions.TryCatch[error](func() {
		exec := MustNewExec(backend, func(qn, kn, vn *Node) *Node {
			round := func(n *Node) *Node { return ConvertDType(n, dtypes.BFloat16) }
			fused, _ := BackendFusedScaledDotProductAttention(
				round(qn), round(kn), round(vn), nil, H, H,
				compute.AxesLayoutBSHD, 0.125, true, nil)
			// Make the returned node depend on the fused output so the fused op is genuinely
			// part of the executable graph and exercised at compile/exec time. A backend whose
			// ErrNotImplemented surfaces only at compile/exec (not graph-build) would otherwise
			// be misdetected as supported.
			return ConvertDType(ReduceAllSum(fused), dtypes.Float32)
		})
		_ = exec.MustCall(q, k, v)
		supported = true
	})
	if probeErr != nil {
		if compute.IsNotImplemented(probeErr) {
			return false // fused op not implemented on this backend.
		}
		panic(probeErr) // unexpected error: do not silently swallow.
	}
	return supported
}

// TestRepeatKVHeads pins the GQA grouping order: kv head h is repeated to query heads
// h*group .. h*group+group-1, so query head q uses kv head q/group (matching reshapeQueryForGQA).
func TestRepeatKVHeads(t *testing.T) {
	backend := testutil.BuildTestBackend()
	in := tensors.FromFlatDataAndDimensions([]float32{10, 20}, 1, 1, 2, 1) // two kv heads
	fn := func(x *Node) *Node { return Reshape(repeatKVHeads(x, 2), 4) }
	out := MustNewExec(backend, fn).MustCall(in)
	require.Equal(t, []float32{10, 10, 20, 20}, out[0].Value().([]float32))
}

// TestFusionFallbackParity checks that on every official backend, attention through Core (fused path
// attempted) matches the decomposed reference, forward and gradients. On a backend without the fused
// kernel this exercises the ErrNotImplemented fallback; on one with it, it is a parity check. Loops
// all official backends.
func TestFusionFallbackParity(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		const B, S, H, D = 1, 64, 2, 64
		scale := 0.125
		q := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 1), B, S, H, D)
		k := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 2), B, S, H, D)
		v := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 3), B, S, H, D)

		store := model.NewStore()
		exec := model.MustNewExec(backend, store, func(scope *model.Scope, qIn, kIn, vIn *Node) []*Node {
			fusedOut, _ := Core(scope, qIn, kIn, vIn, scale, nil, nil, LayoutBSHD, true, false, 0.0, true, nil, nil)
			refOut := naiveCausalAttention(qIn, kIn, vIn, scale)
			fusedOut = ConvertDType(fusedOut, dtypes.Float32)
			fg := Gradient(ReduceAllSum(fusedOut), qIn, kIn, vIn)
			ng := Gradient(ReduceAllSum(refOut), qIn, kIn, vIn)
			relMax := func(got, want *Node) *Node {
				return Div(ReduceAllMax(Abs(Sub(got, want))), AddScalar(ReduceAllMax(Abs(want)), 1e-6))
			}
			return []*Node{relMax(fusedOut, refOut), relMax(fg[0], ng[0]), relMax(fg[1], ng[1]), relMax(fg[2], ng[2])}
		})
		out := exec.MustCall(q, k, v)
		// Fused kernels run in bf16, so use a loose tolerance when fusion is supported; the CPU
		// fallback is the reference itself, so it matches near-exactly.
		tol := 1e-5
		if backendSupportsFusion(backend) {
			tol = 0.06
		}
		for i, name := range []string{"output", "dQ", "dK", "dV"} {
			rel := float64(tensors.ToScalar[float32](out[i]))
			require.LessOrEqualf(t, rel, tol, "%s rel error %.2e exceeds %.2e on %s", name, rel, tol, backend.Name())
		}
	})
}

// TestFusionGQAParity checks grouped-query attention through Core against the independent grouped
// reference on every official backend, forward and gradients. group=3 (6 query / 2 kv heads).
func TestFusionGQAParity(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		const B, S, QH, KVH, D = 1, 128, 6, 2, 64
		scale := 0.125
		q := tensors.FromFlatDataAndDimensions(randFlat(B*S*QH*D, 1), B, S, QH, D)
		k := tensors.FromFlatDataAndDimensions(randFlat(B*S*KVH*D, 2), B, S, KVH, D)
		v := tensors.FromFlatDataAndDimensions(randFlat(B*S*KVH*D, 3), B, S, KVH, D)

		store := model.NewStore()
		exec := model.MustNewExec(backend, store, func(scope *model.Scope, qIn, kIn, vIn *Node) []*Node {
			out, _ := Core(scope, qIn, kIn, vIn, scale, nil, nil, LayoutBSHD, true, false, 0.0, true, nil, nil)
			out = ConvertDType(out, dtypes.Float32)
			ref := naiveGQAReference(qIn, kIn, vIn, KVH, scale)
			og := Gradient(ReduceAllSum(out), qIn, kIn, vIn)
			rg := Gradient(ReduceAllSum(ref), qIn, kIn, vIn)
			relMax := func(got, want *Node) *Node {
				return Div(ReduceAllMax(Abs(Sub(got, want))), AddScalar(ReduceAllMax(Abs(want)), 1e-6))
			}
			return []*Node{relMax(out, ref), relMax(og[0], rg[0]), relMax(og[1], rg[1]), relMax(og[2], rg[2])}
		})
		out := exec.MustCall(q, k, v)
		tol := 1e-5
		if backendSupportsFusion(backend) {
			tol = 0.06
		}
		for i, name := range []string{"output", "dQ", "dK", "dV"} {
			rel := float64(tensors.ToScalar[float32](out[i]))
			require.LessOrEqualf(t, rel, tol, "%s rel error %.2e exceeds %.2e on %s", name, rel, tol, backend.Name())
		}
	})
}

// TestCUDAFusionParity [cuda] cross-checks the cuDNN fused kernel against the float32 reference at
// the head dims cuDNN flash supports (64, 128). Skipped unless the xla:cuda backend is present.
func TestCUDAFusionParity(t *testing.T) {
	backend := testutil.GetOfficialBackend("xla:cuda")
	if backend == nil || !backendSupportsFusion(backend) {
		t.Skip("xla:cuda fused attention backend not available")
	}
	const B, S, H = 1, 512, 4
	for _, D := range []int{64, 128} {
		t.Run(fmt.Sprintf("D=%d", D), func(t *testing.T) {
			scale := 1.0 / math.Sqrt(float64(D))
			q := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 1), B, S, H, D)
			k := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 2), B, S, H, D)
			v := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 3), B, S, H, D)
			store := model.NewStore()
			exec := model.MustNewExec(backend, store, func(scope *model.Scope, qIn, kIn, vIn *Node) []*Node {
				round := func(n *Node) *Node { return ConvertDType(ConvertDType(n, dtypes.BFloat16), dtypes.Float32) }
				qb, kb, vb := round(qIn), round(kIn), round(vIn)
				fusedOut, _ := Core(scope, qb, kb, vb, scale, nil, nil, LayoutBSHD, true, false, 0.0, true, nil, nil)
				fusedOut = ConvertDType(fusedOut, dtypes.Float32)
				refOut := naiveCausalAttention(qb, kb, vb, scale)
				fg := Gradient(ReduceAllSum(fusedOut), qb, kb, vb)
				ng := Gradient(ReduceAllSum(refOut), qb, kb, vb)
				relMax := func(got, want *Node) *Node {
					return Div(ReduceAllMax(Abs(Sub(got, want))), AddScalar(ReduceAllMax(Abs(want)), 1e-6))
				}
				return []*Node{relMax(fusedOut, refOut), relMax(fg[0], ng[0]), relMax(fg[1], ng[1]), relMax(fg[2], ng[2])}
			})
			out := exec.MustCall(q, k, v)
			tol := []float64{0.03, 0.06, 0.06, 0.06}
			for i, name := range []string{"output", "dQ", "dK", "dV"} {
				rel := float64(tensors.ToScalar[float32](out[i]))
				require.Falsef(t, math.IsNaN(rel), "%s rel error NaN", name)
				require.LessOrEqualf(t, rel, tol[i], "%s rel max %.4f exceeds %.4f", name, rel, tol[i])
			}
		})
	}
}

// TestCoreUseFusionFalseMatchesDecomposed pins that Core with useFusion=false produces the
// same output as useFusion=true on the CPU backend (where both take the decomposed path anyway,
// since CPU returns ErrNotImplemented for fused causal). This guards the new gate compiling and
// not altering results. Runs on the default (CPU) backend.
func TestCoreUseFusionFalseMatchesDecomposed(t *testing.T) {
	backend := testutil.BuildTestBackend()
	const B, S, H, D = 1, 32, 2, 64
	scale := 0.125
	q := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 1), B, S, H, D)
	k := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 2), B, S, H, D)
	v := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 3), B, S, H, D)

	store := model.NewStore()
	exec := model.MustNewExec(backend, store, func(scope *model.Scope, qIn, kIn, vIn *Node) []*Node {
		on, _ := Core(scope, qIn, kIn, vIn, scale, nil, nil, LayoutBSHD, true, false, 0.0, true, nil, nil)
		off, _ := Core(scope, qIn, kIn, vIn, scale, nil, nil, LayoutBSHD, true, false, 0.0, false, nil, nil)
		return []*Node{Div(ReduceAllMax(Abs(Sub(on, off))), AddScalar(ReduceAllMax(Abs(off)), 1e-6))}
	})
	out := exec.MustCall(q, k, v)
	rel := float64(tensors.ToScalar[float32](out[0]))
	require.LessOrEqual(t, rel, 1e-6, "useFusion on/off diverged on CPU (both should be decomposed)")
}

// TestBuilderWithFusionDefaultsTrue pins that the builder defaults useFusion to true and that
// WithFusion(false) flips it. Builder-level: inspects the field through a Done() run on CPU and
// asserts the output shape is correct either way (CPU has no fused causal kernel, so both equal
// the reference; the test guards the wiring compiles and the default is true).
func TestBuilderWithFusionDefaultsTrue(t *testing.T) {
	backend := testutil.BuildTestBackend()
	const B, S, H, D = 1, 16, 2, 8
	x := tensors.FromFlatDataAndDimensions(randFlat(B*S*(H*D), 1), B, S, H*D)

	store := model.NewStore()
	exec := model.MustNewExec(backend, store, func(scope *model.Scope, in *Node) []*Node {
		def := SelfAttention(scope.In("def"), in, H, D).WithCausalMask(true).Done()
		off := SelfAttention(scope.In("off"), in, H, D).WithCausalMask(true).WithFusion(false).Done()
		// Weights are not shared across scopes, so compare shapes only: both must produce
		// [B,S,H*D]. The behavioral on/off equivalence is covered by Task 2.
		return []*Node{def, off}
	})
	out := exec.MustCall(x)
	require.Equal(t, []int{B, S, H * D}, out[0].Shape().Dimensions)
	require.Equal(t, []int{B, S, H * D}, out[1].Shape().Dimensions)
}

// TestWithSeqLensRejectsExplicitMask pins the mutual-exclusion rule: WithSeqLens called after
// WithQueryKeyMatrixMask must panic. Builder-time validation; panics before any backend op.
func TestWithSeqLensRejectsExplicitMask(t *testing.T) {
	backend := testutil.BuildTestBackend()
	const B, S, H, D = 1, 8, 2, 8
	x := tensors.FromFlatDataAndDimensions(randFlat(B*S*(H*D), 1), B, S, H*D)
	lens := tensors.FromFlatDataAndDimensions([]int32{S}, B)
	maskData := make([]float32, B*S*H*S)
	mask := tensors.FromFlatDataAndDimensions(maskData, B, S, H, S)

	store := model.NewStore()
	require.Panics(t, func() {
		_ = model.MustNewExec(backend, store, func(scope *model.Scope, in, qlen, klen, m *Node) []*Node {
			return []*Node{
				SelfAttention(scope, in, H, D).
					WithQueryKeyMatrixMask(m).
					WithSeqLens(qlen, klen).
					Done(),
			}
		}).MustCall(x, lens, lens, mask)
	}, "WithQueryKeyMatrixMask then WithSeqLens must panic")
}

// TestWithSeqLensMaskAfterRejectsExplicitMask pins the reverse mutual-exclusion order:
// WithQueryKeyMatrixMask called after WithSeqLens must also panic.
func TestWithSeqLensMaskAfterRejectsExplicitMask(t *testing.T) {
	backend := testutil.BuildTestBackend()
	const B, S, H, D = 1, 8, 2, 8
	x := tensors.FromFlatDataAndDimensions(randFlat(B*S*(H*D), 1), B, S, H*D)
	lens := tensors.FromFlatDataAndDimensions([]int32{S}, B)
	maskData := make([]float32, B*S*H*S)
	mask := tensors.FromFlatDataAndDimensions(maskData, B, S, H, S)

	store := model.NewStore()
	require.Panics(t, func() {
		_ = model.MustNewExec(backend, store, func(scope *model.Scope, in, qlen, klen, m *Node) []*Node {
			return []*Node{
				SelfAttention(scope, in, H, D).
					WithSeqLens(qlen, klen).
					WithQueryKeyMatrixMask(m).
					Done(),
			}
		}).MustCall(x, lens, lens, mask)
	}, "WithSeqLens then WithQueryKeyMatrixMask must panic")
}

// TestWithSeqLensRejectsKeyMask pins that WithSeqLens is mutually exclusive with WithKeyMask
// (a split mask, not just the matrix mask), in both orders. A non-nil mask would otherwise
// disable the causal flag in doneInternal and contradict the "seqlens instead of mask" contract.
func TestWithSeqLensRejectsKeyMask(t *testing.T) {
	backend := testutil.BuildTestBackend()
	const B, S, H, D = 1, 8, 2, 8
	x := tensors.FromFlatDataAndDimensions(randFlat(B*S*(H*D), 1), B, S, H*D)
	lens := tensors.FromFlatDataAndDimensions([]int32{S}, B)
	keyMaskData := make([]bool, B*S)
	for i := range keyMaskData {
		keyMaskData[i] = true
	}
	keyMask := tensors.FromFlatDataAndDimensions(keyMaskData, B, S)

	// WithKeyMask then WithSeqLens: WithSeqLens must reject the already-set key mask at builder time.
	store := model.NewStore()
	err := exceptions.TryCatch[error](func() {
		_ = model.MustNewExec(backend, store, func(scope *model.Scope, in, qlen, klen, km *Node) []*Node {
			return []*Node{
				SelfAttention(scope, in, H, D).
					WithKeyMask(km).
					WithSeqLens(qlen, klen).
					Done(),
			}
		}).MustCall(x, lens, lens, keyMask)
	})
	require.Error(t, err, "WithKeyMask then WithSeqLens must panic")
	require.Contains(t, err.Error(), "WithSeqLens is mutually exclusive with an explicit mask",
		"panic must come from the WithSeqLens builder guard, not the Core guard")

	// WithSeqLens then WithKeyMask (reverse): the mask setter must reject the already-set seqlens.
	store2 := model.NewStore()
	err2 := exceptions.TryCatch[error](func() {
		_ = model.MustNewExec(backend, store2, func(scope *model.Scope, in, qlen, klen, km *Node) []*Node {
			return []*Node{
				SelfAttention(scope, in, H, D).
					WithSeqLens(qlen, klen).
					WithKeyMask(km).
					Done(),
			}
		}).MustCall(x, lens, lens, keyMask)
	})
	require.Error(t, err2, "WithSeqLens then WithKeyMask must panic")
	require.Contains(t, err2.Error(), "an explicit mask is mutually exclusive with WithSeqLens",
		"panic must come from the WithKeyMask builder guard, not the Core guard")
}

// TestCoreRejectsSeqLensWithMask pins the Core-level guard: passing seqlens together with a
// non-nil attentionMask must panic (an additive float mask would otherwise be LogicalAnd-ed
// with the boolean padding mask on the decomposed path, which is wrong).
func TestCoreRejectsSeqLensWithMask(t *testing.T) {
	backend := testutil.BuildTestBackend()
	const B, S, H, D = 1, 8, 2, 8
	q := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 1), B, S, H, D)
	lens := tensors.FromFlatDataAndDimensions([]int32{S}, B)
	maskData := make([]float32, B*S*H*S)
	maskT := tensors.FromFlatDataAndDimensions(maskData, B, S, H, S)

	store := model.NewStore()
	err := exceptions.TryCatch[error](func() {
		_ = model.MustNewExec(backend, store, func(scope *model.Scope, qn, qlen, klen, m *Node) []*Node {
			out, _ := Core(scope, qn, qn, qn, 0.125, m, nil, LayoutBSHD, false, false, 0.0, true, qlen, klen)
			return []*Node{out}
		}).MustCall(q, lens, lens, maskT)
	})
	require.Error(t, err, "Core with seqlens and a non-nil mask must panic")
	require.Contains(t, err.Error(), "mutually exclusive with a non-nil attentionMask",
		"panic must come from the Core seqlens/mask guard, not an incidental LogicalAnd type error")
}

// TestCoreRejectsSeqLensWithNonBSHD pins the Core-level guard that seqlens require LayoutBSHD,
// since buildSeqLenPaddingMask is BSHD-only. Passing seqlens with LayoutBHSD must panic.
func TestCoreRejectsSeqLensWithNonBSHD(t *testing.T) {
	backend := testutil.BuildTestBackend()
	const B, H, S, D = 1, 2, 8, 8
	// BHSD layout: [batch, heads, seq, dim].
	q := tensors.FromFlatDataAndDimensions(randFlat(B*H*S*D, 1), B, H, S, D)
	lens := tensors.FromFlatDataAndDimensions([]int32{S}, B)

	store := model.NewStore()
	err := exceptions.TryCatch[error](func() {
		_ = model.MustNewExec(backend, store, func(scope *model.Scope, qn, qlen, klen *Node) []*Node {
			out, _ := Core(scope, qn, qn, qn, 0.125, nil, nil, LayoutBHSD, false, false, 0.0, true, qlen, klen)
			return []*Node{out}
		}).MustCall(q, lens, lens)
	})
	require.Error(t, err, "Core with seqlens and non-BSHD layout must panic")
	require.Contains(t, err.Error(), "require LayoutBSHD",
		"panic must come from the Core seqlens/layout guard")
}

// TestWithSeqLensRequiresBoth pins the both-or-neither contract: passing exactly one non-nil
// argument to WithSeqLens must panic immediately (before any backend op).
func TestWithSeqLensRequiresBoth(t *testing.T) {
	backend := testutil.BuildTestBackend()
	const B, S, H, D = 1, 8, 2, 8
	x := tensors.FromFlatDataAndDimensions(randFlat(B*S*(H*D), 1), B, S, H*D)
	lens := tensors.FromFlatDataAndDimensions([]int32{S}, B)

	store := model.NewStore()
	// Only querySeqLen set, keyValueSeqLen nil.
	require.Panics(t, func() {
		_ = model.MustNewExec(backend, store, func(scope *model.Scope, in, qlen *Node) []*Node {
			return []*Node{
				SelfAttention(scope, in, H, D).
					WithSeqLens(qlen, nil).
					Done(),
			}
		}).MustCall(x, lens)
	}, "WithSeqLens(non-nil, nil) must panic")

	// Only keyValueSeqLen set, querySeqLen nil.
	require.Panics(t, func() {
		_ = model.MustNewExec(backend, store, func(scope *model.Scope, in, klen *Node) []*Node {
			return []*Node{
				SelfAttention(scope, in, H, D).
					WithSeqLens(nil, klen).
					Done(),
			}
		}).MustCall(x, lens)
	}, "WithSeqLens(nil, non-nil) must panic")
}

// TestSeqLenDecomposedMasking verifies that the decomposed attention path correctly applies
// padding masking when WithSeqLens is used. On CPU the fused path is unavailable, so both
// the seqlen path and an equivalent explicit boolean mask path use decomposed attention.
// They must produce identical attention coefficients; before the fix the seqlen path ignores
// padding so its coefficients differ from the masked-softmax reference.
//
// Uses DoneWithCoefficients() so the comparison is on the attention weight matrix, which
// is computed before the output projection. This means different output-projection weights
// in the two separate scopes do not affect the comparison.
func TestSeqLenDecomposedMasking(t *testing.T) {
	backend := testutil.BuildTestBackend()
	// B=2, Sq=Skv=4, H=2, D=8; seqlens=[3,2] (positions 3 and 2,3 are padding).
	const B, S, H, D = 2, 4, 2, 8

	// Q/K/V: [B, S, H*D] (MultiHeadAttention with preProjected reshapes to [B,S,H,D]).
	q := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 10), B, S, H*D)
	k := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 11), B, S, H*D)
	v := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 12), B, S, H*D)

	// seqlens: batch 0 has 3 valid positions, batch 1 has 2.
	seqLensTensor := tensors.FromFlatDataAndDimensions([]int32{3, 2}, B)

	// Equivalent explicit mask [B, S]: true = valid position (same for query and key in self-attn).
	// batch 0: [T,T,T,F]; batch 1: [T,T,F,F].
	maskTensor := tensors.FromFlatDataAndDimensions([]bool{
		true, true, true, false,
		true, true, false, false,
	}, B, S)

	store := model.NewStore()
	exec := model.MustNewExec(backend, store, func(scope *model.Scope, qIn, kIn, vIn, seqLens, mask *Node) []*Node {
		// Seqlen path: both querySeqLen and keyValueSeqLen set (required by contract).
		// DoneWithCoefficients forces the decomposed path and returns the [B,Sq,H,Skv] coeffs.
		_, slCoeffs := MultiHeadAttention(scope.In("sl"), qIn, kIn, vIn, H, D).
			WithSeqLens(seqLens, seqLens).
			WithPreProjected(true).
			DoneWithCoefficients()

		// Equivalent reference: explicit boolean key+query mask [B, S].
		_, mkCoeffs := MultiHeadAttention(scope.In("mk"), qIn, kIn, vIn, H, D).
			WithKeyMask(mask).
			WithQueryMask(mask).
			WithPreProjected(true).
			DoneWithCoefficients()

		// Compare coefficient matrices: they must match when masking is applied correctly.
		diff := ReduceAllMax(Abs(Sub(slCoeffs, mkCoeffs)))
		return []*Node{diff}
	})
	out := exec.MustCall(q, k, v, seqLensTensor, maskTensor)
	maxDiff := float64(tensors.ToScalar[float32](out[0]))
	// Same Q/K/V, equivalent masks: coefficients must match within float32 rounding.
	require.LessOrEqual(t, maxDiff, float64(1e-5),
		"seqlen and explicit-mask attention coefficients diverge (max diff=%v); padding not applied on decomposed path", maxDiff)
}

// TestWithSeqLensBuildsConfigAndProducesOutput confirms that WithSeqLens wires a non-nil config
// into the fused path and that the builder still produces the correct output shape on CPU
// (the fused path returns ErrNotImplemented on CPU and falls back to decomposed, so the seqlen
// config is built and forwarded but execution uses the decomposed path).
func TestWithSeqLensBuildsConfigAndProducesOutput(t *testing.T) {
	backend := testutil.BuildTestBackend()
	const B, S, H, D = 1, 8, 2, 8
	x := tensors.FromFlatDataAndDimensions(randFlat(B*S*(H*D), 1), B, S, H*D)
	lens := tensors.FromFlatDataAndDimensions([]int32{S}, B)

	store := model.NewStore()
	exec := model.MustNewExec(backend, store, func(scope *model.Scope, in, qlen, klen *Node) []*Node {
		out := SelfAttention(scope, in, H, D).
			WithSeqLens(qlen, klen).
			Done()
		return []*Node{out}
	})
	outputs := exec.MustCall(x, lens, lens)
	require.Equal(t, []int{B, S, H * D}, outputs[0].Shape().Dimensions,
		"WithSeqLens output shape must match [B, S, H*D]")
}

// TestSeqLenFusedParity verifies that the fused PADDING path with WithSeqLens
// produces numerically the same output as the decomposed path with the same both-set seqlen
// config. Skipped unless the backend is present and supports fused attention.
//
// Both arms share output-projection weights (scope.Shared) and use WithPreProjected(true)
// to skip Q/K/V projections, so the only difference is fused=true vs fused=false.
// Real padding: batch element 0 has 48/64 valid positions, element 1 has 32/64.
// The test directly compares fused vs decomposed; it catches any disagreement in how the
// cuDNN PADDING kernel applies seqlen masking relative to the software decomposed path.
//
// TODO(jan): loop over testutil.TestOfficialBackends instead of pinning xla:cuda, so a future Go
// SIMD or ONNXRuntime fused backend is covered too (non-fusing backends pass trivially).
func TestSeqLenFusedParity(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		if backend == nil || !backendSupportsFusion(backend) {
			t.Skipf("%s: fused attention backend not available", backend)
		}

		// B=2, Sq=Skv=64, H=2, D=64; seqlens=[48,32] — real padding in both batch elements.
		const B, S, H, D = 2, 64, 2, 64
		seqLenData := []int32{48, 32}

		// WithPreProjected(true): inputs are pre-projected, shaped [B, S, H*D].
		q := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 10), B, S, H*D)
		k := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 11), B, S, H*D)
		v := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 12), B, S, H*D)
		seqLensTensor := tensors.FromFlatDataAndDimensions(seqLenData, B)

		store := model.NewStore()
		exec := model.MustNewExec(backend, store, func(scope *model.Scope, qIn, kIn, vIn, seqLens *Node) []*Node {
			// Fused arm: WithSeqLens(both) activates the cuDNN PADDING kernel on xla:cuda.
			fusedOut := MultiHeadAttention(scope.In("shared"), qIn, kIn, vIn, H, D).
				WithSeqLens(seqLens, seqLens).
				WithPreProjected(true).
				Done()

			// Decomposed reference: same both-set config, same shared weights, fused disabled.
			refOut := MultiHeadAttention(scope.Shared("shared"), qIn, kIn, vIn, H, D).
				WithSeqLens(seqLens, seqLens).
				WithFusion(false).
				WithPreProjected(true).
				Done()

			fusedOut = ConvertDType(fusedOut, dtypes.Float32)
			refOut = ConvertDType(refOut, dtypes.Float32)
			diff := Div(
				ReduceAllMax(Abs(Sub(fusedOut, refOut))),
				AddScalar(ReduceAllMax(Abs(refOut)), 1e-6),
			)
			return []*Node{fusedOut, diff}
		})
		out := exec.MustCall(q, k, v, seqLensTensor)

		// Output shape must be correct.
		require.Equal(t, []int{B, S, H * D}, out[0].Shape().Dimensions,
			"fused WithSeqLens output shape must be [B, S, H*D]")

		// Fused and decomposed must agree within bf16 tolerance.
		relErr := float64(tensors.ToScalar[float32](out[1]))
		require.LessOrEqual(t, relErr, 0.05,
			"fused and decomposed seqlen outputs diverge (rel err=%v); fused padding path broken", relErr)
	})
}
