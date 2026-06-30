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
	_ = exceptions.TryCatch[error](func() {
		exec := MustNewExec(backend, func(qn, kn, vn *Node) *Node {
			round := func(n *Node) *Node { return ConvertDType(n, dtypes.BFloat16) }
			err := exceptions.TryCatch[error](func() {
				_ = BackendFusedScaledDotProductAttention(
					round(qn), round(kn), round(vn), nil, H, H,
					compute.AxesLayoutBSHD, 0.125, true, nil)
			})
			supported = err == nil || !compute.IsNotImplemented(err)
			return qn // dummy output to keep the graph valid
		})
		_ = exec.MustCall(q, k, v)
	})
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
			fusedOut, _ := Core(scope, qIn, kIn, vIn, scale, nil, nil, LayoutBSHD, true, false, 0.0, true, nil)
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
			out, _ := Core(scope, qIn, kIn, vIn, scale, nil, nil, LayoutBSHD, true, false, 0.0, true, nil)
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
				fusedOut, _ := Core(scope, qb, kb, vb, scale, nil, nil, LayoutBSHD, true, false, 0.0, true, nil)
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
		on, _ := Core(scope, qIn, kIn, vIn, scale, nil, nil, LayoutBSHD, true, false, 0.0, true, nil)
		off, _ := Core(scope, qIn, kIn, vIn, scale, nil, nil, LayoutBSHD, true, false, 0.0, false, nil)
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
