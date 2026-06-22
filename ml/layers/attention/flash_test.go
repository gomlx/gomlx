// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/graphtest"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/stretchr/testify/require"
)

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

func isCUDABackend(b compute.Backend) bool {
	s := strings.ToLower(b.Name() + " " + b.Description())
	return strings.Contains(s, "cuda")
}

func randFlat(n int, seed int64) []float32 {
	r := rand.New(rand.NewSource(seed))
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(r.NormFloat64() * 0.5)
	}
	return out
}

// TestFlashAttentionFallback checks that on a non-cuDNN backend FlashAttention transparently
// falls back to the decomposed reference instead of emitting an unexecutable custom_call. The
// fallback is naiveCausalAttention itself, so output and gradients match near-exactly. Runs on
// the default (non-cuda) test backend, so it guards the fallback in ordinary CI.
func TestFlashAttentionFallback(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	if isCUDABackend(backend) {
		t.Skip("fallback test is for non-cuda backends; this backend runs the flash kernel")
	}

	const (
		B, S, H, D = 1, 64, 2, 64
		scale      = 0.125
	)
	q := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 1), B, S, H, D)
	k := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 2), B, S, H, D)
	v := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 3), B, S, H, D)

	fn := func(qIn, kIn, vIn *Node) []*Node {
		flashOut := ConvertDType(FlashAttention(qIn, kIn, vIn, scale), dtypes.Float32)
		refOut := naiveCausalAttention(qIn, kIn, vIn, scale)
		fg := Gradient(ReduceAllSum(flashOut), qIn, kIn, vIn)
		ng := Gradient(ReduceAllSum(refOut), qIn, kIn, vIn)
		relMax := func(got, want *Node) *Node {
			return Div(ReduceAllMax(Abs(Sub(got, want))), AddScalar(ReduceAllMax(Abs(want)), 1e-6))
		}
		return []*Node{relMax(flashOut, refOut), relMax(fg[0], ng[0]), relMax(fg[1], ng[1]), relMax(fg[2], ng[2])}
	}
	out := MustNewExec(backend, fn).MustCall(q, k, v)
	for i, name := range []string{"output", "dQ", "dK", "dV"} {
		rel := float64(tensors.ToScalar[float32](out[i]))
		require.LessOrEqualf(t, rel, 1e-5, "%s relative error %.2e (fallback should match the reference)", name, rel)
	}
}

// TestFlashAttentionParity checks that the cuDNN flash attention forward output and its
// (flash-kernel) gradients match a decomposed float32 reference. Both paths see the same
// bfloat16-rounded inputs, so the only difference is flash's internal precision. Requires a
// cuDNN GPU backend (GOMLX_BACKEND=xla:cuda); skipped otherwise.
func TestFlashAttentionParity(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	if !isCUDABackend(backend) {
		t.Skipf("flash attention needs a cuDNN (cuda) backend; got %q", backend.Name())
	}

	const (
		B, S, H, D = 1, 512, 4, 64
		scale      = 0.125 // 1/sqrt(D)
	)
	q := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 1), B, S, H, D)
	k := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 2), B, S, H, D)
	v := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 3), B, S, H, D)

	fn := func(qIn, kIn, vIn *Node) []*Node {
		// Round inputs to bf16 (kept as f32) so flash and the reference start identical.
		round := func(n *Node) *Node { return ConvertDType(ConvertDType(n, dtypes.BFloat16), dtypes.Float32) }
		qb, kb, vb := round(qIn), round(kIn), round(vIn)

		flashOut := ConvertDType(FlashAttention(qb, kb, vb, scale), dtypes.Float32)
		refOut := naiveCausalAttention(qb, kb, vb, scale)

		fg := Gradient(ReduceAllSum(flashOut), qb, kb, vb)
		ng := Gradient(ReduceAllSum(refOut), qb, kb, vb)

		relMax := func(got, want *Node) *Node {
			return Div(ReduceAllMax(Abs(Sub(got, want))), AddScalar(ReduceAllMax(Abs(want)), 1e-6))
		}
		return []*Node{
			relMax(flashOut, refOut),
			relMax(fg[0], ng[0]), relMax(fg[1], ng[1]), relMax(fg[2], ng[2]),
		}
	}

	out := MustNewExec(backend, fn).MustCall(q, k, v)
	names := []string{"output", "dQ", "dK", "dV"}
	tol := []float64{0.03, 0.06, 0.06, 0.06}
	for i, name := range names {
		rel := float64(tensors.ToScalar[float32](out[i]))
		require.Falsef(t, math.IsNaN(rel), "%s relative error is NaN", name)
		require.LessOrEqualf(t, rel, tol[i], "%s relative max error %.4f exceeds tolerance %.4f", name, rel, tol[i])
		t.Logf("%s relative max error: %.5f", name, rel)
	}
}

// TestRepeatKVHeads pins the GQA grouping order: kv head h is repeated to query heads
// h*group .. h*group+group-1, so query head q uses kv head q/group (matching reshapeQueryForGQA).
func TestRepeatKVHeads(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	in := tensors.FromFlatDataAndDimensions([]float32{10, 20}, 1, 1, 2, 1) // two kv heads
	fn := func(x *Node) *Node { return Reshape(repeatKVHeads(x, 2), 4) }
	out := MustNewExec(backend, fn).MustCall(in)
	require.Equal(t, []float32{10, 10, 20, 20}, out[0].Value().([]float32))
}

// TestFlashAttentionGQAParity checks grouped-query flash attention (repeat-KV + flash kernel)
// against an independent grouped reference, forward and gradients. Requires a cuDNN GPU backend.
func TestFlashAttentionGQAParity(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	if !isCUDABackend(backend) {
		t.Skipf("flash attention needs a cuDNN (cuda) backend; got %q", backend.Name())
	}

	const (
		B, S, QH, KVH, D = 1, 512, 6, 2, 64 // group = 3, as in lm-100m's 12/4 ratio
		scale            = 0.125
	)
	q := tensors.FromFlatDataAndDimensions(randFlat(B*S*QH*D, 1), B, S, QH, D)
	k := tensors.FromFlatDataAndDimensions(randFlat(B*S*KVH*D, 2), B, S, KVH, D)
	v := tensors.FromFlatDataAndDimensions(randFlat(B*S*KVH*D, 3), B, S, KVH, D)

	fn := func(qIn, kIn, vIn *Node) []*Node {
		round := func(n *Node) *Node { return ConvertDType(ConvertDType(n, dtypes.BFloat16), dtypes.Float32) }
		qb, kb, vb := round(qIn), round(kIn), round(vIn)

		flashOut := ConvertDType(FlashAttention(qb, kb, vb, scale), dtypes.Float32)
		refOut := naiveGQAReference(qb, kb, vb, KVH, scale)
		fg := Gradient(ReduceAllSum(flashOut), qb, kb, vb)
		ng := Gradient(ReduceAllSum(refOut), qb, kb, vb)
		relMax := func(got, want *Node) *Node {
			return Div(ReduceAllMax(Abs(Sub(got, want))), AddScalar(ReduceAllMax(Abs(want)), 1e-6))
		}
		return []*Node{relMax(flashOut, refOut), relMax(fg[0], ng[0]), relMax(fg[1], ng[1]), relMax(fg[2], ng[2])}
	}

	out := MustNewExec(backend, fn).MustCall(q, k, v)
	names := []string{"output", "dQ", "dK", "dV"}
	tol := []float64{0.03, 0.06, 0.06, 0.06}
	for i, name := range names {
		rel := float64(tensors.ToScalar[float32](out[i]))
		require.Falsef(t, math.IsNaN(rel), "%s relative error is NaN", name)
		require.LessOrEqualf(t, rel, tol[i], "%s relative max error %.4f exceeds tolerance %.4f", name, rel, tol[i])
		t.Logf("%s relative max error: %.5f", name, rel)
	}
}
