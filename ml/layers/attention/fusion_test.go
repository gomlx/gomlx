// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
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

// TestRepeatKVHeads pins the GQA grouping order: kv head h is repeated to query heads
// h*group .. h*group+group-1, so query head q uses kv head q/group (matching reshapeQueryForGQA).
func TestRepeatKVHeads(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	in := tensors.FromFlatDataAndDimensions([]float32{10, 20}, 1, 1, 2, 1) // two kv heads
	fn := func(x *Node) *Node { return Reshape(repeatKVHeads(x, 2), 4) }
	out := MustNewExec(backend, fn).MustCall(in)
	require.Equal(t, []float32{10, 10, 20, 20}, out[0].Value().([]float32))
}

// isCUDABackend reports whether b is a cuDNN/CUDA backend.
func isCUDABackend(b compute.Backend) bool {
	s := strings.ToLower(b.Name() + " " + b.Description())
	return strings.Contains(s, "cuda")
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
