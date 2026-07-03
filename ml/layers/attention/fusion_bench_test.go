// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"flag"
	"testing"
	"time"

	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	. "github.com/gomlx/gomlx/support/exceptions"
	"github.com/gomlx/gomlx/support/testutil"
)

var (
	memProbeVariant = flag.String("mem_probe", "", "run the memory probe with this variant: fused|decomposed")
	memProbeSeqLen  = flag.Int("mem_probe_seqlen", 2048, "sequence length for the memory probe")
)

// attentionStep builds one forward+backward attention step through the SAME builder path for both
// fused and non-fused; fusion toggles via WithFusion, not a separate decomposed code path, so the
// benchmark compares the two routes of one implementation. Returns a scalar so reading it forces a
// device sync. query/key/value are pre-projected [B,S,H,D].
//
// The step runs in bf16. The fused cuDNN kernel is gated to half precision (flashSupported returns
// ErrNotImplemented for float32), so a float32 step never engages fusion: both routes collapse to
// the decomposed path and throughput reports a meaningless 1.0x. bf16 is the precision the fused
// path runs in, so this measures the route the model actually takes.
func attentionStep(useFusion bool, qHeads, kvHeads int, scale float64) func(scope *model.Scope, q, k, v *Node) []*Node {
	return func(scope *model.Scope, q, k, v *Node) []*Node {
		q, k, v = ConvertDType(q, dtypes.BFloat16), ConvertDType(k, dtypes.BFloat16), ConvertDType(v, dtypes.BFloat16)
		// Pre-projected [B,S,H,D] in; flatten head dims for the builder's pre-projected path.
		flat := func(n *Node) *Node { d := n.Shape().Dimensions; return Reshape(n, d[0], d[1], d[2]*d[3]) }
		b := MultiHeadAttention(scope, flat(q), flat(k), flat(v), qHeads, q.Shape().Dimensions[3]).
			WithPreProjected(true).WithQueryKeyScale(scale).WithCausalMask(true).WithFusion(useFusion)
		if kvHeads != qHeads {
			b = b.WithNumKVHeads(kvHeads)
		}
		out := ConvertDType(b.Done(), dtypes.Float32)
		g := Gradient(ReduceAllSum(out), q, k, v)
		// Gradients are bf16 (q/k/v are bf16); sum and cast so the harness can read a float32 scalar.
		sum := Add(Add(ReduceAllSum(g[0]), ReduceAllSum(g[1])), ReduceAllSum(g[2]))
		return []*Node{ConvertDType(sum, dtypes.Float32)}
	}
}

// TestFusionThroughput reports per-step (forward+backward) wall time for the fused vs decomposed
// route at a representative GQA attention shape. Requires a backend with fusion support (e.g. cuDNN). [cuda]
func TestFusionThroughput(t *testing.T) {
	backend := testutil.BuildTestBackend()
	if !backendSupportsFusionForBFloat16(backend) {
		t.Skipf("fusion throughput needs a backend with the fused kernel; %q has none", backend.Name())
	}
	const (
		B, S, QH, KVH, D = 2, 2048, 12, 4, 64
		scale            = 0.125
		iters            = 20
	)
	q := tensors.FromFlatDataAndDimensions(randFlat(B*S*QH*D, 1), B, S, QH, D)
	k := tensors.FromFlatDataAndDimensions(randFlat(B*S*KVH*D, 2), B, S, KVH, D)
	v := tensors.FromFlatDataAndDimensions(randFlat(B*S*KVH*D, 3), B, S, KVH, D)

	timeStep := func(useFusion bool) (time.Duration, error) {
		var perStep time.Duration
		err := TryCatch[error](func() {
			store := model.NewStore()
			exec := model.MustNewExec(backend, store, attentionStep(useFusion, QH, KVH, scale))
			exec.MustCall(q, k, v) // warmup + compile
			start := time.Now()
			for range iters {
				out := exec.MustCall(q, k, v)
				_ = tensors.ToScalar[float32](out[0]) // force device sync
				_ = out[0].FinalizeAll()              // release the step's device buffers
			}
			perStep = time.Since(start) / iters
		})
		return perStep, err
	}

	fused, err := timeStep(true)
	if err != nil {
		t.Fatalf("fused step failed: %v", err)
	}
	t.Logf("fused      per-step (fwd+bwd) B=%d S=%d %d/%d heads D=%d: %v", B, S, QH, KVH, D, fused)

	decomposed, err := timeStep(false)
	if err != nil {
		t.Logf("decomposed per-step: did not run (%v)", err)
		return
	}
	t.Logf("decomposed per-step: %v", decomposed)
	t.Logf("fused speedup: %.1fx", float64(decomposed)/float64(fused))
}

// TestFusionMemoryProbe runs one attention variant repeatedly so an external sampler (nvidia-smi,
// XLA preallocation disabled) can read the working-set peak. Enable with
// -mem_probe=fused|decomposed; skipped otherwise. [cuda]
func TestFusionMemoryProbe(t *testing.T) {
	if *memProbeVariant == "" {
		t.Skip("pass -mem_probe=fused|decomposed to run the memory probe")
	}
	backend := testutil.BuildTestBackend()
	if !backendSupportsFusionForBFloat16(backend) {
		t.Skipf("memory probe needs the fused kernel; %q has none", backend.Name())
	}
	const (
		B, QH, KVH, D = 2, 12, 4, 64
		scale         = 0.125
	)
	S := *memProbeSeqLen
	q := tensors.FromFlatDataAndDimensions(randFlat(B*S*QH*D, 1), B, S, QH, D)
	k := tensors.FromFlatDataAndDimensions(randFlat(B*S*KVH*D, 2), B, S, KVH, D)
	v := tensors.FromFlatDataAndDimensions(randFlat(B*S*KVH*D, 3), B, S, KVH, D)

	store := model.NewStore()
	exec := model.MustNewExec(backend, store, attentionStep(*memProbeVariant == "fused", QH, KVH, scale))
	for range 40 {
		out := exec.MustCall(q, k, v)
		_ = tensors.ToScalar[float32](out[0])
		_ = out[0].FinalizeAll()
	}
}
