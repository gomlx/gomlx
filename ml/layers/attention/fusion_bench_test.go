// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"flag"
	"testing"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/support/humanize"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	. "github.com/gomlx/gomlx/support/exceptions"
	"github.com/gomlx/gomlx/support/testutil"
)

var (
	benchFusion     = flag.Bool("bench_fusion", false, "benchmark the fused attention kernel")
	memProbeVariant = flag.String("mem_probe", "", "run the memory probe with this variant: fused|decomposed")
	memProbeSeqLen  = flag.Int("mem_probe_seqlen", 2048, "sequence length for the memory probe")
)

// attentionStep builds numSteps forward+backward attention step through the SAME builder path for both
// fused and non-fused; fusion toggles via WithFusion, not a separate decomposed code path, so the
// benchmark compares the two routes of one implementation. Returns a scalar so reading it forces a
// device sync. query/key/value are pre-projected [B,S,H,D].
//
// The step runs in bf16. The fused cuDNN kernel is gated to half precision (flashSupported returns
// ErrNotImplemented for float32), so a float32 step never engages fusion: both routes collapse to
// the decomposed path and throughput reports a meaningless 1.0x. bf16 is the precision the fused
// path runs in, so this measures the route the model actually takes.
func attentionStep(numSteps int, useFusion bool, qHeads, kvHeads int, scale float64) func(scope *model.Scope, q, k, v *Node) *Node {
	return func(scope *model.Scope, qIn, kIn, vIn *Node) *Node {
		q, k, v := ConvertDType(qIn, dtypes.BFloat16), ConvertDType(kIn, dtypes.BFloat16), ConvertDType(vIn, dtypes.BFloat16)

		flat := func(n *Node) *Node { d := n.Shape().Dimensions; return Reshape(n, d[0], d[1], d[2]*d[3]) }
		unflat := func(n, original *Node) *Node { return Reshape(n, original.Shape().Dimensions...) }
		var out *Node
		for i := range numSteps {
			// Pre-projected [B,S,H,D] in; flatten head dims for the builder's pre-projected path.
			mhaBuilder := MultiHeadAttention(scope.In("layer_%d", i), flat(q), flat(k), flat(v), qHeads, q.Shape().Dimensions[3]).
				WithPreProjected(true).WithQueryKeyScale(scale).WithCausalMask(true).WithFusion(useFusion)
			if kvHeads != qHeads {
				mhaBuilder = mhaBuilder.WithNumKVHeads(kvHeads)
			}
			out = mhaBuilder.Done()
			q = unflat(out, q)
			k = unflat(out, k)
			v = unflat(out, v)
		}
		out = ConvertDType(out, dtypes.Float32)
		g := Gradient(ReduceAllSum(out), qIn, kIn, vIn)
		sum := Add(Add(ReduceAllSum(g[0]), ReduceAllSum(g[1])), ReduceAllSum(g[2]))
		return sum
	}
}

// TestFusionThroughput reports per-step (forward+backward) wall time for the fused vs decomposed
// route at a representative GQA attention shape. Requires a backend with fusion support (e.g. cuDNN). [cuda]
func TestFusionThroughput(t *testing.T) {
	if !*benchFusion {
		t.Skip("pass -bench_fusion to run")
	}
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		if !backendSupportsFusionForBFloat16(backend) {
			t.Skipf("fusion throughput needs a backend with the fused kernel; %q has none", backend.Name())
		}
		const (
			B, S, NumHeads, D = 8, 1024, 16, 128
			scale             = 0.125
			numAttentionSteps = 10
			iters             = 100
		)
		q := tensors.FromFlatDataAndDimensions(randFlat(B*S*NumHeads*D, 1), B, S, NumHeads, D)
		k := tensors.FromFlatDataAndDimensions(randFlat(B*S*NumHeads*D, 2), B, S, NumHeads, D)
		v := tensors.FromFlatDataAndDimensions(randFlat(B*S*NumHeads*D, 3), B, S, NumHeads, D)

		timeStep := func(useFusion bool) (time.Duration, error) {
			var perStep time.Duration
			err := TryCatch[error](func() {
				store := model.NewStore()
				exec := model.MustNewExec1(backend, store, attentionStep(numAttentionSteps, useFusion, NumHeads, NumHeads, scale))
				exec.MustCall(q, k, v) // warmup + compile
				start := time.Now()
				for range iters {
					out := exec.MustCall(q, k, v)
					_ = tensors.ToScalar[float32](out) // force device sync
					_ = out.FinalizeAll()              // release the step's device buffers
				}
				perStep = time.Since(start) / iters
			})
			return perStep, err
		}

		fused, err := timeStep(true)
		if err != nil {
			t.Fatalf("fused step failed: %v", err)
		}
		t.Logf("fused      per-step (fwd+bwd) %d stacked attentions, B=%d S=%d %d/%d heads D=%d: %s", numAttentionSteps, B, S, NumHeads, NumHeads, D, humanize.Duration(fused))

		decomposed, err := timeStep(false)
		if err != nil {
			t.Logf("decomposed per-step: did not run (%v)", err)
			return
		}
		t.Logf("decomposed per-step: %s", humanize.Duration(decomposed))
		t.Logf("fused speedup: %.1fx", float64(decomposed)/float64(fused))
	})
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
	exec := model.MustNewExec1(backend, store, attentionStep(4, *memProbeVariant == "fused", QH, KVH, scale))
	for range 40 {
		out := exec.MustCall(q, k, v)
		_ = tensors.ToScalar[float32](out)
		_ = out.FinalizeAll()
	}
}
