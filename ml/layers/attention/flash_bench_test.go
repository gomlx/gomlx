// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/graphtest"
	"github.com/gomlx/gomlx/core/tensors"
	. "github.com/gomlx/gomlx/support/exceptions"
)

// attentionStep builds one forward+backward attention step (the gradient forces the backward),
// returning a scalar so reading it forces a device sync. useFlash picks the flash kernel; otherwise
// the decomposed reference with kv heads repeated to match the query heads.
func attentionStep(useFlash bool, qHeads, kvHeads int, scale float64) func(q, k, v *Node) []*Node {
	return func(q, k, v *Node) []*Node {
		var out *Node
		if useFlash {
			out = ConvertDType(FlashAttention(q, k, v, scale), dtypes.Float32)
		} else {
			group := qHeads / kvHeads
			out = naiveCausalAttention(q, repeatKVHeads(k, group), repeatKVHeads(v, group), scale)
		}
		g := Gradient(ReduceAllSum(out), q, k, v)
		return []*Node{Add(Add(ReduceAllSum(g[0]), ReduceAllSum(g[1])), ReduceAllSum(g[2]))}
	}
}

// TestFlashThroughput reports per-step (forward+backward) wall time for flash vs the decomposed
// attention at the lm-100m attention shape on the GPU. Requires a cuDNN (cuda) backend.
func TestFlashThroughput(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	if !isCUDABackend(backend) {
		t.Skipf("flash throughput needs a cuDNN (cuda) backend; got %q", backend.Name())
	}

	const (
		B, S, QH, KVH, D = 2, 2048, 12, 4, 64
		scale            = 0.125
		iters            = 20
	)
	q := tensors.FromFlatDataAndDimensions(randFlat(B*S*QH*D, 1), B, S, QH, D)
	k := tensors.FromFlatDataAndDimensions(randFlat(B*S*KVH*D, 2), B, S, KVH, D)
	v := tensors.FromFlatDataAndDimensions(randFlat(B*S*KVH*D, 3), B, S, KVH, D)

	timeStep := func(useFlash bool) (time.Duration, error) {
		var perStep time.Duration
		err := TryCatch[error](func() {
			exec := MustNewExec(backend, attentionStep(useFlash, QH, KVH, scale))
			exec.MustCall(q, k, v) // warmup + compile
			start := time.Now()
			for i := 0; i < iters; i++ {
				out := exec.MustCall(q, k, v)
				_ = tensors.ToScalar[float32](out[0]) // force device sync
			}
			perStep = time.Since(start) / iters
		})
		return perStep, err
	}

	flash, err := timeStep(true)
	if err != nil {
		t.Fatalf("flash step failed: %v", err)
	}
	t.Logf("flash      per-step (fwd+bwd) B=%d S=%d %d/%d heads D=%d: %v", B, S, QH, KVH, D, flash)

	naive, err := timeStep(false)
	if err != nil {
		t.Logf("decomposed per-step: did not run (%v)", err)
		return
	}
	t.Logf("decomposed per-step: %v", naive)
	t.Logf("flash speedup: %.1fx", float64(naive)/float64(flash))
}

// TestFlashMemoryProbe runs one attention variant repeatedly so an external sampler (nvidia-smi,
// with XLA preallocation disabled) can read the working-set peak. Set GOMLX_MEM_PROBE=flash|naive;
// skipped otherwise.
func TestFlashMemoryProbe(t *testing.T) {
	variant := os.Getenv("GOMLX_MEM_PROBE")
	if variant == "" {
		t.Skip("set GOMLX_MEM_PROBE=flash|naive to run the memory probe")
	}
	backend := graphtest.BuildTestBackend()
	if !isCUDABackend(backend) {
		t.Skipf("needs a cuDNN (cuda) backend; got %q", backend.Name())
	}

	const (
		B, QH, KVH, D = 2, 12, 4, 64
		scale         = 0.125
	)
	S := 2048
	if v := os.Getenv("GOMLX_PROBE_S"); v != "" {
		s, err := strconv.Atoi(v)
		if err != nil {
			t.Fatalf("GOMLX_PROBE_S=%q: %v", v, err)
		}
		S = s
	}
	q := tensors.FromFlatDataAndDimensions(randFlat(B*S*QH*D, 1), B, S, QH, D)
	k := tensors.FromFlatDataAndDimensions(randFlat(B*S*KVH*D, 2), B, S, KVH, D)
	v := tensors.FromFlatDataAndDimensions(randFlat(B*S*KVH*D, 3), B, S, KVH, D)

	exec := MustNewExec(backend, attentionStep(variant == "flash", QH, KVH, scale))
	for i := 0; i < 40; i++ {
		out := exec.MustCall(q, k, v)
		_ = tensors.ToScalar[float32](out[0])
	}
}
