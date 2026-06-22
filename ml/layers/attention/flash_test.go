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
