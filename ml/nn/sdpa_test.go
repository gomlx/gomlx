// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn

import (
	"math"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFusedScaledDotProductAttention(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		t.Run("CausalBSHD", func(t *testing.T) {
			exec := MustNewExec(backend, func(q, k, v *Node) *Node {
				return FusedScaledDotProductAttention(q, k, v, compute.AttentionAxesLayoutBSHD, &FusedSDPAConfig{
					Causal: true,
				})
			})
			// [B=1, S=2, H=1, D=2]
			q := [][][][]float32{{{{1, 0}}, {{0, 1}}}}
			k := [][][][]float32{{{{1, 0}}, {{0, 1}}}}
			v := [][][][]float32{{{{10, 20}}, {{30, 40}}}}
			got := exec.MustExec1(q, k, v).Value().([][][][]float32)

			// Row 0 only attends to pos 0 -> [10, 20].
			assert.InDelta(t, 10.0, got[0][0][0][0], 1e-4)
			assert.InDelta(t, 20.0, got[0][0][0][1], 1e-4)

			// Row 1 attends to pos 0 (score 0) and pos 1 (score 1/sqrt(2)).
			s1 := 1.0 / math.Sqrt(2.0)
			w0 := 1.0 / (1.0 + math.Exp(s1))
			w1 := 1.0 - w0
			assert.InDelta(t, w0*10+w1*30, got[0][1][0][0], 1e-4)
			assert.InDelta(t, w0*20+w1*40, got[0][1][0][1], 1e-4)
		})

		t.Run("GQAWithBiasAndBHSD", func(t *testing.T) {
			exec := MustNewExec(backend, func(q, k, v, bias *Node) *Node {
				return FusedScaledDotProductAttention(q, k, v, compute.AttentionAxesLayoutBHSD, &FusedSDPAConfig{
					Bias: bias,
				})
			})
			// Q: [B=1, H=2, Sq=1, D=2], K/V: [B=1, Hkv=1, Skv=2, D=2], Bias: [1, 2, 1, 2]
			q := [][][][]float32{{{{1, 0}}, {{0, 1}}}}
			k := [][][][]float32{{{{1, 0}, {0, 1}}}}
			v := [][][][]float32{{{{10, 0}, {0, 20}}}}
			bias := [][][][]float32{{{{0, -100}}, {{-100, 0}}}}
			got := exec.MustExec1(q, k, v, bias).Value().([][][][]float32)
			require.Len(t, got[0], 2)
			// Head 0 heavily favors key 0 -> [10, 0]; head 1 heavily favors key 1 -> [0, 20].
			assert.InDelta(t, 10.0, got[0][0][0][0], 1e-4)
			assert.InDelta(t, 0.0, got[0][0][0][1], 1e-4)
			assert.InDelta(t, 0.0, got[0][1][0][0], 1e-4)
			assert.InDelta(t, 20.0, got[0][1][0][1], 1e-4)
		})

		t.Run("ValidationPanics", func(t *testing.T) {
			g := NewGraph(backend, "sdpa_validation")
			q := Parameter(g, "q", shapes.Make(dtypes.Float32, 1, 2, 2, 4))
			k := Parameter(g, "k", shapes.Make(dtypes.Float32, 1, 2, 2, 4))
			v := Parameter(g, "v", shapes.Make(dtypes.Float32, 1, 2, 2, 4))
			mask := Parameter(g, "mask", shapes.Make(dtypes.Bool, 1, 2, 1, 2))
			seqLen := Parameter(g, "seq", shapes.Make(dtypes.Int32, 1))
			require.Panics(t, func() {
				FusedScaledDotProductAttention(q, k, v, compute.AttentionAxesLayoutBSHD, &FusedSDPAConfig{
					Causal: true, Mask: mask,
				})
			})
			require.Panics(t, func() {
				FusedScaledDotProductAttention(q, k, v, compute.AttentionAxesLayoutBHSD, &FusedSDPAConfig{
					QuerySeqLen: seqLen,
				})
			})
		})
	})
}
