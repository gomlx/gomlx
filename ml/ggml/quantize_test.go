// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ggml_test

import (
	"encoding/binary"
	"testing"

	"github.com/gomlx/compute"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/graphtest"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/ggml"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

func TestActivationQuantTypeFor(t *testing.T) {
	for _, tc := range []struct {
		wType compute.GGMLQuantType
		want  compute.GGMLQuantType
	}{
		{compute.GGMLQ8_0, compute.GGMLQ8_0},
		{compute.GGMLQ4_0, compute.GGMLQ8_0},
		{compute.GGMLIQ4NL, compute.GGMLQ8_0},
		{compute.GGMLQ2_K, compute.GGMLQ8_K},
		{compute.GGMLQ3_K, compute.GGMLQ8_K},
		{compute.GGMLQ4_K, compute.GGMLQ8_K},
		{compute.GGMLQ5_K, compute.GGMLQ8_K},
		{compute.GGMLQ6_K, compute.GGMLQ8_K},
		{compute.GGMLQ8_K, compute.GGMLQ8_K},
	} {
		got, ok := ggml.ActivationQuantTypeFor(tc.wType)
		assert.True(t, ok, "expected ActivationQuantTypeFor(%s) to succeed", tc.wType)
		assert.Equal(t, tc.want, got)
	}
}

func TestQuantizeDequantize_Q8_0_And_Q8_K(t *testing.T) {
	x3D := make([][][]float32, 2)
	for b := range 2 {
		x3D[b] = make([][]float32, 2)
		for s := range 2 {
			row := make([]float32, 64)
			if b == 1 && s == 0 {
				// Keep one row all-zero to test zero-scale safety.
				x3D[b][s] = row
				continue
			}
			for k := range 64 {
				row[k] = float32((k%13)-6) * 0.25 * float32(b+1)
			}
			x3D[b][s] = row
		}
	}
	graphtest.RunTestGraphFn(t, "Q8_0_3D_RoundTrip", func(g *Graph) ([]*Node, []*Node) {
		xNode := Const(g, x3D)
		q := ggml.Quantize(xNode, compute.GGMLQ8_0)
		// Flatten batch dims to [4, 68] for Dequantize [N, bytesPerRow] -> [4, 64], then reshape back.
		deq := Reshape(ggml.Dequantize(Reshape(q, 4, 68), compute.GGMLQ8_0, 4), 2, 2, 64)
		return []*Node{xNode}, []*Node{deq}
	}, []any{x3D}, 0.03)

	// Verify Q8_K round-trip and exact bsums layout on default backend.
	backend := graphtest.BuildTestBackend()
	row0 := make([]float32, 256)
	for i := range 256 {
		row0[i] = float32((i%19)-9) * 0.2
	}
	row1 := make([]float32, 256) // all zeros

	outs := MustExecOnceN(backend, func(g *Graph) []*Node {
		x := Const(g, [][]float32{row0, row1})
		q := ggml.Quantize(x, compute.GGMLQ8_K)
		deq := ggml.Dequantize(q, compute.GGMLQ8_K, 2)
		return []*Node{q, deq}
	})

	qBytes, err := tensors.CopyFlatData[uint8](outs[0])
	require.NoError(t, err)
	require.Len(t, qBytes, 2*292)

	// Verify bsums[0..15] == sum of each 16-element int8 sub-block of qs[4:260].
	blk0 := qBytes[:292]
	for sb := range 16 {
		var wantSum int16
		for j := range 16 {
			wantSum += int16(int8(blk0[4+sb*16+j]))
		}
		gotSum := int16(binary.LittleEndian.Uint16(blk0[260+sb*2 : 260+(sb+1)*2]))
		assert.Equal(t, wantSum, gotSum, "Q8_K bsums[%d] mismatch", sb)
	}
	for i, b := range qBytes[292:] {
		assert.Zero(t, b, "expected zero byte at offset %d for all-zero Q8_K row", i)
	}

	deqVals, err := tensors.CopyFlatData[float32](outs[1])
	require.NoError(t, err)
	assert.InDeltaSlice(t, row0, deqVals[:256], 0.02)
	assert.InDeltaSlice(t, row1, deqVals[256:], 1e-6)
}
