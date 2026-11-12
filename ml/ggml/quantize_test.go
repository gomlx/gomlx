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
		{compute.GGMLQ4_1, compute.GGMLQ8_0},
		{compute.GGMLQ5_0, compute.GGMLQ8_0},
		{compute.GGMLQ5_1, compute.GGMLQ8_0},
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

func TestDequantize_Q4_1_Q5_0_Q5_1(t *testing.T) {
	const (
		dBits = uint16(0x3800) // IEEE-754 binary16 exact 0.5
		mBits = uint16(0xBD00) // IEEE-754 binary16 exact -1.25
		d     = float32(0.5)
		m     = float32(-1.25)
	)

	// Q4_1: 20 bytes per 32-element block
	q41Row := make([]uint8, 20)
	binary.LittleEndian.PutUint16(q41Row[0:2], dBits)
	binary.LittleEndian.PutUint16(q41Row[2:4], mBits)
	wantQ41 := make([]float32, 32)
	for j := range 16 {
		lo := uint8((j * 3) & 0x0F)
		hi := uint8((j*5 + 7) & 0x0F)
		q41Row[4+j] = lo | (hi << 4)
		wantQ41[j] = d*float32(lo) + m
		wantQ41[16+j] = d*float32(hi) + m
	}
	graphtest.RunTestGraphFn(t, "Dequant_Q4_1", func(g *Graph) ([]*Node, []*Node) {
		w := Const(g, [][]uint8{q41Row})
		return []*Node{w}, []*Node{ggml.Dequantize(w, compute.GGMLQ4_1, 1)}
	}, []any{[][]float32{wantQ41}}, 1e-5)

	// Q5_0: 22 bytes per 32-element block
	q50Row := make([]uint8, 22)
	binary.LittleEndian.PutUint16(q50Row[0:2], dBits)
	var qh50 uint32
	wantQ50 := make([]float32, 32)
	for j := range 16 {
		q0 := uint8((j * 7) % 32)
		q1 := uint8((j*11 + 5) % 32)
		q50Row[6+j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4)
		if q0&0x10 != 0 {
			qh50 |= uint32(1) << uint(j)
		}
		if q1&0x10 != 0 {
			qh50 |= uint32(1) << uint(16+j)
		}
		wantQ50[j] = d * float32(int(q0)-16)
		wantQ50[16+j] = d * float32(int(q1)-16)
	}
	binary.LittleEndian.PutUint32(q50Row[2:6], qh50)
	graphtest.RunTestGraphFn(t, "Dequant_Q5_0", func(g *Graph) ([]*Node, []*Node) {
		w := Const(g, [][]uint8{q50Row})
		return []*Node{w}, []*Node{ggml.Dequantize(w, compute.GGMLQ5_0, 1)}
	}, []any{[][]float32{wantQ50}}, 1e-5)

	// Q5_1: 24 bytes per 32-element block
	q51Row := make([]uint8, 24)
	binary.LittleEndian.PutUint16(q51Row[0:2], dBits)
	binary.LittleEndian.PutUint16(q51Row[2:4], mBits)
	var qh51 uint32
	wantQ51 := make([]float32, 32)
	for j := range 16 {
		q0 := uint8((j*9 + 3) % 32)
		q1 := uint8((j*13 + 1) % 32)
		q51Row[8+j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4)
		if q0&0x10 != 0 {
			qh51 |= uint32(1) << uint(j)
		}
		if q1&0x10 != 0 {
			qh51 |= uint32(1) << uint(16+j)
		}
		wantQ51[j] = d*float32(q0) + m
		wantQ51[16+j] = d*float32(q1) + m
	}
	binary.LittleEndian.PutUint32(q51Row[4:8], qh51)
	graphtest.RunTestGraphFn(t, "Dequant_Q5_1", func(g *Graph) ([]*Node, []*Node) {
		w := Const(g, [][]uint8{q51Row})
		return []*Node{w}, []*Node{ggml.Dequantize(w, compute.GGMLQ5_1, 1)}
	}, []any{[][]float32{wantQ51}}, 1e-5)
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
