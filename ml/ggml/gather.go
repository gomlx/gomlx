// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ggml

import (
	"github.com/gomlx/compute"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/support/exceptions"
)

// EmbeddingLookupDecomposed performs a quantized embedding lookup using graph-level dequantization.
//
// It gathers raw Uint8 rows from the quantized table, then dequantizes only the
// selected rows to Float32.
//
// Parameters:
//   - table: [vocabSize, bytesPerRow] Uint8 with native GGML block layout.
//   - indices: integer tensor with static batch dimensions and last dimension = 1 (Gather convention).
//   - ggmlType: the GGML block format (e.g. Q4_0, Q8_0, IQ4_NL, Q4_K, Q6_K).
//
// Output: Float32 tensor with shape [batch..., K] where K is the logical embedding dimension.
func EmbeddingLookupDecomposed(table, indices *Node, ggmlType compute.GGMLQuantType) *Node {
	indicesShape := indices.Shape()
	if indicesShape.IsDynamic() {
		exceptions.Panicf("ggml.EmbeddingLookupDecomposed does not support dynamic indices shape %s (requires fused QuantizedEmbeddingLookup backend support)", indicesShape)
	}
	tableShape := table.Shape()
	bytesPerRow := tableShape.Dimensions[1]
	bpb := ggmlType.BytesPerBlock()
	vpb := ggmlType.ValuesPerBlock()
	numBlocks := bytesPerRow / bpb
	K := numBlocks * vpb

	// Gather raw Uint8 rows: indices [batch..., 1] → gathered [batch..., bytesPerRow] Uint8
	gathered := Gather(table, indices)

	// Flatten batch dimensions for dequant: [totalRows, bytesPerRow]
	batchDims := indicesShape.Dimensions[:indicesShape.Rank()-1]
	totalRows := 1
	for _, d := range batchDims {
		totalRows *= d
	}
	gathered = Reshape(gathered, totalRows, bytesPerRow)

	// Dequantize: [totalRows, K] Float32
	dequantized := Dequantize(gathered, ggmlType, totalRows)

	// Reshape back to [batch..., K]
	outDims := make([]int, len(batchDims)+1)
	copy(outDims, batchDims)
	outDims[len(batchDims)] = K
	return Reshape(dequantized, outDims...)
}
