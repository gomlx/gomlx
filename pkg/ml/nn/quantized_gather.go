// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn

import (
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// QuantizedGather performs a quantized embedding lookup: gathers rows from a quantized
// embedding table and dequantizes only the selected rows on-the-fly.
//
// This is the quantized analogue of graph.Gather for embedding lookups, similar to
// llama.cpp's ggml_get_rows. The full embedding table stays in native quantized format
// (e.g. GGML Q6_K), and only the rows selected by indices are dequantized to float32.
//
// Parameters:
//   - table: [vocabSize, bytesPerRow] Uint8 with native GGML block layout.
//   - indices: integer tensor with last dimension = 1 (same as Gather convention).
//     For embeddings: [batch, seqLen, 1].
//   - quant: quantization metadata. Only QuantGGML scheme is supported.
//
// Output: float32 tensor with shape [batch..., K] where K is the logical embedding dimension
// derived from the block format: K = (bytesPerRow / bytesPerBlock) * valuesPerBlock.
func QuantizedGather(table, indices *Node, quant *Quantization) *Node {
	if quant.Scheme != backends.QuantGGML {
		panic("nn.QuantizedGather: only QuantGGML scheme is supported")
	}
	// Convert graph-level Quantization to backend-level for the generated function.
	bq := &backends.Quantization{
		Scheme:   quant.Scheme,
		GGMLType: quant.GGMLType,
	}
	return FusedQuantizedGather(table, indices, bq)
}
