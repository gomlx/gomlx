// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn

import (
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/ggml"
)

// QuantizedGather performs a quantized embedding lookup: gathers rows from a quantized
// embedding table and dequantizes only the selected rows on-the-fly.
//
// This is the quantized analogue of graph.Gather for embedding lookups, similar to
// llama.cpp's ggml_get_rows. The full embedding table stays in native quantized format
// and only the rows selected by indices are dequantized to float32.
//
// Types with a decomposed graph-level fallback (Q4_0, Q8_0, IQ4_NL) work on any backend
// (including XLA). K-quant types (Q4_K, Q6_K, etc.) require a backend with a fused
// executor (e.g. simplego); other backends will panic.
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
	if ggml.CanDecompose(quant.GGMLType) {
		return InternalFusedOpCaller(
			func() *Node { return BackendFusedQuantizedGather(table, indices, quant) },
			func() *Node { return ggml.GatherDecomposed(table, indices, quant.GGMLType) },
		)
	}
	return BackendFusedQuantizedGather(table, indices, quant)
}
