// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/support/envutil"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/ggml"
)

// QuantizedGather performs a quantized embedding lookup: gathers rows from a quantized
// embedding table and dequantizes only the selected rows on-the-fly.
//
// This is the quantized analogue of graph.Gather for embedding lookups, similar to
// llama.cpp's ggml_get_rows. The full embedding table stays in native quantized format
// and only the rows selected by indices are dequantized to float32.
//
// All supported GGML types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, Q2_K, Q3_K, Q4_K,
// Q5_K, Q6_K, Q8_K) have a decomposed graph-level fallback and work on any backend
// (including XLA) when batch dimensions are static. Dynamic batch dimensions require
// a backend with a fused executor (e.g. "go"); other backends will panic.
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
	if quant.Scheme != compute.QuantGGML {
		panic("nn.QuantizedGather: only QuantGGML scheme is supported")
	}

	decomposed := func() *Node {
		return ggml.EmbeddingLookupDecomposed(table, indices, quant.GGMLType)
	}

	if enabled, err := envutil.ReadBool(FusionEnv, true); err != nil {
		panic(err)
	} else if !enabled {
		return decomposed()
	}

	backend := table.Graph().Backend()
	if backend != nil && !backend.Capabilities().Operations[compute.OpTypeQuantizedEmbeddingLookup] {
		return decomposed()
	}

	if ggml.CanDecompose(quant.GGMLType) {
		res, _ := InternalFusedOpCaller(
			func() *Node { return BackendQuantizedEmbeddingLookup(table, indices, quant) },
			decomposed,
			false,
		)
		return res
	}
	return BackendQuantizedEmbeddingLookup(table, indices, quant)
}
