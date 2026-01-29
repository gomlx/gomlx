/*
 *	Copyright 2024 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

// Package pos provides positional embedding implementations for attention mechanisms.
// Different positional embedding strategies (RoPE, ALiBi, learned embeddings, etc.)
// can be implemented using the common PositionalEmbedding interface.
package pos

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// PositionalEmbedding is the interface for applying positional information to attention inputs.
// Different implementations can provide various positional encoding strategies like RoPE,
// ALiBi (Attention with Linear Biases), learned positional embeddings, or sinusoidal embeddings.
//
// The interface is designed to work with attention mechanisms where position information
// needs to be incorporated into query and key projections.
type PositionalEmbedding interface {
	// Apply applies the positional embedding to the input tensor.
	//
	// Parameters:
	//   - x: Input tensor, typically shaped [..., seq_len, head_dim] for attention projections
	//   - positionIndices: Position indices tensor shaped [..., seq_len] where each element
	//                      contains the position value for that token. The leading dimensions
	//                      are broadcast to match x's batch dimensions. This allows:
	//                      - Sequential positions: [0, 1, 2, 3]
	//                      - Rotating cache with wrap: [1022, 1023, 0, 1, 2]
	//                      - Batched multi-client: [[5,6,7], [127,128,129]]
	//
	// Returns:
	//   - Tensor with positional information applied. Note: The returned shape may differ
	//     from the input shape depending on the implementation (e.g., some encodings may
	//     append dimensions while others modify in-place).
	//
	// The exact behavior depends on the implementation:
	//   - RoPE: Rotates pairs of dimensions based on position (preserves shape)
	//   - Learned: Adds learned position vectors (may change shape)
	//   - Sinusoidal: Adds fixed sinusoidal position encodings (may change shape)
	Apply(x *Node, positionIndices *Node) *Node
}

// SequentialPositions creates position indices for sequential positions starting from startPos.
//
// Parameters:
//   - g: Graph to create the computation in
//   - startPos: Starting position as a *Node. Can be either:
//   - Scalar: Returns [seqLen] positions for all batches
//   - Shape [batchSize]: Returns [batchSize, seqLen] with each batch at different position
//   - seqLen: Sequence length
//
// Returns:
//   - Position indices with values [startPos, startPos+1, ..., startPos+seqLen-1]
//     Shape is [seqLen] for scalar startPos, or [batchSize, seqLen] for batched startPos.
//     Returns Int32 dtype; the PositionalEmbedding will convert to the appropriate dtype.
//
// Example:
//
//	// Scalar startPos - all batches at same position:
//	posIndices := SequentialPositions(g, Const(g, int32(5)), 4)
//	// Result: [5, 6, 7, 8] with shape [4]
//
//	// Batched startPos - each batch at different position (for multi-client serving):
//	posIndices := SequentialPositions(g, Const(g, []int32{5, 10}), 4)
//	// Result: [[5, 6, 7, 8], [10, 11, 12, 13]] with shape [2, 4]
func SequentialPositions(g *Graph, startPos *Node, seqLen int) *Node {
	// Convert startPos to Int32
	posNode := ConvertDType(startPos, dtypes.Int32)

	// Create [0, 1, 2, ..., seqLen-1]
	offsets := Iota(g, shapes.Make(dtypes.Int32, seqLen), 0)

	// Scalar case
	if posNode.Rank() == 0 || (posNode.Rank() == 1 && posNode.Shape().Dimensions[0] == 1) {
		if posNode.Rank() > 0 {
			posNode = Squeeze(posNode)
		}
		posNode = BroadcastToShape(posNode, offsets.Shape())
		return Add(offsets, posNode)
	}

	// Batched case: startPos has shape [batchSize]
	// Result should be [batchSize, seqLen]
	batchSize := posNode.Shape().Dimensions[0]

	offsets = ExpandDims(offsets, 0)
	offsets = BroadcastToShape(offsets, shapes.Make(dtypes.Int32, batchSize, seqLen))

	posNode = ExpandDims(posNode, -1)
	posNode = BroadcastToShape(posNode, shapes.Make(dtypes.Int32, batchSize, seqLen))

	return Add(offsets, posNode)
}
