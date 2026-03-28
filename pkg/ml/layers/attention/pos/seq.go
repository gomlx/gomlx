package pos

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// SequentialPositions creates position indices for sequential positions starting from startPos.
//
// Parameters:
//   - g: Graph to create the computation in
//   - startPos: Starting position as a *Node. Can be either:
//     Scalar: Returns [seqLen] positions for all batches;
//     Shape [batchSize]: Returns [batchSize, seqLen] with each batch at different position;
//     seqLen: Sequence length.
//
// Returns:
//   - Position indices with values [startPos, startPos+1, ..., startPos+seqLen-1]
//     Shape is [seqLen] for scalar startPos, or [batchSize, seqLen] for batched startPos.
//     Returns Int32 dtype; the Encoder will convert to the appropriate dtype.
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
