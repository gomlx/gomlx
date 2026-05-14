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

// Package pos provides positional encoding implementations for attention mechanisms.
// Different positional encoding strategies (RoPE, learned embeddings, etc.)
// can be implemented using the common PositionalEncoder interface.
package pos

import (
	. "github.com/gomlx/gomlx/core/graph"
)

// Encoder is the interface for applying positional information to attention inputs.
// Different implementations can provide various positional encoding strategies like RoPE,
// ALiBi (Attention with Linear Biases), learned positional embeddings, or sinusoidal embeddings.
//
// One issue is that the positional encoder is applied in different places for different versions.
// So there are different interfaces an Encoder can implement, each will be called in the appropriate
// place in a transformer model / attention layer. This API defines the following positional encoder
// interfaces:
//
//   - PreEmbedder: applies positional information to the initial sequence tensor (e.g. Learned encoder),
//     applied just after the token embedding table lookup.
//   - QKEncoder: applies positional information to query and key vectors (e.g. RoPE)
type Encoder interface {
	// Name returns the name of the encoder.
	Name() string
}

// QKEncoder is the interface for applying positional information to query and key vectors.
// This is the one used by RoPE for instance.
type QKEncoder interface {
	// EncodeQK vectors with positional information.
	//
	// Parameters:
	//   - q, k: Input tensor, shaped [batchSize..., sequenceLength, ...].
	//   - positionIndices: Position indices tensor shaped [[batchSize...,] seqLen] where each element
	//                      contains the position value for that token. The batchSize dimension
	//                      is broadcast to match x's batch dimensions if not set. Examples:
	//                      - Sequential positions: [0, 1, 2, 3]
	//                      - Rotating cache with wrap: [1022, 1023, 0, 1, 2]
	//                      - Batched multi-client: [[5,6,7], [127,128,129]]
	//   - seqAxis: The axis index in x that represents the sequence dimension.
	//              For standard [..., seq_len, head_dim] tensors this is x.Rank()-2.
	//              For BSHD layout [batch, seq, heads, dim] this is 1.
	//              This makes the encoder layout-proof: callers specify which axis is seq.
	//              Negative values are ok, they are counted from the end.
	//
	// Returns:
	//   - Tensor with positional information applied, same shape as x.
	//
	// The exact behavior depends on the implementation:
	//   - RoPE: Rotates pairs of dimensions based on position (preserves shape)
	//   - Learned: Adds learned position vectors (may change shape)
	//   - Sinusoidal: Adds fixed sinusoidal position encodings (may change shape)
	EncodeQK(q, k *Node, positionIndices *Node, seqAxis int) (*Node, *Node)
}

type PreEncoder interface {
	// PreEncode applies an absolute positional information to the initial sequence tensor,
	// usually just after the token embedding table lookup, and before the attention layers.
	//
	// Parameters:
	//   - x: Input tensor, shaped [batchSize..., sequenceLength, ...].
	//   - positionIndices: Position indices tensor shaped [[batchSize...,] seqLen] where each element
	//                      contains the position value for that token. The batchSize dimension
	//                      is broadcast to match x's batch dimensions if not set. Examples:
	//                      - Sequential positions: [0, 1, 2, 3]
	//                      - Rotating cache with wrap: [1022, 1023, 0, 1, 2]
	//                      - Batched multi-client: [[5,6,7], [127,128,129]]
	//   - seqAxis: The axis index in x that represents the sequence dimension.
	//              For standard [..., seq_len, head_dim] tensors this is x.Rank()-2.
	//              For BSHD layout [batch, seq, heads, dim] this is 1.
	//              This makes the encoder layout-proof: callers specify which axis is seq.
	//              Negative values are ok, they are counted from the end.
	//
	// Returns:
	//   - Tensor with positional information applied, same shape as x.
	PreEncode(x, positionIndices *Node, seqAxis int) *Node
}
