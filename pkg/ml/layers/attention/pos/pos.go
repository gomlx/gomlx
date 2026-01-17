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
	. "github.com/gomlx/gomlx/pkg/core/graph"
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
	//   - startPos: Starting position in the sequence as a *Node (scalar int32)
	//               for dynamic positioning (required for graph caching)
	//
	// Returns:
	//   - Modified tensor with positional information applied
	//
	// The exact behavior depends on the implementation:
	//   - RoPE: Rotates pairs of dimensions based on position
	//   - ALiBi: Would modify attention logits (different interface needed)
	//   - Learned: Adds learned position vectors
	//   - Sinusoidal: Adds fixed sinusoidal position encodings
	Apply(x *Node, startPos *Node) *Node
}
