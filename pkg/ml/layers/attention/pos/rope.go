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

package pos

import (
	. "github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// RoPE implements rotary position embeddings ("RoFormer", see [1]).
// It splits the embedding (aka. features) axis into pairs, and rotates
// them at different frequencies, according to position.
//
// [1] "RoFormer: Enhanced Transformer with Rotary Position Embedding", https://arxiv.org/abs/2104.09864
type RoPE struct {
	BaseFreq float64
	DimStart int
	DimEnd   int
}

// NewRoPE creates a RoPE positional embedding that applies to the entire embedding axis.
//
// Parameters:
//   - baseFreq: Base frequency for rotary embeddings, typically 10000.0
//
// Returns:
//   - A RoPE instance configured to apply to the full embedding dimension
//
// Example:
//
//	rope := NewRoPE(10000.0)
//	embedded := rope.Apply(x, positionIndices)
func NewRoPE(baseFreq float64) *RoPE {
	return &RoPE{
		BaseFreq: baseFreq,
		DimStart: 0,
		DimEnd:   -1, // -1 means use full dimension
	}
}

// NewRoPEWithDimRange creates a RoPE that applies only to [dimStart:dimEnd].
// This is useful for applying RoPE to only a slice of the embedding (or feature) axis.
//
// Parameters:
//   - baseFreq: Base frequency for rotary embeddings, typically 10000.0
//   - dimStart: Start index of the dimension range (inclusive), must be >= 0
//   - dimEnd: End index of the dimension range (exclusive), or -1 for end of axis
//
// Returns:
//   - A RoPE instance configured to apply to the specified dimension range
//
// Example:
//
//	// Apply RoPE only to dimensions 0-64 of a 128-dim embedding
//	rope := NewRoPEWithDimRange(10000.0, 0, 64)
//	embedded := rope.Apply(x, positionIndices)
func NewRoPEWithDimRange(baseFreq float64, dimStart, dimEnd int) *RoPE {
	return &RoPE{
		BaseFreq: baseFreq,
		DimStart: dimStart,
		DimEnd:   dimEnd,
	}
}

// Apply implements the PositionalEncoder interface.
// It applies rotary position embeddings to x using the provided position indices.
// The rotation is applied on a range of frequencies multiplied by the position
// of each element, as specified by positionIndices.
//
// Parameters:
//   - x: Input tensor shaped [..., seq_len, head_dim|embed_dim]
//   - positionIndices: Position indices shaped [..., seq_len] with position values for each token
//
// Returns:
//   - Tensor with rotary position embeddings applied, same shape as x
//
// Example:
//
//	rope := NewRoPE(10000.0)
//	// x has shape [batch, seq_len, embed_dim]
//	// positions has shape [batch, seq_len] with values like [0, 1, 2, ...]
//	embedded := rope.Apply(x, positions)
func (r *RoPE) Apply(x *Node, positionIndices *Node) *Node {
	if r.DimStart == 0 && r.DimEnd == -1 {
		// Apply to entire dimension
		return applyRoPE(x, positionIndices, r.BaseFreq)
	}

	// Apply to custom dimension range
	dimEnd := r.DimEnd
	if dimEnd == -1 {
		dimEnd = x.Shape().Dimensions[x.Shape().Rank()-1]
	}

	return applyRoPEWithCustomDim(x, positionIndices, r.BaseFreq, r.DimStart, dimEnd)
}

// applyRoPE applies rotary position embeddings to the last dimension (even length).
// Reference: RoFormer (https://arxiv.org/abs/2104.09864).
//
// Parameters:
//   - x: Input tensor shaped [..., seq_len, head_dim|embed_dim], last dim must be even
//   - positionIndices: Position indices shaped [..., seq_len]
//   - baseFreq: Base frequency for computing rotation angles
//
// Returns:
//   - Tensor with rotary embeddings applied, same shape as x
func applyRoPE(x *Node, positionIndices *Node, baseFreq float64) *Node {
	g := x.Graph()
	shape := x.Shape()
	dtype := shape.DType
	rank := shape.Rank()

	// Extract dimensions - we apply RoPE to the last axis (head_dim or embed_dim)
	seqLen := shape.Dimensions[rank-2] // Second to last axis is sequence length
	embedDim := shape.Dimensions[rank-1]

	// RoPE is applied to pairs of dimensions, so embedDim must be even
	if embedDim%2 != 0 {
		Panicf("RoPE requires even embedding dimension, got %d", embedDim)
	}

	// Use provided position indices and convert to appropriate dtype
	// Position indices should be shaped [..., seq_len]
	positions := ConvertDType(positionIndices, dtype)

	// Validate position indices shape - last dimension must match seqLen
	posRank := positions.Rank()
	if posRank == 0 || positions.Shape().Dimensions[posRank-1] != seqLen {
		Panicf("RoPE positionIndices last dimension must match seq_len=%d, got shape %s", seqLen, positions.Shape())
	}

	// Compute frequency for each pair of dimensions
	// freq_i = 1 / (baseFreq^(2i/embedDim)) for i in [0, embedDim/2)
	// Shape: [embedDim/2]
	halfDim := embedDim / 2
	dimIndices := Iota(g, shapes.Make(dtype, halfDim), 0)
	dimIndices = MulScalar(dimIndices, 2.0/float64(embedDim))

	// Create base frequency tensor and compute freqs = 1 / (baseFreq^dimIndices)
	baseFreqTensor := Const(g, []float64{baseFreq})
	baseFreqTensor = ConvertDType(baseFreqTensor, dtype)
	freqs := Pow(baseFreqTensor, dimIndices)
	freqs = Reciprocal(freqs)

	// Compute angles: outer product of positions and frequencies
	// Shape: [seqLen, embedDim/2]
	positions = ExpandDims(positions, -1) // [seqLen, 1]
	freqs = ExpandDims(freqs, 0)          // [1, embedDim/2]
	angles := Mul(positions, freqs)       // [seqLen, embedDim/2]

	// Compute cos and sin of angles
	cosAngles := Cos(angles) // [seqLen, embedDim/2]
	sinAngles := Sin(angles) // [seqLen, embedDim/2]

	// Split input into first half and second half along the embedding dimension
	// x1: [..., seqLen, embedDim/2] - even indices (first half)
	// x2: [..., seqLen, embedDim/2] - odd indices (second half)
	// Build slice specification for all axes
	sliceSpec := make([]SliceAxisSpec, rank)
	for i := 0; i < rank-1; i++ {
		sliceSpec[i] = AxisRange()
	}
	sliceSpec[rank-1] = AxisRange(0, halfDim)
	x1 := Slice(x, sliceSpec...)

	sliceSpec[rank-1] = AxisRange(halfDim, embedDim)
	x2 := Slice(x, sliceSpec...)

	// Broadcast cos and sin to match input shape
	// We need to expand them to match all leading dimensions of x
	for range rank - 2 {
		cosAngles = ExpandDims(cosAngles, 0)
		sinAngles = ExpandDims(sinAngles, 0)
	}
	// Broadcast to actual shape
	targetShape := shape.Clone()
	targetShape.Dimensions[rank-1] = halfDim
	cosAngles = BroadcastToShape(cosAngles, targetShape)
	sinAngles = BroadcastToShape(sinAngles, targetShape)

	// Apply rotation:
	// rotated_x1 = x1 * cos - x2 * sin
	// rotated_x2 = x1 * sin + x2 * cos
	rotatedX1 := Sub(Mul(x1, cosAngles), Mul(x2, sinAngles))
	rotatedX2 := Add(Mul(x1, sinAngles), Mul(x2, cosAngles))

	// Concatenate back together
	return Concatenate([]*Node{rotatedX1, rotatedX2}, -1)
}

// applyRoPEWithCustomDim applies RoPE to x[..., dimStart:dimEnd] (even length).
// Unchanged dimensions outside [dimStart:dimEnd] are preserved.
//
// Parameters:
//   - x: Input tensor shaped [..., seq_len, embed_dim]
//   - positionIndices: Position indices shaped [..., seq_len]
//   - baseFreq: Base frequency for computing rotation angles
//   - dimStart: Start index of the dimension range (inclusive)
//   - dimEnd: End index of the dimension range (exclusive)
//
// Returns:
//   - Tensor with rotary embeddings applied to the specified range, same shape as x
func applyRoPEWithCustomDim(x *Node, positionIndices *Node, baseFreq float64, dimStart, dimEnd int) *Node {
	rank := x.Shape().Rank()
	// Extract the part to apply RoPE (slice the last axis)
	part := Slice(x, AxisRange().Spacer(), AxisRange(dimStart, dimEnd))

	// Apply RoPE
	rotatedPart := applyRoPE(part, positionIndices, baseFreq)

	// Concatenate with unchanged parts
	if dimStart == 0 && dimEnd == x.Shape().Dimensions[rank-1] {
		// RoPE applied to entire embedding
		return rotatedPart
	}

	parts := make([]*Node, 0, 3)
	if dimStart > 0 {
		parts = append(parts, Slice(x, AxisRange().Spacer(), AxisRange(0, dimStart)))
	}
	parts = append(parts, rotatedPart)
	if dimEnd < x.Shape().Dimensions[rank-1] {
		parts = append(parts, Slice(x, AxisRange().Spacer(), AxisRange(dimEnd, x.Shape().Dimensions[rank-1])))
	}

	return Concatenate(parts, -1)
}
