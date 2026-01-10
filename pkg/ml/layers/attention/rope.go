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

package attention

import (
	. "github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// RoPE applies rotary position embeddings to the last dimension (even length).
// Shapes: x [..., seq_len, head_dim|embed_dim]. Reference: RoFormer.
func RoPE(x *Node, startPos int, baseFreq float64) *Node {
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

	// Compute position indices: [startPos, startPos+1, ..., startPos+seqLen-1]
	// Shape: [seqLen]
	positions := Iota(g, shapes.Make(dtype, seqLen), 0)
	positions = AddScalar(positions, float64(startPos))

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
	for ii := 0; ii < rank-2; ii++ {
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

// RoPEWithCustomDim applies RoPE to x[..., dimStart:dimEnd] (even length).
func RoPEWithCustomDim(x *Node, startPos int, baseFreq float64, dimStart, dimEnd int) *Node {
	rank := x.Shape().Rank()

	// Extract the part to apply RoPE
	part := Slice(x, AxisRange(), AxisRange(dimStart, dimEnd))

	// Apply RoPE
	rotatedPart := RoPE(part, startPos, baseFreq)

	// Concatenate with unchanged parts
	if dimStart == 0 && dimEnd == x.Shape().Dimensions[rank-1] {
		// RoPE applied to entire embedding
		return rotatedPart
	}

	parts := make([]*Node, 0, 3)
	if dimStart > 0 {
		parts = append(parts, Slice(x, AxisRange(), AxisRange(0, dimStart)))
	}
	parts = append(parts, rotatedPart)
	if dimEnd < x.Shape().Dimensions[rank-1] {
		parts = append(parts, Slice(x, AxisRange(), AxisRange(dimEnd, x.Shape().Dimensions[rank-1])))
	}

	return Concatenate(parts, -1)
}
