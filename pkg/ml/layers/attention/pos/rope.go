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
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	. "github.com/gomlx/gomlx/pkg/support/exceptions"
)

// RoPE implements rotary position embeddings ("RoFormer", see [1]).
// It splits the embedding (aka. features) axis into pairs, and rotates
// them at different frequencies, according to position.
//
// RoPE implements the Encoder interface and can be used with MultiHeadAttention.WithPositionEncoder().
//
// [1] "RoFormer: Enhanced Transformer with Rotary Position Embedding", https://arxiv.org/abs/2104.09864
type RoPE struct {
	BaseFreq    float64
	DimStart    int
	DimEnd      int
	Interleaved bool // If true, rotation pairs are at even/odd indices; if false, split first-half/second-half
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

// WithInterleaved sets whether rotation pairs are at interleaved indices.
// If true, pairs are at even/odd indices (x[..., 0], x[..., 1]).
// If false (default), pairs are split first-half/second-half (x[..., :dim/2], x[..., dim/2:]).
//
// Returns the modified RoPE for method chaining.
func (r *RoPE) WithInterleaved(interleaved bool) *RoPE {
	r.Interleaved = interleaved
	return r
}

// Apply implements the Encoder interface.
// It applies rotary position embeddings to x using the provided position indices.
// The rotation is applied on a range of frequencies multiplied by the position
// of each element, as specified by positionIndices.
//
// Parameters:
//   - x: Input tensor with sequence and embedding dimensions.
//   - positionIndices: Position indices shaped [..., seq_len] with position values for each token.
//   - seqAxis: The axis in x that represents the sequence dimension.
//
// Returns:
//   - Tensor with rotary position embeddings applied, same shape as x
//
// Example:
//
//	rope := NewRoPE(10000.0)
//	// x has shape [batch, seq_len, embed_dim]
//	// positions has shape [batch, seq_len] with values like [0, 1, 2, ...]
//	embedded := rope.Apply(x, positions, x.Rank()-2)
func (r *RoPE) Apply(x *Node, positionIndices *Node, seqAxis int) *Node {
	if r.DimStart == 0 && r.DimEnd == -1 {
		return applyRoPE(x, positionIndices, r.BaseFreq, seqAxis, r.Interleaved)
	}
	dimEnd := r.DimEnd
	if dimEnd == -1 {
		dimEnd = x.Shape().Dimensions[x.Shape().Rank()-1]
	}
	return applyRoPEWithCustomDim(x, positionIndices, r.BaseFreq, r.DimStart, dimEnd, seqAxis, r.Interleaved)
}

// RoPEWithCosSin implements the Encoder interface using pre-computed cos and sin tensors.
// This is useful for ONNX inference where cos/sin are provided as cached inputs rather than
// being computed from a base frequency and position indices.
//
// Use NewRoPEWithCosSin to create an instance.
type RoPEWithCosSin struct {
	cos, sin    *Node
	interleaved bool
}

// NewRoPEWithCosSin creates a RoPE encoder using pre-computed cos and sin tensors.
// This is useful for ONNX inference where cos/sin are provided as cached inputs.
//
// Parameters:
//   - cos: Pre-computed cosine values, broadcastable to [..., seq_len, rotary_dim/2]
//   - sin: Pre-computed sine values, same shape as cos
//
// Returns:
//   - A RoPEWithCosSin instance that implements the Encoder interface
//
// Example:
//
//	// For ONNX inference with pre-computed cos/sin:
//	rope := NewRoPEWithCosSin(cosTensor, sinTensor).WithInterleaved(true)
//	embedded := rope.Apply(x, nil, x.Rank()-2) // positionIndices not used
func NewRoPEWithCosSin(cos, sin *Node) *RoPEWithCosSin {
	return &RoPEWithCosSin{
		cos:         cos,
		sin:         sin,
		interleaved: false,
	}
}

// WithInterleaved sets whether rotation pairs are at interleaved indices.
// If true, pairs are at even/odd indices (x[..., 0], x[..., 1]).
// If false (default), pairs are split first-half/second-half (x[..., :dim/2], x[..., dim/2:]).
//
// Returns the modified RoPEWithCosSin for method chaining.
func (r *RoPEWithCosSin) WithInterleaved(interleaved bool) *RoPEWithCosSin {
	r.interleaved = interleaved
	return r
}

// Apply implements the Encoder interface.
// For RoPEWithCosSin, the positionIndices parameter is ignored since cos/sin are pre-computed.
// Pass nil for positionIndices.
//
// Parameters:
//   - x: Input tensor with shape [..., seqLen, ..., headDim], where seqAxis identifies the sequence dimension.
//   - positionIndices: Ignored (pass nil). Position information is already baked into cos/sin.
//   - seqAxis: The axis in x that represents the sequence dimension.
//
// Returns:
//   - Tensor with rotary embeddings applied, same shape as x.
func (r *RoPEWithCosSin) Apply(x *Node, _ *Node, seqAxis int) *Node {
	return applyWithCosSin(x, r.cos, r.sin, seqAxis, r.interleaved)
}

// applyWithCosSin applies rotary position embeddings using pre-computed cos and sin tensors.
// This is the internal implementation used by both RoPE and RoPEWithCosSin.
//
// Parameters:
//   - x: Input tensor shaped [..., seqLen, ..., headDim], where seqAxis identifies the sequence axis.
//   - cos: Pre-computed cosine values, shaped [seqLen, rotary_dim/2].
//   - sin: Pre-computed sine values, same shape as cos.
//   - seqAxis: The axis in x that represents the sequence dimension.
//   - interleaved: If true, rotation pairs are at even/odd indices (x[..., 0], x[..., 1]).
//     If false, pairs are split first-half/second-half (x[..., :dim/2], x[..., dim/2:]).
//
// Returns:
//   - Tensor with rotary embeddings applied, same shape as x.
//     If cos/sin cover fewer dimensions than x's last axis (partial rotation),
//     the remaining dimensions are passed through unchanged.
func applyWithCosSin(x, cos, sin *Node, seqAxis int, interleaved bool) *Node {
	rank := x.Shape().Rank()
	embedDim := x.Shape().Dimensions[rank-1]

	// Determine rotary dimension from cos shape: last dim is rotary_dim/2
	cosRank := cos.Shape().Rank()
	rotaryHalfDim := cos.Shape().Dimensions[cosRank-1]
	rotaryDim := rotaryHalfDim * 2

	// Split into rotatable and pass-through portions if partial rotation
	var xRotate, xPass *Node
	if rotaryDim < embedDim {
		xRotate = Slice(x, AxisRange().Spacer(), AxisRange(0, rotaryDim))
		xPass = Slice(x, AxisRange().Spacer(), AxisRange(rotaryDim, embedDim))
	} else {
		xRotate = x
		xPass = nil
	}

	// Split xRotate into two halves for rotation
	var x1, x2 *Node
	if interleaved {
		// Interleaved: x1 = even indices, x2 = odd indices
		x1 = Slice(xRotate, AxisRange().Spacer(), AxisRange(0, rotaryDim).Stride(2))
		x2 = Slice(xRotate, AxisRange().Spacer(), AxisRange(1, rotaryDim).Stride(2))
	} else {
		// Non-interleaved: split in half
		halfDim := rotaryDim / 2
		x1 = Slice(xRotate, AxisRange().Spacer(), AxisRange(0, halfDim))
		x2 = Slice(xRotate, AxisRange().Spacer(), AxisRange(halfDim, rotaryDim))
	}

	// Broadcast cos/sin to match x1's shape.
	// cos/sin start as [seqLen, rotary_dim/2]. We need to expand them to match x1's full shape
	// which is [..., seqLen, ..., rotary_dim/2] where seqAxis identifies the seq dimension.
	//
	// Strategy: first add leading size-1 axes so that seqAxis aligns, then insert size-1 axes
	// for any dimensions between seqAxis and the last axis, then broadcast.
	x1Shape := x1.Shape()

	// Normalise seqAxis.
	canonicalSeqAxis := seqAxis
	if canonicalSeqAxis < 0 {
		canonicalSeqAxis += rank
	}

	// Step 1: add leading size-1 axes so that the seq dimension of cos/sin sits at canonicalSeqAxis.
	// cos/sin have rank 2: [seqLen, halfDim]. After ExpandLeftToRank(canonicalSeqAxis+2) they become
	// [1, ..., 1, seqLen, halfDim] with seqLen at canonicalSeqAxis.
	cos = ExpandLeftToRank(cos, canonicalSeqAxis+2)
	sin = ExpandLeftToRank(sin, canonicalSeqAxis+2)

	// Step 2: insert size-1 axes for each dimension between seqAxis and the last axis
	// (e.g. the heads axis in BSHD layout).
	for cos.Rank() < x1Shape.Rank() {
		cos = ExpandAxes(cos, -2)
		sin = ExpandAxes(sin, -2)
	}

	cos = BroadcastToShape(cos, x1Shape)
	sin = BroadcastToShape(sin, x1Shape)

	// Apply rotation: rotated_x1 = x1*cos - x2*sin, rotated_x2 = x1*sin + x2*cos
	rotatedX1 := Sub(Mul(x1, cos), Mul(x2, sin))
	rotatedX2 := Add(Mul(x1, sin), Mul(x2, cos))

	// Recombine rotated values
	var xRotated *Node
	if interleaved {
		// Interleave real and imag back together
		// Stack along new axis then reshape to interleave
		stacked := Stack([]*Node{rotatedX1, rotatedX2}, -1)
		stackedDims := stacked.Shape().Dimensions
		// Reshape: merge last two dims to get back rotaryDim
		newDims := make([]int, len(stackedDims)-1)
		copy(newDims, stackedDims[:len(stackedDims)-2])
		newDims[len(newDims)-1] = rotaryDim
		xRotated = Reshape(stacked, newDims...)
	} else {
		// Non-interleaved: concatenate
		xRotated = Concatenate([]*Node{rotatedX1, rotatedX2}, -1)
	}

	// Combine rotated and pass-through portions
	if xPass != nil {
		return Concatenate([]*Node{xRotated, xPass}, -1)
	}
	return xRotated
}

// applyRoPE applies rotary position embeddings to the last dimension (even length).
// Reference: RoFormer (https://arxiv.org/abs/2104.09864).
//
// Parameters:
//   - x: Input tensor shaped [..., seqLen, ..., headDim], where seqAxis identifies the sequence axis.
//     The last dim must be even.
//   - positionIndices: Position indices shaped [..., seq_len]
//   - baseFreq: Base frequency for computing rotation angles
//   - seqAxis: The axis in x that represents the sequence dimension.
//   - interleaved: If true, rotation pairs are at even/odd indices; if false, split first-half/second-half
//
// Returns:
//   - Tensor with rotary embeddings applied, same shape as x
func applyRoPE(x *Node, positionIndices *Node, baseFreq float64, seqAxis int, interleaved bool) *Node {
	g := x.Graph()
	shape := x.Shape()
	dtype := shape.DType
	rank := shape.Rank()

	// Normalise seqAxis.
	canonicalSeqAxis := seqAxis
	if canonicalSeqAxis < 0 {
		canonicalSeqAxis += rank
	}

	// Extract dimensions
	seqLen := shape.Dimensions[canonicalSeqAxis]
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

	return applyWithCosSin(x, cosAngles, sinAngles, seqAxis, interleaved)
}

// applyRoPEWithCustomDim applies RoPE to x[..., dimStart:dimEnd] (even length).
// Unchanged dimensions outside [dimStart:dimEnd] are preserved.
//
// Parameters:
//   - x: Input tensor shaped [..., seqLen, ..., embed_dim], where seqAxis identifies the sequence axis.
//   - positionIndices: Position indices shaped [..., seq_len]
//   - baseFreq: Base frequency for computing rotation angles
//   - dimStart: Start index of the dimension range (inclusive)
//   - dimEnd: End index of the dimension range (exclusive)
//   - seqAxis: The axis in x that represents the sequence dimension.
//   - interleaved: If true, rotation pairs are at even/odd indices; if false, split first-half/second-half
//
// Returns:
//   - Tensor with rotary embeddings applied to the specified range, same shape as x
func applyRoPEWithCustomDim(x *Node, positionIndices *Node, baseFreq float64, dimStart, dimEnd, seqAxis int, interleaved bool) *Node {
	rank := x.Shape().Rank()
	// Extract the part to apply RoPE (slice the last axis)
	part := Slice(x, AxisRange().Spacer(), AxisRange(dimStart, dimEnd))

	// Apply RoPE
	rotatedPart := applyRoPE(part, positionIndices, baseFreq, seqAxis, interleaved)

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
