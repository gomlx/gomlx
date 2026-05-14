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
	"fmt"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

// Learned implements a learned positional embedding.
// It applies learned absolute positional information to the initial sequence tensor,
// usually just after the token embedding table lookup.
//
// It implements both Encoder and PreEncoder interfaces.
type Learned struct {
	ctx             *context.Context
	maxPosEmbedding int
	embedDim        int
}

// Ensure Learned implements Encoder and PreEncoder.
var _ Encoder = &Learned{}
var _ PreEncoder = &Learned{}

// NewLearned creates a Learned positional embedding.
// It takes as input the context, the maximum positional embedding, and the embedding dimension.
func NewLearned(ctx *context.Context, maxPosEmbedding, embedDim int) *Learned {
	return &Learned{
		ctx:             ctx,
		maxPosEmbedding: maxPosEmbedding,
		embedDim:        embedDim,
	}
}

// Name implements the Encoder interface.
func (l *Learned) Name() string {
	return fmt.Sprintf("Learned(maxPosEmbedding=%d, embedDim=%d)", l.maxPosEmbedding, l.embedDim)
}

// PreEncode implements the PreEncoder interface by applying learned positional embeddings
// to the initial sequence tensor based on positionIndices.
func (l *Learned) PreEncode(x, positionIndices *Node, seqAxis int) *Node {
	g := x.Graph()

	// Variable with all the positional embeddings: created if it doesn't exist yet.
	posEmbedFull := l.ctx.In("pos_embed").VariableWithShape("embeddings",
		shapes.Make(x.DType(), l.maxPosEmbedding, l.embedDim)).ValueGraph(g)

	// Ensure positionIndices has an integer dtype for Gather.
	if !positionIndices.DType().IsInt() {
		positionIndices = ConvertDType(positionIndices, dtypes.Int64)
	}

	// Gather the corresponding positional embeddings for the given positions
	// positionIndices is shaped [[batchSize...,] seqLen], ExpandDims makes it [[batchSize...,] seqLen, 1]
	// Gathering from [maxPosEmbedding, embedDim] variable yields [[batchSize...,] seqLen, embedDim]
	posEmbed := Gather(posEmbedFull, ExpandDims(positionIndices, -1))

	// If posEmbed is missing batch dimensions (e.g., posEmbed is [seqLen, embedDim] while x is [batchSize, seqLen, embedDim]),
	// we explicitly broadcast it to match x's shape, similar to what is done with BroadcastPrefix.
	if posEmbed.Rank() < x.Rank() {
		batchSize := x.Shape().Dimensions[0]
		posEmbed = BroadcastPrefix(posEmbed, batchSize)
	}

	// Add the embeddings to the current value of x.
	return Add(x, posEmbed)
}
