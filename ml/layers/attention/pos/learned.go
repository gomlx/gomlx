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
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/nn"
)

// Learned implements a learned positional embedding.
// It applies learned absolute positional information to the initial sequence tensor,
// usually just after the token embedding table lookup.
//
// It implements both Encoder and PreEncoder interfaces.
type Learned struct {
	scope           *model.Scope
	maxPosEmbedding int
	embedDim        int
}

// Ensure Learned implements Encoder and PreEncoder.
var _ Encoder = &Learned{}
var _ PreEncoder = &Learned{}

// NewLearned creates a Learned positional embedding.
// It takes as input the scope, the maximum positional embedding, and the embedding dimension.
func NewLearned(scope *model.Scope, maxPosEmbedding, embedDim int) *Learned {
	return &Learned{
		scope:           scope,
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

	// Ensure positionIndices has an integer dtype for Gather.
	if !positionIndices.DType().IsInt() {
		positionIndices = ConvertDType(positionIndices, dtypes.Int64)
	}
	gatherIndices := ExpandDims(positionIndices, -1)

	// Variable with all the positional embeddings: created if it doesn't exist yet.
	posScope := l.scope.In("pos_embed")
	var posEmbed *Node
	if existing := posScope.GetVariable("embeddings"); existing != nil {
		if existing.Quantization() != nil {
			posEmbed = nn.QuantizedGather(existing.NodeValue(g), gatherIndices, existing.Quantization().ToGraph(g))
		} else {
			posEmbed = Gather(existing.NodeValue(g), gatherIndices)
		}
		if posEmbed.DType() != x.DType() {
			posEmbed = ConvertDType(posEmbed, x.DType())
		}
	} else {
		posEmbedFull := posScope.VariableWithShape("embeddings",
			shapes.Make(x.DType(), l.maxPosEmbedding, l.embedDim)).NodeValue(g)
		posEmbed = Gather(posEmbedFull, gatherIndices)
	}

	// If posEmbed is missing batch dimensions (e.g., posEmbed is [seqLen, embedDim] while x is [batchSize, seqLen, embedDim]),
	// we explicitly broadcast it to match x's shape, similar to what is done with BroadcastPrefix.
	if posEmbed.Rank() < x.Rank() {
		posEmbed = BroadcastLike(posEmbed, x)
	}

	// Add the embeddings to the current value of x.
	return Add(x, posEmbed)
}
