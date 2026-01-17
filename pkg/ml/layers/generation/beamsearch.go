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

package generation

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// BeamSearchConfig: beam search config.
type BeamSearchConfig struct {
	beamSize      int
	lengthPenalty float64
	eosTokenId    int
	maxLength     int
	minLength     int
	numReturnSeqs int
	earlyStopping bool
}

func NewBeamSearch(beamSize, eosTokenId int) *BeamSearchConfig {
	return &BeamSearchConfig{
		beamSize:      beamSize,
		lengthPenalty: 1.0,
		eosTokenId:    eosTokenId,
		maxLength:     100,
		minLength:     0,
		numReturnSeqs: 1,
		earlyStopping: true,
	}
}

// WithLengthPenalty.
func (c *BeamSearchConfig) WithLengthPenalty(penalty float64) *BeamSearchConfig {
	c.lengthPenalty = penalty
	return c
}

// WithMaxLength.
func (c *BeamSearchConfig) WithMaxLength(maxLen int) *BeamSearchConfig {
	c.maxLength = maxLen
	return c
}

// WithMinLength.
func (c *BeamSearchConfig) WithMinLength(minLen int) *BeamSearchConfig {
	c.minLength = minLen
	return c
}

// WithNumReturnSequences.
func (c *BeamSearchConfig) WithNumReturnSequences(num int) *BeamSearchConfig {
	c.numReturnSeqs = num
	return c
}

// WithEarlyStopping.
func (c *BeamSearchConfig) WithEarlyStopping(early bool) *BeamSearchConfig {
	c.earlyStopping = early
	return c
}

// BeamSearchStep: one decoding step.
// Inputs: logits [batch*beam, vocab], sequences [batch*beam, len], scores [batch*beam].
// Outputs: next sequences [batch*beam, len+1], next scores [batch*beam], finished [batch*beam].
func BeamSearchStep(
	logits *Node,
	currentSequences *Node,
	beamScores *Node,
	config *BeamSearchConfig,
	currentLength int,
) (nextSequences, nextBeamScores, isFinished *Node) {
	g := logits.Graph()
	vocabSize := logits.Shape().Dimensions[1]
	batchBeamSize := logits.Shape().Dimensions[0]

	logProbs := LogSoftmax(logits)

	// Suppress EOS below min length
	if currentLength < config.minLength {
		// EOS -> very negative
		eosIdx := config.eosTokenId
		// Mask EOS position
		vocabIndices := Iota(g, logProbs.Shape(), 1)
		eosMask := Equal(vocabIndices, ConstAs(vocabIndices, eosIdx))
		logProbs = Where(eosMask, ConstAs(logProbs, -1e10), logProbs)
	}

	// Add beam scores
	// beamScores: [batch_beam] -> [batch_beam, 1]
	beamScoresExpanded := ExpandDims(beamScores, -1)
	// Broadcast: [batch_beam, vocab]
	beamScoresExpanded = BroadcastToShape(beamScoresExpanded, logProbs.Shape())

	// Combined: [batch_beam, vocab]
	scores := Add(beamScoresExpanded, logProbs)

	// Reshape: [batch, beam*vocab] for TopK across candidates
	batchSize := batchBeamSize / config.beamSize
	scores = Reshape(scores, batchSize, config.beamSize*vocabSize)
	// TopK scores/indices: [batch, beam]
	topKScores, topKIndices := TopK(scores, config.beamSize, -1)

	// beamIdx = topKIndices // vocab_size
	beamIdx := ConvertDType(topKIndices, dtypes.Float32)
	beamIdx = DivScalar(beamIdx, float64(vocabSize))
	beamIdx = ConvertDType(beamIdx, topKIndices.DType())
	// tokenIdx = topKIndices % vocab_size
	tokenIdx := ConvertDType(topKIndices, dtypes.Float32)
	tokenIdx = ModScalar(tokenIdx, float64(vocabSize))
	tokenIdx = ConvertDType(tokenIdx, topKIndices.DType())

	nextBeamScores = Reshape(topKScores, batchBeamSize)
	beamIdx = Reshape(beamIdx, batchBeamSize)
	tokenIdx = Reshape(tokenIdx, batchBeamSize)

	batchIndices := Iota(g, shapes.Make(beamIdx.DType(), batchBeamSize), 0)
	batchIndices = ConvertDType(batchIndices, dtypes.Float32)
	batchIndices = DivScalar(batchIndices, float64(config.beamSize))
	batchIndices = Floor(batchIndices)
	batchIndices = MulScalar(batchIndices, float64(config.beamSize))
	batchIndices = ConvertDType(batchIndices, beamIdx.DType())

	gatherIndices := Add(batchIndices, beamIdx)
	gatherIndices = ConvertDType(gatherIndices, dtypes.Int32)
	gatherIndices = ExpandDims(gatherIndices, -1)

	selectedSequences := Gather(currentSequences, gatherIndices)
	tokenIdx = ExpandDims(tokenIdx, -1)
	nextSequences = Concatenate([]*Node{selectedSequences, tokenIdx}, -1)

	isFinished = Equal(tokenIdx, ConstAs(tokenIdx, config.eosTokenId))
	isFinished = Squeeze(isFinished, -1)

	return nextSequences, nextBeamScores, isFinished
}

func SelectBestSequences(
	sequences *Node,
	scores *Node,
	config *BeamSearchConfig,
) (bestSequences, bestScores *Node) {
	batchBeamSize := sequences.Shape().Dimensions[0]
	_ = sequences.Shape().Dimensions[1] // seqLen - preserved by Gather
	batchSize := batchBeamSize / config.beamSize
	scores = Reshape(scores, batchSize, config.beamSize)
	topScores, topIndices := TopK(scores, config.numReturnSeqs, -1)
	bestScores = Reshape(topScores, batchSize*config.numReturnSeqs)

	g := sequences.Graph()
	batchIndices := Iota(g, shapes.Make(topIndices.DType(), batchSize, config.numReturnSeqs), 0)
	batchOffsets := MulScalar(batchIndices, float64(config.beamSize))
	gatherIndices := Add(batchOffsets, topIndices)
	gatherIndices = Reshape(gatherIndices, batchSize*config.numReturnSeqs, 1)
	// Gather sequences
	bestSequences = Gather(sequences, gatherIndices)

	return bestSequences, bestScores
}

// ApplyLengthPenalty: divide scores by (length^penalty).
func ApplyLengthPenalty(scores *Node, lengths *Node, penalty float64) *Node {
	if penalty == 1.0 {
		return scores
	}

	// score / (length ^ penalty)
	lengthsFloat := ConvertDType(lengths, scores.DType())
	penaltyNode := ConstAs(lengthsFloat, penalty)
	lengthPenalty := Pow(lengthsFloat, penaltyNode)

	// Divide scores by length penalty
	return Div(scores, lengthPenalty)
}
