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

package sample

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// BeamSearchConfig configures beam search decoding for autoregressive generation.
// Beam search explores multiple candidate sequences in parallel (beams), selecting
// the most likely sequences at each step. This typically produces higher quality
// outputs than greedy decoding but requires more computation.
//
// Usage:
//   - beamSize: Number of parallel beams (2-10 is common, higher = better quality but slower)
//   - lengthPenalty: Adjusts preference for longer sequences (>1.0 favors longer, <1.0 favors shorter)
//   - eosTokenId: Token ID that marks end-of-sequence
//   - numReturnSeqs: How many sequences to return (typically 1, must be <= beamSize)
type BeamSearchConfig struct {
	beamSize      int
	lengthPenalty float64
	eosTokenId    int
	maxLength     int
	minLength     int
	numReturnSeqs int
	earlyStopping bool
}

// NewBeamSearch creates a beam search configuration with default settings.
//
// Parameters:
//   - beamSize: Number of parallel beams to maintain (typically 2-10)
//   - eosTokenId: Token ID marking end-of-sequence (-1 to disable EOS handling)
//
// Returns a BeamSearchConfig with defaults:
//   - lengthPenalty: 1.0 (no penalty)
//   - maxLength: 100
//   - minLength: 0
//   - numReturnSeqs: 1 (return only the best sequence)
//   - earlyStopping: true (stop when all beams hit EOS)
//
// Example:
//
//	beamCfg := sample.NewBeamSearch(4, eosTokenId).
//	    WithLengthPenalty(0.9).     // slightly favor shorter sequences
//	    WithMaxLength(150).
//	    WithMinLength(10).          // require at least 10 tokens
//	    WithNumReturnSequences(3)   // return top 3 sequences
//
// Then one would use beamCfg.Step() in a loop to generate tokens.
// See decode.Decoder.Decode() for a higher-level API.
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

// WithLengthPenalty sets the length penalty factor.
// Values > 1.0 favor longer sequences, values < 1.0 favor shorter sequences.
// The penalty is applied as: score / (length ^ lengthPenalty).
// Default is 1.0 (no penalty).
func (c *BeamSearchConfig) WithLengthPenalty(penalty float64) *BeamSearchConfig {
	c.lengthPenalty = penalty
	return c
}

// WithMaxLength sets the maximum generation length.
// Generation stops when this length is reached.
// Default is 100.
func (c *BeamSearchConfig) WithMaxLength(maxLen int) *BeamSearchConfig {
	c.maxLength = maxLen
	return c
}

// WithMinLength sets the minimum generation length.
// EOS tokens are suppressed until this length is reached.
// Default is 0 (no minimum).
func (c *BeamSearchConfig) WithMinLength(minLen int) *BeamSearchConfig {
	c.minLength = minLen
	return c
}

// WithNumReturnSequences sets how many sequences to return.
// Must be <= beamSize. The top numReturnSeqs sequences by score are returned.
// Default is 1 (return only the best sequence).
func (c *BeamSearchConfig) WithNumReturnSequences(num int) *BeamSearchConfig {
	c.numReturnSeqs = num
	return c
}

// WithEarlyStopping controls whether to stop when all beams have hit EOS.
// If true (default), stops as soon as all beams produce EOS tokens.
// If false, continues until maxLength is reached.
func (c *BeamSearchConfig) WithEarlyStopping(early bool) *BeamSearchConfig {
	c.earlyStopping = early
	return c
}

// EarlyStopping returns whether early stopping is enabled.
func (c *BeamSearchConfig) EarlyStopping() bool {
	return c.earlyStopping
}

// Step performs one decoding step of beam search.
// This is a low-level method for implementing beam search generation.
// It takes current beam states and model logits, then selects the top-k
// candidates for the next step based on cumulative scores.
//
// The method:
//  1. Converts logits to log probabilities
//  2. Suppresses EOS tokens if below minLength
//  3. Adds current beam scores to get cumulative scores
//  4. Selects top beamSize candidates across all beams
//  5. Updates sequences with new tokens and tracks finished beams
//
// Parameters:
//   - logits: Model output logits [batch*beam, vocab_size]
//   - currentSequences: Current token sequences [batch*beam, current_length]
//   - beamScores: Cumulative log probability scores [batch*beam]
//   - currentLength: Current sequence length (for minLength enforcement)
//
// Returns:
//   - nextSequences: Updated sequences [batch*beam, current_length+1]
//   - nextBeamScores: Updated cumulative scores [batch*beam]
//   - isFinished: Boolean mask indicating which beams hit EOS [batch*beam]
//
// Note: This method is typically called in a loop by higher-level generation code.
// Most users should use decode.Decoder.Decode() with strategy="beam_search" instead.
func (c *BeamSearchConfig) Step(
	logits *Node,
	currentSequences *Node,
	beamScores *Node,
	currentLength int,
) (nextSequences, nextBeamScores, isFinished *Node) {
	g := logits.Graph()
	vocabSize := logits.Shape().Dimensions[1]
	batchBeamSize := logits.Shape().Dimensions[0]

	logProbs := LogSoftmax(logits)

	// Suppress EOS below min length
	if currentLength < c.minLength {
		// EOS -> very negative
		eosIdx := c.eosTokenId
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
	batchSize := batchBeamSize / c.beamSize
	scores = Reshape(scores, batchSize, c.beamSize*vocabSize)
	// TopK scores/indices: [batch, beam]
	topKScores, topKIndices := TopK(scores, c.beamSize, -1)

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
	batchIndices = DivScalar(batchIndices, float64(c.beamSize))
	batchIndices = Floor(batchIndices)
	batchIndices = MulScalar(batchIndices, float64(c.beamSize))
	batchIndices = ConvertDType(batchIndices, beamIdx.DType())

	gatherIndices := Add(batchIndices, beamIdx)
	gatherIndices = ConvertDType(gatherIndices, dtypes.Int32)
	gatherIndices = ExpandDims(gatherIndices, -1)

	selectedSequences := Gather(currentSequences, gatherIndices)
	tokenIdx = ExpandDims(tokenIdx, -1)
	nextSequences = Concatenate([]*Node{selectedSequences, tokenIdx}, -1)

	isFinished = Equal(tokenIdx, ConstAs(tokenIdx, c.eosTokenId))
	isFinished = Squeeze(isFinished, -1)

	return nextSequences, nextBeamScores, isFinished
}

// SelectBest selects the top scoring sequences from beam search results.
// After beam search completes, this method extracts the best numReturnSeqs
// sequences from each batch based on their cumulative scores.
//
// Parameters:
//   - sequences: All beam sequences [batch*beam, seq_length]
//   - scores: Cumulative scores for each beam [batch*beam]
//
// Returns:
//   - bestSequences: Top sequences [batch*numReturnSeqs, seq_length]
//   - bestScores: Corresponding scores [batch*numReturnSeqs]
//
// Example: If beamSize=4 and numReturnSeqs=1, this returns only the single
// best sequence from each batch.
func (c *BeamSearchConfig) SelectBest(
	sequences *Node,
	scores *Node,
) (bestSequences, bestScores *Node) {
	batchBeamSize := sequences.Shape().Dimensions[0]
	_ = sequences.Shape().Dimensions[1] // seqLen - preserved by Gather
	batchSize := batchBeamSize / c.beamSize
	scores = Reshape(scores, batchSize, c.beamSize)
	topScores, topIndices := TopK(scores, c.numReturnSeqs, -1)
	bestScores = Reshape(topScores, batchSize*c.numReturnSeqs)

	g := sequences.Graph()
	batchIndices := Iota(g, shapes.Make(topIndices.DType(), batchSize, c.numReturnSeqs), 0)
	batchOffsets := MulScalar(batchIndices, float64(c.beamSize))
	gatherIndices := Add(batchOffsets, topIndices)
	gatherIndices = Reshape(gatherIndices, batchSize*c.numReturnSeqs, 1)
	// Gather sequences
	bestSequences = Gather(sequences, gatherIndices)

	return bestSequences, bestScores
}

// applyLengthPenalty applies length normalization to beam search scores.
// Without length penalty, beam search tends to favor shorter sequences since
// they accumulate fewer negative log probabilities. This method normalizes
// scores by sequence length raised to the configured penalty factor.
//
// Formula: adjusted_score = score / (length ^ lengthPenalty)
//
// Note: This is called automatically by SelectBest when lengthPenalty != 1.0.
func (c *BeamSearchConfig) applyLengthPenalty(scores *Node, lengths *Node) *Node {
	if c.lengthPenalty == 1.0 {
		return scores
	}

	// score / (length ^ penalty)
	lengthsFloat := ConvertDType(lengths, scores.DType())
	penaltyNode := ConstAs(lengthsFloat, c.lengthPenalty)
	lengthPenalty := Pow(lengthsFloat, penaltyNode)

	// Divide scores by length penalty
	return Div(scores, lengthPenalty)
}
