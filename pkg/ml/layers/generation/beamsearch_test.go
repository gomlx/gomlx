package generation

import (
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	graphtest "github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/assert"
)

// TestBeamSearchConfig groups tests for NewBeamSearch and its builders.
func TestBeamSearchConfig(t *testing.T) {
	t.Run("Defaults", func(t *testing.T) {
		config := NewBeamSearch(4, 2)
		assert.Equal(t, 4, config.beamSize)
		assert.Equal(t, 2, config.eosTokenId)
		assert.Equal(t, 1.0, config.lengthPenalty)
		assert.Equal(t, 100, config.maxLength)
		assert.Equal(t, 0, config.minLength)
		assert.Equal(t, 1, config.numReturnSeqs)
		assert.True(t, config.earlyStopping)
	})

	t.Run("Builders", func(t *testing.T) {
		config := NewBeamSearch(4, 2).
			WithLengthPenalty(1.2).
			WithMaxLength(50).
			WithMinLength(5).
			WithNumReturnSequences(2).
			WithEarlyStopping(false)

		assert.Equal(t, 1.2, config.lengthPenalty)
		assert.Equal(t, 50, config.maxLength)
		assert.Equal(t, 5, config.minLength)
		assert.Equal(t, 2, config.numReturnSeqs)
		assert.False(t, config.earlyStopping)
	})
}

// TestBeamSearchStep groups tests for BeamSearchStep.
func TestBeamSearchStep(t *testing.T) {
	t.Run("Basic", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		config := NewBeamSearch(2, 999)
		batchSize := 1
		beamSize := 2
		currentLength := 3

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, logits, sequences, scores *Node) []*Node {
			nextSeqs, nextScores, isFinished := BeamSearchStep(logits, sequences, scores, config, currentLength)
			return []*Node{nextSeqs, nextScores, isFinished}
		})

		logitsData := [][]float32{
			{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
			{1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1},
		}
		logits := tensors.FromValue(logitsData)
		sequences := tensors.FromValue([][]int32{{1, 2, 3}, {1, 2, 4}})
		beamScores := tensors.FromValue([]float32{0.0, -0.5})

		results := exec.MustExec(logits, sequences, beamScores)

		nextSeqs := results[0]
		nextScores := results[1]
		isFinished := results[2]

		assert.Equal(t, 2, nextSeqs.Rank())
		assert.Equal(t, batchSize*beamSize, nextSeqs.Shape().Dimensions[0])
		assert.Equal(t, currentLength+1, nextSeqs.Shape().Dimensions[1])

		assert.Equal(t, 1, nextScores.Rank())
		assert.Equal(t, batchSize*beamSize, nextScores.Shape().Dimensions[0])

		assert.Equal(t, 1, isFinished.Rank())
		assert.Equal(t, batchSize*beamSize, isFinished.Shape().Dimensions[0])
	})

	t.Run("EOSSuppression", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		eosToken := 5
		config := NewBeamSearch(2, eosToken).WithMinLength(10)
		currentLength := 3

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, logits, sequences, scores *Node) []*Node {
			nextSeqs, nextScores, isFinished := BeamSearchStep(logits, sequences, scores, config, currentLength)
			return []*Node{nextSeqs, nextScores, isFinished}
		})

		logits := tensors.FromValue([][]float32{
			{0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0},
			{0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0},
		})
		sequences := tensors.FromValue([][]int32{{1, 2, 3}, {1, 2, 4}})
		beamScores := tensors.FromValue([]float32{0.0, 0.0})

		results := exec.MustExec(logits, sequences, beamScores)
		isFinished := results[2]
		assert.Equal(t, 1, isFinished.Rank())

		nextSeqs := results[0]
		selected := nextSeqs.Value().([][]int32)
		for i := range selected {
			last := selected[i][len(selected[i])-1]
			assert.NotEqual(t, int32(eosToken), last)
		}
	})

	t.Run("FinishedDetection", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		eosToken := 3
		config := NewBeamSearch(2, eosToken).WithMinLength(0)
		currentLength := 5

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, logits, sequences, scores *Node) []*Node {
			nextSeqs, nextScores, isFinished := BeamSearchStep(logits, sequences, scores, config, currentLength)
			return []*Node{nextSeqs, nextScores, isFinished}
		})

		logits := tensors.FromValue([][]float32{
			{0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			{0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		})
		sequences := tensors.FromValue([][]int32{{1, 2, 3, 4, 5}, {1, 2, 4, 5, 6}})
		beamScores := tensors.FromValue([]float32{0.0, 0.0})

		results := exec.MustExec(logits, sequences, beamScores)
		nextSeqs := results[0]
		isFinished := results[2]
		assert.Equal(t, 6, nextSeqs.Shape().Dimensions[1])
		assert.Equal(t, 1, isFinished.Rank())
		assert.Equal(t, 2, isFinished.Shape().Dimensions[0])
	})

	t.Run("MultipleBatches", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		config := NewBeamSearch(2, 999)
		batchSize := 3
		beamSize := 2
		currentLength := 2

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, logits, sequences, scores *Node) []*Node {
			nextSeqs, nextScores, isFinished := BeamSearchStep(logits, sequences, scores, config, currentLength)
			return []*Node{nextSeqs, nextScores, isFinished}
		})

		logitsData := make([][]float32, batchSize*beamSize)
		for i := 0; i < batchSize*beamSize; i++ {
			logitsData[i] = []float32{0.1, 0.2, 0.3, 0.4, 0.5}
		}
		logits := tensors.FromValue(logitsData)

		seqsData := make([][]int32, batchSize*beamSize)
		for i := 0; i < batchSize*beamSize; i++ {
			seqsData[i] = []int32{int32(i), int32(i + 1)}
		}
		sequences := tensors.FromValue(seqsData)

		scoresData := make([]float32, batchSize*beamSize)
		for i := 0; i < batchSize*beamSize; i++ {
			scoresData[i] = float32(i) * -0.1
		}
		beamScores := tensors.FromValue(scoresData)

		results := exec.MustExec(logits, sequences, beamScores)

		nextSeqs := results[0]
		nextScores := results[1]
		isFinished := results[2]

		assert.Equal(t, 2, nextSeqs.Rank())
		assert.Equal(t, batchSize*beamSize, nextSeqs.Shape().Dimensions[0])
		assert.Equal(t, currentLength+1, nextSeqs.Shape().Dimensions[1])
		assert.Equal(t, 1, nextScores.Rank())
		assert.Equal(t, batchSize*beamSize, nextScores.Shape().Dimensions[0])
		assert.Equal(t, 1, isFinished.Rank())
		assert.Equal(t, batchSize*beamSize, isFinished.Shape().Dimensions[0])
	})
}

// TestApplyLengthPenalty groups tests for ApplyLengthPenalty.
func TestApplyLengthPenalty(t *testing.T) {
	t.Run("NoPenalty", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, scores, lengths *Node) *Node {
			return ApplyLengthPenalty(scores, lengths, 1.0)
		})
		scores := tensors.FromValue([]float32{-1.0, -2.0, -3.0, -4.0})
		lengths := tensors.FromValue([]int32{5, 10, 15, 20})
		result := exec.MustExec(scores, lengths)[0]
		got := result.Value().([]float32)
		want := []float32{-1.0, -2.0, -3.0, -4.0}
		for i := range want {
			assert.InDelta(t, want[i], got[i], 0.001)
		}
	})

	t.Run("WithPenalty", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		penalty := 1.5
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, scores, lengths *Node) *Node {
			return ApplyLengthPenalty(scores, lengths, penalty)
		})
		scores := tensors.FromValue([]float32{-10.0, -20.0})
		lengths := tensors.FromValue([]int32{4, 8})
		result := exec.MustExec(scores, lengths)[0]
		got := result.Value().([]float32)
		expected0 := (-10.0) / 8.0
		expected1 := (-20.0) / 22.627417
		assert.InDelta(t, expected0, got[0], 0.001)
		assert.InDelta(t, expected1, got[1], 0.001)
	})
}
