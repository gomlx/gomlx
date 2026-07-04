package generate

import (
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/compute/shapes"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors/bucketing"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/zoo/transformer/generate/sample"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestGenerator groups config default and builder tests.
func TestGenerator(t *testing.T) {
	t.Run("Defaults", func(t *testing.T) {
		var modelFn NaiveModelFn = func(scope *model.Scope, tokens *Node, seqLen *Node) *Node { return tokens }
		cfg := New(modelFn)
		assert.Equal(t, 100, cfg.MaxLength)
		assert.Equal(t, sample.StrategyGreedy, cfg.Strategy)
		assert.Equal(t, float32(1.0), cfg.Temperature)
		assert.Equal(t, 50, cfg.TopK)
		assert.Equal(t, float32(0.9), cfg.TopP)
		assert.Equal(t, 4, cfg.BeamSize)
		assert.Equal(t, -1, cfg.EosTokenId)
		assert.False(t, cfg.StopOnEOS)
	})

	t.Run("Builders", func(t *testing.T) {
		var modelFn NaiveModelFn = func(scope *model.Scope, tokens *Node, seqLen *Node) *Node { return tokens }
		cfg := New(modelFn).
			WithMaxLength(50).
			WithTemperature(0.7).
			WithTopK(40).
			WithTopP(0.95).
			WithBeamSize(8).
			WithStrategy(sample.StrategyTemperature).
			WithEOS(2)
		assert.Equal(t, 50, cfg.MaxLength)
		assert.Equal(t, sample.StrategyTemperature, cfg.Strategy)
		assert.Equal(t, float32(0.7), cfg.Temperature)
		assert.Equal(t, 40, cfg.TopK)
		assert.Equal(t, float32(0.95), cfg.TopP)
		assert.Equal(t, 8, cfg.BeamSize)
		assert.Equal(t, 2, cfg.EosTokenId)
		assert.True(t, cfg.StopOnEOS)
	})
}

// TestGeneratorSampling groups non-beam sampling tests.
func TestGeneratorSampling(t *testing.T) {
	t.Run("Greedy", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		store := model.NewStore()
		var modelFn NaiveModelFn = func(scope *model.Scope, tokens *Node, seqLen *Node) *Node {
			vocabSize := 10
			g := tokens.Graph()
			vocabIota := Iota(g, shapes.Make(dtypes.Float32, vocabSize), 0)
			vocabIota = ExpandDims(vocabIota, 0)
			vocabIota = ExpandDims(vocabIota, 0)
			targetDims := []int{tokens.Shape().Dimensions[0], tokens.Shape().Dimensions[1], vocabSize}
			targetAxes := []string{tokens.Shape().AxisName(0), tokens.Shape().AxisName(1), ""}
			targetShape := shapes.MakeDynamic(dtypes.Float32, targetDims, targetAxes)
			logits := BroadcastToShape(vocabIota, targetShape)
			indices := Iota(g, logits.Shape(), 2)
			logits = Where(Equal(indices, ConstAs(indices, 5)), ConstAs(logits, 100.0), logits)
			return logits
		}
		cfg := New(modelFn).WithStrategy(sample.StrategyGreedy).WithMaxLength(10)
		prompt := [][]int32{{1, 2, 3}}
		result, err := cfg.Decode(backend, store.RootScope(), prompt)
		require.NoError(t, err)
		require.NotNil(t, result)
		// Expect full sequence [batch=1, length=10]
		assert.Equal(t, 2, result.Rank())
		assert.Equal(t, []int{1, 10}, result.Shape().Dimensions)
		seq := result.Value().([][]int32)
		// Last generated token should be 5 (greedy)
		assert.Equal(t, int32(5), seq[0][len(seq[0])-1])
	})

	t.Run("Temperature", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		store := model.NewStore()
		var modelFn NaiveModelFn = func(scope *model.Scope, tokens *Node, seqLen *Node) *Node {
			vocabSize := 10
			g := tokens.Graph()
			vocabIota := Iota(g, shapes.Make(dtypes.Float32, vocabSize), 0)
			vocabIota = ExpandDims(vocabIota, 0)
			vocabIota = ExpandDims(vocabIota, 0)
			targetDims := []int{tokens.Shape().Dimensions[0], tokens.Shape().Dimensions[1], vocabSize}
			targetAxes := []string{tokens.Shape().AxisName(0), tokens.Shape().AxisName(1), ""}
			targetShape := shapes.MakeDynamic(dtypes.Float32, targetDims, targetAxes)
			return BroadcastToShape(vocabIota, targetShape)
		}
		cfg := New(modelFn).WithStrategy(sample.StrategyTemperature).WithTemperature(1.5).WithMaxLength(10)
		prompt := [][]int32{{1, 2, 3}}
		result, err := cfg.Decode(backend, store.RootScope(), prompt)
		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Equal(t, 2, result.Rank())
		assert.Equal(t, []int{1, 10}, result.Shape().Dimensions)
		seq := result.Value().([][]int32)
		last := seq[0][len(seq[0])-1]
		assert.GreaterOrEqual(t, last, int32(0))
		assert.Less(t, last, int32(10))
	})

	t.Run("OneDPrompt", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		store := model.NewStore()
		var modelFn NaiveModelFn = func(scope *model.Scope, tokens *Node, seqLen *Node) *Node {
			vocabSize := 10
			g := tokens.Graph()
			vocabIota := Iota(g, shapes.Make(dtypes.Float32, vocabSize), 0)
			vocabIota = ExpandDims(vocabIota, 0)
			vocabIota = ExpandDims(vocabIota, 0)
			targetDims := []int{tokens.Shape().Dimensions[0], tokens.Shape().Dimensions[1], vocabSize}
			targetAxes := []string{tokens.Shape().AxisName(0), tokens.Shape().AxisName(1), ""}
			targetShape := shapes.MakeDynamic(dtypes.Float32, targetDims, targetAxes)
			return BroadcastToShape(vocabIota, targetShape)
		}
		cfg := New(modelFn).WithStrategy(sample.StrategyGreedy).WithMaxLength(10)
		prompt := []int32{1, 2, 3}
		result, err := cfg.Decode(backend, store.RootScope(), prompt)
		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Equal(t, 2, result.Rank())
		assert.Equal(t, []int{1, 10}, result.Shape().Dimensions)
	})

	t.Run("PromptTooLong", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		store := model.NewStore()
		var modelFn NaiveModelFn = func(scope *model.Scope, tokens *Node, seqLen *Node) *Node { return tokens }
		cfg := New(modelFn).WithMaxLength(5)
		prompt := [][]int32{{1, 2, 3, 4, 5, 6, 7, 8}}
		_, err := cfg.Decode(backend, store.RootScope(), prompt)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "prompt length")
	})
}

// TestGenerateBeamSearchNotImplemented remains as a placeholder expectation.
func TestBeamSearch(t *testing.T) {
	backend := testutil.BuildTestBackend()
	store := model.NewStore()
	var modelFn NaiveModelFn = func(scope *model.Scope, tokens *Node, seqLen *Node) *Node {
		vocabSize := 10
		g := tokens.Graph()
		vocabIota := Iota(g, shapes.Make(dtypes.Float32, vocabSize), 0)
		vocabIota = ExpandDims(vocabIota, 0)
		vocabIota = ExpandDims(vocabIota, 0)
		targetDims := []int{tokens.Shape().Dimensions[0], tokens.Shape().Dimensions[1], vocabSize}
		targetAxes := []string{tokens.Shape().AxisName(0), tokens.Shape().AxisName(1), ""}
		targetShape := shapes.MakeDynamic(dtypes.Float32, targetDims, targetAxes)
		return BroadcastToShape(vocabIota, targetShape)
	}
	cfg := New(modelFn).WithStrategy(sample.StrategyBeamSearch).WithBeamSize(4).WithMaxLength(10)
	prompt := [][]int32{{1, 2, 3}}
	result, err := cfg.Decode(backend, store.RootScope(), prompt)
	require.NoError(t, err)
	require.NotNil(t, result)
	// Expect best sequences: shape [batch, seq_len]
	assert.Equal(t, 2, result.Rank())
	assert.Equal(t, []int{1, 10}, result.Shape().Dimensions)
}

// TestStreaming verifies the functional GenerateStreaming method.
func TestStreaming(t *testing.T) {
	backend := testutil.BuildTestBackend()
	store := model.NewStore()
	var modelFn NaiveModelFn = func(scope *model.Scope, tokens *Node, seqLen *Node) *Node {
		vocabSize := 10
		g := tokens.Graph()
		vocabIota := Iota(g, shapes.Make(dtypes.Float32, vocabSize), 0)
		vocabIota = ExpandDims(vocabIota, 0)
		vocabIota = ExpandDims(vocabIota, 0)
		targetDims := []int{tokens.Shape().Dimensions[0], tokens.Shape().Dimensions[1], vocabSize}
		targetAxes := []string{tokens.Shape().AxisName(0), tokens.Shape().AxisName(1), ""}
		targetShape := shapes.MakeDynamic(dtypes.Float32, targetDims, targetAxes)
		return BroadcastToShape(vocabIota, targetShape)
	}
	cfg := New(modelFn).WithMaxLength(10)
	prompt := []int32{1, 2, 3}
	var generated []int
	err := cfg.GenerateStreaming(backend, store.RootScope(), prompt, func(token int) bool {
		generated = append(generated, token)
		return true
	})
	require.NoError(t, err)
	assert.NotEmpty(t, generated)
}

func TestDynamicShapesAndBucketing(t *testing.T) {
	backend := testutil.BuildTestBackend()
	if backend.Name() != "go" {
		goBackend, err := gobackend.New("test-go")
		if err == nil {
			backend = goBackend
		}
	}

	t.Run("DynamicShapesWithNoneStrategy", func(t *testing.T) {
		if !backend.Capabilities().DynamicAxes {
			t.Skipf("Backend %q does not support DynamicAxes", backend.Name())
		}

		store := model.NewStore()
		var modelFn NaiveModelFn = func(scope *model.Scope, tokens *Node, seqLen *Node) *Node {
			vocabSize := 10
			g := tokens.Graph()
			vocabIota := Iota(g, shapes.Make(dtypes.Float32, vocabSize), 0)
			vocabIota = ExpandDims(vocabIota, 0)
			vocabIota = ExpandDims(vocabIota, 0)
			
			targetDims := []int{tokens.Shape().Dimensions[0], tokens.Shape().Dimensions[1], vocabSize}
			targetAxes := []string{tokens.Shape().AxisName(0), tokens.Shape().AxisName(1), ""}
			targetShape := shapes.MakeDynamic(dtypes.Float32, targetDims, targetAxes)
			logits := BroadcastToShape(vocabIota, targetShape)
			return logits
		}

		cfg := New(modelFn).WithStrategy(sample.StrategyGreedy).WithMaxLength(10).WithBucketingStrategy(bucketing.None())
		prompt := [][]int32{{1, 2, 3}}
		result, err := cfg.Decode(backend, store.RootScope(), prompt)
		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Equal(t, []int{1, 10}, result.Shape().Dimensions)

		assert.NotNil(t, cfg.naiveExec)
	})

	t.Run("BucketingPow2Strategy", func(t *testing.T) {
		store := model.NewStore()
		var modelFn NaiveModelFn = func(scope *model.Scope, tokens *Node, seqLen *Node) *Node {
			vocabSize := 10
			g := tokens.Graph()
			vocabIota := Iota(g, shapes.Make(dtypes.Float32, vocabSize), 0)
			vocabIota = ExpandDims(vocabIota, 0)
			vocabIota = ExpandDims(vocabIota, 0)
			
			targetDims := []int{tokens.Shape().Dimensions[0], tokens.Shape().Dimensions[1], vocabSize}
			targetAxes := []string{tokens.Shape().AxisName(0), tokens.Shape().AxisName(1), ""}
			targetShape := shapes.MakeDynamic(dtypes.Float32, targetDims, targetAxes)
			logits := BroadcastToShape(vocabIota, targetShape)
			return logits
		}

		cfg := New(modelFn).WithStrategy(sample.StrategyGreedy).WithMaxLength(10).WithBucketingStrategy(bucketing.Pow2()).WithPadToken(0)
		prompt := [][]int32{{1, 2, 3}}
		result, err := cfg.Decode(backend, store.RootScope(), prompt)
		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Equal(t, []int{1, 10}, result.Shape().Dimensions)

		assert.NotNil(t, cfg.naiveExec)
	})
}
