// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/ctxtest"
	"github.com/gomlx/gomlx/pkg/ml/context/initializers"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMultiHeadAttentionGraph(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	{
		ctx := context.New()
		g := NewGraph(backend, "test")
		batchSize := 3
		key := IotaFull(g, shapes.Make(dtypes.Float32, batchSize, 4, 5, 3))
		query := IotaFull(g, shapes.Make(dtypes.Float32, batchSize, 7, 1, 2))
		value := IotaFull(g, shapes.Make(dtypes.Float32, batchSize, 4, 5, 10))
		attOutput, attCoef := MultiHeadAttention(ctx, query, key, value, 6, 12).
			SetOutputDim(11).DoneWithCoefficients()
		assert.EqualValues(t, []int{batchSize, 7, 1, 11}, attOutput.Shape().Dimensions, "AttentionOutput shape mismatch")
		assert.EqualValues(t, []int{batchSize, 7, 1, 6, 4, 5}, attCoef.Shape().Dimensions, "AttentionCoefficients shape mismatch")
	}

	// Higher-rank with key mask: verifies that masks are correctly flattened alongside Q/K/V
	// when inner axes are > 1, and that the graph executes without errors.
	{
		ctx := context.New()
		batchSize := 2
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, input *Node) (*Node, *Node) {
			g := input.Graph()
			key := IotaFull(g, shapes.Make(dtypes.Float32, batchSize, 3, 4, 5))
			query := IotaFull(g, shapes.Make(dtypes.Float32, batchSize, 6, 1, 3))
			value := IotaFull(g, shapes.Make(dtypes.Float32, batchSize, 3, 4, 7))
			// keyMask shape: [batch, 3, 4] — one mask per key inner element, same for all heads.
			keyMask := Const(g, true)
			keyMask = BroadcastToDims(keyMask, batchSize, 3, 4)
			return MultiHeadAttention(ctx, query, key, value, 2, 8).
				SetKeyMask(keyMask).
				SetOutputDim(9).DoneWithCoefficients()
		})
		// Pass a dummy input to trigger execution.
		results := exec.MustExec(tensors.FromScalar(float32(0)))
		assert.EqualValues(t, []int{batchSize, 6, 1, 9}, results[0].Shape().Dimensions, "Higher-rank masked output shape")
		assert.EqualValues(t, []int{batchSize, 6, 1, 2, 3, 4}, results[1].Shape().Dimensions, "Higher-rank masked coef shape")
	}

	ctxtest.RunTestGraphFn(t, "MultiHeadAttention with masking",
		func(ctx *context.Context, g *Graph) (inputs, outputs []*Node) {
			batchSize := 2
			key := IotaFull(g, shapes.Make(dtypes.Float32, batchSize, 3, 3))
			query := IotaFull(g, shapes.Make(dtypes.Float32, batchSize, 3, 2))
			value := OnePlus(IotaFull(g, shapes.Make(dtypes.Float32, batchSize, 3, 2)))
			attOutput, attCoef := MultiHeadAttention(ctx.WithInitializer(initializers.One),
				query, key, value, 1, 2).
				UseCausalMask().
				SetOutputDim(5).DoneWithCoefficients()
			inputs = []*Node{key, query, value}
			outputs = []*Node{attOutput, attCoef}
			return
		}, []any{
			[][][]float32{
				{{9, 9, 9, 9, 9}, {17, 17, 17, 17, 17}, {25, 25, 25, 25, 25}},
				{{33, 33, 33, 33, 33}, {41, 41, 41, 41, 41}, {49, 49, 49, 49, 49}},
			},
			[][][][]float32{
				// Attention should be mostly (99.9999...%) on the right-most of the valid options,
				// which in the case of causal mask (a lower-triangular matrix) will be on the diagonal.
				{{{1, 0, 0}}, {{0, 1, 0}}, {{0, 0, 1}}},
				{{{1, 0, 0}}, {{0, 1, 0}}, {{0, 0, 1}}},
			},
		}, xslices.Epsilon)
}

// TestMultiHeadAttentionFusedPath verifies that the fused SDPA fast path
// produces the same results as the decomposed path.
func TestMultiHeadAttentionFusedPath(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	// Test that Done() (which may use fused path) matches DoneWithCoefficients()
	// (which always uses decomposed path) with the same weights.
	for _, useCausal := range []bool{false, true} {
		name := "no_causal"
		if useCausal {
			name = "causal"
		}
		t.Run(name, func(t *testing.T) {
			batchSize := 2
			seqLen := 3
			inputDim := 4
			numHeads := 2
			headDim := 2

			// Build two graphs with the same context (shared weights):
			// one using DoneWithCoefficients (decomposed), one using Done (fused).
			ctx := context.New().WithInitializer(initializers.One)

			// Decomposed path.
			decomposedExec := context.MustNewExec(backend, ctx, func(ctx *context.Context, g *Graph) []*Node {
				input := IotaFull(g, shapes.Make(dtypes.Float32, batchSize, seqLen, inputDim))
				builder := MultiHeadAttention(ctx, input, input, input, numHeads, headDim)
				if useCausal {
					builder = builder.UseCausalMask()
				}
				output, _ := builder.DoneWithCoefficients()
				return []*Node{output}
			})
			decomposedOutputs := decomposedExec.MustExec()

			// Fused path (Done() will use fused when available).
			fusedExec := context.MustNewExec(backend, ctx.Reuse(), func(ctx *context.Context, g *Graph) []*Node {
				input := IotaFull(g, shapes.Make(dtypes.Float32, batchSize, seqLen, inputDim))
				builder := MultiHeadAttention(ctx, input, input, input, numHeads, headDim)
				if useCausal {
					builder = builder.UseCausalMask()
				}
				output := builder.Done()
				return []*Node{output}
			})
			fusedOutputs := fusedExec.MustExec()

			// Compare outputs.
			decomposedVal := decomposedOutputs[0].Value()
			fusedVal := fusedOutputs[0].Value()

			require.Truef(t, xslices.SlicesInDelta(decomposedVal, fusedVal, 1e-4),
				"Fused and decomposed paths produce different results.\nDecomposed: %v\nFused: %v",
				decomposedVal, fusedVal)
		})
	}
}

func TestMultiHeadAttentionWithRoPE(t *testing.T) {
	// Verify MHA with RoPE runs without error and produces correct output shape.
	// This test catches the seq axis bug where RoPE was applied on the heads axis
	// instead of the seq axis for BSHD layout.
	backend := graphtest.BuildTestBackend()
	ctx := context.New()

	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x *Node) *Node {
		return SelfAttention(ctx, x, 2, 4).
			WithRoPE(10000.0).
			UseCausalMask().
			Done()
	})

	// [batch=1, seq=4, embed=8]
	input := [][][]float32{{{1, 2, 3, 4, 5, 6, 7, 8},
		{9, 10, 11, 12, 13, 14, 15, 16},
		{17, 18, 19, 20, 21, 22, 23, 24},
		{25, 26, 27, 28, 29, 30, 31, 32}}}

	output := exec.MustExec(input)[0]
	assert.Equal(t, []int{1, 4, 8}, output.Shape().Dimensions)

	// Run a second time with different sequence length to verify re-compilation
	// with the same weights (reuse ctx so Dense parameters are shared).
	exec2 := context.MustNewExec(backend, ctx.Reuse(), func(ctx *context.Context, x *Node) *Node {
		return SelfAttention(ctx, x, 2, 4).
			WithRoPE(10000.0).
			UseCausalMask().
			Done()
	})

	input2 := [][][]float32{{{1, 2, 3, 4, 5, 6, 7, 8},
		{9, 10, 11, 12, 13, 14, 15, 16}}}

	output2 := exec2.MustExec(input2)[0]
	assert.Equal(t, []int{1, 2, 8}, output2.Shape().Dimensions)
}

func TestMultiHeadAttentionWithQKVProjection(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("basic", func(t *testing.T) {
		ctx := context.New()
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x *Node) *Node {
			return SelfAttention(ctx, x, 2, 4).
				UseQKVProjection().
				Done()
		})

		// [batch=2, seq=3, embed=8]
		input := [][][]float32{
			{{1, 2, 3, 4, 5, 6, 7, 8}, {9, 10, 11, 12, 13, 14, 15, 16}, {17, 18, 19, 20, 21, 22, 23, 24}},
			{{25, 26, 27, 28, 29, 30, 31, 32}, {33, 34, 35, 36, 37, 38, 39, 40}, {41, 42, 43, 44, 45, 46, 47, 48}},
		}
		output := exec.MustExec(input)[0]
		assert.Equal(t, []int{2, 3, 8}, output.Shape().Dimensions)
	})

	t.Run("with_causal_mask", func(t *testing.T) {
		ctx := context.New()
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x *Node) *Node {
			return SelfAttention(ctx, x, 2, 4).
				UseQKVProjection().
				UseCausalMask().
				Done()
		})

		input := [][][]float32{
			{{1, 2, 3, 4, 5, 6, 7, 8}, {9, 10, 11, 12, 13, 14, 15, 16}, {17, 18, 19, 20, 21, 22, 23, 24}},
		}
		output := exec.MustExec(input)[0]
		assert.Equal(t, []int{1, 3, 8}, output.Shape().Dimensions)
	})

	t.Run("with_coefficients", func(t *testing.T) {
		ctx := context.New()
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x *Node) []*Node {
			output, coef := SelfAttention(ctx, x, 2, 4).
				UseQKVProjection().
				DoneWithCoefficients()
			return []*Node{output, coef}
		})

		input := [][][]float32{
			{{1, 2, 3, 4, 5, 6, 7, 8}, {9, 10, 11, 12, 13, 14, 15, 16}, {17, 18, 19, 20, 21, 22, 23, 24}},
		}
		outputs := exec.MustExec(input)
		assert.Equal(t, []int{1, 3, 8}, outputs[0].Shape().Dimensions)
		// coefficients: [batch, query_seq, num_heads, key_seq]
		assert.Equal(t, []int{1, 3, 2, 3}, outputs[1].Shape().Dimensions)
	})

	t.Run("no_output_bias", func(t *testing.T) {
		// UseProjectionBias(false) disables only the output projection bias;
		// QKV biases are always present (matching the separate Dense path).
		ctx := context.New()
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x *Node) *Node {
			return SelfAttention(ctx, x, 2, 4).
				UseQKVProjection().
				UseProjectionBias(false).
				Done()
		})

		input := [][][]float32{
			{{1, 2, 3, 4, 5, 6, 7, 8}, {9, 10, 11, 12, 13, 14, 15, 16}, {17, 18, 19, 20, 21, 22, 23, 24}},
		}
		output := exec.MustExec(input)[0]
		assert.Equal(t, []int{1, 3, 8}, output.Shape().Dimensions)
	})
}

func TestMultiHeadAttentionGQA(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("basic", func(t *testing.T) {
		// GQA with 4 query heads, 2 KV heads (2:1 ratio).
		ctx := context.New()
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x *Node) *Node {
			return MultiHeadAttention(ctx, x, x, x, 4, 8).
				SetNumKVHeads(2).
				Done()
		})

		// [batch=2, seq=3, embed=16]
		input := make([][][]float32, 2)
		for b := range input {
			input[b] = make([][]float32, 3)
			for s := range input[b] {
				input[b][s] = make([]float32, 16)
				for d := range input[b][s] {
					input[b][s][d] = float32(b*100+s*10+d) * 0.1
				}
			}
		}
		output := exec.MustExec(input)[0]
		// Output shape: [batch=2, seq=3, embed=16] (outputDim defaults to inputValueDim).
		assert.Equal(t, []int{2, 3, 16}, output.Shape().Dimensions)
	})

	t.Run("MQA", func(t *testing.T) {
		// Multi-Query Attention: 4 query heads, 1 KV head.
		ctx := context.New()
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x *Node) *Node {
			return SelfAttention(ctx, x, 4, 8).
				SetNumKVHeads(1).
				Done()
		})

		input := [][][]float32{
			{{1, 2, 3, 4, 5, 6, 7, 8}, {9, 10, 11, 12, 13, 14, 15, 16}},
		}
		output := exec.MustExec(input)[0]
		assert.Equal(t, []int{1, 2, 8}, output.Shape().Dimensions)
	})

	t.Run("fused_vs_decomposed", func(t *testing.T) {
		// Verify fused (Done) and decomposed (DoneWithCoefficients) paths agree for GQA.
		ctx := context.New().WithInitializer(initializers.One)
		batchSize := 2
		seqLen := 3
		inputDim := 8
		numHeads := 4
		numKVHeads := 2
		headDim := 4

		decomposedExec := context.MustNewExec(backend, ctx, func(ctx *context.Context, g *Graph) []*Node {
			input := IotaFull(g, shapes.Make(dtypes.Float32, batchSize, seqLen, inputDim))
			output, _ := MultiHeadAttention(ctx, input, input, input, numHeads, headDim).
				SetNumKVHeads(numKVHeads).
				DoneWithCoefficients()
			return []*Node{output}
		})
		decomposedOutputs := decomposedExec.MustExec()

		fusedExec := context.MustNewExec(backend, ctx.Reuse(), func(ctx *context.Context, g *Graph) []*Node {
			input := IotaFull(g, shapes.Make(dtypes.Float32, batchSize, seqLen, inputDim))
			output := MultiHeadAttention(ctx, input, input, input, numHeads, headDim).
				SetNumKVHeads(numKVHeads).
				Done()
			return []*Node{output}
		})
		fusedOutputs := fusedExec.MustExec()

		require.Truef(t, xslices.SlicesInDelta(decomposedOutputs[0].Value(), fusedOutputs[0].Value(), 1e-4),
			"Fused and decomposed paths produce different results for GQA.\nDecomposed: %v\nFused: %v",
			decomposedOutputs[0].Value(), fusedOutputs[0].Value())
	})

	t.Run("with_causal_mask", func(t *testing.T) {
		ctx := context.New()
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x *Node) *Node {
			return SelfAttention(ctx, x, 4, 4).
				SetNumKVHeads(2).
				UseCausalMask().
				Done()
		})

		input := [][][]float32{
			{{1, 2, 3, 4, 5, 6, 7, 8}, {9, 10, 11, 12, 13, 14, 15, 16},
				{17, 18, 19, 20, 21, 22, 23, 24}},
		}
		output := exec.MustExec(input)[0]
		assert.Equal(t, []int{1, 3, 8}, output.Shape().Dimensions)
	})

	t.Run("with_rope", func(t *testing.T) {
		ctx := context.New()
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x *Node) *Node {
			return SelfAttention(ctx, x, 4, 4).
				SetNumKVHeads(2).
				WithRoPE(10000.0).
				UseCausalMask().
				Done()
		})

		input := [][][]float32{
			{{1, 2, 3, 4, 5, 6, 7, 8}, {9, 10, 11, 12, 13, 14, 15, 16},
				{17, 18, 19, 20, 21, 22, 23, 24}, {25, 26, 27, 28, 29, 30, 31, 32}},
		}
		output := exec.MustExec(input)[0]
		assert.Equal(t, []int{1, 4, 8}, output.Shape().Dimensions)
	})

	t.Run("coefficients_shape", func(t *testing.T) {
		// Verify coefficient shape uses numHeads (not numKVHeads).
		ctx := context.New()
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x *Node) []*Node {
			output, coef := SelfAttention(ctx, x, 4, 4).
				SetNumKVHeads(2).
				DoneWithCoefficients()
			return []*Node{output, coef}
		})

		input := [][][]float32{
			{{1, 2, 3, 4, 5, 6, 7, 8}, {9, 10, 11, 12, 13, 14, 15, 16},
				{17, 18, 19, 20, 21, 22, 23, 24}},
		}
		outputs := exec.MustExec(input)
		assert.Equal(t, []int{1, 3, 8}, outputs[0].Shape().Dimensions)
		// Coefficients: [batch, seq, numHeads, seq] — numHeads=4, not numKVHeads=2.
		assert.Equal(t, []int{1, 3, 4, 3}, outputs[1].Shape().Dimensions)
	})
}

// buildSyntheticAttentionModelFn builds a model graph building function that does a regression on the elements
// of a sequence, with a learnable positional embedding.
//
// If debug==true allows to control printing out of intermediary results.
func buildSyntheticAttentionModelFn(debug bool) (modelGraphFn func(ctx *context.Context, spec any, inputs []*Node) []*Node) {
	return func(ctx *context.Context, spec any, inputs []*Node) (allLogits []*Node) {
		_ = spec
		input := inputs[0] // shape=[batch, sequence]
		g := input.Graph()
		g.SetTraced(true)

		dtype := input.DType()
		input = InsertAxes(input, -1) // shape=[batch, sequence, 1]
		batchSize := input.Shape().Dimensions[0]
		sequenceSize := input.Shape().Dimensions[1]
		const positionalEmbeddingSize = 16
		noisyCtx := ctx.WithInitializer(initializers.RandomNormalFn(ctx, 1.0))
		positionalVar := noisyCtx.In("positional").VariableWithShape("embeddings", shapes.Make(dtype, sequenceSize, positionalEmbeddingSize))
		positionalEmbedding := positionalVar.ValueGraph(g)
		positionalEmbedding = InsertAxes(positionalEmbedding, 0) // Prefixing with batch dimension.
		dims := positionalEmbedding.Shape().Clone().Dimensions
		dims[0] = batchSize
		positionalEmbedding = BroadcastToDims(positionalEmbedding, dims...)
		logits := Concatenate([]*Node{input, positionalEmbedding}, -1) // Shape=[batch, sequence, 1+positionalEmbeddingSize]
		if debug {
			logits.SetLogged("Input+Positional")
		}
		var coef *Node
		logits, coef = MultiHeadAttention(ctx.In("attention"), logits, logits, logits, 4, 8).DoneWithCoefficients()
		if debug {
			coef.SetLogged("Attention Coefficients")
		}
		residual := logits
		logits = Sigmoid(logits)
		logits = layers.Dense(ctx.In("dense_seq_1"), logits, true, logits.Shape().Dimensions[2])
		logits = Add(residual, logits)
		logits = Sigmoid(logits)
		logits = layers.Dense(ctx.In("dense_seq_0"), logits, true, 1)
		logits = Squeeze(logits, -1)
		allLogits = []*Node{logits}
		return
	}
}

type attentionTestDataset struct {
	name                    string
	batchSize, sequenceSize int
	infinite                bool
	count, maxCount         int
}

func (ds *attentionTestDataset) Name() string {
	return ds.name
}

func (ds *attentionTestDataset) Reset() {
	ds.count = 0
}

func (ds *attentionTestDataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	if !ds.infinite && ds.count+ds.batchSize > ds.maxCount {
		return nil, nil, nil, io.EOF
	}
	ds.count += ds.batchSize

	batch := make([][]float32, ds.batchSize)
	batchLabel := make([][]float32, ds.batchSize)
	for ii := 0; ii < ds.batchSize; ii++ {
		batch[ii] = make([]float32, ds.sequenceSize)
		//batchLabel[ii] = make([]float32, 1)
		batchLabel[ii] = make([]float32, ds.sequenceSize)
		for jj := 0; jj < ds.sequenceSize; jj++ {
			batch[ii][jj] = float32(rand.Intn(2))
			if batch[ii][jj] > 0 && jj > 0 && batch[ii][jj-1] > 0 {
				batchLabel[ii][jj] = 1.0
			}
		}
	}
	inputs = []*tensors.Tensor{tensors.FromValue(batch)}
	labels = []*tensors.Tensor{tensors.FromValue(batchLabel)}
	//fmt.Printf("inputs: %v\n", batch)
	//fmt.Printf("labels: %v\n", labels)
	return
}

// TestMultiHeadAttentionTraining creates a test dataset which to be solved one needs to be able to attend to
// the left/right. The label is the logical-and of the value and the value to the left in a sequence. See
// attentionTestDataset above.
//
// It first learns the model, and then it prints out some example results.
func TestMultiHeadAttentionTraining(t *testing.T) {
	trainDS := &attentionTestDataset{
		name:         "trainDS",
		batchSize:    50,
		sequenceSize: 16,
		infinite:     true,
	}

	// Backend handles creation of ML computation graphs, accelerator resources, etc.
	backend := graphtest.BuildTestBackend()

	// Context and optimizer used for training.
	ctx := context.New()
	opt := optimizers.Adam().LearningRate(0.001).Done()

	trainer := train.NewTrainer(backend, ctx, buildSyntheticAttentionModelFn(false),
		losses.MeanSquaredError,
		opt,
		nil, // trainMetrics
		nil) // evalMetrics
	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.
	metrics, err := loop.RunSteps(trainDS, 1500)
	loss := metrics[1].Value().(float32)
	assert.Truef(t, loss < 0.12, "Expected a loss < 0.12, got %g instead", loss)
	require.NoErrorf(t, err, "Failed training: %+v", err)
	fmt.Printf("Metrics:\n")
	for ii, m := range metrics {
		fmt.Printf("\t%s: %s\n", trainer.TrainMetrics()[ii].Name(), m)
	}

	{
		// Print a sample:
		evalDS := &attentionTestDataset{}
		*evalDS = *trainDS
		evalDS.batchSize = 1
		var results []*tensors.Tensor

		modelFn := buildSyntheticAttentionModelFn(false)
		inferenceFn := func(ctx *context.Context, inputs []*Node) *Node {
			return modelFn(ctx, nil, inputs)[0]
		}
		inferenceExec := context.MustNewExec(backend, ctx.Reuse(), inferenceFn)
		for range 3 {
			_, inputs, labels, err := evalDS.Yield()
			require.NoErrorf(t, err, "Failed datasets: %+v", err)
			fmt.Printf("\nInput:\t%v\n", inputs[0].Value())
			fmt.Printf("Label:\t%v\n", labels[0].Value())
			results = inferenceExec.MustExec(inputs[0])
			tmp := results[0].Value().([][]float32)[0]
			var rounded []int
			for _, v := range tmp {
				rounded = append(rounded, int(math.Round(float64(v))))
			}
			fmt.Printf("Pred:\t[%v]\n", rounded)
		}
	}
}
