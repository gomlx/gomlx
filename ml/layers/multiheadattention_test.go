/*
 *	Copyright 2023 Jan Pfeifer
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

package layers

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"testing"

	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/ctxtest"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xslices"
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
		key := IotaFull(g, shapes.Make(F32, batchSize, 4, 5, 3))
		query := IotaFull(g, shapes.Make(F32, batchSize, 7, 1, 2))
		value := IotaFull(g, shapes.Make(F32, batchSize, 4, 5, 10))
		attOutput, attCoef := MultiHeadAttention(ctx, query, key, value, 6, 12).
			SetOutputDim(11).DoneWithCoefficients()
		assert.EqualValues(t, []int{batchSize, 7, 1, 11}, attOutput.Shape().Dimensions, "AttentionOutput shape mismatch")
		assert.EqualValues(t, []int{batchSize, 7, 1, 6, 4, 5}, attCoef.Shape().Dimensions, "AttentionCoefficients shape mismatch")
	}

	ctxtest.RunTestGraphFn(t, "MultiHeadAttention with masking",
		func(ctx *context.Context, g *Graph) (inputs, outputs []*Node) {
			batchSize := 2
			key := IotaFull(g, shapes.Make(F32, batchSize, 3, 3))
			query := IotaFull(g, shapes.Make(F32, batchSize, 3, 2))
			value := OnePlus(IotaFull(g, shapes.Make(F32, batchSize, 3, 2)))
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
		logits = Dense(ctx.In("dense_seq_1"), logits, true, logits.Shape().Dimensions[2])
		logits = Add(residual, logits)
		logits = Sigmoid(logits)
		logits = Dense(ctx.In("dense_seq_0"), logits, true, 1)
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
		inferenceExec := context.NewExec(backend, ctx.Reuse(), inferenceFn)
		for ii := 0; ii < 3; ii++ {
			_, inputs, labels, err := evalDS.Yield()
			require.NoErrorf(t, err, "Failed datasets: %+v", err)
			fmt.Printf("\nInput:\t%v\n", inputs[0].Value())
			fmt.Printf("Label:\t%v\n", labels[0].Value())
			results = inferenceExec.Call(inputs[0])
			tmp := results[0].Value().([][]float32)[0]
			var rounded []int
			for _, v := range tmp {
				rounded = append(rounded, int(math.Round(float64(v))))
			}
			fmt.Printf("Pred:\t[%v]\n", rounded)
		}
	}
}
