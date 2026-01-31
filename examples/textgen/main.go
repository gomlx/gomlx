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

// Demo: train a small transformer and generate text.
package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/decode"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/gomlx/gomlx/ui/commandline"
)

// Hyperparameter keys
const (
	ParamVocabSize   = "vocab_size"
	ParamEmbedDim    = "embed_dim"
	ParamNumHeads    = "num_heads"
	ParamHeadDim     = "head_dim"
	ParamNumLayers   = "num_layers"
	ParamSeqLen      = "seq_len"
	ParamBatchSize   = "batch_size"
	ParamMaxPosEmbed = "max_pos_embed"
	ParamTrainSteps  = "train_steps"
	ParamUseCache    = "use_cache"
	ParamStrategy    = "strategy"
	ParamTemperature = "temperature"
	ParamMaxLength   = "max_length"
)

var (
	flagPrompt = flag.String("prompt", "The quick", "Prompt text")
	dtype      = dtypes.Float32
)

func createDefaultContext() *context.Context {
	ctx := context.New()
	ctx.SetParams(map[string]any{
		// Model hyperparameters
		ParamVocabSize:   128,
		ParamEmbedDim:    64,
		ParamNumHeads:    4,
		ParamHeadDim:     16,
		ParamNumLayers:   2,
		ParamSeqLen:      32,
		ParamBatchSize:   4,
		ParamMaxPosEmbed: 256,

		// Training hyperparameters
		ParamTrainSteps:              200,
		optimizers.ParamLearningRate: 0.01,

		// Generation hyperparameters
		ParamUseCache:    false,
		ParamStrategy:    "greedy",
		ParamTemperature: 1.0,
		ParamMaxLength:   50,
	})
	return ctx
}

const trainingText = `The quick brown fox jumps over the lazy dog. ` +
	`The quick brown fox jumps over the lazy dog. ` +
	`The quick brown fox jumps over the lazy dog. ` +
	`The quick brown fox jumps over the lazy dog. ` +
	`Pack my box with five dozen liquor jugs. ` +
	`Pack my box with five dozen liquor jugs. ` +
	`How vexingly quick daft zebras jump! ` +
	`How vexingly quick daft zebras jump! `

type CharTokenizer struct {
	vocabSize int
}

func (t *CharTokenizer) Encode(text string) []int {
	tokens := make([]int, len(text))
	for i, char := range text {
		tokens[i] = int(char)
		if tokens[i] >= t.vocabSize {
			tokens[i] = 32 // Use space for unknown chars
		}
	}
	return tokens
}

func (t *CharTokenizer) Decode(tokens []int) string {
	chars := make([]byte, len(tokens))
	for i, token := range tokens {
		if token >= 0 && token < t.vocabSize {
			chars[i] = byte(token)
		} else {
			chars[i] = ' '
		}
	}
	return string(chars)
}

func simpleTransformerModel(ctx *context.Context, inputs []*Node) []*Node {
	vocabSize := context.GetParamOr(ctx, ParamVocabSize, 128)
	embedDim := context.GetParamOr(ctx, ParamEmbedDim, 64)
	numHeads := context.GetParamOr(ctx, ParamNumHeads, 4)
	headDim := context.GetParamOr(ctx, ParamHeadDim, 16)
	numLayers := context.GetParamOr(ctx, ParamNumLayers, 2)
	maxPosEmbed := context.GetParamOr(ctx, ParamMaxPosEmbed, 256)
	useCache := context.GetParamOr(ctx, ParamUseCache, false)

	tokens := inputs[0]
	g := tokens.Graph()
	currentSeqLen := tokens.Shape().Dimensions[1]

	embedded := layers.Embedding(ctx.In("token_embed"), tokens, dtype, vocabSize, embedDim)

	posEmbedFull := ctx.In("pos_embed").VariableWithShape("embeddings",
		shapes.Make(dtype, maxPosEmbed, embedDim)).ValueGraph(g)

	posEmbed := Slice(posEmbedFull, AxisRange(0, currentSeqLen))
	posEmbed = ExpandDims(posEmbed, 0)
	posEmbed = BroadcastToShape(posEmbed, embedded.Shape())

	x := Add(embedded, posEmbed)

	for layer := 0; layer < numLayers; layer++ {
		layerCtx := ctx.In(fmt.Sprintf("layer_%d", layer))
		residual := x

		// Build attention with optional KV cache based on context parameter
		attnBuilder := attention.MultiHeadAttention(layerCtx.In("attn"), x, x, x, numHeads, headDim).
			UseCausalMask()

		// Check if we should use KVCache
		if useCache {
			if posValue, found := ctx.GetParam("_generation_position"); found {
				if positionNode, ok := posValue.(*Node); ok && positionNode != nil {
					attnBuilder = attnBuilder.WithKVCache(maxPosEmbed, positionNode)
				}
			}
		}

		attn := attnBuilder.Done()
		x = layers.LayerNormalization(layerCtx.In("norm1"), Add(residual, attn), -1).Done()
		residual = x
		ff := layers.Dense(layerCtx.In("ff1"), x, true, embedDim*4)
		ff = Tanh(ff)
		ff = layers.Dense(layerCtx.In("ff2"), ff, true, embedDim)
		x = layers.LayerNormalization(layerCtx.In("norm2"), Add(residual, ff), -1).Done()
	}
	logits := layers.Dense(ctx.In("output"), x, false, vocabSize)

	return []*Node{logits}
}

func createTrainingBatch(ctx *context.Context, text string) (inputs [][]int32, targets [][][]int32) {
	batchSize := context.GetParamOr(ctx, ParamBatchSize, 4)
	seqLen := context.GetParamOr(ctx, ParamSeqLen, 32)
	vocabSize := context.GetParamOr(ctx, ParamVocabSize, 128)

	tokenizer := &CharTokenizer{vocabSize: vocabSize}
	tokens := tokenizer.Encode(text)

	for len(tokens) < batchSize*(seqLen+1) {
		tokens = append(tokens, tokens...)
	}

	inputs = make([][]int32, batchSize)
	targets = make([][][]int32, batchSize)

	for i := 0; i < batchSize; i++ {
		start := (i * seqLen) % (len(tokens) - seqLen - 1)
		inputs[i] = make([]int32, seqLen)
		targets[i] = make([][]int32, seqLen)

		for j := 0; j < seqLen; j++ {
			inputs[i][j] = int32(tokens[start+j])
			targets[i][j] = []int32{int32(tokens[start+j+1])}
		}
	}

	return inputs, targets
}

func trainModel(backend backends.Backend, ctx *context.Context) {
	steps := context.GetParamOr(ctx, ParamTrainSteps, 200)
	learningRate := context.GetParamOr(ctx, optimizers.ParamLearningRate, 0.01)
	batchSize := context.GetParamOr(ctx, ParamBatchSize, 4)
	seqLen := context.GetParamOr(ctx, ParamSeqLen, 32)

	fmt.Printf("\nTraining Model\nSteps: %d  LR: %.4f  Batch: %d  SeqLen: %d\n\n", steps, learningRate, batchSize, seqLen)

	// Simple model function wrapper
	modelFn := func(ctx *context.Context, _ any, inputs []*Node) []*Node {
		return simpleTransformerModel(ctx, inputs)
	}

	trainer := train.NewTrainer(backend, ctx, modelFn,
		losses.SparseCategoricalCrossEntropyLogits,
		optimizers.Adam().Done(),
		nil, nil) // no metrics for this simple example

	for step := 0; step < steps; step++ {
		inputData, targetData := createTrainingBatch(ctx, trainingText)
		inputTensor := tensors.FromValue(inputData)
		targetTensor := tensors.FromValue(targetData)

		metrics, err := trainer.TrainStep(nil, []*tensors.Tensor{inputTensor}, []*tensors.Tensor{targetTensor})
		if err != nil {
			log.Fatalf("Training step %d failed: %v", step, err)
		}
		loss := metrics[0].Value().(float32)

		if step%50 == 0 || step == steps-1 {
			fmt.Printf("Step %4d/%d  Loss: %.4f\n", step, steps, loss)
		}
	}
	fmt.Printf("\nTraining complete!\n")
}

func generateText(backend backends.Backend, ctx *context.Context, prompt string) {
	vocabSize := context.GetParamOr(ctx, ParamVocabSize, 128)
	useCache := context.GetParamOr(ctx, ParamUseCache, false)
	strategy := context.GetParamOr(ctx, ParamStrategy, "greedy")
	temperature := context.GetParamOr(ctx, ParamTemperature, 1.0)
	maxLength := context.GetParamOr(ctx, ParamMaxLength, 50)

	tokenizer := &CharTokenizer{vocabSize: vocabSize}
	fmt.Printf("\nGeneration\nStrategy: %s  Temp: %.2f  MaxLen: %d  Cache: %v\nPrompt: %q\n\n", strategy, temperature, maxLength, useCache, prompt)

	promptTokens := tokenizer.Encode(prompt)
	if len(promptTokens) == 0 {
		promptTokens = []int{32}
	}

	var modelFn decode.ModelFn = func(genCtx *context.Context, tokens *Node) *Node {
		outputs := simpleTransformerModel(genCtx, []*Node{tokens})
		return outputs[0]
	}

	genCfg := decode.New(modelFn).
		WithStrategy(strategy).
		WithTemperature(float32(temperature)).
		WithMaxLength(maxLength)

	promptTensor := tensors.FromValue([][]int32{xslices.Map(promptTokens, func(t int) int32 { return int32(t) })})
	generated, err := genCfg.Decode(backend, ctx, promptTensor)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	generatedTokens := extractTokens(generated)
	text := tokenizer.Decode(generatedTokens)
	fmt.Printf("Generated: %q\n", text)
}

func extractTokens(generated *tensors.Tensor) []int {
	genShape := generated.Shape()
	if genShape.Rank() == 2 {
		genData := generated.Value().([][]int32)
		if len(genData) > 0 {
			tokens := make([]int, len(genData[0]))
			for j := range tokens {
				tokens[j] = int(genData[0][j])
			}
			return tokens
		}
	} else if genShape.Rank() == 1 {
		genData := generated.Value().([]int32)
		tokens := make([]int, len(genData))
		for j := range tokens {
			tokens[j] = int(genData[j])
		}
		return tokens
	}
	return nil
}

func main() {
	ctx := createDefaultContext()
	settings := commandline.CreateContextSettingsFlag(ctx, "")
	flag.Parse()
	_, err := commandline.ParseContextSettings(ctx, *settings)
	if err != nil {
		log.Fatalf("Failed to parse context settings: %v", err)
	}

	fmt.Println(commandline.SprintContextSettings(ctx))

	backend, err := backends.New()
	if err != nil {
		log.Fatalf("Failed to create backend: %v", err)
	}

	trainModel(backend, ctx)

	ctx = ctx.Reuse()

	generateText(backend, ctx, *flagPrompt)
}
