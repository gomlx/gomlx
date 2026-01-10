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
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"
	"github.com/gomlx/gomlx/pkg/ml/layers/generation"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gopjrt/dtypes"
)

var (
	flagStrategy     = flag.String("strategy", "greedy", "Sampling strategy: greedy|temperature")
	flagTemperature  = flag.Float64("temperature", 1.0, "Sampling temperature")
	flagMaxLength    = flag.Int("max_length", 50, "Max generation length")
	flagPrompt       = flag.String("prompt", "The quick", "Prompt text")
	flagSteps        = flag.Int("steps", 200, "Training steps")
	flagLearningRate = flag.Float64("lr", 0.01, "Learning rate")
	flagUseCache     = flag.Bool("use_cache", false, "Use KV cache (faster generation)")
)

const (
	vocabSize   = 128
	embedDim    = 64
	numHeads    = 4
	headDim     = 16
	numLayers   = 2
	seqLen      = 32
	batchSize   = 4
	maxPosEmbed = 256
)

var dtype = dtypes.Float32

const trainingText = `The quick brown fox jumps over the lazy dog. ` +
	`The quick brown fox jumps over the lazy dog. ` +
	`The quick brown fox jumps over the lazy dog. ` +
	`The quick brown fox jumps over the lazy dog. ` +
	`Pack my box with five dozen liquor jugs. ` +
	`Pack my box with five dozen liquor jugs. ` +
	`How vexingly quick daft zebras jump! ` +
	`How vexingly quick daft zebras jump! `

type CharTokenizer struct{}

func (t *CharTokenizer) Encode(text string) []int {
	tokens := make([]int, len(text))
	for i, char := range text {
		tokens[i] = int(char)
		if tokens[i] >= vocabSize {
			tokens[i] = 32 // Use space for unknown chars
		}
	}
	return tokens
}

func (t *CharTokenizer) Decode(tokens []int) string {
	chars := make([]byte, len(tokens))
	for i, token := range tokens {
		if token >= 0 && token < vocabSize {
			chars[i] = byte(token)
		} else {
			chars[i] = ' '
		}
	}
	return string(chars)
}

func simpleTransformerModel(ctx *context.Context, inputs []*Node) []*Node {
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
		attn := attention.MultiHeadAttention(layerCtx.In("attn"), x, x, x, numHeads, headDim).
			UseCausalMask().
			Done()
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

func createTrainingBatch(text string, batchSize, seqLen int) (inputs [][]int32, targets [][][]int32) {
	tokenizer := &CharTokenizer{}
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

func trainModel(backend backends.Backend, ctx *context.Context, steps int, learningRate float64) {
	fmt.Printf("\nTraining Model\nSteps: %d  LR: %.4f  Batch: %d  SeqLen: %d\n\n", steps, learningRate, batchSize, seqLen)

	// Set learning rate in context
	ctx.SetParam(optimizers.ParamLearningRate, learningRate)

	var modelFn func(ctx *context.Context, _ any, inputs []*Node) []*Node

	if *flagUseCache {
		// Build cached transformer (training path uses non-cached forward)
		transformerCfg := generation.NewTransformerConfig(vocabSize, embedDim, numLayers, numHeads, headDim).
			WithFFNDim(embedDim * 4).
			WithMaxPosEmbed(maxPosEmbed).
			WithDType(dtype).
			WithLayerNorm(true).
			WithBias(true)

		cachedTransformer := generation.BuildCachedTransformer(ctx, transformerCfg)

		// Use ForTraining() which doesn't use cache
		modelFn = func(ctx *context.Context, _ any, inputs []*Node) []*Node {
			tokens := inputs[0]
			logits := cachedTransformer.ForTraining()(ctx, tokens)
			return []*Node{logits}
		}
	} else {
		// Use simple transformer model (non-cached)
		modelFn = func(ctx *context.Context, _ any, inputs []*Node) []*Node {
			return simpleTransformerModel(ctx, inputs)
		}
	}

	trainer := train.NewTrainer(backend, ctx, modelFn,
		losses.SparseCategoricalCrossEntropyLogits,
		optimizers.Adam().Done(),
		nil, nil) // no metrics for this simple example

	for step := 0; step < steps; step++ {
		inputData, targetData := createTrainingBatch(trainingText, batchSize, seqLen)
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
	tokenizer := &CharTokenizer{}
	fmt.Printf("\nGeneration\nStrategy: %s  Temp: %.2f  MaxLen: %d  Cache: %v\nPrompt: %q\n\n", *flagStrategy, *flagTemperature, *flagMaxLength, *flagUseCache, prompt)

	promptTokens := tokenizer.Encode(prompt)
	if len(promptTokens) == 0 {
		promptTokens = []int{32}
	}

	var genCfg *generation.GenerationConfig

	if *flagUseCache {
		transformerCfg := generation.NewTransformerConfig(vocabSize, embedDim, numLayers, numHeads, headDim).
			WithFFNDim(embedDim * 4).
			WithMaxPosEmbed(maxPosEmbed).
			WithDType(dtype).
			WithLayerNorm(true).
			WithBias(true)

		cachedTransformer := generation.BuildCachedTransformer(ctx, transformerCfg)

		genCfg = generation.NewGenerationConfigCached(
			cachedTransformer.ForGeneration(),
			numLayers,
			numHeads,
			headDim,
			maxPosEmbed,
			dtype,
		).
			WithStrategy(*flagStrategy).
			WithTemperature(float32(*flagTemperature)).
			WithMaxLength(*flagMaxLength)
	} else {
		modelFn := func(genCtx *context.Context, tokens *Node) *Node {
			outputs := simpleTransformerModel(genCtx, []*Node{tokens})
			return outputs[0]
		}

		genCfg = generation.NewGenerationConfig(modelFn).
			WithStrategy(*flagStrategy).
			WithTemperature(float32(*flagTemperature)).
			WithMaxLength(*flagMaxLength)
	}

	promptTensor := tensors.FromValue([][]int32{convertToInt32(promptTokens)})
	generated, err := genCfg.Generate(backend, ctx, promptTensor)
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

func convertToInt32(tokens []int) []int32 {
	result := make([]int32, len(tokens))
	for i, t := range tokens {
		result[i] = int32(t)
	}
	return result
}

func main() {
	flag.Parse()

	backend, err := backends.New()
	if err != nil {
		log.Fatalf("Failed to create backend: %v", err)
	}

	ctx := context.New()

	trainModel(backend, ctx, *flagSteps, *flagLearningRate)

	ctx = ctx.Reuse()

	generateText(backend, ctx, *flagPrompt)
}
