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
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/decode"
	"github.com/gomlx/gomlx/pkg/ml/model/transformer"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/gomlx/gomlx/ui/commandline"
)

// Hyperparameter keys
const (
	ParamSeqLen     = "seq_len"
	ParamBatchSize  = "batch_size"
	ParamTrainSteps = "train_steps"
)

var (
	flagPrompt = flag.String("prompt", "The quick", "Prompt text")
	flagSteps  = flag.Int("steps", 200, "Number of training steps, default 200")
)

func createDefaultContext() *context.Context {
	ctx := context.New()
	ctx.SetParams(map[string]any{
		// Model hyperparameters
		transformer.ParamVocabSize:   128,
		transformer.ParamEmbedDim:    64,
		transformer.ParamNumHeads:    4,
		transformer.ParamHeadDim:     16,
		transformer.ParamNumLayers:   2,
		ParamSeqLen:                  32,
		ParamBatchSize:               32,
		transformer.ParamMaxPosEmbed: 256,
		transformer.ParamDType:       "float32",

		// Training hyperparameters
		ParamTrainSteps:              *flagSteps,
		optimizers.ParamLearningRate: 0.001,

		// Generation hyperparameters
		// decode.ParamStrategy is initialized to "greedy" by default.
		decode.ParamStrategy:    "greedy",
		decode.ParamTemperature: 1.0,
		decode.ParamMaxLength:   50,
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

func createTrainingBatch(ctx *context.Context, text string) (inputs [][]int32, targets [][][]int32) {
	batchSize := context.GetParamOr(ctx, ParamBatchSize, 4)
	seqLen := context.GetParamOr(ctx, ParamSeqLen, 32)
	vocabSize := context.GetParamOr(ctx, transformer.ParamVocabSize, 128)

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
	modelFn := func(ctx *context.Context, _ any, inputs []*graph.Node) []*graph.Node {
		tFn := transformer.NewFromContext(ctx).ForTraining()
		return []*graph.Node{tFn(ctx, inputs[0])}
	}

	trainer := train.NewTrainer(backend, ctx, modelFn,
		losses.SparseCategoricalCrossEntropyLogits,
		optimizers.Adam().Done(),
		nil, nil) // no metrics for this simple example

	for step := range steps {
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
	vocabSize := context.GetParamOr(ctx, transformer.ParamVocabSize, 128)

	tokenizer := &CharTokenizer{vocabSize: vocabSize}

	promptTokens := tokenizer.Encode(prompt)
	if len(promptTokens) == 0 {
		promptTokens = []int{32}
	}

	modelFn := transformer.NewFromContext(ctx).ForGeneration()

	decoder := decode.New(modelFn).FromContext(ctx)
	fmt.Printf("\nGeneration\nStrategy: %s  Temp: %.2f  MaxLen: %d\nPrompt: %q\n\n",
		decoder.Strategy, decoder.Temperature, decoder.MaxLength, prompt)

	promptTensor := tensors.FromValue([][]int32{xslices.Map(promptTokens, func(t int) int32 { return int32(t) })})
	generated, err := decoder.Decode(backend, ctx, promptTensor)
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
	flag.Parse()
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
