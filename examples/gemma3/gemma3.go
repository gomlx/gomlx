// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// gemma3 demonstrates ONNX-based text generation using GoMLX.
//
// It downloads the onnx-community/gemma-3-270m-it-ONNX model from HuggingFace,
// tokenizes a chat prompt, and generates text autoregressively.
//
// Usage:
//
//	go run gemma3.go
//	go run gemma3.go --prompt="What is Go?"
//	go run gemma3.go --max-tokens=50 --temperature=0.6
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/tokenizers"
	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/onnx-gomlx/onnx"
	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/default"
)

const (
	// HuggingFace repository for the Gemma3 270M ONNX model.
	modelRepo = "onnx-community/gemma-3-270m-it-ONNX"
)

var (
	flagPrompt    = flag.String("prompt", "Write a short poem about the sea.", "User message for the chat prompt.")
	flagMaxTokens = flag.Int("max-tokens", 100, "Maximum number of tokens to generate.")
	flagMaxSeqLen = flag.Int("max-seq-len", 256, "Maximum total sequence length (prompt + generated tokens).")
	flagTemp      = flag.Float64("temperature", 0.8, "Sampling temperature (0 = greedy).")
	flagTopK      = flag.Int("top-k", 64, "Top-k sampling (0 = disabled).")
	flagFP16      = flag.Bool("fp16", false, "Use fp16 model variant (570MB instead of 1.14GB).")
	flagBackend   = flag.String("backend", "", "Backend to use (default: auto-detect).")
)

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	if *flagBackend != "" {
		if err := os.Setenv("GOMLX_BACKEND", *flagBackend); err != nil {
			klog.Warningf("Failed to set backend: %v", err)
		}
	}

	// Determine ONNX model file path based on precision.
	modelFile := "onnx/model.onnx"
	if *flagFP16 {
		modelFile = "onnx/model_fp16.onnx"
	}

	// Download and cache model files from HuggingFace.
	// The ONNX model uses external data storage: model.onnx (graph) + model.onnx_data (weights).
	modelDataFile := modelFile + "_data"
	fmt.Printf("Downloading model: %s (%s)\n", modelRepo, modelFile)
	repo := hub.New(modelRepo).WithProgressBar(true)
	if err := repo.DownloadInfo(false); err != nil {
		klog.Fatalf("Failed to get repo info: %+v", err)
	}
	onnxPath, err := repo.DownloadFile(modelFile)
	if err != nil {
		klog.Fatalf("Failed to download %s: %+v", modelFile, err)
	}
	if _, err := repo.DownloadFile(modelDataFile); err != nil {
		klog.Fatalf("Failed to download %s: %+v", modelDataFile, err)
	}
	fmt.Printf("Model downloaded: %s\n\n", onnxPath)

	// Load tokenizer.
	tok, err := tokenizers.New(repo)
	if err != nil {
		klog.Fatalf("Failed to create tokenizer: %+v", err)
	}

	// Load ONNX model.
	model, err := onnx.ReadFile(onnxPath)
	if err != nil {
		klog.Fatalf("Failed to load ONNX model: %+v", err)
	}
	defer model.Close()

	inputNames, inputShapes := model.Inputs()
	outputNames, _ := model.Outputs()
	fmt.Printf("Model inputs (%d):\n", len(inputNames))
	for i, name := range inputNames {
		fmt.Printf("  %s: %v\n", name, inputShapes[i])
	}
	fmt.Printf("Model outputs: %v\n\n", outputNames)

	// Mark KV cache inputs as constants (empty past, no KV cache used).
	// We run full-sequence inference each step, so past_key_values are always empty.
	kvConstants := make(map[string]any)
	for i, name := range inputNames {
		if strings.HasPrefix(name, "past_key_values") {
			dims := inputShapes[i].Dimensions
			// Shape: [batch=1, num_kv_heads, past_seq_len=0, head_dim]
			// Create a zero-element float32 tensor with the right shape.
			kvConstants[name] = tensors.FromShape(shapes.Make(dtypes.Float32, 1, dims[1], 0, dims[3]))
		}
	}
	if len(kvConstants) > 0 {
		model.WithInputsAsConstants(kvConstants)
		fmt.Printf("KV cache: %d inputs marked as empty constants\n", len(kvConstants))
	}

	// Load model weights into context.
	ctx := context.New()
	if err := model.VariablesToContext(ctx); err != nil {
		klog.Fatalf("Failed to load model variables: %+v", err)
	}

	// Initialize backend.
	backend := backends.MustNew()
	fmt.Printf("Backend: %s\n\n", backend.Name())

	// Format chat prompt and tokenize.
	chatPrompt := formatChatPrompt(*flagPrompt)
	promptTokens := tokenizePrompt(tok, chatPrompt)
	fmt.Printf("Prompt: %q\n", *flagPrompt)
	fmt.Printf("Tokenized to %d tokens\n\n", len(promptTokens))

	if len(promptTokens) >= *flagMaxSeqLen {
		klog.Fatalf("Prompt (%d tokens) exceeds max sequence length (%d)", len(promptTokens), *flagMaxSeqLen)
	}

	// Generate text.
	fmt.Println("Generating...")
	fmt.Println("---")
	generate(backend, ctx, model, tok, promptTokens)
	fmt.Println("\n---")
}

// formatChatPrompt wraps the user message in Gemma3's chat template.
func formatChatPrompt(userMessage string) string {
	return fmt.Sprintf("<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", userMessage)
}

// tokenizePrompt encodes the prompt, prepending the BOS token.
func tokenizePrompt(tok api.Tokenizer, prompt string) []int32 {
	bosID, err := tok.SpecialTokenID(api.TokBeginningOfSentence)
	if err != nil {
		bosID = 2 // Gemma default BOS
	}
	encoded := tok.Encode(prompt)
	tokens := make([]int32, 0, len(encoded)+1)
	tokens = append(tokens, int32(bosID))
	for _, t := range encoded {
		tokens = append(tokens, int32(t))
	}
	return tokens
}

// nextPow2 returns the next power of 2 >= n.
func nextPow2(n int) int {
	if n <= 1 {
		return 1
	}
	p := 1
	for p < n {
		p *= 2
	}
	return p
}

// padSequence pads token IDs to targetLen and returns input_ids, attention_mask, and position_ids
// as [][]int64 (batch=1) suitable for passing to context.MustExecOnce.
func padSequence(tokens []int32, padID int32, targetLen int) (inputIDs [][]int64, attentionMask [][]int64, positionIDs [][]int64) {
	ids := make([]int64, targetLen)
	mask := make([]int64, targetLen)
	pos := make([]int64, targetLen)

	for i := range targetLen {
		if i < len(tokens) {
			ids[i] = int64(tokens[i])
			mask[i] = 1
			pos[i] = int64(i)
		} else {
			ids[i] = int64(padID)
			mask[i] = 0
			pos[i] = 0
		}
	}
	return [][]int64{ids}, [][]int64{mask}, [][]int64{pos}
}

// generate runs autoregressive text generation.
func generate(backend backends.Backend, ctx *context.Context, model *onnx.Model, tok api.Tokenizer, promptTokens []int32) {
	padID, err := tok.SpecialTokenID(api.TokPad)
	if err != nil {
		padID = 0
	}
	eosID, err := tok.SpecialTokenID(api.TokEndOfSentence)
	if err != nil {
		eosID = 1
	}

	// Discover which inputs the model expects.
	inputNames, _ := model.Inputs()
	inputSet := make(map[string]bool, len(inputNames))
	for _, name := range inputNames {
		inputSet[name] = true
	}

	hasPositionIDs := inputSet["position_ids"]
	hasAttentionMask := inputSet["attention_mask"]

	tokens := slices.Clone(promptTokens)
	maxSeqLen := *flagMaxSeqLen
	maxTokens := *flagMaxTokens

	startTime := time.Now()
	tokensGenerated := 0

	for range maxTokens {
		seqLen := len(tokens)
		if seqLen >= maxSeqLen {
			fmt.Printf("\n(reached max sequence length %d)", maxSeqLen)
			break
		}

		targetLen := min(nextPow2(seqLen), maxSeqLen)

		inputIDs, attentionMask, positionIDs := padSequence(tokens, int32(padID), targetLen)

		// Build the graph function and execute.
		output := context.MustExecOnce(
			backend, ctx.Reuse(),
			func(ctx *context.Context, idNode, maskNode, posNode *Node) *Node {
				g := idNode.Graph()
				inputs := map[string]*Node{
					"input_ids": idNode,
				}
				if hasAttentionMask {
					inputs["attention_mask"] = maskNode
				}
				if hasPositionIDs {
					inputs["position_ids"] = posNode
				}

				// KV cache inputs are set as constants on the model (empty past).
				outputs := model.CallGraph(ctx, g, inputs)
				// outputs[0] is logits: [batch, seq_len, vocab_size]
				logits := outputs[0]

				// Extract logits at the last real token position.
				lastPos := Const(g, int32(seqLen-1))
				// DynamicSlice to get [1, 1, vocab_size] then squeeze.
				vocabSize := logits.Shape().Dimensions[2]
				lastLogits := DynamicSlice(logits, []*Node{
					Const(g, int32(0)), lastPos, Const(g, int32(0)),
				}, []int{1, 1, vocabSize})
				lastLogits = Reshape(lastLogits, vocabSize)
				return lastLogits
			},
			inputIDs, attentionMask, positionIDs,
		)

		// CPU-side sampling.
		logits := tensors.MustCopyFlatData[float32](output)
		nextToken := sampleToken(logits, *flagTemp, *flagTopK)

		// Decode the token.
		tokenText := tok.Decode([]int{int(nextToken)})

		// Check for EOS.
		if int(nextToken) == eosID || tokenText == "<end_of_turn>" {
			break
		}

		// Print the token.
		fmt.Print(tokenText)

		tokens = append(tokens, nextToken)
		tokensGenerated++
	}

	duration := time.Since(startTime)
	if tokensGenerated > 0 {
		tokensPerSec := float64(tokensGenerated) / duration.Seconds()
		fmt.Printf("\n\nGenerated %d tokens in %.2fs (%.1f tokens/s)\n", tokensGenerated, duration.Seconds(), tokensPerSec)
	}
}

// sampleToken performs CPU-side sampling from logits with temperature scaling and optional top-k filtering.
func sampleToken(logits []float32, temperature float64, topK int) int32 {
	if temperature <= 0 {
		// Greedy: return argmax.
		maxIdx := 0
		maxVal := logits[0]
		for i, v := range logits[1:] {
			if v > maxVal {
				maxVal = v
				maxIdx = i + 1
			}
		}
		return int32(maxIdx)
	}

	// Apply temperature.
	scaled := make([]float64, len(logits))
	for i, v := range logits {
		scaled[i] = float64(v) / temperature
	}

	// Top-k filtering.
	if topK > 0 && topK < len(scaled) {
		type indexedLogit struct {
			index int
			value float64
		}
		indexed := make([]indexedLogit, len(scaled))
		for i, v := range scaled {
			indexed[i] = indexedLogit{i, v}
		}
		sort.Slice(indexed, func(i, j int) bool {
			return indexed[i].value > indexed[j].value
		})
		threshold := indexed[topK-1].value
		for i := range scaled {
			if scaled[i] < threshold {
				scaled[i] = math.Inf(-1)
			}
		}
	}

	// Softmax.
	maxVal := scaled[0]
	for _, v := range scaled[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	var sum float64
	for i := range scaled {
		scaled[i] = math.Exp(scaled[i] - maxVal)
		sum += scaled[i]
	}
	for i := range scaled {
		scaled[i] /= sum
	}

	// Categorical sample.
	r := rand.Float64()
	var cumulative float64
	for i, p := range scaled {
		cumulative += p
		if r < cumulative {
			return int32(i)
		}
	}
	return int32(len(scaled) - 1)
}
