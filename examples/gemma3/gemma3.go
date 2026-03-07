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
//	go run gemma3.go --prompts-file=prompts.txt --warmup=2
package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"math/bits"
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
	flagPrompt      = flag.String("prompt", "Write a short poem about the sea.", "User message for the chat prompt.")
	flagPromptsFile = flag.String("prompts-file", "", "Path to text file with one prompt per line (overrides --prompt).")
	flagWarmup      = flag.Int("warmup", 1, "Number of warmup rounds before measurement (only with --prompts-file).")
	flagMaxTokens   = flag.Int("max-tokens", 100, "Maximum number of tokens to generate.")
	flagMaxSeqLen   = flag.Int("max-seq-len", 256, "Maximum total sequence length (prompt + generated tokens).")
	flagTemp        = flag.Float64("temperature", 0.8, "Sampling temperature (0 = greedy).")
	flagTopK        = flag.Int("top-k", 64, "Top-k sampling (0 = disabled).")
	flagFP16        = flag.Bool("fp16", false, "Use fp16 (float16) model variant (570MB instead of 1.14GB).")
	flagBackend     = flag.String("backend", "", "Backend to use (default: auto-detect).")
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

	// Load model weights into context.
	ctx := context.New()
	if err := model.VariablesToContext(ctx); err != nil {
		klog.Fatalf("Failed to load model variables: %+v", err)
	}

	// Initialize backend.
	backend := backends.MustNew()
	fmt.Printf("Backend: %s\n\n", backend.Name())

	// Load prompts.
	prompts := loadPrompts()

	// Discover which inputs the model expects.
	inputSet := make(map[string]bool, len(inputNames))
	for _, name := range inputNames {
		inputSet[name] = true
	}
	hasPositionIDs := inputSet["position_ids"]
	hasAttentionMask := inputSet["attention_mask"]

	padID, err := tok.SpecialTokenID(api.TokPad)
	if err != nil {
		padID = 0
	}
	eosID, err := tok.SpecialTokenID(api.TokEndOfSentence)
	if err != nil {
		eosID = 1
	}

	maxSeqLen := *flagMaxSeqLen
	maxTokens := *flagMaxTokens

	// Parse KV cache structure from model inputs/outputs.
	kv := parseKVStructure(model)

	if kv.hasOutputs() {
		fmt.Printf("Using KV cache: %d layers, %d heads, dim=%d\n\n", kv.numLayers, kv.kvHeads, kv.headDim)
		cacheSize := int(math.Log2(float64(maxSeqLen))) + 2

		// Create persistent prefill Exec (empty KV as constants).
		emptyKV := make(map[string]any)
		for i := range kv.numLayers {
			emptyKV[kv.inputKeyNames[i]] = tensors.FromShape(shapes.Make(dtypes.Float32, 1, kv.kvHeads, 0, kv.headDim))
			emptyKV[kv.inputValueNames[i]] = tensors.FromShape(shapes.Make(dtypes.Float32, 1, kv.kvHeads, 0, kv.headDim))
		}
		model.WithInputsAsConstants(emptyKV)

		prefillExec := context.MustNewExec(backend, ctx.Reuse(),
			func(ctx *context.Context, idNode, maskNode, posNode, seqLenNode *Node) []*Node {
				g := idNode.Graph()
				inputs := map[string]*Node{"input_ids": idNode}
				if hasAttentionMask {
					inputs["attention_mask"] = maskNode
				}
				if hasPositionIDs {
					inputs["position_ids"] = posNode
				}

				allOutputs := model.CallGraph(ctx, g, inputs)

				// Extract last-token logits: [1, seqLen, vocabSize] -> [vocabSize]
				logits := allOutputs[kv.logitsIndex]
				lastPos := SubScalar(seqLenNode, int32(1))
				vocabSize := logits.Shape().Dimensions[2]
				lastLogits := DynamicSlice(logits, []*Node{
					Const(g, int32(0)), lastPos, Const(g, int32(0)),
				}, []int{1, 1, vocabSize})
				lastLogits = Reshape(lastLogits, vocabSize)

				concatKeys, concatValues := kv.extractOutputs(allOutputs)
				return []*Node{lastLogits, concatKeys, concatValues}
			},
		)
		prefillExec.SetMaxCache(cacheSize)
		defer prefillExec.Finalize()

		// Clear KV constants for decode (KV passed as dynamic inputs).
		model.WithInputsAsConstants(nil)

		decodeExec := context.MustNewExec(backend, ctx.Reuse(),
			func(ctx *context.Context, idNode, maskNode, posNode, concatKeysNode, concatValuesNode, kvInsertPosNode *Node) []*Node {
				g := idNode.Graph()
				inputs := map[string]*Node{"input_ids": idNode}
				if hasAttentionMask {
					inputs["attention_mask"] = maskNode
				}
				if hasPositionIDs {
					inputs["position_ids"] = posNode
				}
				kv.setInputs(inputs, concatKeysNode, concatValuesNode)

				allOutputs := model.CallGraph(ctx, g, inputs)

				// Logits for the single new token: [1, 1, vocabSize] -> [vocabSize]
				logits := allOutputs[kv.logitsIndex]
				vocabSize := logits.Shape().Dimensions[2]
				lastLogits := Reshape(logits, vocabSize)

				presentKeys, presentValues := kv.extractOutputs(allOutputs)
				// presentKeys shape: [numLayers, 1, kvHeads, P+1, headDim]
				// Extract new token's KV at position P (last along seq dim).
				paddedSize := concatKeysNode.Shape().Dimensions[3]
				sliceDims := []int{kv.numLayers, 1, kv.kvHeads, 1, kv.headDim}
				newKeys := DynamicSlice(presentKeys, []*Node{
					Const(g, int32(0)), Const(g, int32(0)), Const(g, int32(0)),
					Const(g, int32(paddedSize)), Const(g, int32(0)),
				}, sliceDims)
				newValues := DynamicSlice(presentValues, []*Node{
					Const(g, int32(0)), Const(g, int32(0)), Const(g, int32(0)),
					Const(g, int32(paddedSize)), Const(g, int32(0)),
				}, sliceDims)

				// Insert new KV at kvInsertPos in the padded buffer.
				updatedKeys := DynamicUpdateSlice(concatKeysNode, newKeys, []*Node{
					Const(g, int32(0)), Const(g, int32(0)), Const(g, int32(0)),
					kvInsertPosNode, Const(g, int32(0)),
				})
				updatedValues := DynamicUpdateSlice(concatValuesNode, newValues, []*Node{
					Const(g, int32(0)), Const(g, int32(0)), Const(g, int32(0)),
					kvInsertPosNode, Const(g, int32(0)),
				})

				return []*Node{lastLogits, updatedKeys, updatedValues}
			},
		)
		// Power-of-2 padding means O(log(maxSeqLen)) unique graph shapes.
		decodeExec.SetMaxCache(cacheSize)
		defer decodeExec.Finalize()

		benchmarkMode := *flagPromptsFile != ""
		warmupRounds := 0
		if benchmarkMode {
			warmupRounds = *flagWarmup
		}
		totalRounds := warmupRounds + 1

		for round := range totalRounds {
			isWarmup := round < warmupRounds
			if isWarmup {
				fmt.Printf("=== Warmup round %d/%d ===\n", round+1, warmupRounds)
			} else if benchmarkMode {
				fmt.Println("=== Measurement round ===")
			}

			var totalTokens int
			var totalDuration time.Duration

			for _, prompt := range prompts {
				promptTokens := tokenizePrompt(tok, formatChatPrompt(prompt))
				if len(promptTokens) >= maxSeqLen {
					fmt.Printf("Warning: prompt %q too long (%d tokens), skipping\n", prompt, len(promptTokens))
					continue
				}

				verbose := !isWarmup
				if verbose {
					fmt.Printf("Prompt: %q\n", prompt)
					fmt.Printf("Tokenized to %d tokens\n\n", len(promptTokens))
					fmt.Println("Generating...")
					fmt.Println("---")
				}

				n, dur := generateOne(backend, prefillExec, decodeExec, tok, promptTokens,
					kv, padID, eosID, maxSeqLen, maxTokens, verbose)

				if verbose {
					fmt.Println("\n---")
					if n > 0 {
						tokensPerSec := float64(n) / dur.Seconds()
						fmt.Printf("Generated %d tokens in %.2fs (%.1f tokens/s)\n\n", n, dur.Seconds(), tokensPerSec)
					}
				}

				totalTokens += n
				totalDuration += dur
			}

			if benchmarkMode && !isWarmup && totalTokens > 0 {
				fmt.Printf("\nAverage: %.1f tokens/s (%d tokens in %.2fs)\n",
					float64(totalTokens)/totalDuration.Seconds(), totalTokens, totalDuration.Seconds())
			}
		}
	} else {
		// Non-KV-cache fallback.
		if kv.numLayers > 0 {
			emptyKV := make(map[string]any)
			for i := range kv.numLayers {
				emptyKV[kv.inputKeyNames[i]] = tensors.FromShape(shapes.Make(dtypes.Float32, 1, kv.kvHeads, 0, kv.headDim))
				emptyKV[kv.inputValueNames[i]] = tensors.FromShape(shapes.Make(dtypes.Float32, 1, kv.kvHeads, 0, kv.headDim))
			}
			model.WithInputsAsConstants(emptyKV)
			fmt.Printf("KV cache: %d past_key_values inputs marked as empty constants\n", len(emptyKV))
		}

		for _, prompt := range prompts {
			promptTokens := tokenizePrompt(tok, formatChatPrompt(prompt))
			if len(promptTokens) >= maxSeqLen {
				fmt.Printf("Warning: prompt %q too long (%d tokens), skipping\n", prompt, len(promptTokens))
				continue
			}

			fmt.Printf("Prompt: %q\n", prompt)
			fmt.Printf("Tokenized to %d tokens\n\n", len(promptTokens))
			fmt.Println("Generating...")
			fmt.Println("---")
			startTime := time.Now()
			n := generateWithoutKVCache(backend, ctx, model, tok, promptTokens,
				padID, eosID, maxSeqLen, maxTokens, hasAttentionMask, hasPositionIDs)
			duration := time.Since(startTime)
			fmt.Println("\n---")
			if n > 0 {
				tokensPerSec := float64(n) / duration.Seconds()
				fmt.Printf("Generated %d tokens in %.2fs (%.1f tokens/s)\n\n", n, duration.Seconds(), tokensPerSec)
			}
		}
	}
}

// loadPrompts returns prompts from either --prompts-file or --prompt.
func loadPrompts() []string {
	if *flagPromptsFile != "" {
		f, err := os.Open(*flagPromptsFile)
		if err != nil {
			klog.Fatalf("Failed to open prompts file: %v", err)
		}
		defer f.Close()

		var prompts []string
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line != "" {
				prompts = append(prompts, line)
			}
		}
		if err := scanner.Err(); err != nil {
			klog.Fatalf("Error reading prompts file: %v", err)
		}
		if len(prompts) == 0 {
			klog.Fatalf("No prompts found in file: %s", *flagPromptsFile)
		}
		fmt.Printf("Loaded %d prompts from %s\n", len(prompts), *flagPromptsFile)
		return prompts
	}
	return []string{*flagPrompt}
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
	if n <= 0 {
		return 1
	}
	return 1 << (bits.UintSize - bits.LeadingZeros(uint(n-1)))
}

// padSequence pads token IDs to targetLen and returns input_ids, attention_mask, and position_ids
// as flat []int64 slices suitable for execution.
func padSequence(tokens []int32, padID int32, targetLen int) (inputIDs []int64, attentionMask []int64, positionIDs []int64) {
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
	return ids, mask, pos
}

// kvStructure holds the KV cache layout parsed from the ONNX model.
type kvStructure struct {
	numLayers       int
	inputKeyNames   []string // past_key_values.{i}.key, ordered by layer
	inputValueNames []string // past_key_values.{i}.value, ordered by layer
	// outputKeyIndices are indices into the model's output list for the present key tensors.
	// The model returns present KV (past KV concatenated with the newly computed token's KV).
	outputKeyIndices []int
	// outputValueIndices are indices into the model's output list for the present value tensors.
	outputValueIndices []int
	logitsIndex        int // index of logits in model.Outputs()
	kvHeads            int // number of KV attention heads
	headDim            int // head dimension
}

// hasOutputs returns true if the model has KV cache outputs (present_key_values).
func (kv *kvStructure) hasOutputs() bool {
	return kv.numLayers > 0 && len(kv.outputKeyIndices) == kv.numLayers
}

// extractOutputs collects per-layer KV outputs from the model and concatenates them
// into [numLayers, 1, kvHeads, seqLen, headDim] tensors.
func (kv *kvStructure) extractOutputs(allOutputs []*Node) (concatKeys, concatValues *Node) {
	keys := make([]*Node, kv.numLayers)
	values := make([]*Node, kv.numLayers)
	for i := range kv.numLayers {
		keys[i] = InsertAxes(allOutputs[kv.outputKeyIndices[i]], 0)
		values[i] = InsertAxes(allOutputs[kv.outputValueIndices[i]], 0)
	}
	return Concatenate(keys, 0), Concatenate(values, 0)
}

// setInputs splits concatenated KV and feeds per-layer inputs into the model input map.
func (kv *kvStructure) setInputs(inputs map[string]*Node, concatKeys, concatValues *Node) {
	for i := range kv.numLayers {
		inputs[kv.inputKeyNames[i]] = Squeeze(Slice(concatKeys, AxisElem(i)), 0)
		inputs[kv.inputValueNames[i]] = Squeeze(Slice(concatValues, AxisElem(i)), 0)
	}
}

// parseKVStructure inspects the ONNX model's inputs and outputs to identify
// the KV cache layout: layer count, input/output names, and shapes.
func parseKVStructure(model *onnx.Model) *kvStructure {
	inputNames, inputShapes := model.Inputs()
	outputNames, _ := model.Outputs()

	kv := &kvStructure{logitsIndex: 0} // default logits at index 0

	// Find KV input names and shapes.
	// Note: fmt.Sscanf returns n=1 once the integer is parsed, even if the trailing
	// literal doesn't match. We reconstruct and compare to ensure an exact match.
	layerKeys := make(map[int]string)
	layerValues := make(map[int]string)
	for i, name := range inputNames {
		var layerIdx int
		if n, _ := fmt.Sscanf(name, "past_key_values.%d.key", &layerIdx); n == 1 && name == fmt.Sprintf("past_key_values.%d.key", layerIdx) {
			layerKeys[layerIdx] = name
			dims := inputShapes[i].Dimensions
			kv.kvHeads = dims[1]
			kv.headDim = dims[3]
		}
		if n, _ := fmt.Sscanf(name, "past_key_values.%d.value", &layerIdx); n == 1 && name == fmt.Sprintf("past_key_values.%d.value", layerIdx) {
			layerValues[layerIdx] = name
		}
	}

	kv.numLayers = len(layerKeys)
	if kv.numLayers == 0 {
		return kv
	}

	// Build ordered lists of KV input names.
	kv.inputKeyNames = make([]string, kv.numLayers)
	kv.inputValueNames = make([]string, kv.numLayers)
	for i := range kv.numLayers {
		kv.inputKeyNames[i] = layerKeys[i]
		kv.inputValueNames[i] = layerValues[i]
	}

	// Find KV output indices and logits index.
	kv.outputKeyIndices = make([]int, kv.numLayers)
	kv.outputValueIndices = make([]int, kv.numLayers)
	foundKeys := 0
	foundValues := 0
	for i, name := range outputNames {
		if name == "logits" {
			kv.logitsIndex = i
		}
		var layerIdx int
		// Try "present.{i}.key" pattern (HuggingFace Optimum).
		if n, _ := fmt.Sscanf(name, "present.%d.key", &layerIdx); n == 1 && name == fmt.Sprintf("present.%d.key", layerIdx) && layerIdx < kv.numLayers {
			kv.outputKeyIndices[layerIdx] = i
			foundKeys++
		}
		if n, _ := fmt.Sscanf(name, "present.%d.value", &layerIdx); n == 1 && name == fmt.Sprintf("present.%d.value", layerIdx) && layerIdx < kv.numLayers {
			kv.outputValueIndices[layerIdx] = i
			foundValues++
		}
		// Try "present_key_values.{i}.key" pattern.
		if n, _ := fmt.Sscanf(name, "present_key_values.%d.key", &layerIdx); n == 1 && name == fmt.Sprintf("present_key_values.%d.key", layerIdx) && layerIdx < kv.numLayers {
			kv.outputKeyIndices[layerIdx] = i
			foundKeys++
		}
		if n, _ := fmt.Sscanf(name, "present_key_values.%d.value", &layerIdx); n == 1 && name == fmt.Sprintf("present_key_values.%d.value", layerIdx) && layerIdx < kv.numLayers {
			kv.outputValueIndices[layerIdx] = i
			foundValues++
		}
	}

	if foundKeys != kv.numLayers || foundValues != kv.numLayers {
		// Model has KV inputs but not matching outputs; can't use KV cache.
		kv.outputKeyIndices = nil
		kv.outputValueIndices = nil
	}

	return kv
}

// generateOne runs a single prompt through prefill and decode with KV cache,
// returning the number of tokens generated and the wall-clock duration.
func generateOne(
	backend backends.Backend,
	prefillExec *context.Exec,
	decodeExec *context.Exec,
	tok api.Tokenizer,
	promptTokens []int32,
	kv *kvStructure,
	padID, eosID, maxSeqLen, maxTokens int,
	verbose bool,
) (int, time.Duration) {
	startTime := time.Now()
	seqLen := len(promptTokens)
	paddedLen := nextPow2(seqLen)

	// Pad prompt and run prefill.
	ids, mask, pos := padSequence(promptTokens, int32(padID), paddedLen)
	prefillResults := prefillExec.MustExec(
		[][]int64{ids}, [][]int64{mask}, [][]int64{pos}, int32(seqLen),
	)

	logitsTensor := prefillResults[0]
	kvKeys := prefillResults[1]   // [numLayers, 1, kvHeads, paddedLen, headDim]
	kvValues := prefillResults[2] // [numLayers, 1, kvHeads, paddedLen, headDim]

	// Sample first token from prefill logits.
	logits := tensors.MustCopyFlatData[float32](logitsTensor)
	nextToken := sampleToken(logits, *flagTemp, *flagTopK)
	tokenText := tok.Decode([]int{int(nextToken)})
	if int(nextToken) == eosID || tokenText == "<end_of_turn>" {
		return 0, time.Since(startTime)
	}
	if verbose {
		fmt.Print(tokenText)
	}
	tokensGenerated := 1

	// Decode loop with power-of-2 KV cache padding.
	// The padded buffer stays at a fixed power-of-2 size, so the decode graph
	// only needs to be recompiled when crossing a power-of-2 boundary.
	realKVLen := seqLen
	paddedSize := paddedLen

	for range maxTokens - 1 {
		if realKVLen+1 >= maxSeqLen {
			if verbose {
				fmt.Printf("\n(reached max sequence length %d)", maxSeqLen)
			}
			break
		}

		// Grow buffer if the next insertion would exceed current padded size.
		if realKVLen >= paddedSize {
			paddedSize = nextPow2(realKVLen + 1)
			kvKeys = growKVBuffer(backend, kvKeys, paddedSize)
			kvValues = growKVBuffer(backend, kvValues, paddedSize)
		}

		// Build attention mask: paddedSize+1 positions (padded past + new token).
		decodeMask := make([]int64, paddedSize+1)
		for i := 0; i < realKVLen; i++ {
			decodeMask[i] = 1
		}
		decodeMask[paddedSize] = 1 // new token position

		results := decodeExec.MustExec(
			[][]int64{{int64(nextToken)}},
			[][]int64{decodeMask},
			[][]int64{{int64(realKVLen)}},
			kvKeys,
			kvValues,
			int32(realKVLen), // kvInsertPos
		)

		logitsTensor = results[0]
		kvKeys = results[1]
		kvValues = results[2]
		realKVLen++

		logits = tensors.MustCopyFlatData[float32](logitsTensor)
		nextToken = sampleToken(logits, *flagTemp, *flagTopK)
		tokenText = tok.Decode([]int{int(nextToken)})
		if int(nextToken) == eosID || tokenText == "<end_of_turn>" {
			break
		}
		if verbose {
			fmt.Print(tokenText)
		}
		tokensGenerated++
	}

	return tokensGenerated, time.Since(startTime)
}

// growKVBuffer creates a new padded KV buffer by zero-padding along the sequence dimension.
func growKVBuffer(backend backends.Backend, kv *tensors.Tensor, targetSeqLen int) *tensors.Tensor {
	oldShape := kv.Shape()
	if oldShape.Dimensions[3] >= targetSeqLen {
		return kv
	}
	newShape := shapes.Make(oldShape.DType,
		oldShape.Dimensions[0], oldShape.Dimensions[1],
		oldShape.Dimensions[2], targetSeqLen, oldShape.Dimensions[4])
	zeroTarget := tensors.FromShape(newShape)
	return context.MustExecOnce(backend, nil,
		func(_ *context.Context, target, old *Node) *Node {
			g := old.Graph()
			return DynamicUpdateSlice(target, old, []*Node{
				Const(g, int32(0)), Const(g, int32(0)), Const(g, int32(0)),
				Const(g, int32(0)), Const(g, int32(0)),
			})
		},
		zeroTarget, kv,
	)
}

// generateWithoutKVCache runs generation without KV cache: each step processes
// the full sequence with power-of-2 padding. A persistent Exec is created
// outside the loop to enable compilation caching across steps.
func generateWithoutKVCache(
	backend backends.Backend, ctx *context.Context, model *onnx.Model, tok api.Tokenizer,
	promptTokens []int32,
	padID, eosID, maxSeqLen, maxTokens int, hasAttentionMask, hasPositionIDs bool,
) int {
	// Create the Exec outside the loop. The seqLen parameter is passed dynamically
	// so the same compiled graph works for different sequence lengths within the
	// same power-of-2 padded shape.
	genExec := context.MustNewExec(backend, ctx.Reuse(),
		func(ctx *context.Context, idNode, maskNode, posNode, seqLenNode *Node) *Node {
			g := idNode.Graph()
			inputs := map[string]*Node{"input_ids": idNode}
			if hasAttentionMask {
				inputs["attention_mask"] = maskNode
			}
			if hasPositionIDs {
				inputs["position_ids"] = posNode
			}

			outputs := model.CallGraph(ctx, g, inputs)
			logits := outputs[0]

			// Extract logits at the last real token position.
			lastPos := SubScalar(seqLenNode, int32(1))
			vocabSize := logits.Shape().Dimensions[2]
			lastLogits := DynamicSlice(logits, []*Node{
				Const(g, int32(0)), lastPos, Const(g, int32(0)),
			}, []int{1, 1, vocabSize})
			lastLogits = Reshape(lastLogits, vocabSize)
			return lastLogits
		},
	)
	defer genExec.Finalize()

	tokens := slices.Clone(promptTokens)
	tokensGenerated := 0

	for range maxTokens {
		seqLen := len(tokens)
		if seqLen >= maxSeqLen {
			fmt.Printf("\n(reached max sequence length %d)", maxSeqLen)
			break
		}

		targetLen := min(nextPow2(seqLen), maxSeqLen)
		ids, mask, pos := padSequence(tokens, int32(padID), targetLen)

		output := genExec.MustExec1([][]int64{ids}, [][]int64{mask}, [][]int64{pos}, int32(seqLen))

		logits := tensors.MustCopyFlatData[float32](output)
		nextToken := sampleToken(logits, *flagTemp, *flagTopK)
		tokenText := tok.Decode([]int{int(nextToken)})
		if int(nextToken) == eosID || tokenText == "<end_of_turn>" {
			break
		}
		fmt.Print(tokenText)
		tokens = append(tokens, nextToken)
		tokensGenerated++
	}

	return tokensGenerated
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
