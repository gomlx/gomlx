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

// kvStructure holds the KV cache layout discovered from the ONNX model.
type kvStructure struct {
	numLayers          int
	inputKeyNames      []string // past_key_values.{i}.key, ordered by layer
	inputValueNames    []string // past_key_values.{i}.value, ordered by layer
	outputKeyIndices   []int    // indices into model.Outputs() for present.{i}.key
	outputValueIndices []int    // indices into model.Outputs() for present.{i}.value
	logitsIndex        int      // index of logits in model.Outputs()
	kvHeads            int      // number of KV attention heads
	headDim            int      // head dimension
}

// hasOutputs returns true if the model has KV cache outputs (present_key_values).
func (kv *kvStructure) hasOutputs() bool {
	return kv.numLayers > 0 && len(kv.outputKeyIndices) == kv.numLayers
}

// discoverKVStructure inspects the ONNX model's inputs and outputs to identify
// the KV cache layout: layer count, input/output names, and shapes.
func discoverKVStructure(model *onnx.Model) *kvStructure {
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

	maxSeqLen := *flagMaxSeqLen
	maxTokens := *flagMaxTokens
	startTime := time.Now()
	tokensGenerated := 0

	// Discover KV cache structure from model inputs/outputs.
	kv := discoverKVStructure(model)

	if kv.hasOutputs() {
		tokensGenerated = generateWithKVCache(backend, ctx, model, tok, promptTokens, kv,
			eosID, maxSeqLen, maxTokens, hasAttentionMask, hasPositionIDs)
	} else {
		// Set any KV inputs as empty constants for the non-KV fallback.
		if kv.numLayers > 0 {
			emptyKV := make(map[string]any)
			for i := range kv.numLayers {
				emptyKV[kv.inputKeyNames[i]] = tensors.FromShape(shapes.Make(dtypes.Float32, 1, kv.kvHeads, 0, kv.headDim))
				emptyKV[kv.inputValueNames[i]] = tensors.FromShape(shapes.Make(dtypes.Float32, 1, kv.kvHeads, 0, kv.headDim))
			}
			model.WithInputsAsConstants(emptyKV)
			fmt.Printf("KV cache: %d past_key_values inputs marked as empty constants\n", len(emptyKV))
		}
		tokensGenerated = generateWithoutKVCache(backend, ctx, model, tok, promptTokens,
			padID, eosID, maxSeqLen, maxTokens, hasAttentionMask, hasPositionIDs)
	}

	duration := time.Since(startTime)
	if tokensGenerated > 0 {
		tokensPerSec := float64(tokensGenerated) / duration.Seconds()
		fmt.Printf("\n\nGenerated %d tokens in %.2fs (%.1f tokens/s)\n", tokensGenerated, duration.Seconds(), tokensPerSec)
	}
}

// generateWithKVCache runs generation using KV cache: a single prefill pass
// for the prompt, then incremental single-token decode steps.
func generateWithKVCache(
	backend backends.Backend, ctx *context.Context, model *onnx.Model, tok api.Tokenizer,
	promptTokens []int32, kv *kvStructure,
	eosID, maxSeqLen, maxTokens int, hasAttentionMask, hasPositionIDs bool,
) int {
	fmt.Printf("Using KV cache: %d layers, %d heads, dim=%d\n", kv.numLayers, kv.kvHeads, kv.headDim)

	// --- Prefill: process the full prompt in one pass ---
	// Set empty KV inputs as constants for the prefill.
	emptyKV := make(map[string]any)
	for i := range kv.numLayers {
		emptyKV[kv.inputKeyNames[i]] = tensors.FromShape(shapes.Make(dtypes.Float32, 1, kv.kvHeads, 0, kv.headDim))
		emptyKV[kv.inputValueNames[i]] = tensors.FromShape(shapes.Make(dtypes.Float32, 1, kv.kvHeads, 0, kv.headDim))
	}
	model.WithInputsAsConstants(emptyKV)

	seqLen := len(promptTokens)
	ids := make([]int64, seqLen)
	mask := make([]int64, seqLen)
	pos := make([]int64, seqLen)
	for i, t := range promptTokens {
		ids[i] = int64(t)
		mask[i] = 1
		pos[i] = int64(i)
	}

	prefillResults := context.MustExecOnceN(
		backend, ctx.Reuse(),
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

			// Extract last-token logits: [1, seqLen, vocabSize] → [vocabSize]
			logits := allOutputs[kv.logitsIndex]
			lastPos := SubScalar(seqLenNode, int32(1))
			vocabSize := logits.Shape().Dimensions[2]
			lastLogits := DynamicSlice(logits, []*Node{
				Const(g, int32(0)), lastPos, Const(g, int32(0)),
			}, []int{1, 1, vocabSize})
			lastLogits = Reshape(lastLogits, vocabSize)

			// Concatenate present KV across layers: [numLayers, 1, kvHeads, seqLen, headDim]
			keys := make([]*Node, kv.numLayers)
			values := make([]*Node, kv.numLayers)
			for i := range kv.numLayers {
				keys[i] = InsertAxes(allOutputs[kv.outputKeyIndices[i]], 0)
				values[i] = InsertAxes(allOutputs[kv.outputValueIndices[i]], 0)
			}
			return []*Node{lastLogits, Concatenate(keys, 0), Concatenate(values, 0)}
		},
		[][]int64{ids}, [][]int64{mask}, [][]int64{pos}, int32(seqLen),
	)

	logitsTensor := prefillResults[0]
	kvKeys := prefillResults[1]   // [numLayers, 1, kvHeads, seqLen, headDim]
	kvValues := prefillResults[2] // [numLayers, 1, kvHeads, seqLen, headDim]

	// Clear KV constants so decode passes KV as dynamic inputs.
	model.WithInputsAsConstants(nil)

	// --- Create persistent decode Exec (outside the loop) ---
	decodeExec := context.MustNewExec(backend, ctx.Reuse(),
		func(ctx *context.Context, idNode, maskNode, posNode, concatKeysNode, concatValuesNode *Node) []*Node {
			g := idNode.Graph()
			inputs := map[string]*Node{"input_ids": idNode}
			if hasAttentionMask {
				inputs["attention_mask"] = maskNode
			}
			if hasPositionIDs {
				inputs["position_ids"] = posNode
			}

			// Split concatenated KV into per-layer inputs.
			// Slice with AxisElem keeps the axis with size 1, so Reshape to remove it.
			for i := range kv.numLayers {
				layerKey := Slice(concatKeysNode, AxisElem(i))
				inputs[kv.inputKeyNames[i]] = Reshape(layerKey, layerKey.Shape().Dimensions[1:]...)
				layerVal := Slice(concatValuesNode, AxisElem(i))
				inputs[kv.inputValueNames[i]] = Reshape(layerVal, layerVal.Shape().Dimensions[1:]...)
			}

			allOutputs := model.CallGraph(ctx, g, inputs)

			// Logits for the single new token: [1, 1, vocabSize] → [vocabSize]
			logits := allOutputs[kv.logitsIndex]
			vocabSize := logits.Shape().Dimensions[2]
			lastLogits := Reshape(logits, vocabSize)

			// Concatenate present KV across layers.
			keys := make([]*Node, kv.numLayers)
			values := make([]*Node, kv.numLayers)
			for i := range kv.numLayers {
				keys[i] = InsertAxes(allOutputs[kv.outputKeyIndices[i]], 0)
				values[i] = InsertAxes(allOutputs[kv.outputValueIndices[i]], 0)
			}
			return []*Node{lastLogits, Concatenate(keys, 0), Concatenate(values, 0)}
		},
	)
	// Each decode step has a unique KV shape (sequence grows by 1), so we need
	// enough cache entries for the full generation. Power-of-2 padding could reduce
	// recompilations further but caching across steps is the main win.
	decodeExec.SetMaxCache(maxTokens + 1)
	defer decodeExec.Finalize()

	// Sample first token from prefill logits.
	logits := tensors.MustCopyFlatData[float32](logitsTensor)
	nextToken := sampleToken(logits, *flagTemp, *flagTopK)
	tokenText := tok.Decode([]int{int(nextToken)})
	if int(nextToken) == eosID || tokenText == "<end_of_turn>" {
		return 0
	}
	fmt.Print(tokenText)
	tokensGenerated := 1

	// --- Decode loop: generate one token at a time using KV cache ---
	currentPos := seqLen
	for range maxTokens - 1 {
		if currentPos >= maxSeqLen {
			fmt.Printf("\n(reached max sequence length %d)", maxSeqLen)
			break
		}

		// Build single-token inputs.
		decodeMask := make([]int64, currentPos+1)
		for i := range decodeMask {
			decodeMask[i] = 1
		}

		results := decodeExec.MustExec(
			[][]int64{{int64(nextToken)}},
			[][]int64{decodeMask},
			[][]int64{{int64(currentPos)}},
			kvKeys,
			kvValues,
		)

		logitsTensor = results[0]
		kvKeys = results[1]
		kvValues = results[2]

		logits = tensors.MustCopyFlatData[float32](logitsTensor)
		nextToken = sampleToken(logits, *flagTemp, *flagTopK)
		tokenText = tok.Decode([]int{int(nextToken)})
		if int(nextToken) == eosID || tokenText == "<end_of_turn>" {
			break
		}
		fmt.Print(tokenText)
		currentPos++
		tokensGenerated++
	}

	return tokensGenerated
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
		inputIDs, attentionMask, positionIDs := padSequence(tokens, int32(padID), targetLen)

		output := genExec.MustExec1(inputIDs, attentionMask, positionIDs, int32(seqLen))

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
