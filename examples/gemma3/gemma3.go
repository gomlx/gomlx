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
	"os"
	"strings"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/tokenizers"
	"github.com/gomlx/go-huggingface/tokenizers/api"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers/attention/kvcache"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/zoo/transformer/generate"
	"github.com/gomlx/gomlx/ml/zoo/transformer/generate/sample"
	"github.com/gomlx/onnx-gomlx/onnx"
	onnxparser "github.com/gomlx/onnx-gomlx/onnx/parser"
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
	flagNaive       = flag.Bool("naive", false, "Use naive generation (re-runs the full prompt at each step) instead of KV cache.")
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
	onnxModel, err := onnxparser.ParseFile(onnxPath)
	if err != nil {
		klog.Fatalf("Failed to load ONNX model: %+v", err)
	}
	defer onnxModel.Close()

	inputNames, _ := onnxModel.Inputs()

	// Load model weights into model.
	scope := model.NewStore().RootScope()
	if err := onnxModel.VariablesToScope(scope); err != nil {
		klog.Fatalf("Failed to load model variables: %+v", err)
	}

	// Initialize backend.
	backend := compute.MustNew()
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

	eosID, err := tok.SpecialTokenID(api.TokEndOfSentence)
	if err != nil {
		eosID = 1
	}

	maxSeqLen := *flagMaxSeqLen
	maxTokens := *flagMaxTokens

	// Parse KV cache structure from model inputs/outputs.
	kv := parseKVStructure(onnxModel)

	// Define NaiveModelFn using generate package's seqLen
	var naiveModelFn generate.NaiveModelFn = func(scope *model.Scope, tokens *Node, seqLen *Node) *Node {
		g := tokens.Graph()
		inputs := map[string]*Node{"input_ids": ConvertType(tokens, dtypes.Int64)}

		if hasAttentionMask {
			// Create attention_mask from seqLen.
			// seqLen has shape [batchSize]. We want mask shape [batchSize, totalSeqLen]
			totalSeqLen := tokens.Shape().Dimensions[1]
			iotaSeq := Iota(g, shapes.Make(dtypes.Int32, 1, totalSeqLen), 1)
			seqLenCol := ExpandDims(seqLen, 1)
			maskNode := Where(LessThan(iotaSeq, seqLenCol), Const(g, int32(1)), Const(g, int32(0)))
			inputs["attention_mask"] = ConvertType(maskNode, dtypes.Int64)
		}
		if hasPositionIDs {
			// Position IDs is [batchSize, totalSeqLen]. We can just use Iota.
			totalSeqLen := tokens.Shape().Dimensions[1]
			batchSize := tokens.Shape().Dimensions[0]
			iotaSeq := Iota(g, shapes.Make(dtypes.Int32, 1, totalSeqLen), 1)
			seqLenCol := ExpandDims(seqLen, 1)
			posNode := Where(LessThan(iotaSeq, seqLenCol), iotaSeq, ConstAs(iotaSeq, 0))
			posNode = BroadcastToDims(posNode, batchSize, totalSeqLen)
			inputs["position_ids"] = ConvertType(posNode, dtypes.Int64)
		}
		if kv.numLayers > 0 {
			batchSize := tokens.Shape().Dimensions[0]
			for i := range kv.numLayers {
				inputs[kv.inputKeyNames[i]] = Zeros(g, shapes.Make(kv.kvDType, batchSize, kv.kvHeads, 0, kv.headDim))
				inputs[kv.inputValueNames[i]] = Zeros(g, shapes.Make(kv.kvDType, batchSize, kv.kvHeads, 0, kv.headDim))
			}
		}

		outputs := onnxModel.CallGraph(scope, g, inputs)
		logits := outputs[kv.logitsIndex]
		return logits
	}

	// Define KVCacheModelFn
	var cacheModelFn generate.KVCacheModelFn = func(scope *model.Scope, newTokens *Node, position *Node, cache kvcache.KVCacheNodes) (*Node, kvcache.KVCacheNodes) {
		g := newTokens.Graph()
		batchSize := newTokens.Shape().Dimensions[0]
		inputs := map[string]*Node{"input_ids": ConvertType(newTokens, dtypes.Int64)}

		// 1. Set past key/value cache inputs from the cache map.
		for i := range kv.numLayers {
			layerScopePath := fmt.Sprintf("/layer_%d", i)
			inputs[kv.inputKeyNames[i]] = cache[layerScopePath+kvcache.KeySuffix]
			inputs[kv.inputValueNames[i]] = cache[layerScopePath+kvcache.ValueSuffix]
		}

		// 2. Create the attention mask of shape [batchSize, cacheSeqLen + newSeqLen].
		cacheSeqLen := inputs[kv.inputKeyNames[0]].Shape().Dimensions[2]
		newSeqLen := newTokens.Shape().Dimensions[1]
		totalSeqLen := cacheSeqLen + newSeqLen
		iotaSeq := Iota(g, shapes.Make(dtypes.Int32, 1, totalSeqLen), 1)
		posCol := Reshape(position, 1, 1)
		posCol = BroadcastToDims(posCol, batchSize, 1)

		isPast := LessThan(iotaSeq, Const(g, int32(cacheSeqLen)))
		isValidPast := And(isPast, LessThan(iotaSeq, posCol))
		isNew := LessOrEqual(Const(g, int32(cacheSeqLen)), iotaSeq)

		maskNode := Or(isValidPast, isNew)
		maskNode = Where(maskNode, Const(g, int32(1)), Const(g, int32(0)))
		inputs["attention_mask"] = ConvertType(maskNode, dtypes.Int64)

		if hasPositionIDs {
			iotaNew := Iota(g, shapes.Make(dtypes.Int32, 1, newSeqLen), 1)
			posNode := Add(posCol, iotaNew)
			inputs["position_ids"] = ConvertType(posNode, dtypes.Int64)
		}

		// 3. Call the ONNX model
		allOutputs := onnxModel.CallGraph(scope, g, inputs)
		logits := allOutputs[kv.logitsIndex]

		// 4. Update the KV Cache
		updatedCache := make(kvcache.KVCacheNodes)
		for k, v := range cache {
			updatedCache[k] = v
		}

		batchIdx := Const(g, int32(0))
		headsIdx := Const(g, int32(0))
		dimIdx := Const(g, int32(0))

		for i := range kv.numLayers {
			presentKeys := allOutputs[kv.outputKeyIndices[i]]
			presentValues := allOutputs[kv.outputValueIndices[i]]

			sliceDims := []int{batchSize, kv.kvHeads, newSeqLen, kv.headDim}
			newKey := DynamicSlice(presentKeys, []*Node{
				Const(g, int32(0)), Const(g, int32(0)), Const(g, int32(cacheSeqLen)), Const(g, int32(0)),
			}, sliceDims)
			newValue := DynamicSlice(presentValues, []*Node{
				Const(g, int32(0)), Const(g, int32(0)), Const(g, int32(cacheSeqLen)), Const(g, int32(0)),
			}, sliceDims)

			layerScopePath := fmt.Sprintf("/layer_%d", i)
			kKey := layerScopePath + kvcache.KeySuffix
			vKey := layerScopePath + kvcache.ValueSuffix
			prevK := updatedCache[kKey]
			prevV := updatedCache[vKey]

			updatedK := prevK
			updatedV := prevV
			for stepIdx := 0; stepIdx < newSeqLen; stepIdx++ {
				writePos := AddScalar(position, stepIdx)
				tokenK := Slice(newKey, AxisRange(), AxisRange(), AxisRange(stepIdx, stepIdx+1), AxisRange())
				tokenV := Slice(newValue, AxisRange(), AxisRange(), AxisRange(stepIdx, stepIdx+1), AxisRange())

				updatedK = DynamicUpdateSlice(updatedK, tokenK, []*Node{batchIdx, headsIdx, writePos, dimIdx})
				updatedV = DynamicUpdateSlice(updatedV, tokenV, []*Node{batchIdx, headsIdx, writePos, dimIdx})
			}

			updatedCache[kKey] = updatedK
			updatedCache[vKey] = updatedV
		}

		return logits, updatedCache
	}

	// Setup KVCache if needed
	var kvCache *kvcache.KVCache
	if !*flagNaive && kv.hasOutputs() {
		kvCache = kvcache.NewKVCache().
			WithSinkPositions(0).
			WithMaxSeqLenPerLayerType([]int{maxSeqLen, maxSeqLen}).
			WithMinSeqLenPerLayerType([]int{32, 32})

		scopes := make([]string, kv.numLayers)
		for i := range kv.numLayers {
			scopes[i] = fmt.Sprintf("/layer_%d", i)
		}
		kvCache.WithOrderedScopes(scopes)
	}

	var decoder *generate.Generator
	if *flagNaive || kvCache == nil {
		fmt.Printf("Using Naive generation (without KV cache)\n\n")
		decoder = generate.New(naiveModelFn).FromScope(scope)
	} else {
		fmt.Printf("Using KV cache: %d layers, %d heads, dim=%d\n\n", kv.numLayers, kv.kvHeads, kv.headDim)
		decoder = generate.New(cacheModelFn).FromScope(scope)
		decoder.WithKVCache(kvCache, kv.kvHeads, kv.headDim, kv.kvDType)
	}

	if *flagTemp > 0 {
		decoder.WithStrategy(sample.StrategyTemperature).WithTemperature(float32(*flagTemp))
	} else {
		decoder.WithStrategy(sample.StrategyGreedy)
	}
	if *flagTopK > 0 {
		decoder.WithTopK(*flagTopK)
	}
	decoder.WithEOS(eosID)
	encodedEOT := tok.Encode("<end_of_turn>")
	if len(encodedEOT) > 0 {
		decoder.WithStopTokens(encodedEOT[len(encodedEOT)-1])
	}

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

			// Adjust max length dynamically for this prompt
			promptLen := len(promptTokens)
			decoder.WithMaxLength(min(promptLen+maxTokens, maxSeqLen))

			var n int
			startTime := time.Now()
			err := decoder.GenerateStreaming(backend, scope, promptTokens, func(token int) bool {
				if verbose {
					tokenText := tok.Decode([]int{token})
					fmt.Print(tokenText)
					_ = os.Stdout.Sync()
				}
				n++
				return true
			})
			if err != nil {
				klog.Fatalf("Generation failed: %+v", err)
			}
			dur := time.Since(startTime)

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

// kvStructure holds the KV cache layout parsed from the ONNX model.
type kvStructure struct {
	numLayers       int
	inputKeyNames   []string // past_key_values.{i}.key, ordered by layer
	inputValueNames []string // past_key_values.{i}.value, ordered by layer
	// outputKeyIndices are indices into the model's output list for the present key tensors.
	outputKeyIndices []int
	// outputValueIndices are indices into the model's output list for the present value tensors.
	outputValueIndices []int
	logitsIndex        int          // index of logits in model.Outputs()
	kvHeads            int          // number of KV attention heads
	headDim            int          // head dimension
	kvDType            dtypes.DType // DType for KV tensors
}

// hasOutputs returns true if the model has KV cache outputs (present_key_values).
func (kv *kvStructure) hasOutputs() bool {
	return kv.numLayers > 0 && len(kv.outputKeyIndices) == kv.numLayers
}

// parseKVStructure inspects the ONNX model's inputs and outputs to identify
// the KV cache layout: layer count, input/output names, and shapes.
func parseKVStructure(onnxModel onnx.Model) *kvStructure {
	inputNames, inputShapes := onnxModel.Inputs()
	outputNames, _ := onnxModel.Outputs()

	kv := &kvStructure{logitsIndex: 0} // default logits at index 0

	layerKeys := make(map[int]string)
	layerValues := make(map[int]string)
	for i, name := range inputNames {
		var layerIdx int
		if n, _ := fmt.Sscanf(name, "past_key_values.%d.key", &layerIdx); n == 1 && name == fmt.Sprintf("past_key_values.%d.key", layerIdx) {
			layerKeys[layerIdx] = name
			dims := inputShapes[i].Dimensions
			kv.kvHeads = dims[1]
			kv.headDim = dims[3]
			kv.kvDType = inputShapes[i].DType
		}
		if n, _ := fmt.Sscanf(name, "past_key_values.%d.value", &layerIdx); n == 1 && name == fmt.Sprintf("past_key_values.%d.value", layerIdx) {
			layerValues[layerIdx] = name
		}
	}

	kv.numLayers = len(layerKeys)
	if kv.numLayers == 0 {
		return kv
	}

	kv.inputKeyNames = make([]string, kv.numLayers)
	kv.inputValueNames = make([]string, kv.numLayers)
	for i := range kv.numLayers {
		kv.inputKeyNames[i] = layerKeys[i]
		kv.inputValueNames[i] = layerValues[i]
	}

	kv.outputKeyIndices = make([]int, kv.numLayers)
	kv.outputValueIndices = make([]int, kv.numLayers)
	foundKeys := 0
	foundValues := 0
	for i, name := range outputNames {
		if name == "logits" {
			kv.logitsIndex = i
		}
		var layerIdx int
		if n, _ := fmt.Sscanf(name, "present.%d.key", &layerIdx); n == 1 && name == fmt.Sprintf("present.%d.key", layerIdx) && layerIdx < kv.numLayers {
			kv.outputKeyIndices[layerIdx] = i
			foundKeys++
		}
		if n, _ := fmt.Sscanf(name, "present.%d.value", &layerIdx); n == 1 && name == fmt.Sprintf("present.%d.value", layerIdx) && layerIdx < kv.numLayers {
			kv.outputValueIndices[layerIdx] = i
			foundValues++
		}
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
		kv.outputKeyIndices = nil
		kv.outputValueIndices = nil
	}

	return kv
}
