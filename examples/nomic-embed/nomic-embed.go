// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// nomic-embed demonstrates text embedding with dynamic shapes using GoMLX.
//
// It downloads the nomic-ai/nomic-embed-text-v1.5 ONNX model from HuggingFace,
// tokenizes texts with task-specific prefixes, and computes dense embeddings.
//
// Texts are padded to a common sequence length and processed in sub-batches
// of varying size. The computation graph is compiled once with a symbolic
// batch dimension and specialized per concrete batch size at execution time,
// demonstrating GoMLX's dynamic shape support.
//
// Supports Matryoshka Representation Learning: embeddings can be truncated to
// smaller dimensions (768, 512, 256, 128, 64) without retraining.
//
// Usage:
//
//	go run nomic-embed.go
//	go run nomic-embed.go --task=search_query "What is machine learning?"
//	go run nomic-embed.go --dim=256
//	go run nomic-embed.go --fp16
package main

import (
	"flag"
	"fmt"
	"math"
	"os"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/tokenizers"
	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/onnx-gomlx/onnx"
	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/default"
)

const (
	// HuggingFace repository for the embedding model.
	modelRepo = "nomic-ai/nomic-embed-text-v1.5"

	// Full embedding dimension before Matryoshka truncation.
	fullDim = 768
)

var (
	flagTask    = flag.String("task", "search_document", "Task prefix: search_document, search_query, clustering, classification")
	flagDim     = flag.Int("dim", fullDim, "Embedding dimension (768, 512, 256, 128, 64)")
	flagMaxLen  = flag.Int("max-length", 512, "Maximum sequence length for tokenization")
	flagFP16    = flag.Bool("fp16", false, "Use fp16 model variant (274MB instead of 547MB)")
	flagBackend = flag.String("backend", "", "Backend to use (default: auto-detect)")
)

// defaultTexts are embedded when no texts are provided as arguments.
var defaultTexts = []string{
	"GoMLX is a machine learning framework written in Go.",
	"The weather forecast calls for sunny skies tomorrow.",
	"Deep learning models can learn complex patterns from data.",
	"I need to buy groceries for dinner tonight.",
	"Transformer architectures have revolutionized natural language processing.",
	"The cat sat on the mat and purred contentedly.",
}

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	if *flagBackend != "" {
		if err := os.Setenv("GOMLX_BACKEND", *flagBackend); err != nil {
			klog.Warningf("Failed to set backend: %v", err)
		}
	}

	// Validate task prefix.
	validTasks := map[string]bool{
		"search_document": true,
		"search_query":    true,
		"clustering":      true,
		"classification":  true,
	}
	if !validTasks[*flagTask] {
		klog.Fatalf("Invalid task %q; must be one of: search_document, search_query, clustering, classification", *flagTask)
	}

	// Validate Matryoshka dimension.
	validDims := map[int]bool{768: true, 512: true, 256: true, 128: true, 64: true}
	if !validDims[*flagDim] {
		klog.Fatalf("Invalid dim %d; must be one of: 768, 512, 256, 128, 64", *flagDim)
	}

	// Determine texts to embed.
	texts := defaultTexts
	if flag.NArg() > 0 {
		texts = flag.Args()
	}

	// Determine ONNX model file.
	modelFile := "onnx/model.onnx"
	if *flagFP16 {
		modelFile = "onnx/model_fp16.onnx"
	}

	// Download and cache model files from HuggingFace.
	fmt.Printf("Downloading model: %s (%s)\n", modelRepo, modelFile)
	repo := hub.New(modelRepo).WithProgressBar(true)
	if err := repo.DownloadInfo(false); err != nil {
		klog.Fatalf("Failed to get repo info: %+v", err)
	}
	onnxPath, err := repo.DownloadFile(modelFile)
	if err != nil {
		klog.Fatalf("Failed to download %s: %+v", modelFile, err)
	}
	fmt.Printf("Model downloaded: %s\n\n", onnxPath)

	// Load tokenizer.
	tok, err := tokenizers.New(repo)
	if err != nil {
		klog.Fatalf("Failed to create tokenizer: %+v", err)
	}

	// Load ONNX model. WithConstantVariables embeds weights as graph constants,
	// required because the model's rotary embeddings use Range ops that need
	// compile-time constant values.
	model, err := onnx.ReadFile(onnxPath)
	if err != nil {
		klog.Fatalf("Failed to load ONNX model: %+v", err)
	}
	defer model.Close()
	model.WithConstantVariables()

	inputNames, _ := model.Inputs()
	outputNames, _ := model.Outputs()
	fmt.Printf("Model inputs: %v\n", inputNames)
	fmt.Printf("Model outputs: %v\n\n", outputNames)

	// Load model weights into context.
	ctx := context.New()
	if err := model.VariablesToContext(ctx); err != nil {
		klog.Fatalf("Failed to load model variables: %+v", err)
	}

	// Initialize backend.
	backend := backends.MustNew()
	fmt.Printf("Backend: %s\n\n", backend.Name())

	// Check which inputs the model expects.
	hasTokenTypeIDs := false
	for _, name := range inputNames {
		if name == "token_type_ids" {
			hasTokenTypeIDs = true
			break
		}
	}

	// Compute embeddings with dynamic batch shapes.
	embeddings := embedDynamic(backend, ctx, model, tok, *flagTask, texts, *flagMaxLen, *flagDim, hasTokenTypeIDs)

	// Display results.
	fmt.Println("Texts:")
	prefix := *flagTask + ": "
	for i, text := range texts {
		fmt.Printf("  [%d] %q\n", i, prefix+text)
	}
	fmt.Println()

	fmt.Printf("Embeddings (%d-dim):\n", *flagDim)
	for i, emb := range embeddings {
		fmt.Printf("  [%d] [%.4f, %.4f, %.4f, ..., %.4f]  norm=%.4f\n",
			i, emb[0], emb[1], emb[2], emb[len(emb)-1], l2norm(emb))
	}
	fmt.Println()

	// Cosine similarity matrix.
	fmt.Println("Cosine similarity:")
	fmt.Printf("       ")
	for i := range texts {
		fmt.Printf("[%d]    ", i)
	}
	fmt.Println()
	for i, a := range embeddings {
		fmt.Printf("  [%d] ", i)
		for _, b := range embeddings {
			fmt.Printf(" %.3f ", cosineSim(a, b))
		}
		fmt.Println()
	}
}

// tokenizedBatch holds tokenized representations of multiple texts,
// padded to a common sequence length.
type tokenizedBatch struct {
	inputIDs      [][]int // [num_texts, seq_len]
	attentionMask [][]int // [num_texts, seq_len]
	tokenTypeIDs  [][]int // [num_texts, seq_len]
	seqLen        int
}

// tokenizeTexts encodes all texts with task prefix, padded to common length.
// The attention mask distinguishes real tokens (1) from padding (0).
func tokenizeTexts(tok tokenizers.Tokenizer, task string, texts []string, maxLength int) tokenizedBatch {
	clsID, err := tok.SpecialTokenID(api.TokClassification)
	if err != nil {
		clsID = 101
	}
	sepID, err := tok.SpecialTokenID(api.TokEndOfSentence)
	if err != nil {
		sepID = 102
	}

	prefix := task + ": "

	// Tokenize each text and find max sequence length.
	allIDs := make([][]int, len(texts))
	maxSeqLen := 0
	for i, text := range texts {
		tokens := tok.Encode(prefix + text)

		// Build: [CLS] tokens [SEP]
		ids := make([]int, 0, len(tokens)+2)
		ids = append(ids, clsID)
		ids = append(ids, tokens...)
		ids = append(ids, sepID)

		if len(ids) > maxLength {
			ids = ids[:maxLength]
		}
		allIDs[i] = ids
		if len(ids) > maxSeqLen {
			maxSeqLen = len(ids)
		}
	}

	// Pad all texts to maxSeqLen.
	batch := tokenizedBatch{
		inputIDs:      make([][]int, len(texts)),
		attentionMask: make([][]int, len(texts)),
		tokenTypeIDs:  make([][]int, len(texts)),
		seqLen:        maxSeqLen,
	}
	for i, ids := range allIDs {
		batch.inputIDs[i] = make([]int, maxSeqLen)
		copy(batch.inputIDs[i], ids)

		batch.attentionMask[i] = make([]int, maxSeqLen)
		for j := range len(ids) {
			batch.attentionMask[i][j] = 1
		}

		batch.tokenTypeIDs[i] = make([]int, maxSeqLen)
	}

	return batch
}

// embedDynamic computes embeddings using dynamic batch shapes.
// All texts are padded to a common sequence length and processed in sub-batches
// of varying size. The computation graph is compiled once with a symbolic batch
// dimension and specialized per concrete batch size at execution time.
func embedDynamic(backend backends.Backend, ctx *context.Context, model *onnx.Model, tok tokenizers.Tokenizer, task string, texts []string, maxLen, dim int, hasTokenTypeIDs bool) [][]float32 {
	// Tokenize all texts, padded to a common sequence length.
	batch := tokenizeTexts(tok, task, texts, maxLen)
	fmt.Printf("Padded sequence length: %d tokens\n", batch.seqLen)

	// Build a persistent Exec with dynamic batch dimension.
	// The graph is compiled once; each call with a different batch size
	// triggers a lightweight shape specialization rather than full recompilation.
	var exec *context.Exec
	if hasTokenTypeIDs {
		exec = context.MustNewExec(backend, ctx,
			func(ctx *context.Context, inputIDs, attentionMask, tokenTypeIDs *Node) *Node {
				return embeddingGraph(ctx, model, inputIDs, attentionMask, tokenTypeIDs, dim)
			},
		)
	} else {
		exec = context.MustNewExec(backend, ctx,
			func(ctx *context.Context, inputIDs, attentionMask *Node) *Node {
				return embeddingGraph(ctx, model, inputIDs, attentionMask, nil, dim)
			},
		)
	}

	// Mark batch dimension as dynamic. Sequence length is fixed (padded).
	// Each input has shape [batch, seq_len] — axis 0 ("batch") is dynamic, axis 1 is static.
	if hasTokenTypeIDs {
		exec.WithDynamicAxes([]string{"batch", ""}, []string{"batch", ""}, []string{"batch", ""})
	} else {
		exec.WithDynamicAxes([]string{"batch", ""}, []string{"batch", ""})
	}
	defer exec.Finalize()

	// Process texts in sub-batches of increasing size to demonstrate dynamic shapes.
	// The graph compiles once with a symbolic batch dimension and specializes per size.
	result := make([][]float32, len(texts))
	offset := 0
	for batchNum := 0; offset < len(texts); batchNum++ {
		// Increasing batch sizes: 1, 2, 3, ...
		size := batchNum + 1
		if offset+size > len(texts) {
			size = len(texts) - offset
		}

		fmt.Printf("  Sub-batch %d: texts [%d..%d) (batch_size=%d)\n", batchNum, offset, offset+size, size)

		subIDs := batch.inputIDs[offset : offset+size]
		subMask := batch.attentionMask[offset : offset+size]
		subTypes := batch.tokenTypeIDs[offset : offset+size]

		var outputs []*tensors.Tensor
		if hasTokenTypeIDs {
			outputs = exec.MustExec(subIDs, subMask, subTypes)
		} else {
			outputs = exec.MustExec(subIDs, subMask)
		}

		embs := splitEmbeddings(outputs[0])
		for i, emb := range embs {
			result[offset+i] = emb
		}

		offset += size
	}
	fmt.Println()

	return result
}

// embeddingGraph builds the computation graph for embedding generation.
// tokenTypeIDs may be nil if the model doesn't require it.
func embeddingGraph(ctx *context.Context, model *onnx.Model, inputIDs, attentionMask, tokenTypeIDs *Node, dim int) *Node {
	g := inputIDs.Graph()

	// Run ONNX model to get last_hidden_state [batch, seq_len, hidden].
	inputs := map[string]*Node{
		"input_ids":      inputIDs,
		"attention_mask": attentionMask,
	}
	if tokenTypeIDs != nil {
		inputs["token_type_ids"] = tokenTypeIDs
	}
	outputs := model.CallGraph(ctx, g, inputs)
	lastHidden := outputs[0]

	// Mean pooling: average token embeddings weighted by attention mask.
	maskF := ConvertDType(attentionMask, lastHidden.DType())                       // [batch, seq_len]
	expandedMask := ExpandDims(maskF, -1)                                          // [batch, seq_len, 1]
	masked := Mul(lastHidden, expandedMask)                                        // [batch, seq_len, hidden]
	summed := ReduceSum(masked, 1)                                                 // [batch, hidden]
	counts := Max(ReduceSum(expandedMask, 1), Scalar(g, lastHidden.DType(), 1e-9)) // [batch, 1]
	pooled := Div(summed, counts)                                                  // [batch, hidden]

	// Layer normalization (no learned parameters): (x - mean) / sqrt(var + eps).
	mean := ReduceAndKeep(pooled, ReduceMean, -1)               // [batch, 1]
	centered := Sub(pooled, mean)
	variance := ReduceAndKeep(Square(centered), ReduceMean, -1) // [batch, 1]
	normed := Div(centered, Sqrt(AddScalar(variance, 1e-5)))

	// Matryoshka truncation.
	if dim < fullDim {
		normed = Slice(normed, AxisRange(), AxisRange(0, dim))
	}

	// L2 normalization.
	return L2Normalize(normed, -1)
}

// splitEmbeddings converts a [batch, dim] tensor into a slice of per-text embedding vectors.
func splitEmbeddings(t *tensors.Tensor) [][]float32 {
	flat := tensors.MustCopyFlatData[float32](t)
	dims := t.Shape().Dimensions
	batchSize := dims[0]
	embDim := dims[1]

	result := make([][]float32, batchSize)
	for i := range batchSize {
		result[i] = make([]float32, embDim)
		copy(result[i], flat[i*embDim:(i+1)*embDim])
	}
	return result
}

func l2norm(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Sqrt(sum)
}

func cosineSim(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
