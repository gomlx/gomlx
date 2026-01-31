// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// mxbai-rerank demonstrates cross-encoder reranking using GoMLX.
//
// It downloads the mixedbread-ai/mxbai-rerank-base-v1 ONNX model from HuggingFace,
// tokenizes query-document pairs, and runs inference to compute relevance scores.
//
// Usage:
//
//	go run mxbai-rerank.go
//	go run mxbai-rerank.go --query="What is deep learning?"
//	go run mxbai-rerank.go --max_length=256
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"sort"

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
	// HuggingFace repository for the reranker model.
	modelRepo = "mixedbread-ai/mxbai-rerank-base-v1"

	// ONNX model filename within the repository.
	modelFile = "onnx/model.onnx"
)

var (
	flagQuery     = flag.String("query", "What is machine learning?", "Query text to rerank documents against.")
	flagMaxLength = flag.Int("max_length", 512, "Maximum sequence length for tokenization.")
)

// documents to rerank against the query.
var defaultDocuments = []string{
	"Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
	"The weather today is sunny with a chance of rain.",
	"Deep learning uses neural networks with many layers to learn representations of data.",
	"Cooking pasta requires boiling water and adding salt.",
	"Natural language processing applies machine learning to understand human language.",
	"The stock market closed higher today driven by tech sector gains.",
}

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	// Download and cache model files from HuggingFace.
	fmt.Printf("Downloading model: %s\n", modelRepo)
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

	// Load ONNX model.
	model, err := onnx.ReadFile(onnxPath)
	if err != nil {
		klog.Fatalf("Failed to load ONNX model: %+v", err)
	}
	defer model.Close()

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

	// Tokenize query-document pairs and run inference.
	query := *flagQuery
	documents := defaultDocuments

	fmt.Printf("Query: %q\n\n", query)

	scores := rerank(backend, ctx, model, tok, query, documents)

	// Display results sorted by score.
	type result struct {
		index int
		score float32
		text  string
	}
	results := make([]result, len(documents))
	for i, doc := range documents {
		results[i] = result{index: i, score: scores[i], text: doc}
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	fmt.Println("Reranked results (highest relevance first):")
	fmt.Println()
	for rank, r := range results {
		fmt.Printf("  %d. [%.4f] %s\n", rank+1, r.score, r.text)
	}
}

// rerank computes relevance scores for a query against multiple documents
// using a cross-encoder model.
func rerank(backend backends.Backend, ctx *context.Context, model *onnx.Model, tok tokenizers.Tokenizer, query string, documents []string) []float32 {
	// Tokenize each query-document pair.
	pairs := encodePairs(tok, query, documents, *flagMaxLength)

	// Build input tensors.
	batchSize := len(documents)
	seqLen := len(pairs.inputIDs[0])

	flatInputIDs := make([]int64, batchSize*seqLen)
	flatAttentionMask := make([]int64, batchSize*seqLen)
	flatTokenTypeIDs := make([]int64, batchSize*seqLen)
	for i := range batchSize {
		for j := range seqLen {
			flatInputIDs[i*seqLen+j] = int64(pairs.inputIDs[i][j])
			flatAttentionMask[i*seqLen+j] = int64(pairs.attentionMask[i][j])
			flatTokenTypeIDs[i*seqLen+j] = int64(pairs.tokenTypeIDs[i][j])
		}
	}

	inputIDsTensor := tensors.FromFlatDataAndDimensions(flatInputIDs, batchSize, seqLen)
	attentionMaskTensor := tensors.FromFlatDataAndDimensions(flatAttentionMask, batchSize, seqLen)
	tokenTypeIDsTensor := tensors.FromFlatDataAndDimensions(flatTokenTypeIDs, batchSize, seqLen)

	// Determine which inputs the model expects.
	inputNames, _ := model.Inputs()
	hasTokenTypeIDs := false
	for _, name := range inputNames {
		if name == "token_type_ids" {
			hasTokenTypeIDs = true
			break
		}
	}

	// Run inference.
	var output *tensors.Tensor
	if hasTokenTypeIDs {
		output = context.MustExecOnce(
			backend, ctx,
			func(ctx *context.Context, inputIDs, attentionMask, tokenTypeIDs *Node) *Node {
				g := inputIDs.Graph()
				outputs := model.CallGraph(ctx, g, map[string]*Node{
					"input_ids":      inputIDs,
					"attention_mask": attentionMask,
					"token_type_ids": tokenTypeIDs,
				})
				return outputs[0]
			},
			inputIDsTensor, attentionMaskTensor, tokenTypeIDsTensor,
		)
	} else {
		output = context.MustExecOnce(
			backend, ctx,
			func(ctx *context.Context, inputIDs, attentionMask *Node) *Node {
				g := inputIDs.Graph()
				outputs := model.CallGraph(ctx, g, map[string]*Node{
					"input_ids":      inputIDs,
					"attention_mask": attentionMask,
				})
				return outputs[0]
			},
			inputIDsTensor, attentionMaskTensor,
		)
	}

	// Extract scores from logits.
	return extractScores(output)
}

// encodedPairs holds the tokenized and padded batch of query-document pairs.
type encodedPairs struct {
	inputIDs      [][]int
	attentionMask [][]int
	tokenTypeIDs  [][]int
}

// encodePairs tokenizes query-document pairs for a cross-encoder model.
// Each pair is encoded as: [CLS] query_tokens [SEP] document_tokens [SEP]
// with appropriate token type IDs (0 for query segment, 1 for document segment).
func encodePairs(tok tokenizers.Tokenizer, query string, documents []string, maxLength int) encodedPairs {
	clsID, err := tok.SpecialTokenID(api.TokClassification)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: no [CLS] token found, using 101\n")
		clsID = 101
	}
	sepID, err := tok.SpecialTokenID(api.TokEndOfSentence)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: no [SEP] token found, using 102\n")
		sepID = 102
	}
	padID, err := tok.SpecialTokenID(api.TokPad)
	if err != nil {
		padID = 0
	}

	queryTokens := tok.Encode(query)

	var allInputIDs [][]int
	var allTokenTypeIDs [][]int
	maxLen := 0

	for _, doc := range documents {
		docTokens := tok.Encode(doc)

		// Build: [CLS] query [SEP] doc [SEP]
		combined := make([]int, 0, 3+len(queryTokens)+len(docTokens))
		combined = append(combined, clsID)
		combined = append(combined, queryTokens...)
		combined = append(combined, sepID)
		combined = append(combined, docTokens...)
		combined = append(combined, sepID)

		// Build token type IDs: 0 for query segment, 1 for document segment.
		typeIDs := make([]int, len(combined))
		// First segment (CLS + query + SEP) is type 0 (default).
		// Second segment (doc + SEP) is type 1.
		docStart := 1 + len(queryTokens) + 1 // after [CLS] query [SEP]
		for i := docStart; i < len(typeIDs); i++ {
			typeIDs[i] = 1
		}

		// Truncate if needed.
		if len(combined) > maxLength {
			combined = combined[:maxLength]
			typeIDs = typeIDs[:maxLength]
		}

		allInputIDs = append(allInputIDs, combined)
		allTokenTypeIDs = append(allTokenTypeIDs, typeIDs)
		if len(combined) > maxLen {
			maxLen = len(combined)
		}
	}

	// Pad all sequences to the same length.
	padded := encodedPairs{
		inputIDs:      make([][]int, len(documents)),
		attentionMask: make([][]int, len(documents)),
		tokenTypeIDs:  make([][]int, len(documents)),
	}
	for i := range documents {
		seqLen := len(allInputIDs[i])
		ids := make([]int, maxLen)
		mask := make([]int, maxLen)
		types := make([]int, maxLen)

		copy(ids, allInputIDs[i])
		copy(types, allTokenTypeIDs[i])
		for j := 0; j < seqLen; j++ {
			mask[j] = 1
		}
		for j := seqLen; j < maxLen; j++ {
			ids[j] = padID
		}

		padded.inputIDs[i] = ids
		padded.attentionMask[i] = mask
		padded.tokenTypeIDs[i] = types
	}

	return padded
}

// extractScores converts model output logits to relevance scores.
// For single-label output (regression): uses logit directly.
// For binary classification (2 labels): applies softmax and takes positive class probability.
func extractScores(output *tensors.Tensor) []float32 {
	shape := output.Shape()
	dims := shape.Dimensions

	if len(dims) == 1 {
		// Shape [batch] - scores directly.
		return tensors.MustCopyFlatData[float32](output)
	}

	// Shape [batch, num_labels]
	batchSize := dims[0]
	numLabels := dims[1]
	flat := tensors.MustCopyFlatData[float32](output)

	scores := make([]float32, batchSize)
	for i := range batchSize {
		offset := i * numLabels
		if numLabels == 1 {
			scores[i] = flat[offset]
		} else if numLabels == 2 {
			// Softmax over [negative, positive] and take positive class.
			scores[i] = softmax(flat[offset], flat[offset+1])
		} else {
			scores[i] = flat[offset]
		}
	}
	return scores
}

// softmax returns the probability of the positive class (second logit) given two logits.
func softmax(neg, pos float32) float32 {
	maxL := neg
	if pos > maxL {
		maxL = pos
	}
	expNeg := math.Exp(float64(neg - maxL))
	expPos := math.Exp(float64(pos - maxL))
	return float32(expPos / (expNeg + expPos))
}
