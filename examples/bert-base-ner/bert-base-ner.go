// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// bert-base-ner demonstrates Named Entity Recognition using the dslim/bert-base-NER
// ONNX model from HuggingFace.
//
// It extracts entities of types: PER (person), ORG (organization), LOC (location), MISC.
//
// Usage:
//
//	go build -o /tmp/bert-base-ner ./examples/bert-base-ner && /tmp/bert-base-ner
//	go build -o /tmp/bert-base-ner ./examples/bert-base-ner && /tmp/bert-base-ner --text="Apple Inc. is based in Cupertino"
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/tokenizers"
	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/onnx-gomlx/onnx"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"
)

var (
	flagText = flag.String("text", "My name is Wolfgang and I live in Berlin",
		"Text to analyze for named entities")

	// HuggingFace model repository
	hfModelID = "dslim/bert-base-NER"

	// Labels for the NER model (BIO tagging scheme from config.json)
	labels = []string{
		"O",      // 0: Outside
		"B-MISC", // 1: Beginning of miscellaneous
		"I-MISC", // 2: Inside miscellaneous
		"B-PER",  // 3: Beginning of person
		"I-PER",  // 4: Inside person
		"B-ORG",  // 5: Beginning of organization
		"I-ORG",  // 6: Inside organization
		"B-LOC",  // 7: Beginning of location
		"I-LOC",  // 8: Inside location
	}
)

// Entity represents a named entity found in the text.
type Entity struct {
	Text  string
	Label string
	Score float32
}

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	// Download model from HuggingFace
	fmt.Printf("Loading model %s...\n", hfModelID)
	hfAuthToken := os.Getenv("HF_TOKEN")
	repo := hub.New(hfModelID).WithAuth(hfAuthToken)

	// Load tokenizer
	tok, err := tokenizers.New(repo)
	if err != nil {
		klog.Fatalf("Failed to load tokenizer: %v", err)
	}

	// Load ONNX model
	onnxPath := must.M1(repo.DownloadFile("onnx/model.onnx"))
	onnxModel := must.M1(onnx.ReadFile(onnxPath))

	// Create context and load model weights
	ctx := context.New()
	must.M(onnxModel.VariablesToContext(ctx))

	// Create backend
	backend := must.M1(backends.New())
	defer backend.Finalize()

	// Tokenize input
	fmt.Printf("\nInput: %s\n", *flagText)
	tokenIDs := tok.Encode(*flagText)

	// Get special token IDs
	clsID, _ := tok.SpecialTokenID(api.TokBeginningOfSentence)
	sepID, _ := tok.SpecialTokenID(api.TokEndOfSentence)

	// Add [CLS] and [SEP] if not already present
	if len(tokenIDs) == 0 || tokenIDs[0] != clsID {
		tokenIDs = append([]int{clsID}, tokenIDs...)
	}
	if tokenIDs[len(tokenIDs)-1] != sepID {
		tokenIDs = append(tokenIDs, sepID)
	}

	// Decode each token to get the token strings
	tokens := make([]string, len(tokenIDs))
	for i, id := range tokenIDs {
		tokens[i] = tok.Decode([]int{id})
	}

	// Prepare tensors
	seqLen := len(tokenIDs)
	inputIDs := make([][]int64, 1)
	attentionMask := make([][]int64, 1)
	tokenTypeIDs := make([][]int64, 1)

	inputIDs[0] = make([]int64, seqLen)
	attentionMask[0] = make([]int64, seqLen)
	tokenTypeIDs[0] = make([]int64, seqLen)

	for i, id := range tokenIDs {
		inputIDs[0][i] = int64(id)
		attentionMask[0][i] = 1
		tokenTypeIDs[0][i] = 0
	}

	// Run inference
	outputs := context.MustExecOnceN(
		backend, ctx,
		func(ctx *context.Context, inputs []*Node) []*Node {
			g := inputs[0].Graph()
			return onnxModel.CallGraph(ctx, g,
				map[string]*Node{
					"input_ids":      inputs[0],
					"attention_mask": inputs[1],
					"token_type_ids": inputs[2],
				},
				"logits")
		},
		inputIDs, attentionMask, tokenTypeIDs)

	// Process logits
	logitsTensor := outputs[0]
	var logits []float32
	logitsTensor.MustConstFlatData(func(flat any) {
		logits = flat.([]float32)
	})

	// Extract entities from predictions
	entities := extractEntities(tokens, logits, len(labels))

	// Print results
	if len(entities) == 0 {
		fmt.Println("\nNo named entities found.")
	} else {
		fmt.Printf("\nFound %d entities:\n", len(entities))
		for _, e := range entities {
			fmt.Printf("  %s => %s (confidence: %.2f)\n", e.Text, e.Label, e.Score)
		}
	}
}

// extractEntities converts model logits to entity spans using BIO tagging.
func extractEntities(tokens []string, logits []float32, numLabels int) []Entity {
	var entities []Entity
	var currentTokens []string
	var currentLabel string
	var scoreSum float32
	var scoreCount int

	seqLen := len(tokens)

	for i := 0; i < seqLen; i++ {
		token := strings.TrimSpace(tokens[i])

		// Skip special tokens
		if token == "[CLS]" || token == "[SEP]" || token == "[PAD]" || token == "" {
			continue
		}

		// Get predicted label (argmax)
		maxIdx := 0
		maxVal := logits[i*numLabels]
		for j := 1; j < numLabels; j++ {
			if logits[i*numLabels+j] > maxVal {
				maxVal = logits[i*numLabels+j]
				maxIdx = j
			}
		}
		label := labels[maxIdx]
		score := softmax(logits[i*numLabels:(i+1)*numLabels], maxIdx)

		if label == "O" {
			// Finalize current entity if any
			if len(currentTokens) > 0 {
				entities = append(entities, Entity{
					Text:  reconstructText(currentTokens),
					Label: currentLabel,
					Score: scoreSum / float32(scoreCount),
				})
				currentTokens = nil
			}
		} else if strings.HasPrefix(label, "B-") {
			// Start new entity
			if len(currentTokens) > 0 {
				entities = append(entities, Entity{
					Text:  reconstructText(currentTokens),
					Label: currentLabel,
					Score: scoreSum / float32(scoreCount),
				})
			}
			currentLabel = label[2:]
			currentTokens = []string{token}
			scoreSum = score
			scoreCount = 1
		} else if strings.HasPrefix(label, "I-") {
			entityType := label[2:]
			if len(currentTokens) > 0 && currentLabel == entityType {
				// Continue current entity
				currentTokens = append(currentTokens, token)
				scoreSum += score
				scoreCount++
			} else {
				// I- without matching B-, treat as new entity
				if len(currentTokens) > 0 {
					entities = append(entities, Entity{
						Text:  reconstructText(currentTokens),
						Label: currentLabel,
						Score: scoreSum / float32(scoreCount),
					})
				}
				currentLabel = entityType
				currentTokens = []string{token}
				scoreSum = score
				scoreCount = 1
			}
		}
	}

	// Finalize remaining entity
	if len(currentTokens) > 0 {
		entities = append(entities, Entity{
			Text:  reconstructText(currentTokens),
			Label: currentLabel,
			Score: scoreSum / float32(scoreCount),
		})
	}

	return entities
}

// reconstructText rebuilds text from WordPiece tokens.
func reconstructText(tokens []string) string {
	var sb strings.Builder
	for i, token := range tokens {
		if strings.HasPrefix(token, "##") {
			sb.WriteString(token[2:])
		} else {
			if i > 0 {
				sb.WriteString(" ")
			}
			sb.WriteString(token)
		}
	}
	return sb.String()
}

// softmax computes the softmax probability for a specific index.
func softmax(logits []float32, idx int) float32 {
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	var sum float32
	for _, v := range logits {
		sum += float32(math.Exp(float64(v - maxVal)))
	}
	return float32(math.Exp(float64(logits[idx]-maxVal))) / sum
}
