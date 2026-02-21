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
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"
	"unicode"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
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
	labelNames = []string{
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

	// Load vocabulary for tokenizer.
	// Note: This model uses an older HuggingFace format without tokenizer.json or tokenizer_class,
	// so we use a simple inline WordPiece tokenizer instead of github.com/gomlx/go-huggingface/tokenizers.
	vocabPath := must.M1(repo.DownloadFile("vocab.txt"))
	tok := must.M1(newTokenizer(vocabPath))

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
	tokenIDs, tokens := tok.encode(*flagText)

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

	// Run inference with argmax and softmax computed in the graph
	outputs := context.MustExecOnceN(
		backend, ctx,
		func(ctx *context.Context, inputs []*Node) []*Node {
			g := inputs[0].Graph()
			logits := onnxModel.CallGraph(ctx, g,
				map[string]*Node{
					"input_ids":      inputs[0],
					"attention_mask": inputs[1],
					"token_type_ids": inputs[2],
				},
				"logits")
			// Compute predictions and scores in the graph (more efficient than doing it in Go)
			predictions := ArgMax(logits[0], -1)
			scores := Softmax(logits[0])
			return []*Node{predictions, scores}
		},
		inputIDs, attentionMask, tokenTypeIDs)

	// Extract predictions and scores from tensors (using MustCopyFlatData which returns a Go-safe copy)
	predictions := tensors.MustCopyFlatData[int32](outputs[0])
	scores := tensors.MustCopyFlatData[float32](outputs[1])

	// Extract entities from predictions
	entities := extractEntities(tokens, predictions, scores, len(labelNames))

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

// tokenizer implements a simple WordPiece tokenizer for BERT.
type tokenizer struct {
	vocab map[string]int
	clsID int
	sepID int
	unkID int
}

func newTokenizer(vocabPath string) (*tokenizer, error) {
	file, err := os.Open(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open vocab file: %w", err)
	}
	defer file.Close()

	vocab := make(map[string]int)
	scanner := bufio.NewScanner(file)
	idx := 0
	for scanner.Scan() {
		vocab[scanner.Text()] = idx
		idx++
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("failed to read vocab file: %w", err)
	}

	t := &tokenizer{vocab: vocab}
	if id, ok := vocab["[CLS]"]; ok {
		t.clsID = id
	}
	if id, ok := vocab["[SEP]"]; ok {
		t.sepID = id
	}
	if id, ok := vocab["[UNK]"]; ok {
		t.unkID = id
	}
	return t, nil
}

func (t *tokenizer) encode(text string) ([]int, []string) {
	// Basic tokenization: split on whitespace and punctuation
	words := t.basicTokenize(text)

	// WordPiece tokenization
	var allTokens []string
	var allIDs []int

	allTokens = append(allTokens, "[CLS]")
	allIDs = append(allIDs, t.clsID)

	for _, word := range words {
		subTokens := t.wordPieceTokenize(word)
		for _, st := range subTokens {
			allTokens = append(allTokens, st)
			if id, ok := t.vocab[st]; ok {
				allIDs = append(allIDs, id)
			} else {
				allIDs = append(allIDs, t.unkID)
			}
		}
	}

	allTokens = append(allTokens, "[SEP]")
	allIDs = append(allIDs, t.sepID)

	return allIDs, allTokens
}

func (t *tokenizer) basicTokenize(text string) []string {
	var words []string
	var current strings.Builder

	for _, r := range text {
		if unicode.IsSpace(r) {
			if current.Len() > 0 {
				words = append(words, current.String())
				current.Reset()
			}
		} else if unicode.IsPunct(r) {
			if current.Len() > 0 {
				words = append(words, current.String())
				current.Reset()
			}
			words = append(words, string(r))
		} else {
			current.WriteRune(r)
		}
	}
	if current.Len() > 0 {
		words = append(words, current.String())
	}
	return words
}

func (t *tokenizer) wordPieceTokenize(word string) []string {
	if len(word) == 0 {
		return nil
	}

	var tokens []string
	start := 0
	runes := []rune(word)

	for start < len(runes) {
		end := len(runes)
		var curToken string
		found := false

		for end > start {
			substr := string(runes[start:end])
			if start > 0 {
				substr = "##" + substr
			}
			if _, ok := t.vocab[substr]; ok {
				curToken = substr
				found = true
				break
			}
			end--
		}

		if !found {
			return []string{"[UNK]"}
		}
		tokens = append(tokens, curToken)
		start = end
	}
	return tokens
}

// extractEntities converts model predictions to entity spans using BIO tagging.
func extractEntities(tokens []string, predictions []int32, scores []float32, numLabels int) []Entity {
	var entities []Entity
	var currentTokens []string
	var currentEntityType string
	var scoreSum float32
	var scoreCount int

	seqLen := len(tokens)

	for i := 0; i < seqLen; i++ {
		token := strings.TrimSpace(tokens[i])

		// Skip special tokens
		if token == "[CLS]" || token == "[SEP]" || token == "[PAD]" || token == "" {
			continue
		}

		// Get prediction and its score
		predIdx := int(predictions[i])
		prediction := labelNames[predIdx]
		score := scores[i*numLabels+predIdx]

		if prediction == "O" {
			if len(currentTokens) > 0 {
				entities = append(entities, Entity{
					Text:  reconstructText(currentTokens),
					Label: currentEntityType,
					Score: scoreSum / float32(scoreCount),
				})
				currentTokens = nil
			}
		} else if strings.HasPrefix(prediction, "B-") {
			if len(currentTokens) > 0 {
				entities = append(entities, Entity{
					Text:  reconstructText(currentTokens),
					Label: currentEntityType,
					Score: scoreSum / float32(scoreCount),
				})
			}
			currentEntityType = prediction[2:]
			currentTokens = []string{token}
			scoreSum = score
			scoreCount = 1
		} else if strings.HasPrefix(prediction, "I-") {
			entityType := prediction[2:]
			if len(currentTokens) > 0 && currentEntityType == entityType {
				currentTokens = append(currentTokens, token)
				scoreSum += score
				scoreCount++
			} else {
				if len(currentTokens) > 0 {
					entities = append(entities, Entity{
						Text:  reconstructText(currentTokens),
						Label: currentEntityType,
						Score: scoreSum / float32(scoreCount),
					})
				}
				currentEntityType = entityType
				currentTokens = []string{token}
				scoreSum = score
				scoreCount = 1
			}
		}
	}

	if len(currentTokens) > 0 {
		entities = append(entities, Entity{
			Text:  reconstructText(currentTokens),
			Label: currentEntityType,
			Score: scoreSum / float32(scoreCount),
		})
	}

	return entities
}

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
