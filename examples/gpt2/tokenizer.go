package main

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"

	"github.com/gomlx/go-huggingface/hub"
)

// Tokenizer handles text encoding/decoding for GPT-2 models using BPE.
type Tokenizer struct {
	encoder     map[string]int // token -> id
	decoder     map[int]string // id -> token
	bpeRanks    map[string]int // merge pairs -> rank
	pattern     *regexp.Regexp // regex pattern for splitting
	byteEncoder map[byte]rune  // byte -> unicode char
	byteDecoder map[rune]byte  // unicode char -> byte
}

// LoadTokenizer loads the BPE tokenizer from the HuggingFace repository
func LoadTokenizer(repo *hub.Repo) (*Tokenizer, error) {
	// Download vocab.json and merges.txt
	vocabPath, err := repo.DownloadFile("vocab.json")
	if err != nil {
		return nil, fmt.Errorf("failed to download vocab.json: %w", err)
	}

	mergesPath, err := repo.DownloadFile("merges.txt")
	if err != nil {
		return nil, fmt.Errorf("failed to download merges.txt: %w", err)
	}

	// Load vocabulary
	vocabData, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read vocab.json: %w", err)
	}

	var encoder map[string]int
	if err := json.Unmarshal(vocabData, &encoder); err != nil {
		return nil, fmt.Errorf("failed to parse vocab.json: %w", err)
	}

	// Create decoder
	decoder := make(map[int]string, len(encoder))
	for token, id := range encoder {
		decoder[id] = token
	}

	// Load merges
	mergesData, err := os.ReadFile(mergesPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read merges.txt: %w", err)
	}

	bpeRanks := make(map[string]int)
	lines := strings.Split(string(mergesData), "\n")
	for i, line := range lines {
		if i == 0 || strings.TrimSpace(line) == "" {
			continue // Skip header and empty lines
		}
		bpeRanks[line] = i - 1
	}

	// GPT-2 uses a special regex pattern for tokenization
	pattern := regexp.MustCompile(`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`)

	// Create byte encoder/decoder for handling all possible bytes
	byteEncoder, byteDecoder := bytesToUnicode()

	return &Tokenizer{
		encoder:     encoder,
		decoder:     decoder,
		bpeRanks:    bpeRanks,
		pattern:     pattern,
		byteEncoder: byteEncoder,
		byteDecoder: byteDecoder,
	}, nil
}

// bytesToUnicode creates a mapping from bytes to unicode characters
func bytesToUnicode() (map[byte]rune, map[rune]byte) {
	// GPT-2 maps printable bytes to themselves, non-printable to 256+offset
	encoder := make(map[byte]rune, 256)
	decoder := make(map[rune]byte, 256)

	// Helper to check if byte should map to itself
	isPrintable := func(b byte) bool {
		return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)
	}

	offset := 0
	for b := 0; b < 256; b++ {
		var r rune
		if isPrintable(byte(b)) {
			r = rune(b)
		} else {
			r = rune(256 + offset)
			offset++
		}
		encoder[byte(b)] = r
		decoder[r] = byte(b)
	}

	return encoder, decoder
}

// bpe performs byte-pair encoding on a token
func (t *Tokenizer) bpe(token string) string {
	if len(token) < 2 {
		return token
	}

	// Start with individual characters
	word := make([]string, 0, len(token))
	for _, r := range token {
		word = append(word, string(r))
	}

	for len(word) > 1 {
		// Find best pair to merge
		bestIdx := -1
		bestRank := int(^uint(0) >> 1)

		for i := 0; i < len(word)-1; i++ {
			pair := word[i] + " " + word[i+1]
			if rank, ok := t.bpeRanks[pair]; ok && rank < bestRank {
				bestRank = rank
				bestIdx = i
			}
		}

		if bestIdx == -1 {
			break
		}

		// Merge the best pair in place
		word[bestIdx] = word[bestIdx] + word[bestIdx+1]
		word = append(word[:bestIdx+1], word[bestIdx+2:]...)
	}

	return strings.Join(word, " ")
}

// Encode converts text to token IDs
func (t *Tokenizer) Encode(text string) ([]int, error) {
	tokens := []int{}

	// Find all matches using the pattern
	matches := t.pattern.FindAllString(text, -1)

	for _, match := range matches {
		// Convert to bytes and then to unicode chars
		encoded := ""
		for i := 0; i < len(match); i++ {
			encoded += string(t.byteEncoder[match[i]])
		}

		// Apply BPE
		bpeToken := t.bpe(encoded)

		// Split and look up token IDs
		for _, token := range strings.Split(bpeToken, " ") {
			if id, ok := t.encoder[token]; ok {
				tokens = append(tokens, id)
			}
		}
	}

	return tokens, nil
}

// Decode converts token IDs back to text
func (t *Tokenizer) Decode(tokens []int) (string, error) {
	text := ""
	for _, id := range tokens {
		if token, ok := t.decoder[id]; ok {
			text += token
		}
	}

	// Convert unicode chars back to bytes
	bytes := []byte{}
	for _, r := range text {
		if b, ok := t.byteDecoder[r]; ok {
			bytes = append(bytes, b)
		}
	}

	return string(bytes), nil
}

// VocabSize returns the vocabulary size
func (t *Tokenizer) VocabSize() int {
	return len(t.encoder)
}
