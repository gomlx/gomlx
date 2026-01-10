package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
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

// LoadTokenizer loads the BPE tokenizer from vocab.json and merges.txt
func LoadTokenizer(checkpointPath string) (*Tokenizer, error) {
	vocabPath := filepath.Join(checkpointPath, "vocab.json")
	mergesPath := filepath.Join(checkpointPath, "merges.txt")

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
	// GPT-2 uses a clever byte encoding to handle all 256 byte values
	// Build list of printable ASCII bytes
	bs := make([]byte, 0, 256)
	// ASCII printable: !, ", #, ..., ~
	for i := 33; i <= 126; i++ { // '!' to '~'
		bs = append(bs, byte(i))
	}
	// Extended ASCII: ¡ to ¬ (bytes 161-172)
	for i := 161; i <= 172; i++ {
		bs = append(bs, byte(i))
	}
	// Extended ASCII: ® to ÿ (bytes 174-255)
	for i := 174; i <= 255; i++ {
		bs = append(bs, byte(i))
	}

	cs := make([]rune, len(bs))
	for i, b := range bs {
		cs[i] = rune(b)
	}

	// Use a map for O(1) lookup instead of O(n) contains check
	bsSet := make(map[byte]bool, len(bs))
	for _, b := range bs {
		bsSet[b] = true
	}

	n := 0
	for b := 0; b < 256; b++ {
		if !bsSet[byte(b)] {
			bs = append(bs, byte(b))
			cs = append(cs, rune(256+n))
			n++
		}
	}

	encoder := make(map[byte]rune, 256)
	decoder := make(map[rune]byte, 256)
	for i := 0; i < 256; i++ {
		encoder[bs[i]] = cs[i]
		decoder[cs[i]] = bs[i]
	}

	return encoder, decoder
}

// bpe performs byte-pair encoding on a token
func (t *Tokenizer) bpe(token string) string {
	if len(token) < 2 {
		return token
	}

	// Split into characters
	word := []string{}
	for _, r := range token {
		word = append(word, string(r))
	}

	pairs := getPairs(word)
	if len(pairs) == 0 {
		return token
	}

	for {
		// Find the pair with the lowest rank
		minPair := ""
		minRank := int(^uint(0) >> 1) // max int

		for _, pair := range pairs {
			if rank, ok := t.bpeRanks[pair]; ok {
				if rank < minRank {
					minRank = rank
					minPair = pair
				}
			}
		}

		if minPair == "" {
			break
		}

		// Merge the pair
		first := strings.Split(minPair, " ")[0]
		second := strings.Split(minPair, " ")[1]

		newWord := []string{}
		i := 0
		for i < len(word) {
			if i < len(word)-1 && word[i] == first && word[i+1] == second {
				newWord = append(newWord, first+second)
				i += 2
			} else {
				newWord = append(newWord, word[i])
				i++
			}
		}

		word = newWord
		if len(word) == 1 {
			break
		}
		pairs = getPairs(word)
	}

	return strings.Join(word, " ")
}

func getPairs(word []string) []string {
	pairs := []string{}
	for i := 0; i < len(word)-1; i++ {
		pairs = append(pairs, word[i]+" "+word[i+1])
	}
	return pairs
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
