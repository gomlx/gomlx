// Package main implements a text generation example using DistilGPT-2.
//
// This demonstrates GoMLX's attention and generation capabilities with a smaller
// transformer model (~82MB), including weight loading, KV caching, and efficient inference.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
)

var (
	flagPrompt       = flag.String("prompt", "Once upon a time", "Input prompt for text generation")
	flagMaxTokens    = flag.Int("max-tokens", 50, "Maximum number of tokens to generate")
	flagTemperature  = flag.Float64("temperature", 0.8, "Sampling temperature (higher = more random)")
	flagTopP         = flag.Float64("top-p", 0.95, "Nucleus sampling threshold")
	flagTopK         = flag.Int("top-k", 0, "Top-k sampling (0 = disabled)")
	flagStrategy     = flag.String("strategy", "temperature", "Sampling strategy: greedy, temperature, top_k, top_p")
	flagDownloadOnly = flag.Bool("download-only", false, "Only download weights, don't run inference")
	flagBackend      = flag.String("backend", "", "Backend to use (default: auto-detect)")
)

func main() {
	flag.Parse()

	// Validate sampling strategy
	validStrategies := map[string]bool{
		"greedy":      true,
		"temperature": true,
		"top_k":       true,
		"top_p":       true,
	}
	if !validStrategies[*flagStrategy] {
		log.Fatalf("Unknown strategy: %s (must be greedy, temperature, top_k, or top_p)", *flagStrategy)
	}

	// Initialize backend
	if *flagBackend != "" {
		err := os.Setenv("GOMLX_BACKEND", *flagBackend)
		if err != nil {
			log.Printf("Warning: failed to set backend: %v", err)
		}
	}
	backend, err := backends.New()
	if err != nil {
		log.Fatalf("Failed to initialize backend: %v", err)
	}
	fmt.Printf("Using backend: %s\n", backend)

	repo := hub.New("distilgpt2")
	if *flagDownloadOnly {
		fmt.Println("Downloading DistilGPT-2 files...")
		files := []string{"model.safetensors", "vocab.json", "merges.txt", "config.json"}
		_, err := repo.DownloadFiles(files...)
		if err != nil {
			log.Fatalf("Failed to download files: %v", err)
		}
		fmt.Println("Download complete. Files cached in ~/.cache/huggingface/hub/")
		return
	}

	// Load model
	fmt.Println("Loading DistilGPT-2 model...")
	model, tokenizer, err := LoadGPT2(backend, repo)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	fmt.Println("Model loaded successfully!")

	// Tokenize prompt
	fmt.Printf("\nPrompt: %q\n", *flagPrompt)
	tokens := tokenizer.Encode(*flagPrompt)
	fmt.Printf("Tokenized to %d tokens\n", len(tokens))

	// Generate text
	fmt.Println("\nGenerating...")
	fmt.Print(*flagPrompt)
	config := GenerationConfig{
		MaxTokens:   *flagMaxTokens,
		Strategy:    *flagStrategy,
		Temperature: *flagTemperature,
		TopK:        *flagTopK,
		TopP:        *flagTopP,
	}
	err = model.Generate(tokens, config, func(token int) bool {
		text := tokenizer.Decode([]int{token})
		fmt.Print(text)
		return true // Continue generation
	})
	if err != nil {
		log.Fatalf("\nGeneration failed: %v", err)
	}

	fmt.Println("\n\nGeneration complete!")
}
