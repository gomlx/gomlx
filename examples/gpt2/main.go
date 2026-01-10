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
	"path/filepath"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
)

var (
	flagPrompt       = flag.String("prompt", "Once upon a time", "Input prompt for text generation")
	flagMaxTokens    = flag.Int("max-tokens", 50, "Maximum number of tokens to generate")
	flagTemperature  = flag.Float64("temperature", 0.8, "Sampling temperature (higher = more random)")
	flagTopP         = flag.Float64("top-p", 0.95, "Nucleus sampling threshold")
	flagCheckpoint   = flag.String("checkpoint", "", "Path to model checkpoint (default: auto-download)")
	flagDownloadOnly = flag.Bool("download-only", false, "Only download weights, don't run inference")
	flagBackend      = flag.String("backend", "", "Backend to use (default: auto-detect)")
)

func main() {
	flag.Parse()

	// Get checkpoint path (download if needed)
	checkpointPath := *flagCheckpoint
	if checkpointPath == "" {
		fmt.Println("Downloading DistilGPT-2 weights from Hugging Face...")
		var err error
		checkpointPath, err = downloadGPT2()
		if err != nil {
			log.Fatalf("Failed to download model: %v", err)
		}
		fmt.Printf("Weights downloaded to: %s\n", checkpointPath)
	}

	if *flagDownloadOnly {
		fmt.Println("Download complete. Use without --download-only to run inference.")
		return
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

	// Load model
	fmt.Println("Loading DistilGPT-2 model...")
	model, tokenizer, err := LoadGPT2(backend, checkpointPath)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	fmt.Println("Model loaded successfully!")

	// Tokenize prompt
	fmt.Printf("\nPrompt: %q\n", *flagPrompt)
	tokens, err := tokenizer.Encode(*flagPrompt)
	if err != nil {
		log.Fatalf("Failed to tokenize prompt: %v", err)
	}
	fmt.Printf("Tokenized to %d tokens\n", len(tokens))

	// Generate text
	fmt.Println("\nGenerating...")
	fmt.Print(*flagPrompt)
	config := GenerationConfig{
		MaxTokens:   *flagMaxTokens,
		Temperature: *flagTemperature,
		TopP:        *flagTopP,
	}
	err = model.Generate(tokens, config, func(token int) bool {
		text, err := tokenizer.Decode([]int{token})
		if err != nil {
			log.Printf("Warning: failed to decode token %d: %v", token, err)
			return true
		}
		fmt.Print(text)
		return true // Continue generation
	})
	if err != nil {
		log.Fatalf("\nGeneration failed: %v", err)
	}

	fmt.Println("\n\nGeneration complete!")
}

// downloadGPT2 downloads the DistilGPT-2 weights from Hugging Face.
// DistilGPT-2 is a distilled version that's 2x smaller (6 layers vs 12).
func downloadGPT2() (string, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to get home directory: %w", err)
	}

	cacheDir := filepath.Join(homeDir, ".cache", "gomlx", "distilgpt2")
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create cache directory: %w", err)
	}

	// Check if already downloaded
	checkpointFile := filepath.Join(cacheDir, "model.safetensors")
	if _, err := os.Stat(checkpointFile); err == nil {
		fmt.Println("Using cached DistilGPT-2 weights")
		return cacheDir, nil
	}

	fmt.Println("Downloading DistilGPT-2 files...")

	baseURL := "https://huggingface.co/distilgpt2/resolve/main"
	files := []string{
		"model.safetensors",
		"vocab.json",
		"merges.txt",
		"config.json",
	}

	for _, file := range files {
		url := fmt.Sprintf("%s/%s", baseURL, file)
		destPath := filepath.Join(cacheDir, file)
		fmt.Printf("  %s...\n", file)
		if err := downloadFile(url, destPath); err != nil {
			return "", fmt.Errorf("failed to download %s: %w", file, err)
		}
	}

	return cacheDir, nil
}
