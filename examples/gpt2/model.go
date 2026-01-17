package main

import (
	"fmt"
	"time"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"
	"github.com/gomlx/gomlx/pkg/ml/layers/generation"
)

// GPT2Config holds the architecture configuration for GPT-2.
type GPT2Config struct {
	VocabSize   int
	HiddenSize  int
	NumLayers   int
	NumHeads    int
	MaxPosEmbed int
	NormEps     float64
	DType       dtypes.DType

	// Sampling configuration
	Strategy    string
	Temperature float64
	TopK        int
	TopP        float64
}

// DefaultGPT2Config returns the configuration for DistilGPT-2.
// DistilGPT-2 has 6 layers (half of GPT-2) making it ~82MB instead of ~500MB.
func DefaultGPT2Config() GPT2Config {
	return GPT2Config{
		VocabSize:   50257,
		HiddenSize:  768,
		NumLayers:   6, // DistilGPT-2 has 6 layers
		NumHeads:    12,
		MaxPosEmbed: 1024,
		NormEps:     1e-5,
		DType:       dtypes.Float32,
	}
}

// GPT2Model represents a loaded GPT-2 model ready for inference.
type GPT2Model struct {
	backend backends.Backend
	ctx     *context.Context
	config  GPT2Config

	// Generation components
	transformer *GPT2TransformerWrapper

	// Single cached exec for all token generations
	tokenExec *context.Exec
}

// GPT2TransformerWrapper wraps the transformer configuration for generation
type GPT2TransformerWrapper struct {
	config     *generation.TransformerConfig
	gpt2Config GPT2Config
	ctx        *context.Context
	Caches     []*attention.KVCache
}

// ForGenerationDynamic returns a model function that accepts position as a Node parameter.
// This enables graph caching by making position a graph input rather than a captured variable.
// The same compiled graph can be reused for all positions, dramatically improving performance.
func (w *GPT2TransformerWrapper) ForGenerationDynamic() func(ctx *context.Context, tokens *Node, positionNode *Node) *Node {
	return func(ctx *context.Context, tokens *Node, positionNode *Node) *Node {
		cfg := w.config
		g := tokens.Graph()
		currentSeqLen := tokens.Shape().Dimensions[1]

		// Token + positional embeddings
		embedded := layers.Embedding(ctx.In("token_embed"), tokens, cfg.DType, cfg.VocabSize, cfg.EmbedDim)
		if embedded.Rank() == 2 {
			embedded = ExpandDims(embedded, 1)
		}

		posEmbedFull := ctx.In("pos_embed").VariableWithShape("embeddings",
			shapes.Make(cfg.DType, cfg.MaxPosEmbed, cfg.EmbedDim)).ValueGraph(g)
		posScalar := Squeeze(ConvertDType(positionNode, dtypes.Int32))
		posEmbed := DynamicSlice(posEmbedFull, []*Node{posScalar, Const(g, int32(0))}, []int{currentSeqLen, cfg.EmbedDim})
		posEmbed = BroadcastToShape(ExpandDims(posEmbed, 0), embedded.Shape())
		x := Add(embedded, posEmbed)

		// Transformer layers (pre-norm architecture)
		for layer := 0; layer < cfg.NumLayers; layer++ {
			layerCtx := ctx.In(fmt.Sprintf("layer_%d", layer))

			// Pre-attention LayerNorm
			attnInput := layers.LayerNormalization(layerCtx.In("norm1"), x, -1).
				Epsilon(w.gpt2Config.NormEps).Done()

			// Self-attention with KV cache
			attn := attention.SelfAttention(layerCtx.In("attn"), attnInput, cfg.NumHeads, cfg.HeadDim).
				WithKVCache(w.Caches[layer], positionNode).
				UseProjectionBias(true).
				UseCausalMask().
				Done()
			x = Add(x, attn)

			// Pre-MLP LayerNorm
			ffInput := layers.LayerNormalization(layerCtx.In("norm2"), x, -1).
				Epsilon(w.gpt2Config.NormEps).Done()

			// Feed-forward network
			ff := layers.Dense(layerCtx.In("ff1"), ffInput, true, cfg.FFNDim)
			ff = activations.Gelu(ff)
			ff = layers.Dense(layerCtx.In("ff2"), ff, true, cfg.EmbedDim)
			x = Add(x, ff)
		}

		// Final layer norm
		x = layers.LayerNormalization(ctx.In("final_norm"), x, -1).
			Epsilon(w.gpt2Config.NormEps).Done()

		// Output projection
		return layers.Dense(ctx.In("output"), x, false, cfg.VocabSize)
	}
}

// LoadGPT2 loads the GPT-2 model from the given HuggingFace repository.
func LoadGPT2(backend backends.Backend, repo *hub.Repo) (*GPT2Model, *Tokenizer, error) {
	config := DefaultGPT2Config()

	// Create context for model parameters
	ctx := context.New()

	// Build the transformer architecture
	// Note: We use BuildCachedTransformer's structure but will add final norm ourselves
	transformerConfig := generation.NewTransformerConfig(
		config.VocabSize,
		config.HiddenSize,
		config.NumLayers,
		config.NumHeads,
		config.HiddenSize/config.NumHeads, // head_dim
	).
		WithFFNDim(config.HiddenSize * 4). // GPT-2 uses 4x expansion
		WithMaxPosEmbed(config.MaxPosEmbed).
		WithDType(config.DType).
		WithLayerNorm(true). // GPT-2 uses layer normalization
		WithBias(true)       // GPT-2 uses bias in dense layers

	transformer := &GPT2TransformerWrapper{
		config:     transformerConfig,
		gpt2Config: config,
		ctx:        ctx,
	}

	// Load checkpoint weights
	fmt.Println("Loading checkpoint weights...")
	if err := loadCheckpoint(ctx, repo); err != nil {
		fmt.Printf("Warning: Failed to load checkpoint: %v\n", err)
		fmt.Println("Continuing with random initialization...")
	} else {
		fmt.Println("✓ Checkpoint loaded successfully")
	}

	model := &GPT2Model{
		backend:     backend,
		ctx:         ctx,
		config:      config,
		transformer: transformer,
	}

	// Load tokenizer
	tokenizer, err := LoadTokenizer(repo)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	return model, tokenizer, nil
}

// GenerationConfig holds parameters for text generation.
type GenerationConfig struct {
	MaxTokens   int
	Strategy    string
	Temperature float64
	TopK        int
	TopP        float64
}

// Generate performs autoregressive text generation.
func (m *GPT2Model) Generate(
	promptTokens []int,
	config GenerationConfig,
	callback func(token int) bool,
) error {
	// Copy sampling config to model for use in exec graphs
	m.config.Strategy = config.Strategy
	m.config.Temperature = config.Temperature
	m.config.TopK = config.TopK
	m.config.TopP = config.TopP

	// Initialize KV caches for each layer
	batchSize := 1
	headDim := m.config.HiddenSize / m.config.NumHeads
	m.transformer.Caches = make([]*attention.KVCache, m.config.NumLayers)
	for i := 0; i < m.config.NumLayers; i++ {
		m.transformer.Caches[i] = attention.NewKVCache(
			m.ctx.In(fmt.Sprintf("layer_%d", i)),
			"attn",
			batchSize,
			m.config.NumHeads,
			m.config.MaxPosEmbed,
			headDim,
			m.config.DType,
		)
	}

	// Convert prompt tokens to int32
	promptTokens32 := make([]int32, len(promptTokens))
	for i, token := range promptTokens {
		promptTokens32[i] = int32(token)
	}

	// Create single exec that handles both prompt and generation
	// Accepts tokens (2D: [batch, seq_len]) and position as graph parameters
	var err error
	m.tokenExec, err = context.NewExec(m.backend, m.ctx.Reuse(),
		func(ctx *context.Context, tokens *Node, position *Node) *Node {
			logits := m.transformer.ForGenerationDynamic()(ctx, tokens, position)
			// Get logits for last token: [batch, seq_len, vocab_size] -> [batch, vocab_size]
			lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
			lastLogits = Squeeze(lastLogits, 1)
			// Use SampleWithStrategy for proper sampling control
			return generation.SampleWithStrategy(
				ctx, lastLogits,
				m.config.Strategy,
				m.config.Temperature,
				m.config.TopK,
				m.config.TopP,
			)
		})
	if err != nil {
		return fmt.Errorf("failed to create generation exec: %w", err)
	}

	// Process prompt (position=0)
	results, _, err := m.tokenExec.ExecWithGraph(
		[][]int32{promptTokens32},
		[]int32{0},
	)
	if err != nil {
		return fmt.Errorf("failed to process prompt: %w", err)
	}

	// Get first generated token
	firstToken := int(results[0].Value().([]int32)[0])
	if callback != nil && !callback(firstToken) {
		return nil
	}

	// Start timing for token generation
	startTime := time.Now()
	tokensGenerated := 1 // Count the first token

	// Generate remaining tokens one by one
	currentPosition := len(promptTokens)
	currentToken := firstToken

	for step := 0; step < config.MaxTokens-1; step++ {
		position := currentPosition + step
		// Get next token from model (includes sampling)
		results, _, err := m.tokenExec.ExecWithGraph(
			[][]int32{{int32(currentToken)}},
			[]int32{int32(position)},
		)
		if err != nil {
			return fmt.Errorf("failed to generate token at step %d: %w", step, err)
		}

		nextToken := int(results[0].Value().([]int32)[0])
		if callback != nil && !callback(nextToken) {
			break
		}
		currentToken = nextToken
		tokensGenerated++
	}

	// Calculate and print timing statistics
	duration := time.Since(startTime)
	tokensPerSecond := float64(tokensGenerated) / duration.Seconds()
	fmt.Printf("\n\nGenerated %d tokens in %.2fs (%.1f tokens/s)\n", tokensGenerated, duration.Seconds(), tokensPerSecond)

	return nil
}

// loadCheckpoint loads model weights from the HuggingFace repository.
func loadCheckpoint(ctx *context.Context, repo *hub.Repo) error {
	// Download the safetensors file
	safetensorsPath, err := repo.DownloadFile("model.safetensors")
	if err != nil {
		return fmt.Errorf("failed to download model.safetensors: %w", err)
	}

	// Get repo info for validation (includes SafeTensorsInfo)
	info := repo.Info()
	if info != nil && info.SafeTensors.Total > 0 {
		fmt.Printf("Model has %d parameters\n", info.SafeTensors.Total)
	}

	// Load safetensors
	if err := LoadSafetensors(ctx, safetensorsPath); err != nil {
		return fmt.Errorf("failed to load safetensors: %w", err)
	}

	return nil
}
