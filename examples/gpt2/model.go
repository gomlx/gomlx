package main

import (
	"fmt"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"
	"github.com/gomlx/gomlx/pkg/ml/layers/generation"
	"github.com/gomlx/gopjrt/dtypes"
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

	// Single cached exec for all token generations (reusable across all positions)
	tokenExec *context.Exec
}

// GPT2TransformerWrapper wraps the transformer configuration for generation
type GPT2TransformerWrapper struct {
	config     *generation.TransformerConfig
	gpt2Config GPT2Config
	ctx        *context.Context
	Caches     []*attention.KVCache
}

func (w *GPT2TransformerWrapper) ForGeneration() generation.IncrementalModelFn {
	return func(ctx *context.Context, tokens *Node, position int) *Node {
		cfg := w.config
		g := tokens.Graph()
		currentSeqLen := tokens.Shape().Dimensions[1]

		// Note: Caches should be pre-initialized in Generate(), not lazily here!
		// The lazy initialization was overwriting the properly-created caches.
		if w.Caches == nil {
			panic("KV Caches must be initialized before calling ForGeneration")
		}

		// Token embeddings
		embedded := layers.Embedding(ctx.In("token_embed"), tokens, cfg.DType, cfg.VocabSize, cfg.EmbedDim)
		if embedded.Rank() == 2 {
			embedded = ExpandDims(embedded, 1)
		}

		// Positional embeddings
		x := embedded
		posEmbedFull := ctx.In("pos_embed").VariableWithShape("embeddings",
			shapes.Make(cfg.DType, cfg.MaxPosEmbed, cfg.EmbedDim)).ValueGraph(g)

		posEmbed := Slice(posEmbedFull, AxisRange(position, position+currentSeqLen))
		posEmbed = ExpandDims(posEmbed, 0)
		posEmbed = BroadcastToShape(posEmbed, embedded.Shape())
		x = Add(embedded, posEmbed)

		// Transformer layers (pre-norm architecture like GPT-2)
		for layer := 0; layer < cfg.NumLayers; layer++ {
			layerCtx := ctx.In(fmt.Sprintf("layer_%d", layer))

			// Pre-attention LayerNorm
			var attnInput *Node
			if cfg.UseLayerNorm {
				attnInput = layers.LayerNormalization(layerCtx.In("norm1"), x, -1).
					Epsilon(w.gpt2Config.NormEps).Done()
			} else {
				attnInput = x
			}

			// Self-attention with KV cache
			cache := w.Caches[layer]
			positionNode := Const(g, int32(position))
			attn := attention.SelfAttention(layerCtx.In("attn"), attnInput, cfg.NumHeads, cfg.HeadDim).
				WithKVCache(cache, positionNode).
				UseProjectionBias(cfg.UseBias).
				UseCausalMask().
				Done()

			// Residual connection
			x = Add(x, attn)

			// Pre-MLP LayerNorm
			var ffInput *Node
			if cfg.UseLayerNorm {
				ffInput = layers.LayerNormalization(layerCtx.In("norm2"), x, -1).
					Epsilon(w.gpt2Config.NormEps).Done()
			} else {
				ffInput = x
			}

			// Feed-forward network
			ff := layers.Dense(layerCtx.In("ff1"), ffInput, cfg.UseBias, cfg.FFNDim)
			ff = activations.Gelu(ff)
			ff = layers.Dense(layerCtx.In("ff2"), ff, cfg.UseBias, cfg.EmbedDim)

			// Residual connection
			x = Add(x, ff)
		}

		// GPT-2's final layer normalization (this is what was missing!)
		x = layers.LayerNormalization(ctx.In("final_norm"), x, -1).
			Epsilon(w.gpt2Config.NormEps).Done()

		// Output projection
		logits := layers.Dense(ctx.In("output"), x, false, cfg.VocabSize)
		return logits
	}
}

// ForGenerationDynamic returns a model function that accepts position as a Node parameter.
// This enables graph caching by making position a graph input rather than a captured variable.
// The same compiled graph can be reused for all positions, dramatically improving performance.
func (w *GPT2TransformerWrapper) ForGenerationDynamic() func(ctx *context.Context, tokens *Node, positionNode *Node) *Node {
	return func(ctx *context.Context, tokens *Node, positionNode *Node) *Node {
		cfg := w.config
		g := tokens.Graph()
		currentSeqLen := tokens.Shape().Dimensions[1]

		if w.Caches == nil {
			panic("KV Caches must be initialized before calling ForGeneration")
		}

		// Token embeddings
		embedded := layers.Embedding(ctx.In("token_embed"), tokens, cfg.DType, cfg.VocabSize, cfg.EmbedDim)
		if embedded.Rank() == 2 {
			embedded = ExpandDims(embedded, 1)
		}

		// Positional embeddings using dynamic slicing with positionNode
		x := embedded
		posEmbedFull := ctx.In("pos_embed").VariableWithShape("embeddings",
			shapes.Make(cfg.DType, cfg.MaxPosEmbed, cfg.EmbedDim)).ValueGraph(g)

		// Convert position node to int32 scalar if needed
		posScalar := ConvertDType(positionNode, dtypes.Int32)
		if posScalar.Rank() > 0 {
			posScalar = Squeeze(posScalar)
		}

		// Use DynamicSlice to extract positional embeddings at runtime
		// DynamicSlice(operand, sizes, startIndices...)
		posEmbed := DynamicSlice(posEmbedFull, []*Node{posScalar, Const(g, int32(0))}, []int{currentSeqLen, cfg.EmbedDim})
		posEmbed = ExpandDims(posEmbed, 0)
		posEmbed = BroadcastToShape(posEmbed, embedded.Shape())
		x = Add(embedded, posEmbed)

		// Transformer layers (pre-norm architecture like GPT-2)
		for layer := 0; layer < cfg.NumLayers; layer++ {
			layerCtx := ctx.In(fmt.Sprintf("layer_%d", layer))

			// Pre-attention LayerNorm
			var attnInput *Node
			if cfg.UseLayerNorm {
				attnInput = layers.LayerNormalization(layerCtx.In("norm1"), x, -1).
					Epsilon(w.gpt2Config.NormEps).Done()
			} else {
				attnInput = x
			}

			// Self-attention with KV cache - now passing position as a Node!
			cache := w.Caches[layer]
			attn := attention.SelfAttention(layerCtx.In("attn"), attnInput, cfg.NumHeads, cfg.HeadDim).
				WithKVCache(cache, positionNode). // Pass position as Node
				UseProjectionBias(cfg.UseBias).
				UseCausalMask().
				Done()

			// Residual connection
			x = Add(x, attn)

			// Pre-MLP LayerNorm
			var ffInput *Node
			if cfg.UseLayerNorm {
				ffInput = layers.LayerNormalization(layerCtx.In("norm2"), x, -1).
					Epsilon(w.gpt2Config.NormEps).Done()
			} else {
				ffInput = x
			}

			// MLP (Feed-Forward Network) - use same scope names as ForGeneration()
			ff := layers.Dense(layerCtx.In("ff1"), ffInput, cfg.UseBias, cfg.FFNDim)
			ff = activations.Gelu(ff)
			ff = layers.Dense(layerCtx.In("ff2"), ff, cfg.UseBias, cfg.EmbedDim)

			// Residual connection
			x = Add(x, ff)
		}

		// Final layer norm
		if cfg.UseLayerNorm {
			x = layers.LayerNormalization(ctx.In("final_norm"), x, -1).
				Epsilon(w.gpt2Config.NormEps).Done()
		}

		// Output projection
		logits := layers.Dense(ctx.In("output"), x, false, cfg.VocabSize)
		return logits
	}
}

// LoadGPT2 loads the GPT-2 model from the given checkpoint.
func LoadGPT2(backend backends.Backend, checkpointPath string) (*GPT2Model, *Tokenizer, error) {
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
	if err := loadCheckpoint(ctx, checkpointPath); err != nil {
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
	tokenizer, err := LoadTokenizer(checkpointPath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	return model, tokenizer, nil
}

// GenerationConfig holds parameters for text generation.
type GenerationConfig struct {
	MaxTokens   int
	Temperature float64
	TopP        float64
}

// Generate performs autoregressive text generation.
func (m *GPT2Model) Generate(
	promptTokens []int,
	config GenerationConfig,
	callback func(token int) bool,
) error {
	// Initialize KV caches for each layer
	batchSize := 1
	headDim := m.config.HiddenSize / m.config.NumHeads
	m.transformer.Caches = make([]*attention.KVCache, m.config.NumLayers)

	// IMPORTANT: Create caches using m.ctx directly, they will internally use Reuse()
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

	// Reset caches before generation - use ctx.Reuse()
	resetExec, err := context.NewExec(m.backend, m.ctx.Reuse(), func(ctx *context.Context, dummy *Node) *Node {
		g := dummy.Graph()
		for _, cache := range m.transformer.Caches {
			cache.Reset(g)
		}
		return dummy
	})
	if err != nil {
		return fmt.Errorf("failed to create cache reset exec: %w", err)
	}
	dummyInput := tensors.FromValue(int32(0))
	_, _, err = resetExec.ExecWithGraph(dummyInput)
	if err != nil {
		return fmt.Errorf("failed to reset caches: %w", err)
	}

	// Convert prompt tokens to int32
	promptTokens32 := make([]int32, len(promptTokens))
	for i, token := range promptTokens {
		promptTokens32[i] = int32(token)
	}

	// Process prompt (position=0) - use ctx.Reuse()
	promptExec, err := context.NewExec(m.backend, m.ctx.Reuse(), func(ctx *context.Context, tokens *Node) *Node {
		logits := m.transformer.ForGeneration()(ctx, tokens, 0)
		// Get logits for last token: [batch, vocab_size]
		lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
		lastLogits = Squeeze(lastLogits, 1)
		return lastLogits
	})
	if err != nil {
		return fmt.Errorf("failed to create prompt exec: %w", err)
	}

	results, _, err := promptExec.ExecWithGraph([][]int32{promptTokens32})
	if err != nil {
		return fmt.Errorf("failed to process prompt: %w", err)
	}

	// Sample first token
	firstLogits := results[0].Value().([][]float32)[0]
	firstToken := m.sampleToken(firstLogits, float32(config.Temperature), float32(config.TopP))

	if callback != nil && !callback(firstToken) {
		return nil
	}

	// Create exec that accepts position as a graph parameter
	// This allows the same compiled graph to be reused for all token positions
	m.tokenExec, err = context.NewExec(m.backend, m.ctx.Reuse(),
		func(ctx *context.Context, token *Node, position *Node) *Node {
			tokenReshaped := ExpandDims(token, -1)
			logits := m.transformer.ForGenerationDynamic()(ctx, tokenReshaped, position)
			lastLogits := Squeeze(logits, 1)
			return lastLogits
		})
	if err != nil {
		return fmt.Errorf("failed to create generation exec: %w", err)
	}

	// Generate remaining tokens one by one
	currentPosition := len(promptTokens)
	currentToken := firstToken

	for step := 0; step < config.MaxTokens-1; step++ {
		position := currentPosition + step

		// Get logits from model
		results, _, err := m.tokenExec.ExecWithGraph(
			[]int32{int32(currentToken)},
			[]int32{int32(position)},
		)
		if err != nil {
			return fmt.Errorf("failed to generate token at step %d: %w", step, err)
		}

		// Sample next token
		logits := results[0].Value().([][]float32)[0]
		nextToken := m.sampleToken(logits, float32(config.Temperature), float32(config.TopP))
		currentToken = nextToken

		if callback != nil && !callback(nextToken) {
			break
		}
	}

	return nil
}

// loadCheckpoint loads model weights from the checkpoint directory.
func loadCheckpoint(ctx *context.Context, checkpointPath string) error {
	safetensorsPath := checkpointPath + "/model.safetensors"

	// Try to load safetensors
	if err := LoadSafetensors(ctx, safetensorsPath); err != nil {
		return fmt.Errorf("checkpoint loading not fully implemented: %w", err)
	}

	return nil
}
