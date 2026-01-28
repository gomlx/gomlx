package main

import (
	"fmt"
	"strings"
	"time"
	"unsafe"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/safetensors"
	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/gomlx/go-huggingface/tokenizers/hftokenizer"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
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
	MaxSeqLen  int // Maximum sequence length for KV cache
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
				WithKVCache(w.MaxSeqLen, positionNode).
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
func LoadGPT2(backend backends.Backend, repo *hub.Repo) (*GPT2Model, api.Tokenizer, error) {
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

	// Load tokenizer - distilgpt2 doesn't have tokenizer_class, so use hftokenizer directly
	tokenizer, err := hftokenizer.New(nil, repo)
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

	// Set max sequence length for KV cache (cache is automatically managed within context)
	m.transformer.MaxSeqLen = m.config.MaxPosEmbed

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
	results, err := m.tokenExec.Exec(
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

	for step := range config.MaxTokens - 1 {
		position := currentPosition + step
		// Get next token from model (includes sampling)
		results, err := m.tokenExec.Exec(
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
	// Get repo info for validation (includes SafeTensorsInfo)
	info := repo.Info()
	if info != nil && info.SafeTensors.Total > 0 {
		fmt.Printf("Model has %d parameters\n", info.SafeTensors.Total)
	}

	// Load safetensors using go-huggingface package
	count := 0
	skipped := 0

	for tensorAndName, err := range safetensors.IterTensorsFromRepo(repo) {
		if err != nil {
			return fmt.Errorf("failed to iterate tensors: %w", err)
		}

		// Map tensor name to GoMLX structure
		scopePath, varName, ok := mapTensorName(tensorAndName.Name)
		if !ok {
			skipped++
			continue
		}

		// Set variable in context
		if err := setContextVariableFromTensor(ctx, scopePath, varName, tensorAndName.Tensor); err != nil {
			fmt.Printf("Warning: failed to load %s -> %s/%s: %v\n",
				tensorAndName.Name, strings.Join(scopePath, "/"), varName, err)
			continue
		}

		// Handle weight tying: GPT-2 shares token embeddings with output projection
		if tensorAndName.Name == "transformer.wte.weight" {
			// Transpose [vocab, hidden] -> [hidden, vocab] for output layer
			shape := tensorAndName.Tensor.Shape().Dimensions
			transposedShape := shapes.Make(tensorAndName.Tensor.DType(), shape[1], shape[0])
			tTransposed := tensors.FromShape(transposedShape)
			transposeFloat32Tensor(tensorAndName.Tensor, tTransposed)

			if err := setContextVariableFromTensor(ctx, []string{"output", "dense"}, "weights", tTransposed); err != nil {
				fmt.Printf("Warning: failed to set tied output weights: %v\n", err)
			}
		}

		count++
	}

	if count > 0 {
		fmt.Printf("✓ Successfully loaded %d/%d tensors from checkpoint\n", count, count+skipped)
	} else {
		fmt.Printf("Warning: No tensors were loaded (%d skipped)\n", skipped)
	}

	return nil
}

// mapTensorName maps safetensors tensor names to GoMLX context variable names
// DistilGPT-2/GPT-2 format: transformer.wte.weight, transformer.h.{N}.attn.c_attn.weight, etc.
// GoMLX format: token_embed/embeddings, layer_{N}/attn/..., etc.
func mapTensorName(safetensorsName string) (scopePath []string, varName string, ok bool) {
	switch {
	case safetensorsName == "transformer.wte.weight":
		return []string{"token_embed"}, "embeddings", true
	case safetensorsName == "transformer.wpe.weight":
		return []string{"pos_embed"}, "embeddings", true
	case safetensorsName == "transformer.ln_f.weight":
		return []string{"final_norm", "layer_normalization"}, "gain", true
	case safetensorsName == "transformer.ln_f.bias":
		return []string{"final_norm", "layer_normalization"}, "offset", true
	}

	// Layer-specific weights: transformer.h.{N}.{component}.{param}
	var layerNum int
	var component string

	// Parse layer number and component
	if n, err := fmt.Sscanf(safetensorsName, "transformer.h.%d.%s", &layerNum, &component); n == 2 && err == nil {
		layerScope := fmt.Sprintf("layer_%d", layerNum)

		// Parse component and parameter
		switch {
		// Attention weights - need special handling for fused QKV
		case component == "attn.c_attn.weight":
			// This returns a marker - we'll handle splitting in the loading code
			return []string{layerScope, "attn", "_fused_qkv"}, "weights", true
		case component == "attn.c_attn.bias":
			return []string{layerScope, "attn", "_fused_qkv"}, "biases", true
		case component == "attn.c_proj.weight":
			return []string{layerScope, "attn", "MultiHeadAttention", "output", "dense"}, "weights", true
		case component == "attn.c_proj.bias":
			return []string{layerScope, "attn", "MultiHeadAttention", "output", "dense"}, "biases", true

		// Layer norm 1 (pre-attention)
		case component == "ln_1.weight":
			return []string{layerScope, "norm1", "layer_normalization"}, "gain", true
		case component == "ln_1.bias":
			return []string{layerScope, "norm1", "layer_normalization"}, "offset", true

		// Feed-forward network
		case component == "mlp.c_fc.weight":
			return []string{layerScope, "ff1", "dense"}, "weights", true
		case component == "mlp.c_fc.bias":
			return []string{layerScope, "ff1", "dense"}, "biases", true
		case component == "mlp.c_proj.weight":
			return []string{layerScope, "ff2", "dense"}, "weights", true
		case component == "mlp.c_proj.bias":
			return []string{layerScope, "ff2", "dense"}, "biases", true

		// Layer norm 2 (pre-FFN)
		case component == "ln_2.weight":
			return []string{layerScope, "norm2", "layer_normalization"}, "gain", true
		case component == "ln_2.bias":
			return []string{layerScope, "norm2", "layer_normalization"}, "offset", true

		// Attention bias (for masking, not used in GoMLX standard attention)
		case component == "attn.bias":
			return nil, "", false // Skip attention bias
		}
	}

	return nil, "", false
}

// setContextVariableFromTensor sets a variable in the context from a tensor
func setContextVariableFromTensor(ctx *context.Context, scopePath []string, varName string, t *tensors.Tensor) error {
	// Check if this is a fused QKV weight that needs splitting
	if len(scopePath) >= 3 && scopePath[len(scopePath)-1] == "_fused_qkv" {
		return splitAndSetQKV(ctx, scopePath, varName, t)
	}

	// Normal case: set single variable
	scopeCtx := ctx
	for _, scope := range scopePath {
		scopeCtx = scopeCtx.In(scope)
	}

	scopeCtx.VariableWithValue(varName, t)
	return nil
}

// splitAndSetQKV splits fused QKV weights/biases into separate Q, K, V tensors
func splitAndSetQKV(ctx *context.Context, scopePath []string, varName string, t *tensors.Tensor) error {
	// Get the actual scope without "_fused_qkv"
	baseScopePath := scopePath[:len(scopePath)-1]
	baseCtx := ctx
	for _, scope := range baseScopePath {
		baseCtx = baseCtx.In(scope)
	}
	baseCtx = baseCtx.In("MultiHeadAttention")

	shape := t.Shape().Dimensions
	var flatData []float32
	t.ConstBytes(func(data []byte) {
		numElements := t.Shape().Size()
		flatData = make([]float32, numElements)
		// Convert bytes to float32
		for i := 0; i < numElements; i++ {
			flatData[i] = float32FromBytes(data[i*4 : (i+1)*4])
		}
	})

	if len(shape) == 2 {
		// Weight matrix: [hiddenSize, 3*hiddenSize]
		hiddenSize := shape[0]
		totalSize := shape[1]
		if totalSize%3 != 0 {
			return fmt.Errorf("expected fused QKV size to be divisible by 3, got %d", totalSize)
		}
		singleSize := totalSize / 3

		numHeads := 12
		headDim := singleSize / numHeads

		qData := make([]float32, hiddenSize*numHeads*headDim)
		kData := make([]float32, hiddenSize*numHeads*headDim)
		vData := make([]float32, hiddenSize*numHeads*headDim)

		for i := 0; i < hiddenSize; i++ {
			for h := 0; h < numHeads; h++ {
				for d := 0; d < headDim; d++ {
					flatIdx := h*headDim + d
					qData[i*numHeads*headDim+h*headDim+d] = flatData[i*totalSize+flatIdx]
					kData[i*numHeads*headDim+h*headDim+d] = flatData[i*totalSize+singleSize+flatIdx]
					vData[i*numHeads*headDim+h*headDim+d] = flatData[i*totalSize+2*singleSize+flatIdx]
				}
			}
		}

		baseCtx.In("query").In("dense").VariableWithValue(varName, tensors.FromFlatDataAndDimensions(qData, hiddenSize, numHeads, headDim))
		baseCtx.In("key").In("dense").VariableWithValue(varName, tensors.FromFlatDataAndDimensions(kData, hiddenSize, numHeads, headDim))
		baseCtx.In("value").In("dense").VariableWithValue(varName, tensors.FromFlatDataAndDimensions(vData, hiddenSize, numHeads, headDim))
	} else if len(shape) == 1 {
		// Bias vector: [3*hiddenSize]
		totalSize := shape[0]
		if totalSize%3 != 0 {
			return fmt.Errorf("expected fused QKV bias size to be divisible by 3, got %d", totalSize)
		}
		singleSize := totalSize / 3

		numHeads := 12
		headDim := singleSize / numHeads

		qData := make([]float32, numHeads*headDim)
		kData := make([]float32, numHeads*headDim)
		vData := make([]float32, numHeads*headDim)

		for h := 0; h < numHeads; h++ {
			for d := 0; d < headDim; d++ {
				flatIdx := h*headDim + d
				qData[h*headDim+d] = flatData[flatIdx]
				kData[h*headDim+d] = flatData[singleSize+flatIdx]
				vData[h*headDim+d] = flatData[2*singleSize+flatIdx]
			}
		}

		baseCtx.In("query").In("dense").VariableWithValue(varName, tensors.FromFlatDataAndDimensions(qData, numHeads, headDim))
		baseCtx.In("key").In("dense").VariableWithValue(varName, tensors.FromFlatDataAndDimensions(kData, numHeads, headDim))
		baseCtx.In("value").In("dense").VariableWithValue(varName, tensors.FromFlatDataAndDimensions(vData, numHeads, headDim))
	} else {
		return fmt.Errorf("unexpected shape for fused QKV: %v", shape)
	}

	return nil
}

// transposeFloat32Tensor transposes a 2D [A, B] tensor to [B, A]
func transposeFloat32Tensor(src, dst *tensors.Tensor) {
	srcShape := src.Shape().Dimensions
	if len(srcShape) != 2 {
		panic("can only transpose 2D tensors")
	}

	rows, cols := srcShape[0], srcShape[1]

	src.ConstBytes(func(srcData []byte) {
		dst.MutableBytes(func(dstData []byte) {
			for i := 0; i < rows; i++ {
				for j := 0; j < cols; j++ {
					srcOffset := (i*cols + j) * 4
					dstOffset := (j*rows + i) * 4
					copy(dstData[dstOffset:dstOffset+4], srcData[srcOffset:srcOffset+4])
				}
			}
		})
	})
}

// float32FromBytes converts 4 bytes (little-endian) to float32
func float32FromBytes(b []byte) float32 {
	bits := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
	return *(*float32)(unsafe.Pointer(&bits))
}
