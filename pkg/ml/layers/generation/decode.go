package generation

import (
	"fmt"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

// ModelFn: full-sequence model, inputs [batch, seq], outputs [batch, seq, vocab].
type ModelFn func(ctx *context.Context, tokens *Node) *Node

// IncrementalModelFn: single-step cached model, inputs [batch, new_len], outputs [batch, new_len, vocab].
type IncrementalModelFn func(ctx *context.Context, newTokens *Node, position int) *Node

// GenerationConfig: config for autoregressive generation.
type GenerationConfig struct {
	// Model functions (exactly one should be set)
	ModelFn            ModelFn            // Standard model for non-cached generation
	IncrementalModelFn IncrementalModelFn // Incremental model for KV-cached generation

	// KV Cache configuration (required for IncrementalModelFn)
	NumLayers   int          // Number of transformer layers
	NumHeads    int          // Number of attention heads per layer
	HeadDim     int          // Dimension of each attention head
	MaxCacheLen int          // Maximum sequence length for cache
	DType       dtypes.DType // Data type for cache (usually same as model)

	// Generation parameters
	MaxLength   int
	Strategy    string // "greedy", "temperature", "top_k", "top_p", "beam_search"
	Temperature float32
	TopK        int
	TopP        float32
	BeamSize    int
	EosTokenId  int
	StopOnEOS   bool
}

// NewGenerationConfig: defaults for non-cached generation.
func NewGenerationConfig(modelFn ModelFn) *GenerationConfig {
	return &GenerationConfig{
		ModelFn:     modelFn,
		MaxLength:   100,
		Strategy:    "greedy",
		Temperature: 1.0,
		TopK:        50,
		TopP:        0.9,
		BeamSize:    4,
		EosTokenId:  -1,
		StopOnEOS:   false,
	}
}

// NewGenerationConfigCached: KV-cached generation using IncrementalAttention.
func NewGenerationConfigCached(
	incrementalModelFn IncrementalModelFn,
	numLayers, numHeads, headDim, maxCacheLen int,
	dtype dtypes.DType,
) *GenerationConfig {
	return &GenerationConfig{
		IncrementalModelFn: incrementalModelFn,
		NumLayers:          numLayers,
		NumHeads:           numHeads,
		HeadDim:            headDim,
		MaxCacheLen:        maxCacheLen,
		DType:              dtype,
		MaxLength:          100,
		Strategy:           "greedy",
		Temperature:        1.0,
		TopK:               50,
		TopP:               0.9,
		BeamSize:           4,
		EosTokenId:         -1,
		StopOnEOS:          false,
	}
}

// WithMaxLength: set maximum generation length.
func (cfg *GenerationConfig) WithMaxLength(maxLength int) *GenerationConfig {
	cfg.MaxLength = maxLength
	return cfg
}

// WithStrategy: set sampling strategy.
func (cfg *GenerationConfig) WithStrategy(strategy string) *GenerationConfig {
	cfg.Strategy = strategy
	return cfg
}

// WithTemperature: set temperature for sampling.
func (cfg *GenerationConfig) WithTemperature(temperature float32) *GenerationConfig {
	cfg.Temperature = temperature
	return cfg
}

// WithTopK: set k for top-k sampling.
func (cfg *GenerationConfig) WithTopK(topK int) *GenerationConfig {
	cfg.TopK = topK
	return cfg
}

// WithTopP: set p for nucleus sampling.
func (cfg *GenerationConfig) WithTopP(topP float32) *GenerationConfig {
	cfg.TopP = topP
	return cfg
}

// WithBeamSize: set beam size for beam search.
func (cfg *GenerationConfig) WithBeamSize(beamSize int) *GenerationConfig {
	cfg.BeamSize = beamSize
	return cfg
}

// WithEOS: set EOS token and enable StopOnEOS.
func (cfg *GenerationConfig) WithEOS(eosTokenId int) *GenerationConfig {
	cfg.EosTokenId = eosTokenId
	cfg.StopOnEOS = true
	return cfg
}

// isCached: true if using KV caching.
func (cfg *GenerationConfig) isCached() bool {
	return cfg.IncrementalModelFn != nil
}

// validate: check configuration validity.
func (cfg *GenerationConfig) validate() error {
	// Exactly one model function must be set
	if cfg.ModelFn != nil && cfg.IncrementalModelFn != nil {
		return fmt.Errorf("cannot set both ModelFn and IncrementalModelFn")
	}
	if cfg.ModelFn == nil && cfg.IncrementalModelFn == nil {
		return fmt.Errorf("must set either ModelFn or IncrementalModelFn")
	}

	// If using incremental model, cache config must be set
	if cfg.IncrementalModelFn != nil {
		if cfg.NumLayers <= 0 {
			return fmt.Errorf("NumLayers must be > 0 for cached generation, got %d", cfg.NumLayers)
		}
		if cfg.NumHeads <= 0 {
			return fmt.Errorf("NumHeads must be > 0 for cached generation, got %d", cfg.NumHeads)
		}
		if cfg.HeadDim <= 0 {
			return fmt.Errorf("HeadDim must be > 0 for cached generation, got %d", cfg.HeadDim)
		}
		if cfg.MaxCacheLen <= 0 {
			return fmt.Errorf("MaxCacheLen must be > 0 for cached generation, got %d", cfg.MaxCacheLen)
		}
		if cfg.DType == dtypes.InvalidDType {
			return fmt.Errorf("DType must be set for cached generation")
		}
	}

	return nil
}

// Generate: autoregressive decoding from prompt.
func (cfg *GenerationConfig) Generate(
	backend backends.Backend,
	ctx *context.Context,
	prompt any,
) (*tensors.Tensor, error) {
	// Validate configuration
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("invalid generation config: %w", err)
	}

	promptTensor := tensors.FromAnyValue(prompt)

	if promptTensor.Rank() == 1 {
		// Reshape is not a method on Tensor, so we need to work with the shape
		// For now, we'll use the prompt as-is in the graph
	}
	if promptTensor.Rank() != 2 && promptTensor.Rank() != 1 {
		return nil, fmt.Errorf("prompt must be 1D or 2D, got rank %d", promptTensor.Rank())
	}

	promptShape := promptTensor.Shape()
	var promptLen int
	if promptTensor.Rank() == 1 {
		promptLen = promptShape.Dimensions[0]
	} else {
		promptLen = promptShape.Dimensions[1]
	}

	if promptLen >= cfg.MaxLength {
		return nil, fmt.Errorf("prompt length %d >= max length %d", promptLen, cfg.MaxLength)
	}

	if cfg.Strategy == "beam_search" {
		return cfg.generateBeamSearch(backend, ctx, promptTensor)
	}

	// Regular sampling-based generation
	return cfg.generateSampling(backend, ctx, promptTensor)
}

// generateSampling: sampling strategies.
func (cfg *GenerationConfig) generateSampling(
	backend backends.Backend,
	ctx *context.Context,
	prompt *tensors.Tensor,
) (*tensors.Tensor, error) {
	promptShape := prompt.Shape()
	var batchSize, promptLen int

	if prompt.Rank() == 1 {
		// Reshape 1D to 2D with batch size 1
		reshapeExec, err := context.NewExec(backend, ctx.Reuse(), func(ctx *context.Context, input *Node) *Node {
			return ExpandDims(input, 0) // [seq_len] -> [1, seq_len]
		})
		if err != nil {
			return nil, fmt.Errorf("failed to create reshape exec: %w", err)
		}

		reshapeResults, _, err := reshapeExec.ExecWithGraph(prompt)
		if err != nil {
			return nil, fmt.Errorf("failed to reshape prompt: %w", err)
		}
		prompt = reshapeResults[0]
		batchSize = 1
		promptLen = promptShape.Dimensions[0]
	} else {
		batchSize = promptShape.Dimensions[0]
		promptLen = promptShape.Dimensions[1]
	}

	if promptLen >= cfg.MaxLength {
		return nil, fmt.Errorf("prompt length %d >= max length %d", promptLen, cfg.MaxLength)
	}

	if cfg.isCached() {
		return cfg.generateSamplingCached(backend, ctx, prompt, batchSize, promptLen)
	}
	return cfg.generateSamplingNonCached(backend, ctx, prompt, promptLen)
}

// generateSamplingNonCached.
func (cfg *GenerationConfig) generateSamplingNonCached(
	backend backends.Backend,
	ctx *context.Context,
	prompt *tensors.Tensor,
	promptLen int,
) (*tensors.Tensor, error) {
	concatExec, err := context.NewExec(backend, ctx.Reuse(), func(ctx *context.Context, seq, token *Node) *Node {
		tokenReshaped := ExpandDims(token, -1) // [batch, 1]
		return Concatenate([]*Node{seq, tokenReshaped}, 1)
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create concat graph: %w", err)
	}
	concatExec.SetMaxCache(-1) // No limit - sequence length varies at each step

	predCtx := ctx.Reuse()

	currentSeq := prompt

	numTokensToGenerate := cfg.MaxLength - promptLen
	if numTokensToGenerate <= 0 {
		return prompt, nil
	}

	for step := 0; step < numTokensToGenerate; step++ {
		exec, err := context.NewExec(backend, predCtx, func(ctx *context.Context, currentSeq *Node) *Node {
			logits := cfg.ModelFn(ctx, currentSeq)
			lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
			lastLogits = Squeeze(lastLogits, 1) // Remove the seq_len dimension
			nextToken := SampleWithStrategy(ctx, lastLogits, cfg.Strategy, float64(cfg.Temperature), cfg.TopK, float64(cfg.TopP))
			return nextToken
		})
		if err != nil {
			return nil, fmt.Errorf("failed to create generation graph: %w", err)
		}

		outputs, _, err := exec.ExecWithGraph(currentSeq)
		if err != nil {
			return nil, fmt.Errorf("generation step %d failed: %w", step, err)
		}

		nextToken := outputs[0]

		if cfg.StopOnEOS && cfg.EosTokenId >= 0 {
			if cfg.checkEOS(nextToken) {
				concatResults, _, err := concatExec.ExecWithGraph(currentSeq, nextToken)
				if err != nil {
					return nil, fmt.Errorf("final concatenation failed: %w", err)
				}
				return concatResults[0], nil
			}
		}

		concatResults, _, err := concatExec.ExecWithGraph(currentSeq, nextToken)
		if err != nil {
			return nil, fmt.Errorf("concatenation step %d failed: %w", step, err)
		}

		currentSeq = concatResults[0]
	}

	return currentSeq, nil
}

// generateSamplingCached.
func (cfg *GenerationConfig) generateSamplingCached(
	backend backends.Backend,
	ctx *context.Context,
	prompt *tensors.Tensor,
	_, promptLen int,
) (*tensors.Tensor, error) {
	// Note: KV caches are now automatically managed within the context by the attention layers.
	// Each call to WithKVCache in the model function will create/reuse cache variables in the context.

	promptExec, err := context.NewExec(backend, ctx.Reuse(), func(ctx *context.Context, tokens *Node) *Node {
		logits := cfg.IncrementalModelFn(ctx, tokens, 0)
		lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
		lastLogits = Squeeze(lastLogits, 1) // [batch, vocab_size]
		nextToken := SampleWithStrategy(ctx, lastLogits, cfg.Strategy, float64(cfg.Temperature), cfg.TopK, float64(cfg.TopP))
		return nextToken
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create prompt exec: %w", err)
	}

	outputs, _, err := promptExec.ExecWithGraph(prompt)
	if err != nil {
		return nil, fmt.Errorf("failed to process prompt: %w", err)
	}

	firstToken := outputs[0]

	if cfg.StopOnEOS && cfg.EosTokenId >= 0 && cfg.checkEOS(firstToken) {
		concatExec, _ := context.NewExec(backend, ctx.Reuse(), func(ctx *context.Context, seq, token *Node) *Node {
			tokenReshaped := ExpandDims(token, -1)
			return Concatenate([]*Node{seq, tokenReshaped}, 1)
		})
		concatExec.SetMaxCache(-1)
		result, _, _ := concatExec.ExecWithGraph(prompt, firstToken)
		return result[0], nil
	}

	outputTokens := make([]*tensors.Tensor, 0, cfg.MaxLength)
	outputTokens = append(outputTokens, prompt, firstToken)

	numTokensToGenerate := cfg.MaxLength - promptLen - 1
	if numTokensToGenerate <= 0 {
		return cfg.concatenateTokens(backend, ctx, outputTokens)
	}

	genCtx := ctx.Reuse()

	currentPosition := promptLen
	for step := 0; step < numTokensToGenerate; step++ {
		position := currentPosition + step

		exec, err := context.NewExec(backend, genCtx, func(ctx *context.Context, token *Node) *Node {
			tokenReshaped := ExpandDims(token, -1)
			logits := cfg.IncrementalModelFn(ctx, tokenReshaped, position)
			lastLogits := Squeeze(logits, 1)
			nextToken := SampleWithStrategy(ctx, lastLogits, cfg.Strategy, float64(cfg.Temperature), cfg.TopK, float64(cfg.TopP))
			return nextToken
		})
		if err != nil {
			return nil, fmt.Errorf("failed to create generation exec at position %d: %w", position, err)
		}

		prevToken := outputTokens[len(outputTokens)-1]

		outputs, _, err := exec.ExecWithGraph(prevToken)
		if err != nil {
			return nil, fmt.Errorf("generation step %d (position %d) failed: %w", step, position, err)
		}

		nextToken := outputs[0]

		if cfg.StopOnEOS && cfg.EosTokenId >= 0 && cfg.checkEOS(nextToken) {
			outputTokens = append(outputTokens, nextToken)
			break
		}

		outputTokens = append(outputTokens, nextToken)
	}

	return cfg.concatenateTokens(backend, ctx, outputTokens)
}

// checkEOS.
func (cfg *GenerationConfig) checkEOS(token *tensors.Tensor) bool {
	tokenValue := token.Value()
	switch v := tokenValue.(type) {
	case []int32:
		for _, t := range v {
			if int(t) == cfg.EosTokenId {
				return true
			}
		}
	case int32:
		return int(v) == cfg.EosTokenId
	case []int64:
		for _, t := range v {
			if int(t) == cfg.EosTokenId {
				return true
			}
		}
	case int64:
		return int(v) == cfg.EosTokenId
	}
	return false
}

// concatenateTokens.
func (cfg *GenerationConfig) concatenateTokens(
	backend backends.Backend,
	ctx *context.Context,
	tokens []*tensors.Tensor,
) (*tensors.Tensor, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("no tokens to concatenate")
	}
	if len(tokens) == 1 {
		return tokens[0], nil
	}

	result := tokens[0]
	concatExec, err := context.NewExec(backend, ctx.Reuse(), func(ctx *context.Context, seq, token *Node) *Node {
		tokenReshaped := ExpandDims(token, -1) // [batch] -> [batch, 1]
		return Concatenate([]*Node{seq, tokenReshaped}, 1)
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create concatenation exec: %w", err)
	}
	concatExec.SetMaxCache(-1)

	for i := 1; i < len(tokens); i++ {
		outputs, _, err := concatExec.ExecWithGraph(result, tokens[i])
		if err != nil {
			return nil, fmt.Errorf("failed to concatenate token %d: %w", i, err)
		}
		result = outputs[0]
	}

	return result, nil
}

// generateBeamSearch: beam search (cached or non-cached).
func (cfg *GenerationConfig) generateBeamSearch(
	backend backends.Backend,
	ctx *context.Context,
	prompt *tensors.Tensor,
) (*tensors.Tensor, error) {
	// Ensure prompt 2D
	promptShape := prompt.Shape()
	var batchSize, promptLen int

	if prompt.Rank() == 1 {
		reshapeExec, err := context.NewExec(backend, ctx.Reuse(), func(ctx *context.Context, input *Node) *Node {
			return ExpandDims(input, 0)
		})
		if err != nil {
			return nil, fmt.Errorf("failed to create reshape exec: %w", err)
		}

		reshapeResults, _, err := reshapeExec.ExecWithGraph(prompt)
		if err != nil {
			return nil, fmt.Errorf("failed to reshape prompt: %w", err)
		}
		prompt = reshapeResults[0]
		batchSize = 1
		promptLen = promptShape.Dimensions[0]
	} else {
		batchSize = promptShape.Dimensions[0]
		promptLen = promptShape.Dimensions[1]
	}

	if promptLen >= cfg.MaxLength {
		return nil, fmt.Errorf("prompt length %d >= max length %d", promptLen, cfg.MaxLength)
	}

	// Dispatch cached or non-cached
	if cfg.isCached() {
		return cfg.generateBeamSearchCached(backend, ctx, prompt, batchSize, promptLen)
	}
	return cfg.generateBeamSearchNonCached(backend, ctx, prompt, batchSize, promptLen)
}

// generateBeamSearchNonCached: beam search without KV caches.
func (cfg *GenerationConfig) generateBeamSearchNonCached(
	backend backends.Backend,
	ctx *context.Context,
	prompt *tensors.Tensor,
	batchSize, promptLen int,
) (*tensors.Tensor, error) {
	beamSize := cfg.BeamSize
	batchBeamSize := batchSize * beamSize

	// Replicate prompt for each beam
	replicateExec, err := context.NewExec(backend, ctx.Reuse(), func(ctx *context.Context, prompt *Node) *Node {
		// prompt: [batch, seq_len]
		// Replicate beamSize times along batch dimension
		// Result: [batch * beam_size, seq_len]
		replicated := make([]*Node, beamSize)
		for i := range beamSize {
			replicated[i] = prompt
		}
		return Concatenate(replicated, 0)
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create replicate exec: %w", err)
	}

	replicatedResults, _, err := replicateExec.ExecWithGraph(prompt)
	if err != nil {
		return nil, fmt.Errorf("failed to replicate prompt: %w", err)
	}

	currentSequences := replicatedResults[0]

	// Initialize beam scores: first beam 0, others -1e10
	initialScores := make([]float32, batchBeamSize)
	for i := 0; i < batchBeamSize; i++ {
		if i%beamSize == 0 {
			initialScores[i] = 0.0
		} else {
			initialScores[i] = -1e10
		}
	}
	beamScores := tensors.FromValue(initialScores)

	// Beam search configuration
	beamConfig := NewBeamSearch(beamSize, cfg.EosTokenId).
		WithMaxLength(cfg.MaxLength).
		WithLengthPenalty(1.0)

	// Main loop
	predCtx := ctx.Reuse()
	numSteps := cfg.MaxLength - promptLen

	for step := 0; step < numSteps; step++ {
		currentLength := promptLen + step

		// Create exec for this step
		exec, err := context.NewExec(backend, predCtx, func(ctx *context.Context, sequences, scores *Node) (*Node, *Node, *Node) {
			// Run model
			logits := cfg.ModelFn(ctx, sequences)

			// Last token logits: [batch_beam_size, vocab_size]
			lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
			lastLogits = Squeeze(lastLogits, 1)

			// Beam search step
			nextSeqs, nextScores, isFinished := beamConfig.Step(
				lastLogits,
				sequences,
				scores,
				currentLength,
			)

			return nextSeqs, nextScores, isFinished
		})
		if err != nil {
			return nil, fmt.Errorf("failed to create beam search exec at step %d: %w", step, err)
		}

		outputs, _, err := exec.ExecWithGraph(currentSequences, beamScores)
		if err != nil {
			return nil, fmt.Errorf("beam search step %d failed: %w", step, err)
		}

		currentSequences = outputs[0]
		beamScores = outputs[1]
		isFinished := outputs[2]

		// All beams finished?
		if beamConfig.earlyStopping {
			finishedValue := isFinished.Value()
			allFinished := true
			switch v := finishedValue.(type) {
			case []bool:
				for _, f := range v {
					if !f {
						allFinished = false
						break
					}
				}
			}
			if allFinished {
				break
			}
		}
	}

	// Apply length penalty; select best sequences
	selectExec, err := context.NewExec(backend, ctx.Reuse(), func(ctx *context.Context, sequences, scores *Node) (*Node, *Node) {
		// Select best sequences (length penalty is applied automatically)
		bestSeqs, bestScores := beamConfig.SelectBest(sequences, scores)
		return bestSeqs, bestScores
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create select exec: %w", err)
	}

	selectResults, _, err := selectExec.ExecWithGraph(currentSequences, beamScores)
	if err != nil {
		return nil, fmt.Errorf("failed to select best sequences: %w", err)
	}

	return selectResults[0], nil
}

// generateBeamSearchCached: beam search with KV caches per beam.
func (cfg *GenerationConfig) generateBeamSearchCached(
	backend backends.Backend,
	ctx *context.Context,
	prompt *tensors.Tensor,
	batchSize, promptLen int,
) (*tensors.Tensor, error) {
	beamSize := cfg.BeamSize

	// Note: KV caches are now automatically managed within the context by the attention layers.
	// Each call to WithKVCache in the model function will create/reuse cache variables in the context.

	// Replicate prompt for each beam
	replicateExec, err := context.NewExec(backend, ctx.Reuse(), func(ctx *context.Context, prompt *Node) *Node {
		replicated := make([]*Node, beamSize)
		for i := range beamSize {
			replicated[i] = prompt
		}
		return Concatenate(replicated, 0)
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create replicate exec: %w", err)
	}

	replicatedResults, _, err := replicateExec.ExecWithGraph(prompt)
	if err != nil {
		return nil, fmt.Errorf("failed to replicate prompt: %w", err)
	}

	replicatedPrompt := replicatedResults[0]

	// Process prompt to populate caches
	promptExec, err := context.NewExec(backend, ctx.Reuse(), func(ctx *context.Context, tokens *Node) *Node {
		logits := cfg.IncrementalModelFn(ctx, tokens, 0)
		lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
		return Squeeze(lastLogits, 1)
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create prompt exec: %w", err)
	}

	_, _, err = promptExec.ExecWithGraph(replicatedPrompt)
	if err != nil {
		return nil, fmt.Errorf("failed to process prompt: %w", err)
	}

	batchBeamSize := batchSize * beamSize

	// Initialize beam scores: first beam 0, others -1e10
	initialScores := make([]float32, batchBeamSize)
	for i := range batchBeamSize {
		if i%beamSize == 0 {
			initialScores[i] = 0.0
		} else {
			initialScores[i] = -1e10
		}
	}
	beamScores := tensors.FromValue(initialScores)
	currentSequences := replicatedPrompt

	// Beam search configuration
	beamConfig := NewBeamSearch(beamSize, cfg.EosTokenId).
		WithMaxLength(cfg.MaxLength).
		WithLengthPenalty(1.0)

	// Main loop
	genCtx := ctx.Reuse()
	numSteps := cfg.MaxLength - promptLen

	for step := 0; step < numSteps; step++ {
		position := promptLen + step

		// Cached: process single token; beam step needs logits from all beams.
		// Extract last token from each sequence.
		extractExec, err := context.NewExec(backend, ctx.Reuse(), func(ctx *context.Context, sequences *Node) *Node {
			// Last token: [batch_beam_size]
			lastTokens := Slice(sequences, AxisRange(), AxisElem(-1))
			return lastTokens
		})
		if err != nil {
			return nil, fmt.Errorf("failed to create extract exec: %w", err)
		}

		extractResults, _, err := extractExec.ExecWithGraph(currentSequences)
		if err != nil {
			return nil, fmt.Errorf("failed to extract last tokens: %w", err)
		}

		lastTokens := extractResults[0]

		// Generate next token logits using cached model
		exec, err := context.NewExec(backend, genCtx, func(ctx *context.Context, tokens, scores, sequences *Node) (*Node, *Node, *Node) {
			// tokens: [batch_beam_size]
			tokensReshaped := ExpandDims(tokens, -1) // [batch_beam_size, 1]

			// Process with incremental model
			logits := cfg.IncrementalModelFn(ctx, tokensReshaped, position)
			logits = Squeeze(logits, 1) // [batch_beam_size, vocab_size]

			// Beam search step
			nextSeqs, nextScores, isFinished := beamConfig.Step(
				logits,
				sequences,
				scores,
				position,
			)

			return nextSeqs, nextScores, isFinished
		})
		if err != nil {
			return nil, fmt.Errorf("failed to create beam step exec: %w", err)
		}

		outputs, _, err := exec.ExecWithGraph(lastTokens, beamScores, currentSequences)
		if err != nil {
			return nil, fmt.Errorf("beam search step %d failed: %w", step, err)
		}

		currentSequences = outputs[0]
		beamScores = outputs[1]
		isFinished := outputs[2]

		// Early stopping check
		if beamConfig.earlyStopping {
			finishedValue := isFinished.Value()
			allFinished := true
			switch v := finishedValue.(type) {
			case []bool:
				for _, f := range v {
					if !f {
						allFinished = false
						break
					}
				}
			}
			if allFinished {
				break
			}
		}
	}

	// Select best sequences
	selectExec, err := context.NewExec(backend, ctx.Reuse(), func(ctx *context.Context, sequences, scores *Node) (*Node, *Node) {
		// Select best sequences (length penalty is applied automatically)
		bestSeqs, bestScores := beamConfig.SelectBest(sequences, scores)
		return bestSeqs, bestScores
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create select exec: %w", err)
	}

	selectResults, _, err := selectExec.ExecWithGraph(currentSequences, beamScores)
	if err != nil {
		return nil, fmt.Errorf("failed to select best sequences: %w", err)
	}

	return selectResults[0], nil
}

// GenerateStreaming performs streaming generation, yielding tokens as they are generated.
//
// NOTE: This is a placeholder for future streaming support.
// GenerateStreaming: streaming generation (placeholder).
func (cfg *GenerationConfig) GenerateStreaming(
	backend backends.Backend,
	ctx *context.Context,
	prompt any,
	callback func(token int) bool,
) error {
	return fmt.Errorf("streaming generation not yet implemented")
}
