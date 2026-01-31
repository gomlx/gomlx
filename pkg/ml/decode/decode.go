// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package decode

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/simplego"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/generation"
	"github.com/pkg/errors"
)

// Hyperparameter keys for context configuration
const (
	ParamMaxLength   = "decode_max_length"
	ParamStrategy    = "decode_strategy"
	ParamTemperature = "decode_temperature"
	ParamTopK        = "decode_top_k"
	ParamTopP        = "decode_top_p"
	ParamBeamSize    = "decode_beam_size"
	ParamEosTokenId  = "decode_eos_token_id"
	ParamStopOnEOS   = "decode_stop_on_eos"
)

// ModelFn represents a full-sequence model function.
// Takes a token sequence and returns logits for all positions.
//
// Parameters:
//   - ctx: Context containing model parameters and variables
//   - tokens: Input token sequence [batch, seqLen]
//
// Returns:
//   - logits: Output logits [batch, seqLen, vocabSize]
type ModelFn func(ctx *context.Context, tokens *Node) *Node

// IncrementalModelFn represents an incremental model function with KV caching.
// Processes new tokens at a specific position, reusing cached key/value projections.
//
// Parameters:
//   - ctx: Context containing model parameters and KV cache variables
//   - newTokens: New tokens to process [batch, newLen]
//   - position: Position in the sequence (for cache indexing)
//
// Returns:
//   - logits: Output logits [batch, newLen, vocabSize]
type IncrementalModelFn func(ctx *context.Context, newTokens *Node, position int) *Node

// Decoder configures and executes autoregressive text generation.
// Supports multiple sampling strategies (greedy, temperature, top-k, nucleus, beam search)
// with optional KV caching for efficient incremental generation.
type Decoder struct {
	// Model functions (exactly one should be set)
	ModelFn            ModelFn            // Standard model for non-cached generation
	IncrementalModelFn IncrementalModelFn // Incremental model for KV-cached generation

	// Generation parameters
	MaxLength   int
	Strategy    string // "greedy", "temperature", "top_k", "top_p", "beam_search"
	Temperature float32
	TopK        int
	TopP        float32
	BeamSize    int
	EosTokenId  int
	StopOnEOS   bool

	// Internal backend for small operations (concat, reshape)
	// Uses simplego for faster compilation and dynamic shape support
	simpleBackend backends.Backend
}

// New creates a decoder for autoregressive text generation.
// Accepts either ModelFn (non-cached, full sequence) or IncrementalModelFn (KV-cached).
// The decoder automatically detects which type is provided and configures itself accordingly.
//
// Parameters:
//   - modelFn: Either ModelFn for full-sequence processing or IncrementalModelFn for cached generation
//
// Returns:
//   - Decoder with default generation parameters (greedy sampling, maxLength=100)
//
// Example:
//
//	// Non-cached generation
//	decoder := decode.New(model.FullSequence())
//
//	// KV-cached generation
//	decoder := decode.New(model.Incremental())
//	decoder.WithStrategy("temperature").WithTemperature(0.8)
func New[M interface{ ModelFn | IncrementalModelFn }](modelFn M) *Decoder {
	var decoder *Decoder
	switch typedModelFn := any(modelFn).(type) {
	case ModelFn:
		decoder = &Decoder{
			ModelFn: typedModelFn,
		}
	case IncrementalModelFn:
		decoder = &Decoder{
			IncrementalModelFn: typedModelFn,
		}
	}

	// Set default parameters
	decoder.MaxLength = 100
	decoder.Strategy = "greedy"
	decoder.Temperature = 1.0
	decoder.TopK = 50
	decoder.TopP = 0.9
	decoder.BeamSize = 4
	decoder.EosTokenId = -1
	decoder.StopOnEOS = false

	// Initialize simplego backend for small operations (concat, reshape)
	// Faster than XLA for small ops and supports dynamic shapes
	decoder.simpleBackend, _ = simplego.New("decode")

	return decoder
}

// FromContext configures the decoder with hyperparameters from the context.
// This allows fine-tuning an existing decoder configuration.
//
// Supported hyperparameters:
//   - decode_max_length: Maximum generation length (default: unchanged)
//   - decode_strategy: Sampling strategy ("greedy", "temperature", "top_k", "top_p", "beam_search")
//   - decode_temperature: Temperature for sampling (default: unchanged)
//   - decode_top_k: k for top-k sampling (default: unchanged)
//   - decode_top_p: p for nucleus sampling (default: unchanged)
//   - decode_beam_size: Beam size for beam search (default: unchanged)
//   - decode_eos_token_id: End-of-sequence token ID (default: unchanged)
//   - decode_stop_on_eos: Whether to stop on EOS (default: unchanged)
//
// Example:
//
//	ctx.SetParams(map[string]any{
//	    "decode_strategy": "temperature",
//	    "decode_temperature": 0.8,
//	    "decode_max_length": 200,
//	})
//	decoder.FromContext(ctx)
func (cfg *Decoder) FromContext(ctx *context.Context) *Decoder {
	// Optional parameters with defaults
	cfg.MaxLength = context.GetParamOr(ctx, ParamMaxLength, cfg.MaxLength)
	cfg.Strategy = context.GetParamOr(ctx, ParamStrategy, cfg.Strategy)
	cfg.Temperature = context.GetParamOr(ctx, ParamTemperature, cfg.Temperature)
	cfg.TopK = context.GetParamOr(ctx, ParamTopK, cfg.TopK)
	cfg.TopP = context.GetParamOr(ctx, ParamTopP, cfg.TopP)
	cfg.BeamSize = context.GetParamOr(ctx, ParamBeamSize, cfg.BeamSize)
	cfg.EosTokenId = context.GetParamOr(ctx, ParamEosTokenId, cfg.EosTokenId)
	cfg.StopOnEOS = context.GetParamOr(ctx, ParamStopOnEOS, cfg.StopOnEOS)

	return cfg
}

// WithMaxLength sets the maximum generation length (including prompt).
func (cfg *Decoder) WithMaxLength(maxLength int) *Decoder {
	cfg.MaxLength = maxLength
	return cfg
}

// WithStrategy sets the sampling strategy.
// Options: "greedy", "temperature", "top_k", "top_p", "beam_search".
func (cfg *Decoder) WithStrategy(strategy string) *Decoder {
	cfg.Strategy = strategy
	return cfg
}

// WithTemperature sets the temperature for sampling.
// Higher values (>1.0) increase randomness, lower values (<1.0) make output more deterministic.
func (cfg *Decoder) WithTemperature(temperature float32) *Decoder {
	cfg.Temperature = temperature
	return cfg
}

// WithTopK sets k for top-k sampling.
// Only the k most likely tokens are considered at each step.
func (cfg *Decoder) WithTopK(topK int) *Decoder {
	cfg.TopK = topK
	return cfg
}

// WithTopP sets p for nucleus sampling.
// Tokens with cumulative probability up to p are considered.
func (cfg *Decoder) WithTopP(topP float32) *Decoder {
	cfg.TopP = topP
	return cfg
}

// WithBeamSize sets the beam size for beam search.
// Higher values explore more candidates but are slower.
func (cfg *Decoder) WithBeamSize(beamSize int) *Decoder {
	cfg.BeamSize = beamSize
	return cfg
}

// WithEOS sets the end-of-sequence token ID and enables early stopping.
// Generation stops when this token is produced.
func (cfg *Decoder) WithEOS(eosTokenId int) *Decoder {
	cfg.EosTokenId = eosTokenId
	cfg.StopOnEOS = true
	return cfg
}

// isCached returns true if this decoder uses KV caching.
func (cfg *Decoder) isCached() bool {
	return cfg.IncrementalModelFn != nil
}

// validate checks that the decoder configuration is valid.
// Returns an error if required fields are missing or invalid.
func (cfg *Decoder) validate() error {
	// Exactly one model function must be set
	if cfg.ModelFn != nil && cfg.IncrementalModelFn != nil {
		return errors.Errorf("cannot set both ModelFn and IncrementalModelFn")
	}
	if cfg.ModelFn == nil && cfg.IncrementalModelFn == nil {
		return errors.Errorf("must set either ModelFn or IncrementalModelFn")
	}

	return nil
}

// Decode performs autoregressive text generation from a prompt.
// The prompt can be 1D [seqLen] or 2D [batch, seqLen] tensor of token IDs.
//
// Parameters:
//   - backend: Backend for computation
//   - ctx: Context containing model parameters
//   - prompt: Input token sequence (1D or 2D tensor)
//
// Returns:
//   - Generated token sequence including the prompt
//   - Error if generation fails or configuration is invalid
//
// Example:
//
//	prompt := []int32{1, 2, 3}  // Token IDs
//	output, err := decoder.Decode(backend, ctx, prompt)
func (cfg *Decoder) Decode(
	backend backends.Backend,
	ctx *context.Context,
	prompt any,
) (*tensors.Tensor, error) {
	// Validate configuration
	if err := cfg.validate(); err != nil {
		return nil, errors.WithMessagef(err, "invalid generation config")
	}

	promptTensor := tensors.FromAnyValue(prompt)

	if promptTensor.Rank() == 1 {
		// Reshape is not a method on Tensor, so we need to work with the shape
		// For now, we'll use the prompt as-is in the graph
	}
	if promptTensor.Rank() != 2 && promptTensor.Rank() != 1 {
		return nil, errors.Errorf("prompt must be 1D or 2D, got rank %d", promptTensor.Rank())
	}

	promptShape := promptTensor.Shape()
	var promptLen int
	if promptTensor.Rank() == 1 {
		promptLen = promptShape.Dimensions[0]
	} else {
		promptLen = promptShape.Dimensions[1]
	}

	if promptLen >= cfg.MaxLength {
		return nil, errors.Errorf("prompt length %d >= max length %d", promptLen, cfg.MaxLength)
	}

	if cfg.Strategy == "beam_search" {
		return cfg.generateBeamSearch(backend, ctx, promptTensor)
	}

	// Regular sampling-based generation
	return cfg.generateSampling(backend, ctx, promptTensor)
}

// generateSampling performs generation using sampling strategies.
// Handles both cached and non-cached generation.
//
// Parameters:
//   - backend: Backend for computation
//   - ctx: Context containing model parameters
//   - prompt: Input token sequence [seqLen] or [batch, seqLen]
//
// Returns:
//   - Generated sequence [batch, totalLen] where totalLen <= MaxLength
//   - Error if generation fails
func (cfg *Decoder) generateSampling(
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
			return nil, errors.WithMessagef(err, "failed to create reshape exec")
		}

		reshapeResults, _, err := reshapeExec.ExecWithGraph(prompt)
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to reshape prompt")
		}
		prompt = reshapeResults[0]
		batchSize = 1
		promptLen = promptShape.Dimensions[0]
	} else {
		batchSize = promptShape.Dimensions[0]
		promptLen = promptShape.Dimensions[1]
	}

	if promptLen >= cfg.MaxLength {
		return nil, errors.Errorf("prompt length %d >= max length %d", promptLen, cfg.MaxLength)
	}

	if cfg.isCached() {
		return cfg.generateSamplingCached(backend, ctx, prompt, batchSize, promptLen)
	}
	return cfg.generateSamplingNonCached(backend, ctx, prompt, promptLen)
}

// generateSamplingNonCached performs sampling-based generation without KV caching.
// Processes the full sequence at each step.
//
// Parameters:
//   - backend: Backend for computation
//   - ctx: Context containing model parameters
//   - prompt: Input token sequence [batch, seqLen]
//   - promptLen: Length of the prompt sequence
//
// Returns:
//   - Generated sequence [batch, totalLen] where totalLen <= MaxLength
//   - Error if generation fails
func (cfg *Decoder) generateSamplingNonCached(
	backend backends.Backend,
	ctx *context.Context,
	prompt *tensors.Tensor,
	promptLen int,
) (*tensors.Tensor, error) {
	concatExec, err := NewExec(backend, func(seq, token *Node) *Node {
		tokenReshaped := ExpandDims(token, -1) // [batch, 1]
		return Concatenate([]*Node{seq, tokenReshaped}, 1)
	})
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to create concat graph")
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
			nextToken := generation.SampleWithStrategy(ctx, lastLogits, cfg.Strategy, float64(cfg.Temperature), cfg.TopK, float64(cfg.TopP))
			return nextToken
		})
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to create generation graph")
		}

		outputs, err := exec.Exec(currentSeq)
		if err != nil {
			return nil, errors.WithMessagef(err, "generation step %d failed", step)
		}

		nextToken := outputs[0]

		if cfg.StopOnEOS && cfg.EosTokenId >= 0 {
			if cfg.checkEOS(nextToken) {
				concatResults, err := concatExec.Exec(currentSeq, nextToken)
				if err != nil {
					return nil, errors.WithMessagef(err, "final concatenation failed")
				}
				return concatResults[0], nil
			}
		}

		concatResults, err := concatExec.Exec(currentSeq, nextToken)
		if err != nil {
			return nil, errors.WithMessagef(err, "concatenation step %d failed", step)
		}

		currentSeq = concatResults[0]
	}

	return currentSeq, nil
}

// generateSamplingCached performs sampling-based generation with KV caching.
// Processes only new tokens at each step, reusing cached attention.
//
// Parameters:
//   - backend: Backend for computation
//   - ctx: Context containing model parameters and KV cache
//   - prompt: Input token sequence [batch, seqLen]
//   - batchSize: Batch size (unused, kept for consistency)
//   - promptLen: Length of the prompt sequence
//
// Returns:
//   - Generated sequence [batch, totalLen] where totalLen <= MaxLength
//   - Error if generation fails
func (cfg *Decoder) generateSamplingCached(
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
		nextToken := generation.SampleWithStrategy(ctx, lastLogits, cfg.Strategy, float64(cfg.Temperature), cfg.TopK, float64(cfg.TopP))
		return nextToken
	})
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to create prompt exec")
	}

	outputs, err := promptExec.Exec(prompt)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to process prompt")
	}

	firstToken := outputs[0]

	if cfg.StopOnEOS && cfg.EosTokenId >= 0 && cfg.checkEOS(firstToken) {
		concatExec, _ := NewExec(backend, func(seq, token *Node) *Node {
			tokenReshaped := ExpandDims(token, -1)
			return Concatenate([]*Node{seq, tokenReshaped}, 1)
		})

		concatExec.SetMaxCache(-1)
		result, err := concatExec.Exec(prompt, firstToken)
		if err != nil {
			return nil, errors.WithMessagef(err, "final concatenation failed")
		}

		return result[0], nil
	}

	outputTokens := make([]*tensors.Tensor, 0, cfg.MaxLength)
	outputTokens = append(outputTokens, prompt, firstToken)

	numTokensToGenerate := cfg.MaxLength - promptLen - 1
	if numTokensToGenerate <= 0 {
		return cfg.concatenateTokens(backend, outputTokens)
	}

	genCtx := ctx.Reuse()

	currentPosition := promptLen
	for step := 0; step < numTokensToGenerate; step++ {
		position := currentPosition + step

		exec, err := context.NewExec(backend, genCtx, func(ctx *context.Context, token *Node) *Node {
			tokenReshaped := ExpandDims(token, -1)
			logits := cfg.IncrementalModelFn(ctx, tokenReshaped, position)
			lastLogits := Squeeze(logits, 1)
			nextToken := generation.SampleWithStrategy(ctx, lastLogits, cfg.Strategy, float64(cfg.Temperature), cfg.TopK, float64(cfg.TopP))
			return nextToken
		})
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to create generation exec at position %d", position)
		}

		prevToken := outputTokens[len(outputTokens)-1]

		outputs, err := exec.Exec(prevToken)
		if err != nil {
			return nil, errors.WithMessagef(err, "generation step %d (position %d) failed", step, position)
		}

		nextToken := outputs[0]

		if cfg.StopOnEOS && cfg.EosTokenId >= 0 && cfg.checkEOS(nextToken) {
			outputTokens = append(outputTokens, nextToken)
			break
		}

		outputTokens = append(outputTokens, nextToken)
	}

	return cfg.concatenateTokens(backend, outputTokens)
}

// checkEOS returns true if any token in the tensor matches the EOS token ID.
func (cfg *Decoder) checkEOS(token *tensors.Tensor) bool {
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

// concatenateTokens concatenates a sequence of token tensors along the sequence dimension.
func (cfg *Decoder) concatenateTokens(
	backend backends.Backend,
	tokens []*tensors.Tensor,
) (*tensors.Tensor, error) {
	if len(tokens) == 0 {
		return nil, errors.Errorf("no tokens to concatenate")
	}
	if len(tokens) == 1 {
		return tokens[0], nil
	}

	result := tokens[0]
	concatExec, err := NewExec(backend, func(seq, token *Node) *Node {
		tokenReshaped := ExpandDims(token, -1) // [batch] -> [batch, 1]
		return Concatenate([]*Node{seq, tokenReshaped}, 1)
	})
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to create concatenation exec")
	}
	concatExec.SetMaxCache(-1)

	for i := 1; i < len(tokens); i++ {
		outputs, err := concatExec.Exec(result, tokens[i])
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to concatenate token %d", i)
		}
		result = outputs[0]
	}

	return result, nil
}

// generateBeamSearch performs beam search generation.
// Dispatches to cached or non-cached implementation based on configuration.
func (cfg *Decoder) generateBeamSearch(
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
			return nil, errors.WithMessagef(err, "failed to create reshape exec")
		}

		reshapeResults, _, err := reshapeExec.ExecWithGraph(prompt)
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to reshape prompt")
		}
		prompt = reshapeResults[0]
		batchSize = 1
		promptLen = promptShape.Dimensions[0]
	} else {
		batchSize = promptShape.Dimensions[0]
		promptLen = promptShape.Dimensions[1]
	}

	if promptLen >= cfg.MaxLength {
		return nil, errors.Errorf("prompt length %d >= max length %d", promptLen, cfg.MaxLength)
	}

	// Dispatch cached or non-cached
	if cfg.isCached() {
		return cfg.generateBeamSearchCached(backend, ctx, prompt, batchSize, promptLen)
	}
	return cfg.generateBeamSearchNonCached(backend, ctx, prompt, batchSize, promptLen)
}

// generateBeamSearchNonCached performs beam search without KV caching.
// Maintains multiple beam hypotheses and selects the best sequence.
func (cfg *Decoder) generateBeamSearchNonCached(
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
		return nil, errors.WithMessagef(err, "failed to create replicate exec")
	}

	replicatedResults, _, err := replicateExec.ExecWithGraph(prompt)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to replicate prompt")
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
	beamConfig := generation.NewBeamSearch(beamSize, cfg.EosTokenId).
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
			return nil, errors.WithMessagef(err, "failed to create beam search exec at step %d", step)
		}

		outputs, _, err := exec.ExecWithGraph(currentSequences, beamScores)
		if err != nil {
			return nil, errors.WithMessagef(err, "beam search step %d failed", step)
		}

		currentSequences = outputs[0]
		beamScores = outputs[1]
		isFinished := outputs[2]

		// All beams finished?
		if beamConfig.EarlyStopping() {
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
		return nil, errors.WithMessagef(err, "failed to create select exec")
	}

	selectResults, _, err := selectExec.ExecWithGraph(currentSequences, beamScores)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to select best sequences")
	}

	return selectResults[0], nil
}

// generateBeamSearchCached performs beam search with KV caching.
// Each beam maintains its own cache for efficient incremental generation.
func (cfg *Decoder) generateBeamSearchCached(
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
		return nil, errors.WithMessagef(err, "failed to create replicate exec")
	}

	replicatedResults, _, err := replicateExec.ExecWithGraph(prompt)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to replicate prompt")
	}

	replicatedPrompt := replicatedResults[0]

	// Process prompt to populate caches
	promptExec, err := context.NewExec(backend, ctx.Reuse(), func(ctx *context.Context, tokens *Node) *Node {
		logits := cfg.IncrementalModelFn(ctx, tokens, 0)
		lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
		return Squeeze(lastLogits, 1)
	})
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to create prompt exec")
	}

	_, _, err = promptExec.ExecWithGraph(replicatedPrompt)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to process prompt")
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
	beamConfig := generation.NewBeamSearch(beamSize, cfg.EosTokenId).
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
			return nil, errors.WithMessagef(err, "failed to create extract exec")
		}

		extractResults, _, err := extractExec.ExecWithGraph(currentSequences)
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to extract last tokens")
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
			return nil, errors.WithMessagef(err, "failed to create beam step exec")
		}

		outputs, _, err := exec.ExecWithGraph(lastTokens, beamScores, currentSequences)
		if err != nil {
			return nil, errors.WithMessagef(err, "beam search step %d failed", step)
		}

		currentSequences = outputs[0]
		beamScores = outputs[1]
		isFinished := outputs[2]

		// Early stopping check
		if beamConfig.EarlyStopping() {
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
		return nil, errors.WithMessagef(err, "failed to create select exec")
	}

	selectResults, _, err := selectExec.ExecWithGraph(currentSequences, beamScores)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to select best sequences")
	}

	return selectResults[0], nil
}

// GenerateStreaming performs streaming generation, yielding tokens as they are generated.
// The callback function is called for each generated token and can return false to stop generation.
//
// Parameters:
//   - backend: Backend for computation
//   - ctx: Context containing model parameters
//   - prompt: Input token sequence (1D or 2D tensor)
//   - callback: Function called with each generated token ID; return false to stop
//
// Returns:
//   - Error if generation fails
//
// Note: This is a placeholder for future streaming support.
func (cfg *Decoder) GenerateStreaming(
	backend backends.Backend,
	ctx *context.Context,
	prompt any,
	callback func(token int) bool,
) error {
	return errors.Errorf("streaming generation not yet implemented")
}
