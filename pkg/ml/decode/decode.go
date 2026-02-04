// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package decode

import (
	"sync"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/simplego"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/decode/sample"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
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

// IterativeModelFn represents a full-sequence model function,
// that is used iteratively during decoding, by re-feeding the
// whole previous state.
//
// Only use this if your backend supports dynamic shapes, otherwise
// this will likely be very slow.
//
// Parameters:
//   - ctx: Model context passed along by the Decoder.
//   - tokens: Input token sequence, shaped [batch, seqLen].
//
// Returns:
//   - logits: Output logits per candidate token (vocabSize), shaped [batch, seqLen, vocabSize]
type IterativeModelFn func(ctx *context.Context, tokens *Node) *Node

// IncrementalModelFn represents an incremental model function, which is expected to have
// some form of caching (stored in the variables in the context).
//
// It processes new tokens at a specific position, presumably reusing cached previous results
// (typically in the form of a "KVCache" or cached key/value projections, stored in the context).
//
// Currently, it doesn't support concurrent generation -- it's expected to have only one
// cache during execution.
//
// Parameters:
//   - ctx: Model context passed along by the Decoder. Likely contains the cache during the execution.
//   - newTokens: New tokens to process [batch, newLen]
//   - position: Position in the sequence (for cache indexing)
//
// Returns:
//   - logits: Output logits [batch, newLen, vocabSize]
type IncrementalModelFn func(ctx *context.Context, newTokens *Node, position int) *Node

// Decoder configures and executes autoregressive text generation.
// Supports multiple sampling strategies (greedy, temperature, top-k, nucleus, beam search)
// and incremeantal models, usually using some form of cache (e.g.: a "KV-cache").
type Decoder struct {
	// Model functions (exactly one should be set)
	ModelFn            IterativeModelFn   // Standard model for non-cached generation
	IncrementalModelFn IncrementalModelFn // Incremental model for KV-cached generation

	// Generation parameters
	MaxLength   int
	Strategy    sample.Strategy
	Temperature float32
	TopK        int
	TopP        float32
	BeamSize    int
	EosTokenId  int
	StopOnEOS   bool

	// Internal backend for small operations (concat, reshape)
	// Uses simplego for faster compilation and dynamic shape support
	simpleBackend backends.Backend

	// Cached executors to avoid recompilation
	promptExec   *context.Exec         // Cached executor for processing initial prompt in incremental generation
	genExecCache map[int]*context.Exec // Cached executors for each position in incremental generation
	fullExec     *context.Exec         // Cached executor for non-cached full sequence generation

	// err is a delayed error set during initialization, and returned by Decode.
	err error
	mu  sync.Mutex
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
//	decoder := decode.New(model.Incremental()).
//		WithStrategy(sample.StrategyTemperature).WithTemperature(0.8)
//	output, err := decoder.Decode(prompt)
func New[M interface {
	IterativeModelFn | IncrementalModelFn
}](modelFn M) *Decoder {
	var decoder *Decoder
	switch typedModelFn := any(modelFn).(type) {
	case IterativeModelFn:
		klog.Warning("Using Decoder to generate text with interactive model is not yet well supported, and will likely be very slow. Consider using an IncrementalModelFn instead.")
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
	decoder.Strategy = sample.StrategyGreedy
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
func (dec *Decoder) FromContext(ctx *context.Context) *Decoder {
	dec.mu.Lock()
	defer dec.mu.Unlock()
	if dec.promptExec != nil {
		dec.err = errors.Errorf("cannot change configuration of decode.Decoder, once it was used for generation -- create a new Decoder if you need a different configuration.")
		return dec
	}
	// Optional parameters with defaults
	dec.MaxLength = context.GetParamOr(ctx, ParamMaxLength, dec.MaxLength)
	strategyName := context.GetParamOr(ctx, ParamStrategy, "")
	if strategyName != "" {
		var err error
		dec.Strategy, err = sample.StrategyString(strategyName)
		if err != nil {
			dec.err = errors.Wrapf(err, "failed to parse sampling strategy %q, valid values are %v", strategyName, sample.StrategyStrings())
			return dec
		}
	}
	dec.Temperature = context.GetParamOr(ctx, ParamTemperature, dec.Temperature)
	dec.TopK = context.GetParamOr(ctx, ParamTopK, dec.TopK)
	dec.TopP = context.GetParamOr(ctx, ParamTopP, dec.TopP)
	dec.BeamSize = context.GetParamOr(ctx, ParamBeamSize, dec.BeamSize)
	dec.EosTokenId = context.GetParamOr(ctx, ParamEosTokenId, dec.EosTokenId)
	dec.StopOnEOS = context.GetParamOr(ctx, ParamStopOnEOS, dec.StopOnEOS)
	return dec
}

// initializePromptExec creates the cached executor for processing prompts in incremental generation.
// This executor is reused across multiple generation calls to avoid recompilation overhead.
func (dec *Decoder) initializePromptExec(backend backends.Backend, ctx *context.Context) error {
	if dec.promptExec != nil || dec.IncrementalModelFn == nil {
		return nil
	}

	var err error
	dec.promptExec, err = context.NewExec(backend, ctx.Reuse(), func(ctx *context.Context, tokens *Node) *Node {
		logits := dec.IncrementalModelFn(ctx, tokens, 0)
		lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
		lastLogits = Squeeze(lastLogits, 1) // [batch, vocab_size]
		nextToken := sample.SampleWithStrategy(ctx, lastLogits, dec.Strategy, float64(dec.Temperature), dec.TopK, float64(dec.TopP))
		return nextToken
	})
	if err != nil {
		return errors.WithMessagef(err, "failed to create prompt exec")
	}

	return nil
}

// WithMaxLength sets the maximum generation length (including prompt).
func (dec *Decoder) WithMaxLength(maxLength int) *Decoder {
	dec.mu.Lock()
	defer dec.mu.Unlock()
	if dec.promptExec != nil {
		dec.err = errors.Errorf("cannot change configuration of decode.Decoder, once it was used for generation -- create a new Decoder if you need a different configuration.")
		return dec
	}
	dec.MaxLength = maxLength
	return dec
}

// WithStrategy sets the sampling strategy.
// The default strategy is sample.StrategyGreedy, which always take the immediately most likely next token.
func (dec *Decoder) WithStrategy(strategy sample.Strategy) *Decoder {
	dec.mu.Lock()
	defer dec.mu.Unlock()
	if dec.promptExec != nil {
		dec.err = errors.Errorf("cannot change configuration of decode.Decoder, once it was used for generation -- create a new Decoder if you need a different configuration.")
		return dec
	}
	dec.Strategy = strategy
	return dec
}

// WithTemperature sets the temperature for sampling.
// Higher values (>1.0) increase randomness, lower values (<1.0) make output more deterministic.
//
// You have to also set the strategy to sample.StrategyTemperature or sample.StrategyTopK or
// sample.StrategyTopP to use this parameter
func (dec *Decoder) WithTemperature(temperature float32) *Decoder {
	dec.mu.Lock()
	defer dec.mu.Unlock()
	if dec.promptExec != nil {
		dec.err = errors.Errorf("cannot change configuration of decode.Decoder, once it was used for generation -- create a new Decoder if you need a different configuration.")
		return dec
	}
	dec.Temperature = temperature
	return dec
}

// WithTopK sets k for top-k sampling.
// Only the k most likely tokens are considered at each step.
func (dec *Decoder) WithTopK(topK int) *Decoder {
	dec.mu.Lock()
	defer dec.mu.Unlock()
	if dec.promptExec != nil {
		dec.err = errors.Errorf("cannot change configuration of decode.Decoder, once it was used for generation -- create a new Decoder if you need a different configuration.")
		return dec
	}
	dec.TopK = topK
	return dec
}

// WithTopP sets p for nucleus sampling.
// Tokens with cumulative probability up to p are considered.
func (dec *Decoder) WithTopP(topP float32) *Decoder {
	dec.mu.Lock()
	defer dec.mu.Unlock()
	if dec.promptExec != nil {
		dec.err = errors.Errorf("cannot change configuration of decode.Decoder, once it was used for generation -- create a new Decoder if you need a different configuration.")
		return dec
	}
	dec.TopP = topP
	return dec
}

// WithBeamSize sets the beam size for beam search.
// Higher values explore more candidates but are slower.
func (dec *Decoder) WithBeamSize(beamSize int) *Decoder {
	dec.mu.Lock()
	defer dec.mu.Unlock()
	if dec.promptExec != nil {
		dec.err = errors.Errorf("cannot change configuration of decode.Decoder, once it was used for generation -- create a new Decoder if you need a different configuration.")
		return dec
	}
	dec.BeamSize = beamSize
	return dec
}

// WithEOS sets the end-of-sequence token ID and enables early stopping.
// Generation stops when this token is produced.
func (dec *Decoder) WithEOS(eosTokenId int) *Decoder {
	dec.mu.Lock()
	defer dec.mu.Unlock()
	if dec.promptExec != nil {
		dec.err = errors.Errorf("cannot change configuration of decode.Decoder, once it was used for generation -- create a new Decoder if you need a different configuration.")
		return dec
	}
	dec.EosTokenId = eosTokenId
	dec.StopOnEOS = true
	return dec
}

// isCached returns true if this decoder uses KV caching.
func (dec *Decoder) isCached() bool {
	return dec.IncrementalModelFn != nil
}

// validate checks that the decoder configuration is valid.
// Returns an error if required fields are missing or invalid.
func (dec *Decoder) validate() error {
	if dec.err != nil {
		return errors.WithMessagef(dec.err, "Decoder failed during configuration")
	}
	// Exactly one model function must be set
	if dec.ModelFn != nil && dec.IncrementalModelFn != nil {
		return errors.Errorf("cannot set both ModelFn and IncrementalModelFn")
	}
	if dec.ModelFn == nil && dec.IncrementalModelFn == nil {
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
func (dec *Decoder) Decode(
	backend backends.Backend,
	ctx *context.Context,
	prompt any,
) (*tensors.Tensor, error) {
	dec.mu.Lock()
	defer dec.mu.Unlock()

	// Validate configuration
	if err := dec.validate(); err != nil {
		return nil, errors.WithMessagef(err, "invalid Decoder config")
	}

	// Initialize cached executors for incremental generation
	if err := dec.initializePromptExec(backend, ctx); err != nil {
		return nil, err
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

	if promptLen >= dec.MaxLength {
		return nil, errors.Errorf("prompt length %d >= max length %d", promptLen, dec.MaxLength)
	}

	if dec.Strategy == sample.StrategyBeamSearch {
		return dec.generateBeamSearch(backend, ctx, promptTensor)
	}

	// Regular sampling-based generation
	return dec.generateSampling(backend, ctx, promptTensor)
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
func (dec *Decoder) generateSampling(
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

		reshapeResults, err := reshapeExec.Exec(prompt)
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

	if promptLen >= dec.MaxLength {
		return nil, errors.Errorf("prompt length %d >= max length %d", promptLen, dec.MaxLength)
	}

	if dec.isCached() {
		return dec.generateSamplingIncremental(backend, ctx, prompt, batchSize, promptLen)
	}
	return dec.generateSamplingFull(backend, ctx, prompt, promptLen)
}

// generateSamplingFull performs sampling-based generation without KV caching.
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
func (dec *Decoder) generateSamplingFull(
	backend backends.Backend,
	ctx *context.Context,
	prompt *tensors.Tensor,
	promptLen int,
) (*tensors.Tensor, error) {
	numTokensToGenerate := dec.MaxLength - promptLen
	if numTokensToGenerate <= 0 {
		return prompt, nil
	}

	// Extract batch size from prompt shape
	promptShape := prompt.Shape()
	batchSize := promptShape.Dimensions[0]

	// Store generated tokens as int32 values [batch][position]
	outputTokens := make([][]int32, batchSize)
	for i := range outputTokens {
		outputTokens[i] = make([]int32, 0, dec.MaxLength)
	}

	// Add prompt tokens to output
	promptValues := prompt.Value().([][]int32)
	for i := range batchSize {
		outputTokens[i] = append(outputTokens[i], promptValues[i]...)
	}

	// Create or reuse cached executor for full sequence generation
	if dec.fullExec == nil {
		predCtx := ctx.Reuse()
		var err error
		dec.fullExec, err = context.NewExec(backend, predCtx, func(ctx *context.Context, currentSeq *Node) *Node {
			logits := dec.ModelFn(ctx, currentSeq)
			lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
			lastLogits = Squeeze(lastLogits, 1) // Remove the seq_len dimension
			nextToken := sample.SampleWithStrategy(ctx, lastLogits, dec.Strategy, float64(dec.Temperature), dec.TopK, float64(dec.TopP))
			return nextToken
		})
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to create generation exec")
		}
	}

	for step := range numTokensToGenerate {
		// Build current sequence tensor from accumulated tokens
		currentSeq := tensors.FromValue(outputTokens)

		outputs, err := dec.fullExec.Exec(currentSeq)
		if err != nil {
			return nil, errors.WithMessagef(err, "generation step %d failed", step)
		}

		nextToken := outputs[0]
		nextTokenValues := nextToken.Value().([]int32)

		// Check for EOS and append tokens
		allEOS := true
		for i := range batchSize {
			outputTokens[i] = append(outputTokens[i], nextTokenValues[i])
			if dec.StopOnEOS && dec.EosTokenId >= 0 && int(nextTokenValues[i]) != dec.EosTokenId {
				allEOS = false
			}
		}

		if dec.StopOnEOS && dec.EosTokenId >= 0 && allEOS {
			break
		}
	}

	return tensors.FromValue(outputTokens), nil
}

// generateSamplingIncremental performs sampling-based generation with KV caching.
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
func (dec *Decoder) generateSamplingIncremental(
	backend backends.Backend,
	ctx *context.Context,
	prompt *tensors.Tensor,
	_, promptLen int,
) (*tensors.Tensor, error) {
	// Use cached promptExec (initialized in Decode)
	outputs, err := dec.promptExec.Exec(prompt)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to process prompt")
	}

	firstToken := outputs[0]

	if dec.StopOnEOS && dec.EosTokenId >= 0 && dec.checkEOS(firstToken) {
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

	// Extract batch size from prompt shape
	promptShape := prompt.Shape()
	batchSize := promptShape.Dimensions[0]

	// Store generated tokens as int32 values [batch][position]
	outputTokens := make([][]int32, batchSize)
	for i := range outputTokens {
		outputTokens[i] = make([]int32, 0, dec.MaxLength)
	}

	// Add prompt tokens to output
	promptValues := prompt.Value().([][]int32)
	for i := range batchSize {
		outputTokens[i] = append(outputTokens[i], promptValues[i]...)
	}

	// Add first generated token
	firstTokenValues := firstToken.Value().([]int32)
	for i := range batchSize {
		outputTokens[i] = append(outputTokens[i], firstTokenValues[i])
	}

	numTokensToGenerate := dec.MaxLength - promptLen - 1
	if numTokensToGenerate <= 0 {
		return tensors.FromValue(outputTokens), nil
	}

	genCtx := ctx.Reuse()

	// Initialize cache if needed
	if dec.genExecCache == nil {
		dec.genExecCache = make(map[int]*context.Exec)
	}

	currentPosition := promptLen
	for step := 0; step < numTokensToGenerate; step++ {
		position := currentPosition + step

		// Get or create cached executor for this position
		exec, ok := dec.genExecCache[position]
		if !ok {
			var err error
			exec, err = context.NewExec(backend, genCtx, func(ctx *context.Context, token *Node) *Node {
				tokenReshaped := ExpandDims(token, -1)
				logits := dec.IncrementalModelFn(ctx, tokenReshaped, position)
				lastLogits := Squeeze(logits, 1)
				nextToken := sample.SampleWithStrategy(ctx, lastLogits, dec.Strategy, float64(dec.Temperature), dec.TopK, float64(dec.TopP))
				return nextToken
			})
			if err != nil {
				return nil, errors.WithMessagef(err, "failed to create generation exec at position %d", position)
			}
			dec.genExecCache[position] = exec
		}

		// Get previous token from each batch as tensor
		prevTokens := make([]int32, batchSize)
		for i := range batchSize {
			prevTokens[i] = outputTokens[i][len(outputTokens[i])-1]
		}
		prevTokenTensor := tensors.FromValue(prevTokens)

		outputs, err := exec.Exec(prevTokenTensor)
		if err != nil {
			return nil, errors.WithMessagef(err, "generation step %d (position %d) failed", step, position)
		}

		nextToken := outputs[0]
		nextTokenValues := nextToken.Value().([]int32)

		// Check for EOS and append tokens
		allEOS := true
		for i := range batchSize {
			outputTokens[i] = append(outputTokens[i], nextTokenValues[i])
			if dec.StopOnEOS && dec.EosTokenId >= 0 && int(nextTokenValues[i]) != dec.EosTokenId {
				allEOS = false
			}
		}

		if dec.StopOnEOS && dec.EosTokenId >= 0 && allEOS {
			break
		}
	}

	return tensors.FromValue(outputTokens), nil
}

// checkEOS returns true if any token in the tensor matches the EOS token ID.
func (dec *Decoder) checkEOS(token *tensors.Tensor) bool {
	tokenValue := token.Value()
	switch v := tokenValue.(type) {
	case []int32:
		for _, t := range v {
			if int(t) == dec.EosTokenId {
				return true
			}
		}
	case int32:
		return int(v) == dec.EosTokenId
	case []int64:
		for _, t := range v {
			if int(t) == dec.EosTokenId {
				return true
			}
		}
	case int64:
		return int(v) == dec.EosTokenId
	}
	return false
}

// generateBeamSearch performs beam search generation.
// Dispatches to cached or non-cached implementation based on configuration.
func (dec *Decoder) generateBeamSearch(
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

		reshapeResults, err := reshapeExec.Exec(prompt)
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

	if promptLen >= dec.MaxLength {
		return nil, errors.Errorf("prompt length %d >= max length %d", promptLen, dec.MaxLength)
	}

	// Dispatch cached or non-cached
	if dec.isCached() {
		return dec.generateBeamSearchCached(backend, ctx, prompt, batchSize, promptLen)
	}
	return dec.generateBeamSearchNonCached(backend, ctx, prompt, batchSize, promptLen)
}

// generateBeamSearchNonCached performs beam search without KV caching.
// Maintains multiple beam hypotheses and selects the best sequence.
func (dec *Decoder) generateBeamSearchNonCached(
	backend backends.Backend,
	ctx *context.Context,
	prompt *tensors.Tensor,
	batchSize, promptLen int,
) (*tensors.Tensor, error) {
	beamSize := dec.BeamSize
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

	replicatedResults, err := replicateExec.Exec(prompt)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to replicate prompt")
	}

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
	currentSequences := replicatedResults[0]

	// Beam search configuration
	beamConfig := sample.NewBeamSearch(beamSize, dec.EosTokenId).
		WithMaxLength(dec.MaxLength).
		WithLengthPenalty(1.0)

	// Main loop
	predCtx := ctx.Reuse()
	numSteps := dec.MaxLength - promptLen

	for step := 0; step < numSteps; step++ {
		currentLength := promptLen + step

		// TODO: It seems I cannot cache this exec because currentLength changes each iteration
		// and is used as a compile-time constant in the graph (passed to beamConfig.Step)
		// I leave it like this for now as I think we need the dynamic shape support of the simplego backend.
		exec, err := context.NewExec(backend, predCtx, func(ctx *context.Context, sequences, scores *Node) (*Node, *Node, *Node) {
			// Run model
			logits := dec.ModelFn(ctx, sequences)

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

		outputs, err := exec.Exec(currentSequences, beamScores)
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

	selectResults, err := selectExec.Exec(currentSequences, beamScores)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to select best sequences")
	}

	return selectResults[0], nil
}

// generateBeamSearchCached performs beam search with KV caching.
// Each beam maintains its own cache for efficient incremental generation.
func (dec *Decoder) generateBeamSearchCached(
	backend backends.Backend,
	ctx *context.Context,
	prompt *tensors.Tensor,
	batchSize, promptLen int,
) (*tensors.Tensor, error) {
	beamSize := dec.BeamSize

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

	replicatedResults, err := replicateExec.Exec(prompt)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to replicate prompt")
	}

	replicatedPrompt := replicatedResults[0]

	// Process prompt to populate caches
	promptExec, err := context.NewExec(backend, ctx.Reuse(), func(ctx *context.Context, tokens *Node) *Node {
		logits := dec.IncrementalModelFn(ctx, tokens, 0)
		lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
		return Squeeze(lastLogits, 1)
	})
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to create prompt exec")
	}

	_, err = promptExec.Exec(replicatedPrompt)
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
	beamConfig := sample.NewBeamSearch(beamSize, dec.EosTokenId).
		WithMaxLength(dec.MaxLength).
		WithLengthPenalty(1.0)

	// Main loop
	genCtx := ctx.Reuse()
	numSteps := dec.MaxLength - promptLen

	for step := 0; step < numSteps; step++ {
		position := promptLen + step

		// Note: Cannot cache extractExec across decode calls because it would need
		// to be invalidated when sequences change shape. Creating it per-step is acceptable
		// since it's a trivial slice operation.
		extractExec, err := context.NewExec(backend, ctx.Reuse(), func(ctx *context.Context, sequences *Node) *Node {
			// Last token: [batch_beam_size]
			lastTokens := Slice(sequences, AxisRange(), AxisElem(-1))
			return lastTokens
		})
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to create extract exec")
		}

		extractResults, err := extractExec.Exec(currentSequences)
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to extract last tokens")
		}

		lastTokens := extractResults[0]

		// Note: Cannot cache this exec because position changes each iteration
		// and is used as a compile-time constant in the graph (passed to IncrementalModelFn)
		exec, err := context.NewExec(backend, genCtx, func(ctx *context.Context, tokens, scores, sequences *Node) (*Node, *Node, *Node) {
			// tokens: [batch_beam_size]
			tokensReshaped := ExpandDims(tokens, -1) // [batch_beam_size, 1]

			// Process with incremental model
			logits := dec.IncrementalModelFn(ctx, tokensReshaped, position)
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

		outputs, err := exec.Exec(lastTokens, beamScores, currentSequences)
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

	selectResults, err := selectExec.Exec(currentSequences, beamScores)
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
func (dec *Decoder) GenerateStreaming(
	backend backends.Backend,
	ctx *context.Context,
	prompt any,
	callback func(token int) bool,
) error {
	dec.mu.Lock()
	defer dec.mu.Unlock()
	return errors.Errorf("streaming generation not yet implemented")
}
