// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package generate implements a generator for transformer models, including the sampling strategies: greedy, top-k, top-p and beam search.
//
// It defines a Generator object that can be initialized with a KVCacheModelFn or NaiveModelFn, depending
// if you want to use a KVCache (the usual, more performant way) or the "naive" generator (recalculate everything for each new token,
// waistful in memory and performance, but simpler and useful for testing).
//
// See New for details.
package generate

import (
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers/attention"
	"github.com/gomlx/gomlx/ml/layers/attention/kvcache"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/zoo/transformer/generate/sample"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// Hyperparameter keys for scope configuration
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

// NaiveModelFn represents a full-sequence model function,
// that is used iteratively during decoding, by re-feeding
// all the tokens each time.
//
// This is a slow way of generating, since it will re-run all the computation
// for the whole sequence each time.
//
// Parameters:
//   - scope: Model scope passed along by the Decoder.
//   - tokens: Input token sequence, shaped [batch, paddedSeqLen].
//   - length: Number of tokens that are non-padding.
//
// Returns:
//   - logits: Output logits per candidate token (vocabSize), shaped [batch, seqLen, vocabSize]
type NaiveModelFn func(scope *model.Scope, tokens *Node, length int) *Node

// KVCacheModelFn represents an incremental model function, which is expected to have
// an associated KVCache.
//
// It processes new tokens at a specific position, presumably reusing cached previous results
// (typically in the form of a "KVCache" or cached key/value projections, stored in the scope).
//
// Parameters:
//   - scope: Model scope passed along by the Decoder. Likely contains the cache during the execution.
//   - newTokens: New tokens to process [batch, newLen]
//   - position: Position (scalar Int32) in the sequence (for cache indexing)
//   - cache: KVCache nodes, usually in the form of KeyValueCaches
//
// Returns:
//   - logits: Output logits [batch, newLen, vocabSize]
//   - updatedCache: Updated KVCache nodes
type KVCacheModelFn func(scope *model.Scope, newTokens *Node, position *Node, cache kvcache.KVCacheNodes) (logits *Node, updatedCache kvcache.KVCacheNodes)

// Generator configures and executes autoregressive text generation.
// Supports multiple sampling strategies (greedy, temperature, top-k, nucleus, beam search)
// and incremeantal models, usually using some form of cache (e.g.: a "KV-cache").
type Generator struct {
	// Model functions (exactly one should be set)
	naiveModelFn NaiveModelFn   // Standard model for non-cached generation
	cacheModelFn KVCacheModelFn // Incremental model for KV-cached generation

	KVCache    *kvcache.KVCache
	numKVHeads int
	headDim    int
	dtype      dtypes.DType

	// Generation parameters
	MaxLength   int
	Strategy    sample.Strategy
	Temperature float32
	TopK        int
	TopP        float32
	BeamSize    int
	EosTokenId  int
	EosTokenIds []int
	StopOnEOS   bool

	// Internal backend for small operations (concat, reshape)
	// Uses simplego for faster compilation and dynamic shape support
	simpleBackend compute.Backend

	// Cached executors to avoid recompilation
	promptExecCache map[int]*model.Exec // Cached executors for processing initial prompt in incremental generation
	genExecCache    map[int]*model.Exec // Cached executors for each position in incremental generation
	fullExec        *model.Exec         // Cached executor for non-cached full sequence generation

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
	NaiveModelFn | KVCacheModelFn
}](modelFn M) *Generator {
	var decoder *Generator
	switch typedModelFn := any(modelFn).(type) {
	case NaiveModelFn:
		klog.Warning("Using Decoder to generate text with interactive model is not yet well supported, and will likely be very slow. Consider using an IncrementalModelFn instead.")
		decoder = &Generator{
			naiveModelFn: typedModelFn,
		}
	case KVCacheModelFn:
		decoder = &Generator{
			cacheModelFn: typedModelFn,
		}
	}

	decoder.promptExecCache = make(map[int]*model.Exec)
	decoder.genExecCache = make(map[int]*model.Exec)

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
	decoder.simpleBackend, _ = gobackend.New("decode")

	return decoder
}

func (gen *Generator) WithKVCache(kvCache *kvcache.KVCache, numKVHeads, headDim int, dtype dtypes.DType) *Generator {
	gen.mu.Lock()
	defer gen.mu.Unlock()
	gen.KVCache = kvCache
	gen.numKVHeads = numKVHeads
	gen.headDim = headDim
	gen.dtype = dtype
	return gen
}

func (gen *Generator) wasUsed() bool {
	return len(gen.promptExecCache) > 0 || len(gen.genExecCache) > 0 || gen.fullExec != nil
}

// FromScope configures the decoder with hyperparameters from the model.
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
//	scope.SetParams(map[string]any{
//	    "decode_strategy": "temperature",
//	    "decode_temperature": 0.8,
//	    "decode_max_length": 200,
//	})
//	decoder.FromScope(scope)
func (gen *Generator) FromScope(scope *model.Scope) *Generator {
	gen.mu.Lock()
	defer gen.mu.Unlock()
	if gen.wasUsed() {
		gen.err = errors.Errorf("cannot change configuration of decode.Decoder, once it was used for generation -- create a new Decoder if you need a different configuration.")
		return gen
	}
	// Optional parameters with defaults
	gen.MaxLength = model.GetParamOr(scope, ParamMaxLength, gen.MaxLength)
	strategyName := model.GetParamOr(scope, ParamStrategy, "")
	if strategyName != "" {
		var err error
		gen.Strategy, err = sample.StrategyString(strategyName)
		if err != nil {
			gen.err = errors.Wrapf(err, "failed to parse sampling strategy %q, valid values are %v", strategyName, sample.StrategyStrings())
			return gen
		}
	}
	gen.Temperature = model.GetParamOr(scope, ParamTemperature, gen.Temperature)
	gen.TopK = model.GetParamOr(scope, ParamTopK, gen.TopK)
	gen.TopP = model.GetParamOr(scope, ParamTopP, gen.TopP)
	gen.BeamSize = model.GetParamOr(scope, ParamBeamSize, gen.BeamSize)
	gen.EosTokenId = model.GetParamOr(scope, ParamEosTokenId, gen.EosTokenId)
	gen.StopOnEOS = model.GetParamOr(scope, ParamStopOnEOS, gen.StopOnEOS)
	return gen
}

func (gen *Generator) getPromptExec(backend compute.Backend, scope *model.Scope, cacheSeqLen int) (*model.Exec, error) {
	if gen.promptExecCache == nil {
		gen.promptExecCache = make(map[int]*model.Exec)
	}
	exec, ok := gen.promptExecCache[cacheSeqLen]
	if ok {
		return exec, nil
	}

	var err error
	exec, err = model.NewExec(backend, scope.Store(), func(scope *model.Scope, inputs []*Node) []*Node {
		tokens := inputs[0]
		cacheNodes := inputs[1:]
		cache := gen.KVCache.DeserializeNodes(cacheNodes)
		g := tokens.Graph()
		positionNode := Const(g, int32(0))
		logits, updatedCache := gen.cacheModelFn(scope, tokens, positionNode, cache)
		serializedUpdatedCache, err := gen.KVCache.SerializeNodes(updatedCache)
		if err != nil {
			panic(err)
		}
		lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
		lastLogits = Squeeze(lastLogits, 1)
		nextToken := sample.SampleWithStrategy(scope, lastLogits, gen.Strategy, float64(gen.Temperature), gen.TopK, float64(gen.TopP))
		res := make([]*Node, 1+len(serializedUpdatedCache))
		res[0] = nextToken
		copy(res[1:], serializedUpdatedCache)
		return res
	})
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to create prompt exec")
	}
	gen.promptExecCache[cacheSeqLen] = exec
	return exec, nil
}

// WithMaxLength sets the maximum generation length (including prompt).
func (gen *Generator) WithMaxLength(maxLength int) *Generator {
	gen.mu.Lock()
	defer gen.mu.Unlock()
	if gen.wasUsed() {
		gen.err = errors.Errorf("cannot change configuration of decode.Decoder, once it was used for generation -- create a new Decoder if you need a different configuration.")
		return gen
	}
	gen.MaxLength = maxLength
	return gen
}

// WithStrategy sets the sampling strategy.
// The default strategy is sample.StrategyGreedy, which always take the immediately most likely next token.
func (gen *Generator) WithStrategy(strategy sample.Strategy) *Generator {
	gen.mu.Lock()
	defer gen.mu.Unlock()
	if gen.wasUsed() {
		gen.err = errors.Errorf("cannot change configuration of decode.Decoder, once it was used for generation -- create a new Decoder if you need a different configuration.")
		return gen
	}
	gen.Strategy = strategy
	return gen
}

// WithTemperature sets the temperature for sampling.
// Higher values (>1.0) increase randomness, lower values (<1.0) make output more deterministic.
//
// You have to also set the strategy to sample.StrategyTemperature or sample.StrategyTopK or
// sample.StrategyTopP to use this parameter
func (gen *Generator) WithTemperature(temperature float32) *Generator {
	gen.mu.Lock()
	defer gen.mu.Unlock()
	if gen.wasUsed() {
		gen.err = errors.Errorf("cannot change configuration of decode.Decoder, once it was used for generation -- create a new Decoder if you need a different configuration.")
		return gen
	}
	gen.Temperature = temperature
	return gen
}

// WithTopK sets k for top-k sampling.
// Only the k most likely tokens are considered at each step.
func (gen *Generator) WithTopK(topK int) *Generator {
	gen.mu.Lock()
	defer gen.mu.Unlock()
	if gen.wasUsed() {
		gen.err = errors.Errorf("cannot change configuration of decode.Decoder, once it was used for generation -- create a new Decoder if you need a different configuration.")
		return gen
	}
	gen.TopK = topK
	return gen
}

// WithTopP sets p for nucleus sampling.
// Tokens with cumulative probability up to p are considered.
func (gen *Generator) WithTopP(topP float32) *Generator {
	gen.mu.Lock()
	defer gen.mu.Unlock()
	if gen.wasUsed() {
		gen.err = errors.Errorf("cannot change configuration of decode.Decoder, once it was used for generation -- create a new Decoder if you need a different configuration.")
		return gen
	}
	gen.TopP = topP
	return gen
}

// WithBeamSize sets the beam size for beam search.
// Higher values explore more candidates but are slower.
func (gen *Generator) WithBeamSize(beamSize int) *Generator {
	gen.mu.Lock()
	defer gen.mu.Unlock()
	if gen.wasUsed() {
		gen.err = errors.Errorf("cannot change configuration of decode.Decoder, once it was used for generation -- create a new Decoder if you need a different configuration.")
		return gen
	}
	gen.BeamSize = beamSize
	return gen
}

// WithEOS sets the end-of-sequence token ID and enables early stopping.
// Generation stops when this token is produced.
func (gen *Generator) WithEOS(eosTokenId int) *Generator {
	gen.mu.Lock()
	defer gen.mu.Unlock()
	if gen.wasUsed() {
		gen.err = errors.Errorf("cannot change configuration of decode.Decoder, once it was used for generation -- create a new Decoder if you need a different configuration.")
		return gen
	}
	gen.EosTokenId = eosTokenId
	gen.StopOnEOS = true
	return gen
}

// WithStopTokens sets the token IDs that trigger early stopping.
func (gen *Generator) WithStopTokens(tokenIds ...int) *Generator {
	gen.mu.Lock()
	defer gen.mu.Unlock()
	if gen.wasUsed() {
		gen.err = errors.Errorf("cannot change configuration of decode.Decoder, once it was used for generation -- create a new Decoder if you need a different configuration.")
		return gen
	}
	gen.EosTokenIds = append(gen.EosTokenIds, tokenIds...)
	gen.StopOnEOS = true
	return gen
}

// isCached returns true if this decoder uses KV caching.
func (gen *Generator) isCached() bool {
	return gen.cacheModelFn != nil
}

// validate checks that the decoder configuration is valid.
// Returns an error if required fields are missing or invalid.
func (gen *Generator) validate() error {
	if gen.err != nil {
		return errors.WithMessagef(gen.err, "Decoder failed during configuration")
	}
	// Exactly one model function must be set
	if gen.naiveModelFn != nil && gen.cacheModelFn != nil {
		return errors.Errorf("cannot set both ModelFn and IncrementalModelFn")
	}
	if gen.naiveModelFn == nil && gen.cacheModelFn == nil {
		return errors.Errorf("must set either ModelFn or IncrementalModelFn")
	}
	if gen.cacheModelFn != nil && gen.KVCache == nil {
		return errors.Errorf("must set KVCache configuration when using IncrementalModelFn")
	}

	return nil
}

// Decode performs autoregressive text generation from a prompt.
// The prompt can be 1D [seqLen] or 2D [batch, seqLen] tensor of token IDs.
//
// Parameters:
//   - backend: Backend for computation
//   - scope: Scope containing model parameters
//   - prompt: Input token sequence (1D or 2D tensor)
//
// Returns:
//   - Generated token sequence including the prompt
//   - Error if generation fails or configuration is invalid
//
// Example:
//
//	prompt := []int32{1, 2, 3}  // Token IDs
//	output, err := decoder.Decode(backend, scope, prompt)
func (gen *Generator) Decode(
	backend compute.Backend,
	scope *model.Scope,
	prompt any,
) (*tensors.Tensor, error) {
	gen.mu.Lock()
	defer gen.mu.Unlock()

	// Validate configuration
	if err := gen.validate(); err != nil {
		return nil, errors.WithMessagef(err, "invalid Decoder config")
	}

	// Prompt executors are now retrieved dynamically

	promptTensor, err := tensors.FromAnyValue(prompt)
	if err != nil {
		return nil, err
	}
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

	if promptLen >= gen.MaxLength {
		return nil, errors.Errorf("prompt length %d >= max length %d", promptLen, gen.MaxLength)
	}

	if gen.Strategy == sample.StrategyBeamSearch {
		return gen.generateBeamSearch(backend, scope, promptTensor)
	}

	// Regular sampling-based generation
	return gen.generateSampling(backend, scope, promptTensor)
}

// generateSampling performs generation using sampling strategies.
// Handles both cached and non-cached generation.
//
// Parameters:
//   - backend: Backend for computation
//   - scope: Scope containing model parameters
//   - prompt: Input token sequence [seqLen] or [batch, seqLen]
//
// Returns:
//   - Generated sequence [batch, totalLen] where totalLen <= MaxLength
//   - Error if generation fails
func (gen *Generator) generateSampling(
	backend compute.Backend,
	scope *model.Scope,
	prompt *tensors.Tensor,
) (*tensors.Tensor, error) {
	promptShape := prompt.Shape()
	var batchSize, promptLen int

	if prompt.Rank() == 1 {
		// Reshape 1D to 2D with batch size 1
		reshapeExec, err := model.NewExec(backend, scope.Store(), func(scope *model.Scope, input *Node) *Node {
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

	if promptLen >= gen.MaxLength {
		return nil, errors.Errorf("prompt length %d >= max length %d", promptLen, gen.MaxLength)
	}

	if gen.isCached() {
		return gen.generateSamplingIncremental(backend, scope, prompt, batchSize, promptLen)
	}
	return gen.generateSamplingFull(backend, scope, prompt, promptLen)
}

// generateSamplingFull performs sampling-based generation without KV caching.
// Processes the full sequence at each step.
//
// Parameters:
//   - backend: Backend for computation
//   - scope: Scope containing model parameters
//   - prompt: Input token sequence [batch, seqLen]
//   - promptLen: Length of the prompt sequence
//
// Returns:
//   - Generated sequence [batch, totalLen] where totalLen <= MaxLength
//   - Error if generation fails
func (gen *Generator) generateSamplingFull(
	backend compute.Backend,
	scope *model.Scope,
	prompt *tensors.Tensor,
	promptLen int,
) (*tensors.Tensor, error) {
	numTokensToGenerate := gen.MaxLength - promptLen
	if numTokensToGenerate <= 0 {
		return prompt, nil
	}

	// Extract batch size from prompt shape
	promptShape := prompt.Shape()
	batchSize := promptShape.Dimensions[0]

	// Store generated tokens as int32 values [batch][position]
	outputTokens := make([][]int32, batchSize)
	for i := range outputTokens {
		outputTokens[i] = make([]int32, 0, gen.MaxLength)
	}

	// Add prompt tokens to output
	promptValues, err := getPromptValues(prompt.Value())
	if err != nil {
		return nil, err
	}
	for i := range batchSize {
		outputTokens[i] = append(outputTokens[i], promptValues[i]...)
	}

	// Create or reuse cached executor for full sequence generation
	if gen.fullExec == nil {
		predScope := scope
		var err error
		gen.fullExec, err = model.NewExec(backend, predScope.Store(), func(scope *model.Scope, currentSeq *Node) *Node {
			logits := gen.naiveModelFn(scope, currentSeq)
			lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
			lastLogits = Squeeze(lastLogits, 1) // Remove the seq_len dimension
			nextToken := sample.SampleWithStrategy(scope, lastLogits, gen.Strategy, float64(gen.Temperature), gen.TopK, float64(gen.TopP))
			return nextToken
		})
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to create generation exec")
		}
	}

	for step := range numTokensToGenerate {
		// Build current sequence tensor from accumulated tokens
		currentSeq := tensors.FromValue(outputTokens)

		outputs, err := gen.fullExec.Exec(currentSeq)
		if err != nil {
			return nil, errors.WithMessagef(err, "generation step %d failed", step)
		}

		nextToken := outputs[0]
		nextTokenValues, err := get1DTokenValues(nextToken.Value())
		if err != nil {
			return nil, err
		}

		// Check for EOS and append tokens
		allEOS := true
		for i := range batchSize {
			outputTokens[i] = append(outputTokens[i], nextTokenValues[i])
			if gen.StopOnEOS && !gen.isEOSToken(int(nextTokenValues[i])) {
				allEOS = false
			}
		}

		if gen.StopOnEOS && allEOS {
			break
		}
	}

	return tensors.FromValue(outputTokens), nil
}

func (gen *Generator) generateSamplingIncremental(
	backend compute.Backend,
	scope *model.Scope,
	prompt *tensors.Tensor,
	batchSize, promptLen int,
) (*tensors.Tensor, error) {
	// Initialize cache
	lt := attention.GlobalLayer
	if len(gen.KVCache.OrderedScopes) > 0 {
		lt = gen.KVCache.GetLayerType(gen.KVCache.OrderedScopes[0])
	}
	cacheSeqLen := gen.KVCache.CacheSeqLen(lt, promptLen)

	kvCacheTensors := gen.KVCache.InitializeTensors(batchSize, gen.numKVHeads, gen.headDim, gen.dtype, promptLen)
	kvCacheTensorsSerialized, err := gen.KVCache.SerializeTensors(kvCacheTensors)
	if err != nil {
		return nil, err
	}

	promptExec, err := gen.getPromptExec(backend, scope, cacheSeqLen)
	if err != nil {
		return nil, err
	}

	promptInputs := make([]any, 1+len(kvCacheTensorsSerialized))
	promptInputs[0] = prompt
	for i, t := range kvCacheTensorsSerialized {
		promptInputs[i+1] = t
	}

	outputs, err := promptExec.Exec(promptInputs...)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to process prompt")
	}

	firstToken := outputs[0]
	kvCacheTensors = gen.KVCache.DeserializeTensors(outputs[1:])

	if gen.StopOnEOS && gen.checkEOS(firstToken) {
		concatExec, _ := NewExec(backend, func(seq, token *Node) *Node {
			tokenReshaped := ExpandDims(token, -1)
			return Concatenate([]*Node{seq, tokenReshaped}, 1)
		})

		concatExec.SetMaxCache(-1)
		result, err := concatExec.Call(prompt, firstToken)
		if err != nil {
			return nil, errors.WithMessagef(err, "final concatenation failed")
		}

		return result[0], nil
	}

	// Store generated tokens as int32 values [batch][position]
	outputTokens := make([][]int32, batchSize)
	for i := range outputTokens {
		outputTokens[i] = make([]int32, 0, gen.MaxLength)
	}

	// Add prompt tokens to output
	promptValues, err := getPromptValues(prompt.Value())
	if err != nil {
		return nil, err
	}
	for i := range batchSize {
		outputTokens[i] = append(outputTokens[i], promptValues[i]...)
	}

	// Add first generated token
	firstTokenValues, err := get1DTokenValues(firstToken.Value())
	if err != nil {
		return nil, err
	}
	for i := range batchSize {
		outputTokens[i] = append(outputTokens[i], firstTokenValues[i])
	}

	numTokensToGenerate := gen.MaxLength - promptLen - 1
	if numTokensToGenerate <= 0 {
		return tensors.FromValue(outputTokens), nil
	}

	genScope := scope

	currentPosition := promptLen
	for step := 0; step < numTokensToGenerate; step++ {
		position := currentPosition + step

		// Pad cache if needed
		kvCacheTensors, err = gen.KVCache.PadTensors(kvCacheTensors, position)
		if err != nil {
			return nil, err
		}
		kvCacheTensorsSerialized, err = gen.KVCache.SerializeTensors(kvCacheTensors)
		if err != nil {
			return nil, err
		}

		currentCacheSeqLen := 0
		if len(kvCacheTensorsSerialized) > 0 {
			currentCacheSeqLen = kvCacheTensorsSerialized[0].Shape().Dimensions[2]
		}

		if gen.genExecCache == nil {
			gen.genExecCache = make(map[int]*model.Exec)
		}

		// Get or create cached executor for this cache size
		exec, ok := gen.genExecCache[currentCacheSeqLen]
		if !ok {
			var err error
			exec, err = model.NewExec(backend, genScope.Store(), func(scope *model.Scope, inputs []*Node) []*Node {
				token := inputs[0]
				positionNode := inputs[1]
				cacheNodes := inputs[2:]
				tokenReshaped := ExpandDims(token, -1)
				cache := gen.KVCache.DeserializeNodes(cacheNodes)
				logits, updatedCache := gen.cacheModelFn(scope, tokenReshaped, positionNode, cache)
				serializedUpdatedCache, err := gen.KVCache.SerializeNodes(updatedCache)
				if err != nil {
					panic(err)
				}
				lastLogits := Squeeze(logits, 1)
				nextToken := sample.SampleWithStrategy(scope, lastLogits, gen.Strategy, float64(gen.Temperature), gen.TopK, float64(gen.TopP))
				res := make([]*Node, 1+len(serializedUpdatedCache))
				res[0] = nextToken
				copy(res[1:], serializedUpdatedCache)
				return res
			})
			if err != nil {
				return nil, errors.WithMessagef(err, "failed to create generation exec for cache size %d", currentCacheSeqLen)
			}
			gen.genExecCache[currentCacheSeqLen] = exec
		}

		// Get previous token from each batch as tensor
		prevTokens := make([]int32, batchSize)
		for i := range batchSize {
			prevTokens[i] = outputTokens[i][len(outputTokens[i])-1]
		}
		prevTokenTensor := tensors.FromValue(prevTokens)
		positionTensor := tensors.FromValue(int32(position))

		stepInputs := make([]any, 2+len(kvCacheTensorsSerialized))
		stepInputs[0] = prevTokenTensor
		stepInputs[1] = positionTensor
		for i, t := range kvCacheTensorsSerialized {
			stepInputs[i+2] = t
		}

		outputs, err := exec.Exec(stepInputs...)
		if err != nil {
			return nil, errors.WithMessagef(err, "generation step %d (position %d) failed", step, position)
		}

		nextToken := outputs[0]
		kvCacheTensors = gen.KVCache.DeserializeTensors(outputs[1:])

		nextTokenValues, err := get1DTokenValues(nextToken.Value())
		if err != nil {
			return nil, err
		}

		// Check for EOS and append tokens
		allEOS := true
		for i := range batchSize {
			outputTokens[i] = append(outputTokens[i], nextTokenValues[i])
			if gen.StopOnEOS && !gen.isEOSToken(int(nextTokenValues[i])) {
				allEOS = false
			}
		}

		if gen.StopOnEOS && allEOS {
			break
		}
	}

	return tensors.FromValue(outputTokens), nil
}

// checkEOS returns true if any token in the tensor matches any EOS token ID.
func (gen *Generator) checkEOS(token *tensors.Tensor) bool {
	tokenValue := token.Value()
	switch v := tokenValue.(type) {
	case []int32:
		for _, t := range v {
			if gen.isEOSToken(int(t)) {
				return true
			}
		}
	case int32:
		return gen.isEOSToken(int(v))
	case []int64:
		for _, t := range v {
			if gen.isEOSToken(int(t)) {
				return true
			}
		}
	case int64:
		return gen.isEOSToken(int(v))
	}
	return false
}

func (gen *Generator) isEOSToken(tokenId int) bool {
	if gen.EosTokenId >= 0 && tokenId == gen.EosTokenId {
		return true
	}
	for _, id := range gen.EosTokenIds {
		if id >= 0 && tokenId == id {
			return true
		}
	}
	return false
}

// generateBeamSearch performs beam search generation.
// Dispatches to cached or non-cached implementation based on configuration.
func (gen *Generator) generateBeamSearch(
	backend compute.Backend,
	scope *model.Scope,
	prompt *tensors.Tensor,
) (*tensors.Tensor, error) {
	// Ensure prompt 2D
	promptShape := prompt.Shape()
	var batchSize, promptLen int

	if prompt.Rank() == 1 {
		reshapeExec, err := model.NewExec(backend, scope.Store(), func(scope *model.Scope, input *Node) *Node {
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

	if promptLen >= gen.MaxLength {
		return nil, errors.Errorf("prompt length %d >= max length %d", promptLen, gen.MaxLength)
	}

	// Dispatch cached or non-cached
	if gen.isCached() {
		return gen.generateBeamSearchCached(backend, scope, prompt, batchSize, promptLen)
	}
	return gen.generateBeamSearchNonCached(backend, scope, prompt, batchSize, promptLen)
}

// generateBeamSearchNonCached performs beam search without KV caching.
// Maintains multiple beam hypotheses and selects the best sequence.
func (gen *Generator) generateBeamSearchNonCached(
	backend compute.Backend,
	scope *model.Scope,
	prompt *tensors.Tensor,
	batchSize, promptLen int,
) (*tensors.Tensor, error) {
	beamSize := gen.BeamSize
	batchBeamSize := batchSize * beamSize

	// Replicate prompt for each beam
	replicateExec, err := model.NewExec(backend, scope.Store(), func(scope *model.Scope, prompt *Node) *Node {
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
	beamConfig := sample.NewBeamSearch(beamSize, gen.EosTokenId).
		WithMaxLength(gen.MaxLength).
		WithLengthPenalty(1.0)

	// Main loop
	predScope := scope
	numSteps := gen.MaxLength - promptLen

	for step := 0; step < numSteps; step++ {
		currentLength := promptLen + step

		// TODO: It seems I cannot cache this exec because currentLength changes each iteration
		// and is used as a compile-time constant in the graph (passed to beamConfig.Step)
		// I leave it like this for now as I think we need the dynamic shape support of the simplego backend.
		exec, err := model.NewExec(backend, predScope.Store(), func(scope *model.Scope, sequences, scores *Node) (*Node, *Node, *Node) {
			// Run model
			logits := gen.naiveModelFn(scope, sequences)

			// Last token logits: [batch_beam_size, vocab_size]
			lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
			lastLogits = Squeeze(lastLogits, 1)

			// Beam search step
			nextSeqs, nextScores, isFinished, _ := beamConfig.Step(
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
	selectExec, err := model.NewExec(backend, scope.Store(), func(scope *model.Scope, sequences, scores *Node) (*Node, *Node) {
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
func (gen *Generator) generateBeamSearchCached(
	backend compute.Backend,
	scope *model.Scope,
	prompt *tensors.Tensor,
	batchSize, promptLen int,
) (*tensors.Tensor, error) {
	beamSize := gen.BeamSize
	batchBeamSize := batchSize * beamSize

	// Replicate prompt for each beam
	replicateExec, err := model.NewExec(backend, scope.Store(), func(scope *model.Scope, prompt *Node) *Node {
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

	// Initialize KV cache for batchBeamSize
	lt := attention.GlobalLayer
	if len(gen.KVCache.OrderedScopes) > 0 {
		lt = gen.KVCache.GetLayerType(gen.KVCache.OrderedScopes[0])
	}
	cacheSeqLen := gen.KVCache.CacheSeqLen(lt, promptLen)

	kvCacheTensors := gen.KVCache.InitializeTensors(batchBeamSize, gen.numKVHeads, gen.headDim, gen.dtype, promptLen)
	kvCacheTensorsSerialized, err := gen.KVCache.SerializeTensors(kvCacheTensors)
	if err != nil {
		return nil, err
	}

	promptExec, err := gen.getPromptExec(backend, scope, cacheSeqLen)
	if err != nil {
		return nil, err
	}

	promptInputs := make([]any, 1+len(kvCacheTensorsSerialized))
	promptInputs[0] = replicatedPrompt
	for i, t := range kvCacheTensorsSerialized {
		promptInputs[i+1] = t
	}

	outputs, err := promptExec.Exec(promptInputs...)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to process prompt")
	}

	kvCacheTensors = gen.KVCache.DeserializeTensors(outputs[1:])

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
	beamConfig := sample.NewBeamSearch(beamSize, gen.EosTokenId).
		WithMaxLength(gen.MaxLength).
		WithLengthPenalty(1.0)

	// Main loop
	genScope := scope
	numSteps := gen.MaxLength - promptLen

	for step := 0; step < numSteps; step++ {
		position := promptLen + step

		// Pad cache if needed
		kvCacheTensors, err = gen.KVCache.PadTensors(kvCacheTensors, position)
		if err != nil {
			return nil, err
		}
		kvCacheTensorsSerialized, err = gen.KVCache.SerializeTensors(kvCacheTensors)
		if err != nil {
			return nil, err
		}

		extractExec, err := model.NewExec(backend, scope.Store(), func(scope *model.Scope, sequences *Node) *Node {
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

		positionTensor := tensors.FromValue(int32(position))

		exec, err := model.NewExec(backend, genScope.Store(), func(scope *model.Scope, inputs []*Node) []*Node {
			tokens := inputs[0]
			scores := inputs[1]
			sequences := inputs[2]
			positionNode := inputs[3]
			cacheNodes := inputs[4:]
			// tokens: [batch_beam_size]
			tokensReshaped := ExpandDims(tokens, -1) // [batch_beam_size, 1]

			// Process with incremental model
			cache := gen.KVCache.DeserializeNodes(cacheNodes)
			logits, updatedCache := gen.cacheModelFn(scope, tokensReshaped, positionNode, cache)
			logits = Squeeze(logits, 1) // [batch_beam_size, vocab_size]

			// Beam search step
			nextSeqs, nextScores, isFinished, gatherIndices := beamConfig.Step(
				logits,
				sequences,
				scores,
				position,
			)

			// Re-order the cache nodes using gatherIndices
			for kKey, node := range updatedCache {
				updatedCache[kKey] = Gather(node, gatherIndices)
			}

			serializedUpdatedCache, err := gen.KVCache.SerializeNodes(updatedCache)
			if err != nil {
				panic(err)
			}

			res := make([]*Node, 3+len(serializedUpdatedCache))
			res[0] = nextSeqs
			res[1] = nextScores
			res[2] = isFinished
			copy(res[3:], serializedUpdatedCache)
			return res
		})
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to create beam step exec")
		}

		stepInputs := make([]any, 4+len(kvCacheTensorsSerialized))
		stepInputs[0] = lastTokens
		stepInputs[1] = beamScores
		stepInputs[2] = currentSequences
		stepInputs[3] = positionTensor
		for i, t := range kvCacheTensorsSerialized {
			stepInputs[i+4] = t
		}

		outputs, err := exec.Exec(stepInputs...)
		if err != nil {
			return nil, errors.WithMessagef(err, "beam search step %d failed", step)
		}

		currentSequences = outputs[0]
		beamScores = outputs[1]
		isFinished := outputs[2]
		kvCacheTensors = gen.KVCache.DeserializeTensors(outputs[3:])

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
			case bool:
				allFinished = v
			}
			if allFinished {
				break
			}
		}
	}

	// Select best sequence
	selectExec, err := model.NewExec(backend, scope.Store(), func(scope *model.Scope, sequences, scores *Node) *Node {
		bestSeqs, _ := beamConfig.SelectBest(sequences, scores)
		return bestSeqs
	})
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to create select exec")
	}

	bestResults, err := selectExec.Exec(currentSequences, beamScores)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to select best sequences")
	}

	return bestResults[0], nil
}

// GenerateStreaming performs streaming generation, yielding tokens as they are generated.
// The callback function is called for each generated token and can return false to stop generation.
//
// Parameters:
//   - backend: Backend for computation
//   - scope: Scope containing model parameters
//   - prompt: Input token sequence (1D or 2D tensor)
//   - callback: Function called with each generated token ID; return false to stop
//
// Returns:
//   - Error if generation fails
//
// Note: This is a placeholder for future streaming support.
func (gen *Generator) GenerateStreaming(
	backend compute.Backend,
	scope *model.Scope,
	prompt any,
	callback func(token int) bool,
) error {
	gen.mu.Lock()
	defer gen.mu.Unlock()
	return errors.Errorf("streaming generation not yet implemented")
}

func getPromptValues(promptVal any) ([][]int32, error) {
	switch pVal := promptVal.(type) {
	case [][]int32:
		return pVal, nil
	case [][]int64:
		res := make([][]int32, len(pVal))
		for i, row := range pVal {
			res[i] = make([]int32, len(row))
			for j, v := range row {
				res[i][j] = int32(v)
			}
		}
		return res, nil
	case [][]int:
		res := make([][]int32, len(pVal))
		for i, row := range pVal {
			res[i] = make([]int32, len(row))
			for j, v := range row {
				res[i][j] = int32(v)
			}
		}
		return res, nil
	default:
		return nil, errors.Errorf("unsupported prompt tensor value type: %T", promptVal)
	}
}

func get1DTokenValues(tokenVal any) ([]int32, error) {
	switch fVal := tokenVal.(type) {
	case []int32:
		return fVal, nil
	case []int64:
		res := make([]int32, len(fVal))
		for i, v := range fVal {
			res[i] = int32(v)
		}
		return res, nil
	case []int:
		res := make([]int32, len(fVal))
		for i, v := range fVal {
			res[i] = int32(v)
		}
		return res, nil
	default:
		return nil, errors.Errorf("unsupported token tensor value type: %T", tokenVal)
	}
}
