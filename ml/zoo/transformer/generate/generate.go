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
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/core/tensors/bucketing"
	"github.com/gomlx/gomlx/ml/layers/attention"
	"github.com/gomlx/gomlx/ml/layers/attention/kvcache"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/zoo/transformer/generate/sample"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// Hyperparameter keys for scope configuration
const (
	ParamMaxLength   = "generate_max_length"
	ParamStrategy    = "generate_strategy"
	ParamTemperature = "generate_temperature"
	ParamTopK        = "generate_top_k"
	ParamTopP        = "generate_top_p"
	ParamBeamSize    = "generate_beam_size"
	ParamEosTokenId  = "generate_eos_token_id"
	ParamStopOnEOS   = "generate_stop_on_eos"
)

// NaiveModelFn represents a full-sequence model function,
// that is used iteratively during generation, by re-feeding
// all the tokens each time.
//
// This is a slow way of generating, since it will re-run all the computation
// for the whole sequence each time.
//
// Parameters:
//   - scope: Model scope passed along by the Generator.
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
//   - scope: Model scope passed along by the Generator. Likely contains the cache during the execution.
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
// and incremental models, usually using some form of cache (e.g.: a "KV-cache").
type Generator struct {
	// Model functions (exactly one should be set)
	naiveModelFn NaiveModelFn   // Standard model for non-cached generation
	cacheModelFn KVCacheModelFn // Incremental model for KV-cached generation

	KVCache    *kvcache.KVCache
	numKVHeads int
	headDim    int
	dtype      dtypes.DType

	// Generation parameters
	MaxLength         int
	Strategy          sample.Strategy
	Temperature       float32
	TopK              int
	TopP              float32
	BeamSize          int
	EosTokenId        int
	EosTokenIds       []int
	StopOnEOS         bool
	BucketingStrategy bucketing.Strategy
	PadToken          int

	// Internal backend for small operations (concat, reshape)
	// Uses simplego for faster compilation and dynamic shape support
	simpleBackend compute.Backend

	// Cached executors to avoid recompilation
	promptExec                *model.Exec // Cached executor for processing initial prompt in incremental generation
	genExec                   *model.Exec // Cached executor for each position in incremental generation
	naiveExec                 *model.Exec // Cached executor for non-cached full sequence generation
	beamExec                  *model.Exec // Cached executor for non-cached beam search
	beamKVCacheExec           *model.Exec // Cached executor for beam search with KV cache
	updateCurrentSeqExec      *model.Exec // Cached executor to update a single token in currentSeq on accelerator
	growCurrentSeqExec        *model.Exec // Cached executor to grow currentSeq to a new bucket length on accelerator
	growCurrentSeqDynamicExec *model.Exec // Cached executor to grow currentSeq dynamically on accelerator

	// err is a delayed error set during initialization, and returned by Decode.
	err error
	mu  sync.Mutex
}

// New creates a generator for autoregressive text generation.
// Accepts either ModelFn (non-cached, full sequence) or IncrementalModelFn (KV-cached).
// The generator automatically detects which type is provided and configures itself accordingly.
//
// Parameters:
//   - modelFn: Either ModelFn for full-sequence processing or IncrementalModelFn for cached generation
//
// Returns:
//   - Generator with default generation parameters (greedy sampling, maxLength=100)
//
// Example:
//
//	generator := generate.New(model.Incremental()).
//		WithStrategy(sample.StrategyTemperature).WithTemperature(0.8)
//	output, err := generator.Decode(prompt)
func New[M interface {
	NaiveModelFn | KVCacheModelFn
}](modelFn M) *Generator {
	var generator *Generator
	switch typedModelFn := any(modelFn).(type) {
	case NaiveModelFn:
		klog.Warning("Using Generator to generate text with interactive model is not yet well supported, and will likely be very slow. Consider using an IncrementalModelFn instead.")
		generator = &Generator{
			naiveModelFn: typedModelFn,
		}
	case KVCacheModelFn:
		generator = &Generator{
			cacheModelFn: typedModelFn,
		}
	}

	// Set default parameters
	generator.MaxLength = 100
	generator.Strategy = sample.StrategyGreedy
	generator.Temperature = 1.0
	generator.TopK = 50
	generator.TopP = 0.9
	generator.BeamSize = 4
	generator.EosTokenId = -1
	generator.StopOnEOS = false
	generator.PadToken = 0

	// Initialize simplego backend for small operations (concat, reshape)
	// Faster than XLA for small ops and supports dynamic shapes
	generator.simpleBackend, _ = gobackend.New("generate")

	return generator
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
	return gen.promptExec != nil || gen.genExec != nil || gen.naiveExec != nil || gen.beamExec != nil || gen.beamKVCacheExec != nil || gen.updateCurrentSeqExec != nil || gen.growCurrentSeqExec != nil || gen.growCurrentSeqDynamicExec != nil
}

// FromScope configures the generator with hyperparameters from the model.
// This allows fine-tuning an existing generator configuration.
//
// Supported hyperparameters:
//   - generate_max_length: Maximum generation length (default: unchanged)
//   - generate_strategy: Sampling strategy ("greedy", "temperature", "top_k", "top_p", "beam_search")
//   - generate_temperature: Temperature for sampling (default: unchanged)
//   - generate_top_k: k for top-k sampling (default: unchanged)
//   - generate_top_p: p for nucleus sampling (default: unchanged)
//   - generate_beam_size: Beam size for beam search (default: unchanged)
//   - generate_eos_token_id: End-of-sequence token ID (default: unchanged)
//   - generate_stop_on_eos: Whether to stop on EOS (default: unchanged)
//
// Example:
//
//	scope.SetParams(map[string]any{
//	    "generate_strategy": "temperature",
//	    "generate_temperature": 0.8,
//	    "generate_max_length": 200,
//	})
//	generator.FromScope(scope)
func (gen *Generator) FromScope(scope *model.Scope) *Generator {
	gen.mu.Lock()
	defer gen.mu.Unlock()
	if gen.wasUsed() {
		gen.err = errors.Errorf("cannot change configuration of generate.Generator, once it was used for generation -- create a new Generator if you need a different configuration.")
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
	if gen.promptExec != nil {
		return gen.promptExec, nil
	}

	var err error
	gen.promptExec, err = model.NewExec(backend, scope.Store(), func(scope *model.Scope, inputs []*Node) []*Node {
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
	return gen.promptExec, nil
}

// WithMaxLength sets the maximum generation length (including prompt).
func (gen *Generator) WithMaxLength(maxLength int) *Generator {
	gen.mu.Lock()
	defer gen.mu.Unlock()
	if gen.wasUsed() {
		gen.err = errors.Errorf("cannot change configuration of generate.Generator, once it was used for generation -- create a new Generator if you need a different configuration.")
		return gen
	}
	gen.MaxLength = maxLength
	return gen
}

// WithBucketingStrategy sets the bucketing strategy to reduce the number of unique compiled graphs.
//
// This is used for naive generation (not is using KVCache), and defaults to [bucketing.Pow2] -- tokens tensor doubles
// each time it runs out of space -- or [bucketing.None] is the backend supports dynamic shapes, in which case bucketing is not needed.
func (gen *Generator) WithBucketingStrategy(strategy bucketing.Strategy) *Generator {
	gen.mu.Lock()
	defer gen.mu.Unlock()
	if gen.wasUsed() {
		gen.err = errors.Errorf("cannot change configuration of generate.Generator, once it was used for generation -- create a new Generator if you need a different configuration.")
		return gen
	}
	gen.BucketingStrategy = strategy
	return gen
}

// WithPadToken sets the padding token ID (default to 0).
func (gen *Generator) WithPadToken(padToken int) *Generator {
	gen.mu.Lock()
	defer gen.mu.Unlock()
	if gen.wasUsed() {
		gen.err = errors.Errorf("cannot change configuration of generate.Generator, once it was used for generation -- create a new Generator if you need a different configuration.")
		return gen
	}
	gen.PadToken = padToken
	return gen
}

// getBucketingStrategy resolves the bucketing strategy to use.
// If not explicitly set, it defaults to None if the backend supports dynamic shapes,
// as dynamic shapes avoid the need for bucketing.
// Otherwise, default to bucketing.Pow2.
func (gen *Generator) getBucketingStrategy(backend compute.Backend) bucketing.Strategy {
	if gen.BucketingStrategy != nil {
		return gen.BucketingStrategy
	}
	if backend.Capabilities().DynamicAxes {
		return bucketing.None()
	}
	return bucketing.Pow2()
}

// padAndMakeTensor pads outputTokens slices to bucketedLen using PadToken, and returns a new Tensor.
func (gen *Generator) padAndMakeTensor(outputTokens [][]int32, bucketedLen int) *tensors.Tensor {
	batchSize := len(outputTokens)
	padded := make([][]int32, batchSize)
	for i := range batchSize {
		padded[i] = make([]int32, bucketedLen)
		copy(padded[i], outputTokens[i])
		for j := len(outputTokens[i]); j < bucketedLen; j++ {
			padded[i][j] = int32(gen.PadToken)
		}
	}
	return tensors.FromValue(padded)
}

// WithStrategy sets the sampling strategy.
// The default strategy is [sample.StrategyGreedy], which always take the immediately most likely next token.
//
// Alternatively, simply use [WithTemperature], [WithTopK], [WithTopP], or [WithBeamSize] to set the strategy and relevant parameters.
func (gen *Generator) WithStrategy(strategy sample.Strategy) *Generator {
	gen.mu.Lock()
	defer gen.mu.Unlock()
	if gen.wasUsed() {
		gen.err = errors.Errorf("cannot change configuration of generate.Generator, once it was used for generation -- create a new Generator if you need a different configuration.")
		return gen
	}
	gen.Strategy = strategy
	return gen
}

// WithTemperature sets the temperature for sampling.
// Higher values (>1.0) increase randomness, lower values (<1.0) make output more deterministic.
//
// It sets the sampling strategy to [sample.StrategyTemperature] if temperature is non-zero.
func (gen *Generator) WithTemperature(temperature float32) *Generator {
	gen.mu.Lock()
	defer gen.mu.Unlock()
	if gen.wasUsed() {
		gen.err = errors.Errorf("cannot change configuration of generate.Generator, once it was used for generation -- create a new Generator if you need a different configuration.")
		return gen
	}
	gen.Temperature = temperature
	if temperature != 0 {
		gen.Strategy = sample.StrategyTemperature
	}
	return gen
}

// WithTopK sets k for top-k sampling.
// Only the k most likely tokens are considered at each step, their probabilities are renormalized, and then the next token is sampled from them.
//
// It sets the sampling strategy to [sample.StrategyTopK] if topK is non-zero.
func (gen *Generator) WithTopK(topK int) *Generator {
	gen.mu.Lock()
	defer gen.mu.Unlock()
	if gen.wasUsed() {
		gen.err = errors.Errorf("cannot change configuration of generate.Generator, once it was used for generation -- create a new Generator if you need a different configuration.")
		return gen
	}
	gen.TopK = topK
	if topK != 0 {
		gen.Strategy = sample.StrategyTopK
	}
	return gen
}

// WithTopP sets p for nucleus sampling.
// Tokens with cumulative probability up to p are considered, their probabilities are renormalized, and then the next token is sampled from them.
//
// It sets the sampling strategy to [sample.StrategyTopP] if topP is non-zero.
func (gen *Generator) WithTopP(topP float32) *Generator {
	gen.mu.Lock()
	defer gen.mu.Unlock()
	if gen.wasUsed() {
		gen.err = errors.Errorf("cannot change configuration of generate.Generator, once it was used for generation -- create a new Generator if you need a different configuration.")
		return gen
	}
	gen.TopP = topP
	if topP != 0 {
		gen.Strategy = sample.StrategyTopP
	}
	return gen
}

// WithBeamSize sets the beam size for beam search.
// Higher values explore more candidates but are slower.
//
// It sets the sampling strategy to [sample.StrategyBeamSearch] if beamSize is non-zero.
func (gen *Generator) WithBeamSize(beamSize int) *Generator {
	gen.mu.Lock()
	defer gen.mu.Unlock()
	if gen.wasUsed() {
		gen.err = errors.Errorf("cannot change configuration of generate.Generator, once it was used for generation -- create a new Generator if you need a different configuration.")
		return gen
	}
	gen.BeamSize = beamSize
	if beamSize != 0 {
		gen.Strategy = sample.StrategyBeamSearch
	}
	return gen
}

// WithEOS sets the end-of-sequence token ID and enables early stopping.
// Generation stops when this token is produced.
func (gen *Generator) WithEOS(eosTokenId int) *Generator {
	gen.mu.Lock()
	defer gen.mu.Unlock()
	if gen.wasUsed() {
		gen.err = errors.Errorf("cannot change configuration of generate.Generator, once it was used for generation -- create a new Generator if you need a different configuration.")
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
		gen.err = errors.Errorf("cannot change configuration of generate.Generator, once it was used for generation -- create a new Generator if you need a different configuration.")
		return gen
	}
	gen.EosTokenIds = append(gen.EosTokenIds, tokenIds...)
	gen.StopOnEOS = true
	return gen
}

// isCached returns true if this generator uses KV caching.
func (gen *Generator) isCached() bool {
	return gen.cacheModelFn != nil
}

// validate checks that the generator configuration is valid.
// Returns an error if required fields are missing or invalid.
func (gen *Generator) validate() error {
	if gen.err != nil {
		return errors.WithMessagef(gen.err, "Generator failed during configuration")
	}
	if gen.Temperature == 0 && (gen.Strategy == sample.StrategyTemperature || gen.Strategy == sample.StrategyTopK || gen.Strategy == sample.StrategyTopP) {
		gen.Temperature = 1.0
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
//	output, err := generator.Decode(backend, scope, prompt)
func (gen *Generator) Decode(
	backend compute.Backend,
	scope *model.Scope,
	prompt any,
) (*tensors.Tensor, error) {
	gen.mu.Lock()
	defer gen.mu.Unlock()

	// Validate configuration
	if err := gen.validate(); err != nil {
		return nil, errors.WithMessagef(err, "invalid Generator config")
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
	return gen.generateSampling(backend, scope, promptTensor, nil)
}

func finalizeTensors(ts []*tensors.Tensor) {
	for _, t := range ts {
		if t != nil {
			t.FinalizeAll()
		}
	}
}

func finalizeKVCache(cache kvcache.KVCacheTensors) {
	for _, t := range cache {
		if t != nil {
			t.FinalizeAll()
		}
	}
}

// updateCurrentSeq updates currentSeq in-place at position with nextToken on the accelerator.
//
// Expected shapes:
//   - currentSeq: [batchSize, seqLen]
//   - nextToken: [batchSize]
func (gen *Generator) updateCurrentSeq(backend compute.Backend, scope *model.Scope, currentSeq *tensors.Tensor, nextToken *tensors.Tensor, position int32) (*tensors.Tensor, error) {
	if gen.updateCurrentSeqExec == nil {
		var err error
		gen.updateCurrentSeqExec, err = model.NewExec(backend, scope.Store(), func(scope *model.Scope, currentSeqNode, nextTokenNode, positionNode *Node) *Node {
			g := currentSeqNode.Graph()
			update := ExpandDims(nextTokenNode, -1) // [batchSize] -> [batchSize, 1]
			batchIdx := Const(g, int32(0))
			updatedSeq := DynamicUpdateSlice(currentSeqNode, update, []*Node{batchIdx, positionNode})
			return updatedSeq
		})
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to create updateCurrentSeqExec")
		}
		if backend.Capabilities().DynamicAxes {
			gen.updateCurrentSeqExec.WithDynamicAxes([]string{"batch", "seq_len"}, []string{""}, []string{})
		}
	}
	res, err := gen.updateCurrentSeqExec.Call(currentSeq, nextToken, tensors.FromValue(position))
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to run updateCurrentSeqExec")
	}
	return res[0], nil
}

// growCurrentSeq grows currentSeq by appending PadToken values on the accelerator.
// The returned grown length is defined by the bucketing strategy.
//
// Expected shapes:
//   - currentSeq: [batchSize, currentSeqLen]
func (gen *Generator) growCurrentSeq(backend compute.Backend, scope *model.Scope, currentSeq *tensors.Tensor) (*tensors.Tensor, error) {
	if gen.growCurrentSeqExec == nil {
		var err error
		gen.growCurrentSeqExec, err = model.NewExec(backend, scope.Store(), func(scope *model.Scope, currentSeqNode *Node) *Node {
			g := currentSeqNode.Graph()
			backend := g.Backend()
			strategy := gen.getBucketingStrategy(backend)
			batchSize := currentSeqNode.Shape().Dimensions[0]
			currentSeqLen := currentSeqNode.Shape().Dimensions[1]
			targetLen := strategy.Bucket(currentSeqLen + 1)
			paddingLen := targetLen - currentSeqLen

			// Create padding filled with PadToken
			scalarNode := Const(g, int32(gen.PadToken))
			padding := BroadcastToShape(scalarNode, shapes.Make(dtypes.Int32, batchSize, paddingLen))

			return Concatenate([]*Node{currentSeqNode, padding}, 1)
		})
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to create growCurrentSeqExec")
		}
		if backend.Capabilities().DynamicAxes {
			gen.growCurrentSeqExec.WithDynamicAxes([]string{"batch", "seq_len"})
		}
	}
	res, err := gen.growCurrentSeqExec.Call(currentSeq)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to run growCurrentSeqExec")
	}
	return res[0], nil
}

// growCurrentSeqDynamic grows currentSeq by 1 by appending nextToken on the accelerator.
//
// Expected shapes:
//   - currentSeq: [batchSize, currentSeqLen]
//   - nextToken: [batchSize]
func (gen *Generator) growCurrentSeqDynamic(backend compute.Backend, scope *model.Scope, currentSeq *tensors.Tensor, nextToken *tensors.Tensor) (*tensors.Tensor, error) {
	if gen.growCurrentSeqDynamicExec == nil {
		var err error
		gen.growCurrentSeqDynamicExec, err = model.NewExec(backend, scope.Store(), func(scope *model.Scope, currentSeqNode *Node, nextTokenNode *Node) *Node {
			update := ExpandDims(nextTokenNode, -1) // [batchSize] -> [batchSize, 1]
			return Concatenate([]*Node{currentSeqNode, update}, 1)
		})
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to create growCurrentSeqDynamicExec")
		}
		if backend.Capabilities().DynamicAxes {
			gen.growCurrentSeqDynamicExec.WithDynamicAxes([]string{"batch", "seq_len"}, []string{""})
		}
	}
	res, err := gen.growCurrentSeqDynamicExec.Call(currentSeq, nextToken)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to run growCurrentSeqDynamicExec")
	}
	return res[0], nil
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
	callback func(token int) bool,
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

		reshapeResults, err := reshapeExec.Call(prompt)
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
		return gen.generateSamplingWithKVCache(backend, scope, prompt, batchSize, promptLen, callback)
	}
	return gen.generateSamplingNaive(backend, scope, prompt, promptLen, callback)
}

// generateSamplingNaive performs sampling-based generation using the "naive"
// processes the full sequence at each step algorithm.
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
func (gen *Generator) generateSamplingNaive(
	backend compute.Backend,
	scope *model.Scope,
	prompt *tensors.Tensor,
	promptLen int,
	callback func(token int) bool,
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

	// Checks whether to use dynamic shapes (if available) or bucketing.
	strategy := gen.getBucketingStrategy(backend)
	_, isNoneStrategy := strategy.(bucketing.NoneStrategy)
	useDynamic := isNoneStrategy && backend.Capabilities().DynamicAxes

	// Initialize currentSeq on accelerator
	var currentSeq *tensors.Tensor
	if useDynamic {
		currentSeq = prompt
	} else {
		bucketedLen := strategy.Bucket(promptLen)
		if bucketedLen > promptLen {
			currentSeq = gen.padAndMakeTensor(outputTokens, bucketedLen)
		} else {
			currentSeq = prompt
		}
	}

	defer func() {
		if currentSeq != nil && currentSeq != prompt {
			currentSeq.FinalizeAll()
		}
	}()

	// Create or reuse cached executor for sequence generation
	if gen.naiveExec == nil {
		gen.naiveExec, err = model.NewExec(backend, scope.Store(), func(scope *model.Scope, inputs []*Node) *Node {
			currentSeqNode := inputs[0]
			if useDynamic {
				seqLenNode := inputs[1]
				logits := gen.naiveModelFn(scope, currentSeqNode, -1)
				lastIndex := Sub(seqLenNode, ConstAs(seqLenNode, 1))
				transposed := Transpose(logits, 0, 1)
				lastLogits := Gather(transposed, lastIndex)
				nextToken := sample.SampleWithStrategy(scope, lastLogits, gen.Strategy, float64(gen.Temperature), gen.TopK, float64(gen.TopP))
				return nextToken
			} else {
				lengthDummy := inputs[1]
				currentSeqLen := lengthDummy.Shape().Dimensions[0]
				logits := gen.naiveModelFn(scope, currentSeqNode, currentSeqLen)
				g := logits.Graph()
				lastIndex := Const(g, int32(currentSeqLen-1))
				transposed := Transpose(logits, 0, 1)
				lastLogits := Gather(transposed, lastIndex)
				nextToken := sample.SampleWithStrategy(scope, lastLogits, gen.Strategy, float64(gen.Temperature), gen.TopK, float64(gen.TopP))
				return nextToken
			}
		})
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to create naiveExec")
		}
		if backend.Capabilities().DynamicAxes {
			gen.naiveExec.WithDynamicAxes([]string{"batch", "seq_len"}, []string{""})
		}
	}

	for step := range numTokensToGenerate {
		currentSeqLen := promptLen + step

		var outputs []*tensors.Tensor
		if useDynamic {
			outputs, err = gen.naiveExec.Call(currentSeq, tensors.FromValue(int32(currentSeqLen)))
		} else {
			lengthDummy := tensors.FromShape(shapes.Make(dtypes.Int32, currentSeqLen))
			outputs, err = gen.naiveExec.Call(currentSeq, lengthDummy)
			lengthDummy.FinalizeAll()
		}

		if err != nil {
			return nil, errors.WithMessagef(err, "generation step %d failed", step)
		}

		nextToken := outputs[0]
		nextTokenValues, err := get1DTokenValues(nextToken.Value())
		if err != nil {
			nextToken.FinalizeAll()
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

		if callback != nil {
			if !callback(int(nextTokenValues[0])) {
				nextToken.FinalizeAll()
				break
			}
		}

		if gen.StopOnEOS && allEOS {
			nextToken.FinalizeAll()
			break
		}

		// Update or grow currentSeq on accelerator
		var nextSeq *tensors.Tensor
		if useDynamic {
			nextSeq, err = gen.growCurrentSeqDynamic(backend, scope, currentSeq, nextToken)
		} else {
			bucketedLen := strategy.Bucket(currentSeqLen)
			nextBucketedLen := strategy.Bucket(currentSeqLen + 1)
			if nextBucketedLen > bucketedLen {
				grownSeq, err := gen.growCurrentSeq(backend, scope, currentSeq)
				if err != nil {
					nextToken.FinalizeAll()
					return nil, err
				}
				if currentSeq != prompt {
					currentSeq.FinalizeAll()
				}
				currentSeq = grownSeq
			}
			nextSeq, err = gen.updateCurrentSeq(backend, scope, currentSeq, nextToken, int32(currentSeqLen))
		}
		nextToken.FinalizeAll()
		if err != nil {
			return nil, err
		}
		if currentSeq != prompt {
			currentSeq.FinalizeAll()
		}
		currentSeq = nextSeq
	}

	return tensors.FromValue(outputTokens), nil
}

func (gen *Generator) generateSamplingWithKVCache(
	backend compute.Backend,
	scope *model.Scope,
	prompt *tensors.Tensor,
	batchSize, promptLen int,
	callback func(token int) bool,
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
		finalizeKVCache(kvCacheTensors)
		return nil, err
	}

	promptInputs := make([]any, 1+len(kvCacheTensorsSerialized))
	promptInputs[0] = prompt
	for i, t := range kvCacheTensorsSerialized {
		donated, err := DonateTensorBuffer(t, backend, 0)
		if err != nil {
			finalizeKVCache(kvCacheTensors)
			return nil, err
		}
		promptInputs[i+1] = donated
	}

	outputs, err := promptExec.Call(promptInputs...)
	if err != nil {
		finalizeKVCache(kvCacheTensors)
		return nil, errors.WithMessagef(err, "failed to process prompt")
	}

	firstToken := outputs[0]
	// Clean up initial serialized cache tensors is no longer needed since they were donated.

	kvCacheTensors = gen.KVCache.DeserializeTensors(outputs[1:])

	defer func() {
		for _, t := range kvCacheTensors {
			if t != nil {
				t.FinalizeAll()
			}
		}
	}()

	if gen.StopOnEOS && gen.checkEOS(firstToken) {
		concatExec, _ := NewExec(backend, func(seq, token *Node) *Node {
			tokenReshaped := ExpandDims(token, -1)
			return Concatenate([]*Node{seq, tokenReshaped}, 1)
		})

		concatExec.SetMaxCache(-1)
		result, err := concatExec.Call(prompt, firstToken)
		firstToken.FinalizeAll()
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
		firstToken.FinalizeAll()
		return nil, err
	}
	for i := range batchSize {
		outputTokens[i] = append(outputTokens[i], promptValues[i]...)
	}

	// Add first generated token
	firstTokenValues, err := get1DTokenValues(firstToken.Value())
	firstToken.FinalizeAll()
	if err != nil {
		return nil, err
	}
	for i := range batchSize {
		outputTokens[i] = append(outputTokens[i], firstTokenValues[i])
	}

	if callback != nil {
		if !callback(int(firstTokenValues[0])) {
			return tensors.FromValue(outputTokens), nil
		}
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
		paddedKVCacheTensors, err := gen.KVCache.PadTensors(kvCacheTensors, position)
		if err != nil {
			return nil, err
		}
		// Finalize old tensors that were replaced by padded ones
		for path, oldT := range kvCacheTensors {
			if newT, ok := paddedKVCacheTensors[path]; ok && newT != oldT {
				oldT.FinalizeAll()
			}
		}
		kvCacheTensors = paddedKVCacheTensors

		kvCacheTensorsSerialized, err = gen.KVCache.SerializeTensors(kvCacheTensors)
		if err != nil {
			return nil, err
		}

		// Get or create cached executor
		if gen.genExec == nil {
			var err error
			gen.genExec, err = model.NewExec(backend, genScope.Store(), func(scope *model.Scope, inputs []*Node) []*Node {
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
				return nil, errors.WithMessagef(err, "failed to create genExec")
			}
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
			donated, err := DonateTensorBuffer(t, backend, 0)
			if err != nil {
				return nil, err
			}
			stepInputs[i+2] = donated
		}

		outputs, err := gen.genExec.Call(stepInputs...)
		prevTokenTensor.FinalizeAll()
		positionTensor.FinalizeAll()
		if err != nil {
			return nil, errors.WithMessagef(err, "generation step %d (position %d) failed", step, position)
		}

		nextToken := outputs[0]
		newKVCacheTensors := gen.KVCache.DeserializeTensors(outputs[1:])

		// No need to finalize the old cache tensors because they were donated.
		kvCacheTensors = newKVCacheTensors

		nextTokenValues, err := get1DTokenValues(nextToken.Value())
		nextToken.FinalizeAll()
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

		if callback != nil {
			if !callback(int(nextTokenValues[0])) {
				break
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

		reshapeResults, err := reshapeExec.Call(prompt)
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
		return gen.generateBeamSearchWithKVCache(backend, scope, prompt, batchSize, promptLen)
	}
	return gen.generateBeamSearchNaive(backend, scope, prompt, batchSize, promptLen)
}

// generateBeamSearchNaive performs beam search with the naive model.
// Maintains multiple beam hypotheses and selects the best sequence.
func (gen *Generator) generateBeamSearchNaive(
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

	replicatedResults, err := replicateExec.Call(prompt)
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

	for step := range numSteps {
		if gen.beamExec == nil {
			var err error
			gen.beamExec, err = model.NewExec(backend, predScope.Store(), func(scope *model.Scope, sequences, scores *Node) (*Node, *Node, *Node) {
				g := sequences.Graph()
				backend := g.Backend()
				strategy := gen.getBucketingStrategy(backend)
				currentLength := sequences.Shape().Dimensions[1]
				bucketedLen := strategy.Bucket(currentLength)

				// Run model
				var paddedSequences *Node
				if bucketedLen != currentLength {
					padVals := make([]int32, batchBeamSize*(bucketedLen-currentLength))
					if gen.PadToken != 0 {
						for i := range padVals {
							padVals[i] = int32(gen.PadToken)
						}
					}
					padding := ConstAs(sequences, padVals)
					padding = Reshape(padding, batchBeamSize, bucketedLen-currentLength)
					paddedSequences = Concatenate([]*Node{sequences, padding}, 1)
				} else {
					paddedSequences = sequences
				}
				logits := gen.naiveModelFn(scope, paddedSequences, currentLength)

				// Last token logits: [batch_beam_size, vocab_size]
				lastIndex := Const(g, int32(currentLength-1))
				transposed := Transpose(logits, 0, 1)
				lastLogits := Gather(transposed, lastIndex)

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
				currentSequences.FinalizeAll()
				beamScores.FinalizeAll()
				return nil, errors.WithMessagef(err, "failed to create beam search exec")
			}
		}

		outputs, err := gen.beamExec.Call(currentSequences, beamScores)
		if err != nil {
			currentSequences.FinalizeAll()
			beamScores.FinalizeAll()
			return nil, errors.WithMessagef(err, "beam search step %d failed", step)
		}

		nextSequences := outputs[0]
		nextScores := outputs[1]
		isFinished := outputs[2]

		currentSequences.FinalizeAll()
		beamScores.FinalizeAll()

		currentSequences = nextSequences
		beamScores = nextScores

		// All beams finished?
		var allFinished bool
		if beamConfig.EarlyStopping() {
			finishedValue := isFinished.Value()
			allFinished = true
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
		}
		isFinished.FinalizeAll()

		if beamConfig.EarlyStopping() && allFinished {
			break
		}
	}

	// Apply length penalty; select best sequences
	selectExec, err := model.NewExec(backend, scope.Store(), func(scope *model.Scope, sequences, scores *Node) (*Node, *Node) {
		// Select best sequences (length penalty is applied automatically)
		bestSeqs, bestScores := beamConfig.SelectBest(sequences, scores)
		return bestSeqs, bestScores
	})
	if err != nil {
		currentSequences.FinalizeAll()
		beamScores.FinalizeAll()
		return nil, errors.WithMessagef(err, "failed to create select exec")
	}

	selectResults, err := selectExec.Call(currentSequences, beamScores)
	currentSequences.FinalizeAll()
	beamScores.FinalizeAll()
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to select best sequences")
	}

	bestSeqs := selectResults[0]
	bestScores := selectResults[1]
	bestScores.FinalizeAll()
	return bestSeqs, nil
}

// generateBeamSearchWithKVCache performs beam search with KV caching.
// Each beam maintains its own cache for efficient incremental generation.
func (gen *Generator) generateBeamSearchWithKVCache(
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

	replicatedResults, err := replicateExec.Call(prompt)
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
		replicatedPrompt.FinalizeAll()
		return nil, err
	}

	promptExec, err := gen.getPromptExec(backend, scope, cacheSeqLen)
	if err != nil {
		replicatedPrompt.FinalizeAll()
		finalizeKVCache(kvCacheTensors)
		return nil, err
	}

	promptInputs := make([]any, 1+len(kvCacheTensorsSerialized))
	promptInputs[0] = replicatedPrompt
	for i, t := range kvCacheTensorsSerialized {
		donated, err := DonateTensorBuffer(t, backend, 0)
		if err != nil {
			replicatedPrompt.FinalizeAll()
			finalizeKVCache(kvCacheTensors)
			return nil, err
		}
		promptInputs[i+1] = donated
	}

	outputs, err := promptExec.Call(promptInputs...)
	if err != nil {
		replicatedPrompt.FinalizeAll()
		finalizeKVCache(kvCacheTensors)
		return nil, errors.WithMessagef(err, "failed to process prompt")
	}

	// Finalize outputs[0] (first prompt-pred token) because it's leaked/not used
	outputs[0].FinalizeAll()
	// Clean up of initial serialized cache tensors is no longer needed since they were donated.

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
		paddedKVCacheTensors, err := gen.KVCache.PadTensors(kvCacheTensors, position)
		if err != nil {
			currentSequences.FinalizeAll()
			beamScores.FinalizeAll()
			finalizeKVCache(kvCacheTensors)
			return nil, err
		}
		for path, oldT := range kvCacheTensors {
			if newT, ok := paddedKVCacheTensors[path]; ok && newT != oldT {
				oldT.FinalizeAll()
			}
		}
		kvCacheTensors = paddedKVCacheTensors

		kvCacheTensorsSerialized, err = gen.KVCache.SerializeTensors(kvCacheTensors)
		if err != nil {
			currentSequences.FinalizeAll()
			beamScores.FinalizeAll()
			finalizeKVCache(kvCacheTensors)
			return nil, err
		}

		extractExec, err := model.NewExec(backend, scope.Store(), func(scope *model.Scope, sequences *Node) *Node {
			// Last token: [batch_beam_size]
			lastTokens := Slice(sequences, AxisRange(), AxisElem(-1))
			return lastTokens
		})
		if err != nil {
			currentSequences.FinalizeAll()
			beamScores.FinalizeAll()
			finalizeKVCache(kvCacheTensors)
			return nil, errors.WithMessagef(err, "failed to create extract exec")
		}

		extractResults, err := extractExec.Call(currentSequences)
		if err != nil {
			currentSequences.FinalizeAll()
			beamScores.FinalizeAll()
			finalizeKVCache(kvCacheTensors)
			return nil, errors.WithMessagef(err, "failed to extract last tokens")
		}

		lastTokens := extractResults[0]
		positionTensor := tensors.FromValue(int32(position))

		if gen.beamKVCacheExec == nil {
			gen.beamKVCacheExec, err = model.NewExec(backend, genScope.Store(), func(scope *model.Scope, inputs []*Node) []*Node {
				tokens := inputs[0]
				scores := inputs[1]
				sequences := inputs[2]
				positionNode := inputs[3]
				cacheNodes := inputs[4:]
				tokensReshaped := ExpandDims(tokens, -1) // [batch_beam_size, 1]

				// Process with incremental model
				cache := gen.KVCache.DeserializeNodes(cacheNodes)
				logits, updatedCache := gen.cacheModelFn(scope, tokensReshaped, positionNode, cache)
				logits = Squeeze(logits, 1) // [batch_beam_size, vocab_size]

				// Beam search step
				positionVal := sequences.Shape().Dimensions[1]
				nextSeqs, nextScores, isFinished, gatherIndices := beamConfig.Step(
					logits,
					sequences,
					scores,
					positionVal,
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
				currentSequences.FinalizeAll()
				beamScores.FinalizeAll()
				lastTokens.FinalizeAll()
				positionTensor.FinalizeAll()
				finalizeKVCache(kvCacheTensors)
				return nil, errors.WithMessagef(err, "failed to create beamKVCacheExec")
			}
		}

		stepInputs := make([]any, 4+len(kvCacheTensorsSerialized))
		stepInputs[0] = lastTokens
		stepInputs[1] = beamScores
		stepInputs[2] = currentSequences
		stepInputs[3] = positionTensor
		for i, t := range kvCacheTensorsSerialized {
			donated, err := DonateTensorBuffer(t, backend, 0)
			if err != nil {
				currentSequences.FinalizeAll()
				beamScores.FinalizeAll()
				lastTokens.FinalizeAll()
				positionTensor.FinalizeAll()
				finalizeKVCache(kvCacheTensors)
				return nil, err
			}
			stepInputs[i+4] = donated
		}

		outputs, err := gen.beamKVCacheExec.Call(stepInputs...)
		lastTokens.FinalizeAll()
		positionTensor.FinalizeAll()
		if err != nil {
			currentSequences.FinalizeAll()
			beamScores.FinalizeAll()
			finalizeKVCache(kvCacheTensors)
			return nil, errors.WithMessagef(err, "beam search step %d failed", step)
		}

		nextSeqs := outputs[0]
		nextScores := outputs[1]
		isFinished := outputs[2]
		newKVCacheTensors := gen.KVCache.DeserializeTensors(outputs[3:])

		currentSequences.FinalizeAll()
		beamScores.FinalizeAll()
		// No need to finalize kvCacheTensors because they were donated.

		currentSequences = nextSeqs
		beamScores = nextScores
		kvCacheTensors = newKVCacheTensors

		// Early stopping check
		var allFinished bool
		if beamConfig.EarlyStopping() {
			finishedValue := isFinished.Value()
			allFinished = true
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
		}
		isFinished.FinalizeAll()

		if beamConfig.EarlyStopping() && allFinished {
			break
		}
	}

	// Select best sequence
	selectExec, err := model.NewExec(backend, scope.Store(), func(scope *model.Scope, sequences, scores *Node) *Node {
		bestSeqs, _ := beamConfig.SelectBest(sequences, scores)
		return bestSeqs
	})
	if err != nil {
		currentSequences.FinalizeAll()
		beamScores.FinalizeAll()
		finalizeKVCache(kvCacheTensors)
		return nil, errors.WithMessagef(err, "failed to create select exec")
	}

	bestResults, err := selectExec.Call(currentSequences, beamScores)
	currentSequences.FinalizeAll()
	beamScores.FinalizeAll()
	finalizeKVCache(kvCacheTensors)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to select best sequences")
	}

	return bestResults[0], nil
}

// GenerateStreaming performs streaming generation, yielding tokens as they are generated.
// The callback function is called for each generated token and can return false to stop generation.
//
// It doesn't work with [sampling.StrategyBeamSearch].
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

	// Validate configuration
	if err := gen.validate(); err != nil {
		return errors.WithMessagef(err, "invalid Generator config")
	}

	promptTensor, err := tensors.FromAnyValue(prompt)
	if err != nil {
		return err
	}
	if promptTensor.Rank() != 2 && promptTensor.Rank() != 1 {
		return errors.Errorf("prompt must be 1D or 2D, got rank %d", promptTensor.Rank())
	}

	if gen.Strategy == sample.StrategyBeamSearch {
		return errors.Errorf("streaming generation is not supported with beam search")
	}

	_, err = gen.generateSampling(backend, scope, promptTensor, callback)
	return err
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
	case int32:
		return []int32{fVal}, nil
	case int64:
		return []int32{int32(fVal)}, nil
	case int:
		return []int32{int32(fVal)}, nil
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
