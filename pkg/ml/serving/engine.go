package serving

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	mlctx "github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/decode"
	"github.com/gomlx/gomlx/pkg/ml/decode/sample"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"
)

var (
	// ErrEngineStopped is returned by Submit when the engine has been stopped.
	ErrEngineStopped = errors.New("engine is stopped")

	// ErrPromptEmpty is returned by Submit when the input tokens slice is empty.
	ErrPromptEmpty = errors.New("prompt is empty")
)

// Config holds engine configuration.
type Config struct {
	// MaxSeqLen is the maximum total sequence length (prompt + generated).
	// This determines the KV cache size per request.
	MaxSeqLen int

	// MaxBatchSize is the maximum number of concurrent requests in a single
	// forward pass. Only used in batched mode (NewBatched).
	// Default: 8.
	MaxBatchSize int

	// SubmitQueueSize is the capacity of the internal submit channel.
	// Default: 256.
	SubmitQueueSize int

	// Preemption configures request preemption under memory pressure.
	// nil means preemption is disabled.
	Preemption *PreemptionPolicy

	// Speculative configures speculative decoding.
	// nil means speculative decoding is disabled.
	Speculative *SpeculativeConfig
}

// DefaultConfig returns a default engine configuration.
func DefaultConfig() Config {
	return Config{
		MaxSeqLen:       2048,
		MaxBatchSize:    8,
		SubmitQueueSize: 256,
	}
}

// applyDefaults fills in zero-valued fields with sensible defaults.
func (c *Config) applyDefaults() {
	if c.SubmitQueueSize <= 0 {
		c.SubmitQueueSize = 256
	}
	if c.MaxSeqLen <= 0 {
		c.MaxSeqLen = 2048
	}
	if c.MaxBatchSize <= 0 {
		c.MaxBatchSize = 8
	}
}

// Engine provides concurrent inference with token streaming.
//
// Multiple goroutines may call Submit concurrently. The engine processes
// requests via a single background loop, streaming generated tokens back
// through channels.
//
// Two modes are supported:
//   - Sequential (New): processes one request at a time using IncrementalModelFn.
//   - Batched (NewBatched): true continuous batching using BatchedModelFn with
//     per-element positions, slot management, and power-of-2 padded batch sizes.
type Engine struct {
	backend   backends.Backend
	modelCtx  *mlctx.Context
	modelFn   decode.IncrementalModelFn // Phase 1 sequential model
	batchedFn decode.BatchedModelFn     // Phase 2 batched model
	tokenizer Tokenizer
	config    Config

	// Batched mode infrastructure.
	batchedMode bool
	slotMgr     *slotManager
	sched       *scheduler

	// Paged KV cache (Phase 3).
	pagedMode bool
	blockMgr  *attention.BlockManager
	pagedCfg  attention.PagedKVCacheConfig

	// Speculative decoding (Phase 4).
	specConfig     *SpeculativeConfig
	draftExecCache map[int]*mlctx.Exec
	verifyExec     *mlctx.Exec // cached verify executor (returns all-position logits)

	// Preemption (Phase 4).
	preemptMgr *preemptionManager

	// Prefix cache (Phase 4, paged mode).
	prefixCache *attention.PrefixCache

	// Request tracking.
	mu       sync.Mutex
	requests map[uint64]*engineRequest
	nextID   uint64

	// Concurrency control.
	submitCh chan *engineRequest
	stopCh   chan struct{}
	stopped  atomic.Bool
	stopOnce sync.Once
	submitMu sync.RWMutex // guards stopped check + submitCh send atomicity
	wg       sync.WaitGroup

	// Cached executors — sequential mode (keyed by position).
	promptExec   *mlctx.Exec
	genExecCache map[int]*mlctx.Exec

	// Cached executors — batched mode (keyed by paddedBatchSize).
	batchedPromptExec      *mlctx.Exec
	batchedDecodeExecCache map[int]*mlctx.Exec // paddedBatchSize → exec
}

// New creates and starts a new Engine. The background step loop begins
// immediately. Call Stop to shut down.
//
// Parameters:
//   - backend: Backend for computation (e.g., SimpleGo or XLA).
//   - ctx: Model context containing weights. The engine calls ctx.Reuse()
//     internally for executor creation.
//   - modelFn: An IncrementalModelFn for KV-cached generation.
//   - tokenizer: Tokenizer for output decoding and EOS detection.
//   - config: Engine configuration.
func New(
	backend backends.Backend,
	ctx *mlctx.Context,
	modelFn decode.IncrementalModelFn,
	tokenizer Tokenizer,
	config Config,
) *Engine {
	config.applyDefaults()

	e := &Engine{
		backend:      backend,
		modelCtx:     ctx,
		modelFn:      modelFn,
		tokenizer:    tokenizer,
		config:       config,
		requests:     make(map[uint64]*engineRequest),
		genExecCache: make(map[int]*mlctx.Exec),
		submitCh:     make(chan *engineRequest, config.SubmitQueueSize),
		stopCh:       make(chan struct{}),
	}

	e.wg.Add(1)
	go e.runStepLoop()

	return e
}

// NewBatched creates and starts a batched Engine that uses a BatchedModelFn
// for continuous batching. Multiple requests share a single forward pass with
// per-element positions, dramatically improving throughput.
//
// Parameters:
//   - backend: Backend for computation (e.g., SimpleGo or XLA).
//   - ctx: Model context containing weights.
//   - batchedFn: A BatchedModelFn with tensor positions.
//   - tokenizer: Tokenizer for output decoding and EOS detection.
//   - config: Engine configuration (MaxBatchSize is used).
func NewBatched(
	backend backends.Backend,
	ctx *mlctx.Context,
	batchedFn decode.BatchedModelFn,
	tokenizer Tokenizer,
	config Config,
) *Engine {
	e := newBatchedEngine(backend, ctx, batchedFn, tokenizer, config)
	e.wg.Add(1)
	go e.runStepLoop()
	return e
}

// NewPaged creates a batched Engine with paged KV cache for memory-efficient
// inference. Instead of pre-allocating maxSeqLen per batch slot, KV entries
// are stored in fixed-size blocks allocated on demand.
//
// Parameters:
//   - backend: Backend for computation.
//   - ctx: Model context containing weights.
//   - batchedFn: A BatchedModelFn with tensor positions.
//   - tokenizer: Tokenizer for output decoding and EOS detection.
//   - config: Engine configuration.
//   - pagedCfg: Paged KV cache configuration (block size, num blocks, etc.).
func NewPaged(
	backend backends.Backend,
	ctx *mlctx.Context,
	batchedFn decode.BatchedModelFn,
	tokenizer Tokenizer,
	config Config,
	pagedCfg attention.PagedKVCacheConfig,
) *Engine {
	e := newBatchedEngine(backend, ctx, batchedFn, tokenizer, config)
	e.pagedMode = true
	e.blockMgr = attention.NewBlockManager(pagedCfg)
	e.pagedCfg = pagedCfg
	e.prefixCache = attention.NewPrefixCache(0)

	e.wg.Add(1)
	go e.runStepLoop()
	return e
}

// newBatchedEngine creates a batched Engine with common initialization shared
// by NewBatched and NewPaged. The caller must start the step loop.
func newBatchedEngine(
	backend backends.Backend,
	ctx *mlctx.Context,
	batchedFn decode.BatchedModelFn,
	tokenizer Tokenizer,
	config Config,
) *Engine {
	config.applyDefaults()

	e := &Engine{
		backend:                backend,
		modelCtx:               ctx,
		batchedFn:              batchedFn,
		tokenizer:              tokenizer,
		config:                 config,
		batchedMode:            true,
		slotMgr:                newSlotManager(config.MaxBatchSize),
		sched:                  newScheduler(config.MaxBatchSize),
		requests:               make(map[uint64]*engineRequest),
		batchedDecodeExecCache: make(map[int]*mlctx.Exec),
		submitCh:               make(chan *engineRequest, config.SubmitQueueSize),
		stopCh:                 make(chan struct{}),
	}

	if config.Preemption != nil {
		e.preemptMgr = newPreemptionManager(*config.Preemption)
	}
	if config.Speculative != nil {
		e.specConfig = config.Speculative
	}

	return e
}

// Submit submits a generation request with pre-tokenized input.
//
// Returns channels for streaming output and errors. The caller should
// range over the output channel until it is closed, then read from the
// error channel (which will also be closed; nil error means success).
//
// The context controls cancellation: if ctx is cancelled, the engine will
// stop generating for this request, send the context error on errChan,
// and close both channels.
func (e *Engine) Submit(
	ctx context.Context,
	inputTokens []int32,
	opts RequestOptions,
) (<-chan SequenceDelta, <-chan error, error) {
	if len(inputTokens) == 0 {
		return nil, nil, ErrPromptEmpty
	}

	if opts.MaxNewTokens <= 0 {
		opts.MaxNewTokens = 100
	}

	outputChan := make(chan SequenceDelta, 256)
	errChan := make(chan error, 1)

	req := &engineRequest{
		inputTokens: make([]int32, len(inputTokens)),
		opts:        opts,
		outputChan:  outputChan,
		errChan:     errChan,
		ctx:         ctx,
		startTime:   time.Now(),
	}
	copy(req.inputTokens, inputTokens)

	// Atomic check-and-send: RLock prevents Stop from closing submitCh
	// while we're checking stopped + sending.
	e.submitMu.RLock()
	if e.stopped.Load() {
		e.submitMu.RUnlock()
		return nil, nil, ErrEngineStopped
	}
	select {
	case e.submitCh <- req:
	case <-ctx.Done():
		e.submitMu.RUnlock()
		return nil, nil, ctx.Err()
	}
	e.submitMu.RUnlock()

	return outputChan, errChan, nil
}

// Stop signals the engine to stop and waits for the background loop to drain.
// After Stop returns, no further requests will be processed. Submit calls
// after Stop returns will return ErrEngineStopped.
func (e *Engine) Stop() {
	e.stopOnce.Do(func() {
		e.stopped.Store(true)
		close(e.stopCh)
	})
	e.wg.Wait()

	// Drain any requests that were queued after the loop exited.
	e.submitMu.Lock()
	defer e.submitMu.Unlock()
	for {
		select {
		case req := <-e.submitCh:
			select {
			case req.errChan <- ErrEngineStopped:
			default:
			}
			close(req.outputChan)
			close(req.errChan)
		default:
			return
		}
	}
}

// SetSpeculativeConfig enables speculative decoding with the given config.
// Prefer setting Config.Speculative before construction.
// If called after construction, must be called before submitting requests.
func (e *Engine) SetSpeculativeConfig(config SpeculativeConfig) {
	if config.NumSpecTokens <= 0 {
		config.NumSpecTokens = 4
	}
	e.mu.Lock()
	e.specConfig = &config
	// Invalidate cached executors compiled for the old speculative config.
	e.verifyExec = nil
	e.draftExecCache = nil
	e.mu.Unlock()
}

// EnablePreemption enables request preemption with the given policy.
// Prefer setting Config.Preemption before construction.
// If called after construction, must be called before submitting requests.
func (e *Engine) EnablePreemption(policy PreemptionPolicy) {
	e.mu.Lock()
	e.preemptMgr = newPreemptionManager(policy)
	e.mu.Unlock()
}

// initPromptExec lazily initializes the prompt executor.
func (e *Engine) initPromptExec() error {
	if e.promptExec != nil {
		return nil
	}

	var err error
	e.promptExec, err = mlctx.NewExec(e.backend, e.modelCtx.Reuse(),
		func(ctx *mlctx.Context, tokens *Node) *Node {
			logits := e.modelFn(ctx, tokens, 0)
			// Extract last token logits: [batch, seqLen, vocab] -> [batch, vocab]
			lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
			lastLogits = Squeeze(lastLogits, 1)
			return lastLogits
		},
	)
	return err
}

// getGenExec returns a cached executor for the given position, creating one if needed.
func (e *Engine) getGenExec(position int) (*mlctx.Exec, error) {
	if exec, ok := e.genExecCache[position]; ok {
		return exec, nil
	}

	exec, err := mlctx.NewExec(e.backend, e.modelCtx.Reuse(),
		func(ctx *mlctx.Context, token *Node) *Node {
			tokenReshaped := ExpandDims(token, -1) // [batch] -> [batch, 1]
			logits := e.modelFn(ctx, tokenReshaped, position)
			lastLogits := Squeeze(logits, 1) // [batch, 1, vocab] -> [batch, vocab]
			return lastLogits
		},
	)
	if err != nil {
		return nil, err
	}
	e.genExecCache[position] = exec
	return exec, nil
}

// greedySample returns the index of the maximum value in logits (argmax).
// This is CPU-side greedy sampling used by sequential mode and speculative decoding.
func greedySample(logits []float32) int32 {
	maxIdx := int32(0)
	maxVal := logits[0]
	for i := int32(1); i < int32(len(logits)); i++ {
		if logits[i] > maxVal {
			maxVal = logits[i]
			maxIdx = i
		}
	}
	return maxIdx
}

// initBatchedPromptExec lazily initializes the batched prompt executor.
// The prompt executor takes [1, promptLen] tokens + [1] positions and returns
// [1, vocabSize] logits. Prefills are done one-at-a-time because prompt
// lengths vary.
func (e *Engine) initBatchedPromptExec() error {
	if e.batchedPromptExec != nil {
		return nil
	}

	var err error
	e.batchedPromptExec, err = mlctx.NewExec(e.backend, e.modelCtx.Reuse(),
		func(ctx *mlctx.Context, tokens *Node, positions *Node) *Node {
			logits := e.batchedFn(ctx, tokens, positions)
			// Extract last-token logits: [1, seqLen, vocab] -> [1, vocab]
			lastLogits := Slice(logits, AxisRange(), AxisElem(-1), AxisRange())
			lastLogits = Squeeze(lastLogits, 1)
			return lastLogits
		},
	)
	return err
}

// getBatchedDecodeExec returns a cached decode executor for the given padded
// batch size, creating one if needed. The executor takes:
//   - tokens: [paddedBatch, 1] int32
//   - positions: [paddedBatch] int32
//
// and returns logits [paddedBatch, vocabSize] with greedy sampling applied.
func (e *Engine) getBatchedDecodeExec(paddedBatch int) (*mlctx.Exec, error) {
	if exec, ok := e.batchedDecodeExecCache[paddedBatch]; ok {
		return exec, nil
	}

	exec, err := mlctx.NewExec(e.backend, e.modelCtx.Reuse(),
		func(ctx *mlctx.Context, tokens *Node, positions *Node) *Node {
			logits := e.batchedFn(ctx, tokens, positions)
			// logits: [paddedBatch, 1, vocab] -> [paddedBatch, vocab]
			lastLogits := Squeeze(logits, 1)

			// GPU-side greedy sampling: argmax per batch element.
			sampled := sample.Greedy(lastLogits)
			return sampled // [paddedBatch] int32
		},
	)
	if err != nil {
		return nil, err
	}
	e.batchedDecodeExecCache[paddedBatch] = exec
	return exec, nil
}

// buildPaddedTokens creates a [paddedBatch, 1] int32 token tensor and
// [paddedBatch] int32 positions tensor from the batch. Padding elements
// use token 0 and position 0 (their outputs are discarded).
func buildPaddedTokens(b *batch, paddedSize int) (tokens [][]int32, positions []int32) {
	tokens = make([][]int32, paddedSize)
	positions = make([]int32, paddedSize)
	for i, req := range b.requests {
		prevToken := req.generatedTokens[len(req.generatedTokens)-1]
		tokens[i] = []int32{prevToken}
		positions[i] = b.positions[i]
	}
	// Fill padding slots with zeros.
	for i := len(b.requests); i < paddedSize; i++ {
		tokens[i] = []int32{0}
		positions[i] = 0
	}
	return
}

// addRequest assigns an ID (and in batched mode, a KV cache slot) to the request.
// Block allocation and preemption happen without holding e.mu to avoid deadlocks.
func (e *Engine) addRequest(req *engineRequest) {
	// Assign ID under lock, then release for allocations.
	e.mu.Lock()
	req.id = e.nextID
	e.nextID++
	e.mu.Unlock()

	if e.batchedMode {
		slot, err := e.slotMgr.Allocate(req.id)
		if err != nil {
			e.failRequest(req, err)
			return
		}
		req.slot = slot
	}

	// In paged mode, pre-allocate blocks for the full sequence
	// (prompt + all generated tokens). This avoids needing to grow blocks
	// during decode, which could fail when free blocks are exhausted.
	if e.pagedMode {
		promptLen := len(req.inputTokens)
		blocksNeeded := promptLen + req.opts.MaxNewTokens

		// Check prefix cache for reusable KV blocks.
		hash := attention.HashTokens(req.inputTokens)
		req.prefixHash = hash
		if cachedBlocks, cachedTokens, ok := e.prefixCache.LookupAndRef(hash); ok {
			// Cache hit -- reuse prefix blocks and only allocate for the rest.
			req.prefixBlocks = cachedBlocks
			req.prefixLen = cachedTokens
			req.hasPrefixHit = true
			blocksNeeded = max(promptLen-cachedTokens+req.opts.MaxNewTokens, 1)

		}

		err := e.blockMgr.EnsureBlocks(req.id, blocksNeeded)

		// If allocation failed, try evicting prefix cache entries first
		// (cheaper than preemption — no request needs to re-prefill).
		if err != nil && e.prefixCache != nil {
			for err != nil {
				freed := e.prefixCache.EvictLRU()
				if len(freed) == 0 {
					break
				}
				e.blockMgr.RecycleBlocks(freed)
				err = e.blockMgr.EnsureBlocks(req.id, blocksNeeded)
			}
		}

		// If still not enough, try preemption (may need multiple victims).
		if err != nil && e.preemptMgr != nil {
			for err != nil {
				victimID := e.preemptLowestPriority()
				if victimID == 0 {
					break
				}
				err = e.blockMgr.EnsureBlocks(req.id, blocksNeeded)
			}
		}

		if err != nil {
			if e.batchedMode {
				e.slotMgr.Free(req.slot)
			}
			// Unref prefix blocks if we had a cache hit.
			if req.hasPrefixHit {
				freed := e.prefixCache.Unref(req.prefixBlocks)
				if len(freed) > 0 {
					e.blockMgr.RecycleBlocks(freed)
				}
			}
			e.failRequest(req, err)
			return
		}
	}

	e.mu.Lock()
	e.requests[req.id] = req
	e.mu.Unlock()
}

// failRequest sends an error on the request's channels and closes them.
// Used when a request cannot be admitted (no slots, no blocks, etc.).
func (e *Engine) failRequest(req *engineRequest, err error) {
	select {
	case req.errChan <- err:
	default:
	}
	close(req.outputChan)
	close(req.errChan)
}

// removeRequest removes a request from the active map.
func (e *Engine) removeRequest(id uint64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	delete(e.requests, id)
}

// hasActiveRequests returns true if there are any in-flight requests.
func (e *Engine) hasActiveRequests() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.requests) > 0
}

// resetKVCache clears KV cache variables between requests.
func (e *Engine) resetKVCache() {
	attention.KVCacheReset(e.modelCtx)
}
