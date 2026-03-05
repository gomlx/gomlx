package serving

import (
	"context"
	"time"

	"github.com/gomlx/gomlx/pkg/ml/decode/sample"
)

// SequenceDelta represents one incremental output from the engine.
type SequenceDelta struct {
	// Token is the decoded text for this generated token.
	// May be empty if the tokenizer returned a partial multi-byte character.
	Token string

	// TokenID is the raw token ID that was generated.
	TokenID int32

	// EOSReached is true when the end-of-sequence token was generated.
	// No further deltas will be sent after this.
	EOSReached bool
}

// RequestOptions configures generation for a single request.
type RequestOptions struct {
	// MaxNewTokens is the maximum number of new tokens to generate.
	// Default: 100.
	MaxNewTokens int

	// Strategy is the sampling strategy.
	// Default: sample.StrategyGreedy.
	Strategy sample.Strategy

	// Temperature controls randomness. Higher values increase randomness.
	// Only used with temperature-based strategies.
	// Default: 1.0.
	Temperature float32

	// TopK is the k parameter for top-k sampling.
	// Default: 50.
	TopK int

	// TopP is the p parameter for nucleus (top-p) sampling.
	// Default: 0.9.
	TopP float32
}

// DefaultRequestOptions returns sensible defaults for generation.
func DefaultRequestOptions() RequestOptions {
	return RequestOptions{
		MaxNewTokens: 100,
		Strategy:     sample.StrategyGreedy,
		Temperature:  1.0,
		TopK:         50,
		TopP:         0.9,
	}
}

// engineRequest is the internal state for one in-flight generation request.
type engineRequest struct {
	id          uint64
	inputTokens []int32
	opts        RequestOptions

	outputChan chan SequenceDelta
	errChan    chan error
	ctx        context.Context

	// Generation state, mutated by the step loop.
	generatedTokens []int32
	position        int  // current absolute position in sequence (prompt + generated)
	slot            int  // KV cache batch slot index (Phase 2+)
	eosReached      bool
	startTime       time.Time

	// Prefix cache state (Phase 4, paged mode only).
	prefixBlocks []int     // physical block indices shared via prefix cache
	prefixLen    int       // number of tokens covered by cached prefix
	prefixHash   [32]byte  // hash of prefix tokens for cache key
	hasPrefixHit bool      // true if this request reused cached prefix blocks
}
