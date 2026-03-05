// Package engine provides a continuous batching inference engine for GoMLX.
//
// The engine accepts concurrent Submit calls from multiple goroutines,
// processes requests via a single background loop, and streams generated
// tokens back through channels. It follows the same concurrency model as
// ONNX Runtime GenAI's Engine API but is implemented in pure Go on top
// of GoMLX's graph execution infrastructure.
package serving

// Tokenizer is the interface the engine uses for streaming decoded text
// back to callers and for detecting end-of-sequence tokens.
//
// The engine itself works with pre-tokenized input ([]int32), so this
// interface handles only the output side: incremental decoding of generated
// token IDs back into text, and EOS detection.
//
// Implementations should maintain internal state for incremental decoding
// (e.g., partial multi-byte characters) and reset it between requests via
// the Reset method.
type Tokenizer interface {
	// Decode incrementally decodes a single token ID into text.
	// It may return "" if the token is a partial multi-byte character
	// that needs subsequent tokens to complete.
	Decode(tokenID int32) (string, error)

	// IsEOS returns true if the given token ID is an end-of-sequence token.
	IsEOS(tokenID int32) bool

	// Reset resets the incremental decoding state.
	// Called when starting a new request.
	Reset()
}
