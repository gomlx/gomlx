package engine

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/gomlx/gomlx/backends/simplego"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	mlctx "github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/decode"
	"github.com/gomlx/gomlx/pkg/ml/decode/sample"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"
)

// mockTokenizer implements Tokenizer for testing.
type mockTokenizer struct {
	eosID int32
}

func (t *mockTokenizer) Decode(tokenID int32) (string, error) {
	return string(rune('a' + tokenID%26)), nil
}

func (t *mockTokenizer) IsEOS(tokenID int32) bool {
	return tokenID == t.eosID
}

func (t *mockTokenizer) Reset() {}

// makeConstantModel creates a model that always returns logits where
// token ID `outputToken` has the highest value.
func makeConstantModel(vocabSize int, outputToken int32) decode.IncrementalModelFn {
	return func(ctx *mlctx.Context, newTokens *Node, position int) *Node {
		g := newTokens.Graph()
		batchSize := newTokens.Shape().Dimensions[0]
		seqLen := newTokens.Shape().Dimensions[1]

		// All zeros: greedy will pick token 0.
		logits := Zeros(g, shapes.Make(dtypes.Float32, batchSize, seqLen, vocabSize))

		if outputToken != 0 {
			// Create a one-hot bump at outputToken.
			// For now, just add a scalar at the target index using broadcasting.
			// This is complex in graph form, so we use a different approach:
			// create logits as a negative constant and add 10.0 at target.
			logits = AddScalar(logits, -1.0)
			// Build one-hot: [vocabSize] with 11.0 at outputToken, else 0.
			oneHot := OneHot(Const(g, outputToken), vocabSize, dtypes.Float32)
			oneHot = MulScalar(oneHot, 11.0)
			// Broadcast to [batch, seqLen, vocabSize]
			oneHot = BroadcastPrefix(oneHot, batchSize, seqLen)
			logits = Add(logits, oneHot)
		}
		return logits
	}
}

// makeEOSAfterN creates a model that outputs `normalToken` for the first N
// tokens, then outputs `eosToken`.
func makeEOSAfterN(vocabSize int, normalToken, eosToken int32, n int) decode.IncrementalModelFn {
	return func(ctx *mlctx.Context, newTokens *Node, position int) *Node {
		g := newTokens.Graph()
		batchSize := newTokens.Shape().Dimensions[0]
		seqLen := newTokens.Shape().Dimensions[1]

		var target int32
		if position >= n {
			target = eosToken
		} else {
			target = normalToken
		}

		logits := AddScalar(
			Zeros(g, shapes.Make(dtypes.Float32, batchSize, seqLen, vocabSize)),
			-1.0,
		)
		oneHot := OneHot(Const(g, target), vocabSize, dtypes.Float32)
		oneHot = MulScalar(oneHot, 11.0)
		oneHot = BroadcastPrefix(oneHot, batchSize, seqLen)
		return Add(logits, oneHot)
	}
}

func setupEngine(t *testing.T, modelFn decode.IncrementalModelFn, eosToken int32) *Engine {
	t.Helper()
	backend, err := simplego.New("")
	if err != nil {
		t.Fatalf("Failed to create backend: %v", err)
	}
	ctx := mlctx.New()
	tok := &mockTokenizer{eosID: eosToken}
	config := DefaultConfig()
	config.MaxSeqLen = 128
	return New(backend, ctx, modelFn, tok, config)
}

func TestEngineSubmitAndReceive(t *testing.T) {
	vocabSize := 10
	outputToken := int32(3)
	engine := setupEngine(t, makeConstantModel(vocabSize, outputToken), -1)
	defer engine.Stop()

	maxTokens := 5
	outCh, errCh, err := engine.Submit(
		context.Background(),
		[]int32{1},
		RequestOptions{MaxNewTokens: maxTokens, Strategy: sample.StrategyGreedy},
	)
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	var deltas []SequenceDelta
	for d := range outCh {
		deltas = append(deltas, d)
	}

	// Check no errors.
	for err := range errCh {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(deltas) != maxTokens {
		t.Fatalf("Expected %d deltas, got %d", maxTokens, len(deltas))
	}

	for i, d := range deltas {
		if d.TokenID != outputToken {
			t.Errorf("Delta %d: expected token %d, got %d", i, outputToken, d.TokenID)
		}
		if d.EOSReached {
			t.Errorf("Delta %d: unexpected EOS", i)
		}
	}
}

func TestEngineEOS(t *testing.T) {
	vocabSize := 10
	normalToken := int32(3)
	eosToken := int32(9)
	// Model outputs normalToken for first 3 positions, then eosToken.
	engine := setupEngine(t, makeEOSAfterN(vocabSize, normalToken, eosToken, 3), eosToken)
	defer engine.Stop()

	outCh, errCh, err := engine.Submit(
		context.Background(),
		[]int32{1},
		RequestOptions{MaxNewTokens: 100, Strategy: sample.StrategyGreedy},
	)
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	var deltas []SequenceDelta
	for d := range outCh {
		deltas = append(deltas, d)
	}

	for err := range errCh {
		t.Fatalf("Unexpected error: %v", err)
	}

	// The prompt is at position 0. Decode starts at position 1.
	// Position 0 (prefill): outputs normalToken (position=0 < 3)
	// Position 1: outputs normalToken (position=1 < 3)
	// Position 2: outputs normalToken (position=2 < 3)
	// Position 3: outputs eosToken (position=3 >= 3)
	// Total: 3 normal + 1 EOS = 4 deltas
	if len(deltas) < 1 {
		t.Fatalf("Expected at least 1 delta, got %d", len(deltas))
	}

	lastDelta := deltas[len(deltas)-1]
	if !lastDelta.EOSReached {
		t.Errorf("Expected last delta to have EOSReached=true")
	}
	if lastDelta.TokenID != eosToken {
		t.Errorf("Expected EOS token %d, got %d", eosToken, lastDelta.TokenID)
	}

	// Should have stopped before MaxNewTokens (100).
	if len(deltas) > 10 {
		t.Errorf("Expected early stop on EOS, got %d deltas", len(deltas))
	}
}

func TestEngineContextCancellation(t *testing.T) {
	vocabSize := 10
	// Model that never produces EOS — always outputs token 3.
	engine := setupEngine(t, makeConstantModel(vocabSize, 3), -1)
	defer engine.Stop()

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	outCh, errCh, err := engine.Submit(
		ctx,
		[]int32{1},
		RequestOptions{MaxNewTokens: 100000, Strategy: sample.StrategyGreedy},
	)
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	// Drain output channel.
	for range outCh {
	}

	// Should get a context error.
	gotErr := false
	for err := range errCh {
		if err != nil {
			gotErr = true
		}
	}
	if !gotErr {
		t.Error("Expected context cancellation error, got none")
	}
}

func TestEngineStop(t *testing.T) {
	vocabSize := 10
	engine := setupEngine(t, makeConstantModel(vocabSize, 3), -1)

	// Submit a long-running request.
	outCh, errCh, err := engine.Submit(
		context.Background(),
		[]int32{1},
		RequestOptions{MaxNewTokens: 100000, Strategy: sample.StrategyGreedy},
	)
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	// Let it generate a few tokens.
	time.Sleep(50 * time.Millisecond)

	// Stop the engine.
	engine.Stop()

	// Drain channels — should complete without hanging.
	for range outCh {
	}
	for range errCh {
	}

	// Submit after stop should fail.
	_, _, err = engine.Submit(
		context.Background(),
		[]int32{1},
		RequestOptions{MaxNewTokens: 5},
	)
	if err != ErrEngineStopped {
		t.Errorf("Expected ErrEngineStopped, got %v", err)
	}
}

func TestEngineMaxNewTokens(t *testing.T) {
	vocabSize := 10
	engine := setupEngine(t, makeConstantModel(vocabSize, 3), -1)
	defer engine.Stop()

	maxTokens := 3
	outCh, errCh, err := engine.Submit(
		context.Background(),
		[]int32{1},
		RequestOptions{MaxNewTokens: maxTokens, Strategy: sample.StrategyGreedy},
	)
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	count := 0
	for range outCh {
		count++
	}
	for err := range errCh {
		t.Fatalf("Unexpected error: %v", err)
	}

	if count != maxTokens {
		t.Errorf("Expected %d tokens, got %d", maxTokens, count)
	}
}

func TestEngineConcurrentSubmit(t *testing.T) {
	vocabSize := 10
	engine := setupEngine(t, makeConstantModel(vocabSize, 5), -1)
	defer engine.Stop()

	numRequests := 5
	maxTokens := 3

	var wg sync.WaitGroup
	errors := make(chan error, numRequests)

	for i := range numRequests {
		wg.Go(func() {
			outCh, errCh, err := engine.Submit(
				context.Background(),
				[]int32{int32(i + 1)},
				RequestOptions{MaxNewTokens: maxTokens, Strategy: sample.StrategyGreedy},
			)
			if err != nil {
				errors <- err
				return
			}

			count := 0
			for range outCh {
				count++
			}
			for err := range errCh {
				errors <- err
				return
			}

			if count != maxTokens {
				errors <- fmt.Errorf("expected %d tokens, got %d", maxTokens, count)
			}
		})
	}

	wg.Wait()
	close(errors)

	for err := range errors {
		t.Errorf("Concurrent request error: %v", err)
	}
}

func TestEngineEmptyPrompt(t *testing.T) {
	vocabSize := 10
	engine := setupEngine(t, makeConstantModel(vocabSize, 0), -1)
	defer engine.Stop()

	_, _, err := engine.Submit(
		context.Background(),
		[]int32{},
		DefaultRequestOptions(),
	)
	if err != ErrPromptEmpty {
		t.Errorf("Expected ErrPromptEmpty, got %v", err)
	}

	_, _, err = engine.Submit(
		context.Background(),
		nil,
		DefaultRequestOptions(),
	)
	if err != ErrPromptEmpty {
		t.Errorf("Expected ErrPromptEmpty for nil, got %v", err)
	}
}

// --- Phase 2: Batched Engine Tests ---

// makeConstantBatchedModel creates a BatchedModelFn that always returns logits
// where token ID `outputToken` has the highest value.
// positions is a *Node [batchSize] but is ignored for this constant model.
func makeConstantBatchedModel(vocabSize int, outputToken int32) decode.BatchedModelFn {
	return func(ctx *mlctx.Context, newTokens *Node, positions *Node) *Node {
		g := newTokens.Graph()
		batchSize := newTokens.Shape().Dimensions[0]
		seqLen := newTokens.Shape().Dimensions[1]

		logits := Zeros(g, shapes.Make(dtypes.Float32, batchSize, seqLen, vocabSize))

		if outputToken != 0 {
			logits = AddScalar(logits, -1.0)
			oneHot := OneHot(Const(g, outputToken), vocabSize, dtypes.Float32)
			oneHot = MulScalar(oneHot, 11.0)
			oneHot = BroadcastPrefix(oneHot, batchSize, seqLen)
			logits = Add(logits, oneHot)
		}
		return logits
	}
}

// makeEOSAfterNBatched creates a BatchedModelFn that outputs eosToken after
// any batch element's position exceeds n. Since positions are tensors,
// we use a graph-level comparison.
func makeEOSAfterNBatched(vocabSize int, normalToken, eosToken int32, n int) decode.BatchedModelFn {
	return func(ctx *mlctx.Context, newTokens *Node, positions *Node) *Node {
		g := newTokens.Graph()
		batchSize := newTokens.Shape().Dimensions[0]
		seqLen := newTokens.Shape().Dimensions[1]

		// Build logits for normalToken.
		normalLogits := AddScalar(
			Zeros(g, shapes.Make(dtypes.Float32, batchSize, seqLen, vocabSize)),
			-1.0,
		)
		normalOneHot := OneHot(Const(g, normalToken), vocabSize, dtypes.Float32)
		normalOneHot = MulScalar(normalOneHot, 11.0)
		normalOneHot = BroadcastPrefix(normalOneHot, batchSize, seqLen)
		normalLogits = Add(normalLogits, normalOneHot)

		// Build logits for eosToken.
		eosLogits := AddScalar(
			Zeros(g, shapes.Make(dtypes.Float32, batchSize, seqLen, vocabSize)),
			-1.0,
		)
		eosOneHot := OneHot(Const(g, eosToken), vocabSize, dtypes.Float32)
		eosOneHot = MulScalar(eosOneHot, 11.0)
		eosOneHot = BroadcastPrefix(eosOneHot, batchSize, seqLen)
		eosLogits = Add(eosLogits, eosOneHot)

		// Per-element condition: positions >= n → use eosLogits, else normalLogits.
		// positions: [batchSize], compare with scalar n.
		cond := GreaterOrEqual(ConvertDType(positions, dtypes.Float32), Const(g, float32(n)))
		// cond: [batchSize] bool → [batchSize, 1, 1] for broadcast
		cond = ExpandDims(cond, -1)
		cond = ExpandDims(cond, -1)
		cond = BroadcastToShape(cond, shapes.Make(dtypes.Bool, batchSize, seqLen, vocabSize))

		return Where(cond, eosLogits, normalLogits)
	}
}

func setupBatchedEngine(t *testing.T, modelFn decode.BatchedModelFn, eosToken int32) *Engine {
	t.Helper()
	backend, err := simplego.New("")
	if err != nil {
		t.Fatalf("Failed to create backend: %v", err)
	}
	ctx := mlctx.New()
	tok := &mockTokenizer{eosID: eosToken}
	config := DefaultConfig()
	config.MaxSeqLen = 128
	config.MaxBatchSize = 4
	return NewBatched(backend, ctx, modelFn, tok, config)
}

func TestBatchedEngineSubmitAndReceive(t *testing.T) {
	vocabSize := 10
	outputToken := int32(3)
	engine := setupBatchedEngine(t, makeConstantBatchedModel(vocabSize, outputToken), -1)
	defer engine.Stop()

	maxTokens := 5
	outCh, errCh, err := engine.Submit(
		context.Background(),
		[]int32{1},
		RequestOptions{MaxNewTokens: maxTokens, Strategy: sample.StrategyGreedy},
	)
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	var deltas []SequenceDelta
	for d := range outCh {
		deltas = append(deltas, d)
	}

	for err := range errCh {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(deltas) != maxTokens {
		t.Fatalf("Expected %d deltas, got %d", maxTokens, len(deltas))
	}

	for i, d := range deltas {
		if d.TokenID != outputToken {
			t.Errorf("Delta %d: expected token %d, got %d", i, outputToken, d.TokenID)
		}
	}
}

func TestBatchedEngineEOS(t *testing.T) {
	vocabSize := 10
	normalToken := int32(3)
	eosToken := int32(9)
	engine := setupBatchedEngine(t, makeEOSAfterNBatched(vocabSize, normalToken, eosToken, 3), eosToken)
	defer engine.Stop()

	outCh, errCh, err := engine.Submit(
		context.Background(),
		[]int32{1},
		RequestOptions{MaxNewTokens: 100, Strategy: sample.StrategyGreedy},
	)
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	var deltas []SequenceDelta
	for d := range outCh {
		deltas = append(deltas, d)
	}

	for err := range errCh {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(deltas) < 1 {
		t.Fatalf("Expected at least 1 delta, got %d", len(deltas))
	}

	lastDelta := deltas[len(deltas)-1]
	if !lastDelta.EOSReached {
		t.Errorf("Expected last delta to have EOSReached=true")
	}
	if lastDelta.TokenID != eosToken {
		t.Errorf("Expected EOS token %d, got %d", eosToken, lastDelta.TokenID)
	}

	if len(deltas) > 10 {
		t.Errorf("Expected early stop on EOS, got %d deltas", len(deltas))
	}
}

func TestBatchedEngineConcurrentSubmit(t *testing.T) {
	vocabSize := 10
	engine := setupBatchedEngine(t, makeConstantBatchedModel(vocabSize, 5), -1)
	defer engine.Stop()

	numRequests := 4 // = MaxBatchSize
	maxTokens := 3

	var wg sync.WaitGroup
	errors := make(chan error, numRequests)

	for i := range numRequests {
		wg.Go(func() {
			outCh, errCh, err := engine.Submit(
				context.Background(),
				[]int32{int32(i + 1)},
				RequestOptions{MaxNewTokens: maxTokens, Strategy: sample.StrategyGreedy},
			)
			if err != nil {
				errors <- err
				return
			}

			count := 0
			for range outCh {
				count++
			}
			for err := range errCh {
				errors <- err
				return
			}

			if count != maxTokens {
				errors <- fmt.Errorf("expected %d tokens, got %d", maxTokens, count)
			}
		})
	}

	wg.Wait()
	close(errors)

	for err := range errors {
		t.Errorf("Concurrent request error: %v", err)
	}
}

func TestBatchedEngineMaxNewTokens(t *testing.T) {
	vocabSize := 10
	engine := setupBatchedEngine(t, makeConstantBatchedModel(vocabSize, 3), -1)
	defer engine.Stop()

	maxTokens := 3
	outCh, errCh, err := engine.Submit(
		context.Background(),
		[]int32{1},
		RequestOptions{MaxNewTokens: maxTokens, Strategy: sample.StrategyGreedy},
	)
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	count := 0
	for range outCh {
		count++
	}
	for err := range errCh {
		t.Fatalf("Unexpected error: %v", err)
	}

	if count != maxTokens {
		t.Errorf("Expected %d tokens, got %d", maxTokens, count)
	}
}

// --- Phase 4 Integration Tests ---

func TestSpeculativeDecodeIntegration(t *testing.T) {
	vocabSize := 10
	outputToken := int32(3)

	// Main model and draft model both produce the same constant token.
	// This means all draft tokens should be accepted.
	mainModel := makeConstantBatchedModel(vocabSize, outputToken)
	draftModel := makeConstantBatchedModel(vocabSize, outputToken)

	backend, err := simplego.New("")
	if err != nil {
		t.Fatalf("Failed to create backend: %v", err)
	}
	ctx := mlctx.New()
	tok := &mockTokenizer{eosID: -1}
	config := DefaultConfig()
	config.MaxSeqLen = 128
	config.MaxBatchSize = 4

	specCfg := SpeculativeConfig{
		DraftModelFn:  draftModel,
		NumSpecTokens: 3,
	}
	config.Speculative = &specCfg
	eng := NewBatched(backend, ctx, mainModel, tok, config)
	defer eng.Stop()

	maxTokens := 5
	outCh, errCh, submitErr := eng.Submit(
		context.Background(),
		[]int32{1, 2},
		RequestOptions{MaxNewTokens: maxTokens, Strategy: sample.StrategyGreedy},
	)
	if submitErr != nil {
		t.Fatalf("Submit failed: %v", submitErr)
	}

	var deltas []SequenceDelta
	for d := range outCh {
		deltas = append(deltas, d)
	}

	for err := range errCh {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(deltas) == 0 {
		t.Fatal("Expected at least 1 delta, got 0")
	}
	if len(deltas) > maxTokens {
		t.Errorf("Got %d deltas, expected at most %d", len(deltas), maxTokens)
	}

	// All tokens should be the expected output token.
	for i, d := range deltas {
		if d.TokenID != outputToken {
			t.Errorf("Delta %d: expected token %d, got %d", i, outputToken, d.TokenID)
		}
	}
}

func TestPreemptionIntegration(t *testing.T) {
	vocabSize := 10
	outputToken := int32(3)
	model := makeConstantBatchedModel(vocabSize, outputToken)

	backend, err := simplego.New("")
	if err != nil {
		t.Fatalf("Failed to create backend: %v", err)
	}
	ctx := mlctx.New()
	tok := &mockTokenizer{eosID: -1}
	config := DefaultConfig()
	config.MaxSeqLen = 128
	config.MaxBatchSize = 4

	// Create paged engine with very few blocks to force preemption.
	pagedCfg := attention.PagedKVCacheConfig{
		NumBlocks:  4, // very small — will run out
		BlockSize:  4,
		NumKVHeads: 1,
		HeadDim:    4,
		DType:      dtypes.Float32,
	}

	preemptPolicy := PreemptRecompute
	config.Preemption = &preemptPolicy
	eng := NewPaged(backend, ctx, model, tok, config, pagedCfg)
	defer eng.Stop()

	// First request consumes blocks.
	maxTokens := 2
	outCh1, errCh1, err := eng.Submit(
		context.Background(),
		[]int32{1, 2},
		RequestOptions{MaxNewTokens: maxTokens, Strategy: sample.StrategyGreedy},
	)
	if err != nil {
		t.Fatalf("First submit failed: %v", err)
	}

	// Drain first request.
	count1 := 0
	for range outCh1 {
		count1++
	}
	for err := range errCh1 {
		t.Fatalf("First request error: %v", err)
	}

	// Second request — blocks freed from first should be available.
	outCh2, errCh2, err := eng.Submit(
		context.Background(),
		[]int32{4, 5},
		RequestOptions{MaxNewTokens: maxTokens, Strategy: sample.StrategyGreedy},
	)
	if err != nil {
		t.Fatalf("Second submit failed: %v", err)
	}

	count2 := 0
	for range outCh2 {
		count2++
	}
	for err := range errCh2 {
		t.Fatalf("Second request error: %v", err)
	}

	if count1 != maxTokens {
		t.Errorf("First request: expected %d tokens, got %d", maxTokens, count1)
	}
	if count2 != maxTokens {
		t.Errorf("Second request: expected %d tokens, got %d", maxTokens, count2)
	}
}

func TestPrefixCacheIntegration(t *testing.T) {
	vocabSize := 10
	outputToken := int32(3)
	model := makeConstantBatchedModel(vocabSize, outputToken)

	backend, err := simplego.New("")
	if err != nil {
		t.Fatalf("Failed to create backend: %v", err)
	}
	ctx := mlctx.New()
	tok := &mockTokenizer{eosID: -1}
	config := DefaultConfig()
	config.MaxSeqLen = 128
	config.MaxBatchSize = 4

	pagedCfg := attention.PagedKVCacheConfig{
		NumBlocks:  16,
		BlockSize:  4,
		NumKVHeads: 1,
		HeadDim:    4,
		DType:      dtypes.Float32,
	}

	eng := NewPaged(backend, ctx, model, tok, config, pagedCfg)
	defer eng.Stop()

	// Verify prefix cache starts empty.
	if eng.prefixCache.NumEntries() != 0 {
		t.Fatalf("Expected empty prefix cache, got %d entries", eng.prefixCache.NumEntries())
	}

	// First request with a specific prompt — cache miss, stores prefix.
	prompt := []int32{1, 2, 3, 4, 5}
	maxTokens := 3
	outCh1, errCh1, err := eng.Submit(
		context.Background(),
		prompt,
		RequestOptions{MaxNewTokens: maxTokens, Strategy: sample.StrategyGreedy},
	)
	if err != nil {
		t.Fatalf("First submit failed: %v", err)
	}

	count1 := 0
	for range outCh1 {
		count1++
	}
	for err := range errCh1 {
		t.Fatalf("First request error: %v", err)
	}

	if count1 != maxTokens {
		t.Errorf("First request: expected %d tokens, got %d", maxTokens, count1)
	}

	// After first request, prefix cache should have an entry.
	if eng.prefixCache.NumEntries() != 1 {
		t.Errorf("Expected 1 prefix cache entry, got %d", eng.prefixCache.NumEntries())
	}

	// Second request with the same prompt — should hit cache.
	outCh2, errCh2, err := eng.Submit(
		context.Background(),
		prompt,
		RequestOptions{MaxNewTokens: maxTokens, Strategy: sample.StrategyGreedy},
	)
	if err != nil {
		t.Fatalf("Second submit failed: %v", err)
	}

	count2 := 0
	for range outCh2 {
		count2++
	}
	for err := range errCh2 {
		t.Fatalf("Second request error: %v", err)
	}

	if count2 != maxTokens {
		t.Errorf("Second request: expected %d tokens, got %d", maxTokens, count2)
	}

	// Prefix cache should still have exactly 1 entry (same prompt).
	if eng.prefixCache.NumEntries() != 1 {
		t.Errorf("Expected 1 prefix cache entry after reuse, got %d", eng.prefixCache.NumEntries())
	}

	// Third request with different prompt — cache miss, stores new entry.
	outCh3, errCh3, err := eng.Submit(
		context.Background(),
		[]int32{10, 20, 30},
		RequestOptions{MaxNewTokens: maxTokens, Strategy: sample.StrategyGreedy},
	)
	if err != nil {
		t.Fatalf("Third submit failed: %v", err)
	}

	count3 := 0
	for range outCh3 {
		count3++
	}
	for err := range errCh3 {
		t.Fatalf("Third request error: %v", err)
	}

	if count3 != maxTokens {
		t.Errorf("Third request: expected %d tokens, got %d", maxTokens, count3)
	}

	// Should now have 2 prefix cache entries.
	if eng.prefixCache.NumEntries() != 2 {
		t.Errorf("Expected 2 prefix cache entries, got %d", eng.prefixCache.NumEntries())
	}
}
