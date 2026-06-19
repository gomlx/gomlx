package generate

import (
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers/attention/kvcache"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGeneratorConfigFreeze(t *testing.T) {
	backend := testutil.BuildTestBackend()
	store := model.NewStore()

	// Create a simple incremental model function
	var incrementalFn KVCacheModelFn = func(scope *model.Scope, newTokens *graph.Node, position *graph.Node, cache kvcache.KVCacheNodes) (*graph.Node, kvcache.KVCacheNodes) {
		g := newTokens.Graph()
		batchSize := newTokens.Shape().Dimensions[0]
		seqLen := newTokens.Shape().Dimensions[1]
		vocabSize := 10
		return graph.Zeros(g, shapes.Make(dtypes.Float32, batchSize, seqLen, vocabSize)), cache
	}

	gen := New(incrementalFn).WithKVCache(kvcache.NewKVCache(), 1, 1, dtypes.Float32)

	// Config before use - should be fine
	gen.WithMaxLength(50)
	assert.NoError(t, gen.err)

	// Run Decode to trigger initialization of promptExec.
	// We use a valid prompt.
	prompt := []int32{1, 2, 3}
	// The model is dummy, but compatible shapes.
	// Decode might succeed or fail, but promptExec should be initialized.
	_, _ = gen.Decode(backend, store.RootScope(), prompt)

	// Config after use - should fail
	gen.WithMaxLength(60)
	require.Error(t, gen.err)
	assert.Contains(t, gen.err.Error(), "cannot change configuration")

	// Reset error to test another method
	gen.err = nil
	gen.WithTemperature(0.5)
	require.Error(t, gen.err)
	assert.Contains(t, gen.err.Error(), "cannot change configuration")
}
