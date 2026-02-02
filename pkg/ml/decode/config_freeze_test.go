package decode

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	graphtest "github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDecoderConfigFreeze(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New()

	// Create a simple incremental model function
	var incrementalFn IncrementalModelFn = func(ctx *context.Context, newTokens *graph.Node, position int) *graph.Node {
		g := newTokens.Graph()
		batchSize := newTokens.Shape().Dimensions[0]
		seqLen := newTokens.Shape().Dimensions[1]
		vocabSize := 10
		return graph.Zeros(g, shapes.Make(dtypes.Float32, batchSize, seqLen, vocabSize))
	}

	dec := New(incrementalFn)

	// Config before use - should be fine
	dec.WithMaxLength(50)
	assert.NoError(t, dec.err)

	// Run Decode to trigger initialization of promptExec.
	// We use a valid prompt.
	prompt := []int32{1, 2, 3}
	// The model is dummy, but compatible shapes.
	// Decode might succeed or fail, but promptExec should be initialized.
	_, _ = dec.Decode(backend, ctx, prompt)

	// Config after use - should fail
	dec.WithMaxLength(60)
	require.Error(t, dec.err)
	assert.Contains(t, dec.err.Error(), "cannot change configuration")

	// Reset error to test another method
	dec.err = nil
	dec.WithTemperature(0.5)
	require.Error(t, dec.err)
	assert.Contains(t, dec.err.Error(), "cannot change configuration")
}
