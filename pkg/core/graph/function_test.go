package graph_test

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/stretchr/testify/assert"
)

func TestFunction(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("IsMain", func(t *testing.T) {
		g := graph.NewGraph(backend, t.Name())
		_ = graph.Parameter(g, "x", shapes.Make(dtypes.F32))
		assert.True(t, g.IsMainFunc())
		assert.True(t, g.CurrentFunc().IsMain())
		g.Finalize()
	})

}
