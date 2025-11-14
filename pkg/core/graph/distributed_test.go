package graph_test

import (
	"fmt"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/distributed"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
)

func must1[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

func TestDistributedAllReduce(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	if backend.NumDevices() <= 1 {
		t.Skipf("Skipping distributed test because there are only 1 device available for backend %q.", backend.Name())
	}

	t.Run("scalar", func(t *testing.T) {
		mesh := must1(distributed.NewDeviceMesh(backend, []int{2}, []string{"replica"}))
		g := graph.NewGraph(backend, t.Name()).
			WithDistributedStrategy(distributed.SPMD).
			WithDeviceMesh(mesh)
		g.AssertBuilding()
		x := graph.Parameter(g, "x", shapes.Make(dtypes.Float32))
		fmt.Printf("Graph:\n%s\n", g)
		reduced := g.Distributed().AllReduce(x, backends.ReduceOpSum)
		g.Compile(reduced)
		fmt.Printf("Graph:\n%s\n", g)
		outputs := g.Run(float32(1), float32(3))
		require.Len(t, outputs, mesh.NumDevices())
		for i, output := range outputs {
			require.Equalf(t, float32(4), output.Value(), "device #%d got %s", i, output)
		}
	})
}
