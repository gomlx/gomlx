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
		reduced := g.Distributed().AllReduce(x, backends.ReduceOpSum)
		g.Compile(reduced)
		outputs := g.Run(float32(1), float32(3))
		require.Len(t, outputs, mesh.NumDevices())
		for i, output := range outputs {
			fmt.Printf("\t- device #%d: %s\n", i, output)
			require.Equalf(t, float32(4), output.Value(), "device #%d got %s", i, output)
		}
	})

	t.Run("multiple values, same dtype", func(t *testing.T) {
		mesh := must1(distributed.NewDeviceMesh(backend, []int{2}, []string{"replica"}))
		g := graph.NewGraph(backend, t.Name()).
			WithDistributedStrategy(distributed.SPMD).
			WithDeviceMesh(mesh)
		g.AssertBuilding()
		x := graph.Parameter(g, "x", shapes.Make(dtypes.Float32))
		y := graph.Parameter(g, "y", shapes.Make(dtypes.Float32, 2))
		reduced := g.Distributed().AllReduceMany([]*graph.Node{x, y}, backends.ReduceOpSum)
		g.Compile(reduced...)
		outputs := g.Run(
			float32(1), []float32{10, 11}, // Replica 0
			float32(3), []float32{20, 21}) // Replica 1
		const numOutputs = 2
		require.Len(t, outputs, numOutputs*mesh.NumDevices())
		for deviceIdx := range mesh.NumDevices() {
			reducedX, reducedY := outputs[deviceIdx*numOutputs], outputs[deviceIdx*numOutputs+1]
			fmt.Printf("\t- device #%d reduced-sum outputs: x=%s, y=%s\n", deviceIdx, reducedX, reducedY)
			require.Equalf(t, float32(4), reducedX.Value(), "device #%d got reduced-sum x=%s", deviceIdx, reducedX)
			require.Equalf(t, []float32{30, 32}, reducedY.Value(), "device #%d got reduced-sum x=%s", deviceIdx, reducedY)
		}
	})

	t.Run("multiple values, different dtype", func(t *testing.T) {
		mesh := must1(distributed.NewDeviceMesh(backend, []int{2}, []string{"replica"}))
		g := graph.NewGraph(backend, t.Name()).
			WithDistributedStrategy(distributed.SPMD).
			WithDeviceMesh(mesh)
		g.AssertBuilding()
		x := graph.Parameter(g, "x", shapes.Make(dtypes.Float32))
		y := graph.Parameter(g, "y", shapes.Make(dtypes.Float32, 2))
		z := graph.Parameter(g, "z", shapes.Make(dtypes.Float64, 3))
		reduced := g.Distributed().AllReduceMany([]*graph.Node{x, y, z}, backends.ReduceOpSum)
		g.Compile(reduced...)
		outputs := g.Run(
			float32(1), []float32{10, 11}, []float64{0.1, 0.2, 0.3}, // Replica 0
			float32(3), []float32{20, 21}, []float64{10, 100, 1000}) // Replica 1
		const numOutputs = 3
		require.Len(t, outputs, numOutputs*mesh.NumDevices())
		for deviceIdx := range mesh.NumDevices() {
			reducedX, reducedY, reducedZ := outputs[deviceIdx*numOutputs], outputs[deviceIdx*numOutputs+1], outputs[deviceIdx*numOutputs+2]
			fmt.Printf("\t- device #%d reduced-sum outputs: x=%s, y=%s, z=%s\n", deviceIdx, reducedX, reducedY, reducedZ)
			require.Equalf(t, float32(4), reducedX.Value(), "device #%d got reduced-sum x=%s", deviceIdx, reducedX)
			require.Equalf(t, []float32{30, 32}, reducedY.Value(), "device #%d got reduced-sum y=%s", deviceIdx, reducedY)
			require.Equalf(t, []float64{10.1, 100.2, 1000.3}, reducedZ.Value(), "device #%d got reduced-sum z=%s", deviceIdx, reducedZ)
		}
	})

}
