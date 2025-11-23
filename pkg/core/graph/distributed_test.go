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
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func must1[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

// TestDistributedPortable tests that a computation is properly compiled as "portable" if
// using no distribution strategy and not assigned to any device.
//
// This can be used to manually distribute computation by executing the program concurrently on different devices.
//
// Portable compilation may not work on all PJRT types or different backends.
// Please add a blacklist of backends to skip the test if the backend doesn't support it.
func TestDistributedPortable(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	if backend.NumDevices() <= 1 {
		t.Skipf("Skipping distributed test because there are only 1 device available for backend %q.", backend.Name())
	}
	g := graph.NewGraph(backend, "portable")
	x := graph.Parameter(g, "x", shapes.Make(dtypes.Float32))
	negX := graph.Neg(x)
	g.Compile(negX)
	outputs := g.Run(float32(1))
	require.Len(t, outputs, 1)
	require.Equalf(t, float32(-1), outputs[0].Value(), "got %s", outputs[0])
}

func TestCollective(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	if backend.NumDevices() <= 1 {
		t.Skipf("Skipping distributed test because there are only 1 device available for backend %q.", backend.Name())
	}

	t.Run("AllReduce:scalar", func(t *testing.T) {
		mesh := must1(distributed.NewDeviceMesh([]int{2}, []string{"replica"}))
		g := graph.NewGraph(backend, t.Name())
		require.NoError(t, g.SetSPMD(mesh))
		x := graph.Parameter(g, "x", shapes.Make(dtypes.Float32))
		reduced := g.Distributed().AllReduceOne(x, backends.ReduceOpSum)
		g.Compile(reduced)
		outputs := g.Run(float32(1), float32(3))
		require.Len(t, outputs, mesh.NumDevices())
		for i, output := range outputs {
			fmt.Printf("\t- device #%d: %s\n", i, output)
			assert.Equalf(t, float32(4), output.Value(), "result for device #%d got %s", i, output)
		}
	})

	t.Run("AllReduce:multiple values, same dtype", func(t *testing.T) {
		mesh := must1(distributed.NewDeviceMesh([]int{2}, []string{"replica"}))
		g := graph.NewGraph(backend, t.Name())
		require.NoError(t, g.SetSPMD(mesh))
		g.AssertBuilding()
		x := graph.Parameter(g, "x", shapes.Make(dtypes.Float32))
		y := graph.Parameter(g, "y", shapes.Make(dtypes.Float32, 2))
		reduced := g.Distributed().AllReduce([]*graph.Node{x, y}, backends.ReduceOpSum)
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
			require.Equalf(
				t,
				[]float32{30, 32},
				reducedY.Value(),
				"device #%d got reduced-sum x=%s",
				deviceIdx,
				reducedY,
			)
		}
	})

	t.Run("AllReduce:multiple values, different dtype", func(t *testing.T) {
		mesh := must1(distributed.NewDeviceMesh([]int{2}, []string{"replica"}))
		g := graph.NewGraph(backend, t.Name())
		require.NoError(t, g.SetSPMD(mesh))
		g.AssertBuilding()
		x := graph.Parameter(g, "x", shapes.Make(dtypes.Float32))
		y := graph.Parameter(g, "y", shapes.Make(dtypes.Float32, 2))
		z := graph.Parameter(g, "z", shapes.Make(dtypes.Float64, 3))
		reduced := g.Distributed().AllReduce([]*graph.Node{x, y, z}, backends.ReduceOpSum)
		g.Compile(reduced...)
		outputs := g.Run(
			float32(1), []float32{10, 11}, []float64{0.1, 0.2, 0.3}, // Replica 0
			float32(3), []float32{20, 21}, []float64{10, 100, 1000}) // Replica 1
		const numOutputs = 3
		require.Len(t, outputs, numOutputs*mesh.NumDevices())
		for deviceIdx := range mesh.NumDevices() {
			reducedX, reducedY, reducedZ := outputs[deviceIdx*numOutputs], outputs[deviceIdx*numOutputs+1], outputs[deviceIdx*numOutputs+2]
			fmt.Printf(
				"\t- device #%d reduced-sum outputs: x=%s, y=%s, z=%s\n",
				deviceIdx,
				reducedX,
				reducedY,
				reducedZ,
			)
			require.Equalf(t, float32(4), reducedX.Value(), "device #%d got reduced-sum x=%s", deviceIdx, reducedX)
			require.Equalf(
				t,
				[]float32{30, 32},
				reducedY.Value(),
				"device #%d got reduced-sum y=%s",
				deviceIdx,
				reducedY,
			)
			require.Equalf(
				t,
				[]float64{10.1, 100.2, 1000.3},
				reducedZ.Value(),
				"device #%d got reduced-sum z=%s",
				deviceIdx,
				reducedZ,
			)
		}
	})
}
