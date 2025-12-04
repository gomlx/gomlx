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

// TestPortable tests that a computation is properly compiled as "portable" if
// using no distribution strategy and not assigned to any device.
//
// This can be used to manually distribute computation by executing the program concurrently on different devices.
//
// Portable compilation may not work on all PJRT types or different backends.
// Please add a blacklist of backends to skip the test if the backend doesn't support it.
func TestPortable(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	if backend.NumDevices() <= 1 {
		t.Skipf("Skipping distributed test because there are only 1 device available for backend %q.", backend.Name())
	}

	t.Run("Graph", func(t *testing.T) {
		g := graph.NewGraph(backend, "portable")
		x := graph.Parameter(g, "x", shapes.Make(dtypes.Float32))
		negX := graph.Neg(x)
		g.Compile(negX)

		numDevices := backend.NumDevices()
		for deviceNum := range numDevices {
			outputs := g.RunOnDevice(backends.DeviceNum(deviceNum), float32(deviceNum))
			require.Len(t, outputs, 1)
			output := outputs[0]
			outputDevice := must1(output.Device())
			fmt.Printf("- device #%d: output in device #%d\n", deviceNum, outputDevice)
			assert.Equalf(t, float32(-deviceNum), outputs[0].Value(), "got %s", outputs[0])
		}
	})

	t.Run("Exec", func(t *testing.T) {
		e := graph.MustNewExec(backend, func(x *graph.Node) *graph.Node {
			return graph.Neg(x)
		})
		numDevices := backend.NumDevices()
		for deviceNum := range numDevices {
			outputs, err := e.ExecOnDevice(backends.DeviceNum(deviceNum), float32(deviceNum))
			require.NoError(t, err)
			require.Len(t, outputs, 1)
			output := outputs[0]
			outputDevice := must1(output.Device())
			fmt.Printf("- device #%d: output in device #%d\n", deviceNum, outputDevice)
			assert.Equalf(t, float32(-deviceNum), outputs[0].Value(), "got %s", outputs[0])
		}
	})
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

// TestAutoSharding distributed strategy.
func TestAutoSharding(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	if backend.NumDevices() <= 1 {
		t.Skipf("Skipping distributed test because there are only 1 device available for backend %q.", backend.Name())
	}

	// Test AutoSharding using the Graph API (no Exec).
	t.Run("Graph", func(t *testing.T) {
		mesh := must1(distributed.NewDeviceMesh([]int{2}, []string{"sharded"}))
		g := graph.NewGraph(backend, t.Name())
		require.NoError(t, g.SetAutoSharding(mesh))
		x := graph.ShardedParameter(g, "x", shapes.Make(dtypes.Float32, 4),
			must1(distributed.BuildSpec(mesh).S("sharded").Done()))
		y := graph.ReduceAllSum(x)
		g.Compile(y)
		outputs := g.Run([]float32{1, 2}, []float32{0.1, 0.2})
		require.Len(t, outputs, mesh.NumDevices())
		for i, output := range outputs {
			fmt.Printf("\t- device #%d: %s\n", i, output)
			require.Equalf(t, float32(3.3), output.Value(), "result for device #%d got %s", i, output)
		}
	})

	// Test AutoSharding using the Graph API (no Exec).
	t.Run("Graph with sharded output", func(t *testing.T) {
		mesh := must1(distributed.NewDeviceMesh([]int{2}, []string{"sharded"}))
		g := graph.NewGraph(backend, t.Name())
		require.NoError(t, g.SetAutoSharding(mesh))
		x := graph.ShardedParameter(g, "x", shapes.Make(dtypes.Float32, 2, 2),
			must1(distributed.BuildSpec(mesh).S("sharded").R().Done()))
		y := graph.ReduceSum(x, 0) // Force an AllReduce
		g.CompileWithSharding([]*graph.Node{y},
			[]*distributed.ShardingSpec{
				must1(distributed.BuildSpec(mesh).S("sharded").Done()),
			})
		outputs := g.Run([][]float32{{1, 2}}, [][]float32{{0.1, 0.2}})
		require.Len(t, outputs, mesh.NumDevices())
		want := []any{[]float32{1.1}, []float32{2.2}} // Sharded output, reduce-summed on axis 0.
		for shardIdx, output := range outputs {
			fmt.Printf("\t- device #%d: %s\n", shardIdx, output)
			require.Equalf(t, want[shardIdx], output.Value(), "result for device #%d got %s", shardIdx, output)
		}
	})

	// Simple sharding of the input and output.
	mesh := must1(distributed.NewDeviceMesh([]int{2}, []string{"sharded"}))
	spec := must1(distributed.BuildSpec(mesh).S("sharded").Done())
	t.Run("Exec", func(t *testing.T) {
		exec := graph.MustNewExec(backend,
			func(x, w *graph.Node) *graph.Node {
				// x is scalar, w is sharded [4].
				return graph.Add(x, w)
			}).
			AutoSharding(mesh).
			WithInputShardingSpecs(nil, spec). // x is replicated (nil), w is sharded (spec).
			WithOutputShardingSpecs(spec)      // output y is sharded (spec).
		outputs, err := exec.Exec(
			float32(10), []float32{0, 1}, // Device 0
			float32(10), []float32{2, 3}, // Device 1
		)
		require.NoError(t, err)
		require.Len(t, outputs, 2) // One output per device.

		// Check outputs.
		// Device 0 should have [10, 11]
		// Device 1 should have [12, 13]
		fmt.Printf("\t- device #0: %s\n", outputs[0])
		fmt.Printf("\t- device #1: %s\n", outputs[1])
		require.Equal(t, []float32{10, 11}, outputs[0].Value())
		require.Equal(t, []float32{12, 13}, outputs[1].Value())
	})

	// Same test, but including logged values.
	t.Run("With Logged Nodes", func(t *testing.T) {
		exec := graph.MustNewExec(backend,
			func(x, w *graph.Node) *graph.Node {
				// x is scalar, w is sharded [4].
				y := graph.Add(x, w)
				graph.ReduceAllSum(y).SetLogged("ReduceSum(y)")
				return y
			}).
			AutoSharding(mesh).
			WithInputShardingSpecs(nil, spec).
			WithOutputShardingSpecs(spec)
		outputs, err := exec.Exec(
			float32(10), []float32{0, 1}, // Device 0
			float32(10), []float32{2, 3}, // Device 1
		)
		require.NoError(t, err)
		require.Len(t, outputs, 2)
		fmt.Printf("\t- device #0: %s\n", outputs[0])
		fmt.Printf("\t- device #1: %s\n", outputs[1])
		require.Equal(t, []float32{10, 11}, outputs[0].Value())
		require.Equal(t, []float32{12, 13}, outputs[1].Value())
	})

	// Test correct handling of sharding for a variable number of inputs and outputs.
	t.Run("variable-length inputs and outputs", func(t *testing.T) {
		exec := graph.MustNewExec(backend,
			func(inputs []*graph.Node) []*graph.Node {
				// x is a scalar; w0 and w1 are sharded [4].
				x, w0, w1 := inputs[0], inputs[1], inputs[2]
				y0 := graph.Add(x, w0)
				y1 := graph.Add(x, w1)
				graph.ReduceAllSum(y0).SetLogged("ReduceSum(y)")
				graph.ReduceAllSum(y1).SetLogged("ReduceSum(y)")
				return []*graph.Node{y0, y1}
			}).
			AutoSharding(mesh).
			WithInputShardingSpecs(nil, spec). // the last spec is repeated for tail of inputs.
			WithOutputShardingSpecs(spec)      // the last spec is repeated for tail of outputs.
		outputs, err := exec.Exec(
			float32(100), []float32{0, 1}, []float32{10, 11}, // Device 0
			float32(100), []float32{2, 3}, []float32{12, 13}, // Device 1
		)
		require.NoError(t, err)
		require.Len(t, outputs, 4)
		fmt.Printf("\t- device #0: y0^0=%s, y1^0%s\n", outputs[0], outputs[1])
		fmt.Printf("\t- device #1: y0^1=%s, y1^1%s\n", outputs[2], outputs[3])
		// y0 = {10, 11, 12, 13}
		require.Equal(t, []float32{100, 101}, outputs[0].Value())
		require.Equal(t, []float32{102, 103}, outputs[2].Value())
		// y1 = {110, 111, 112, 113}
		require.Equal(t, []float32{110, 111}, outputs[1].Value())
		require.Equal(t, []float32{112, 113}, outputs[3].Value())
	})
}
