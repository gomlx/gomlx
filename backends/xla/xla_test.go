package xla

import (
	"flag"
	"fmt"
	"math/rand"
	"runtime"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
)

var flagPlugin = flag.String("plugin", "cpu", "Plugin to use for testing for xla backend")

func TestRepeatedClients(t *testing.T) {
	fmt.Println("Creating and destroying 10 backend clients: one at a time")
	for range 10 {
		backend, err := New(*flagPlugin)
		require.NoError(t, err)
		backend.Finalize()
	}

	fmt.Println("Creating and destroying 10 backend clients: all at once")
	testBackends := make([]backends.Backend, 0, 10)
	for range 10 {
		backend, err := New(*flagPlugin)
		require.NoError(t, err)
		testBackends = append(testBackends, backend)
	}
	for _, backend := range testBackends {
		backend.Finalize()
	}

	fmt.Println("Creating and destroying 10 backend and graphs: one at a time")
	for ii := range 100 {
		backend, err := New(*flagPlugin)
		require.NoError(t, err)
		builder := backend.Builder(fmt.Sprintf("builder_#%d", ii))
		var exec backends.Executable
		{
			x, err := builder.Parameter("x", shapes.Make(dtypes.Float64, 3), nil)
			require.NoError(t, err)
			for range rand.Intn(10) {
				x, err = builder.Add(x, x)
				require.NoError(t, err)
			}
			x2, err := builder.Mul(x, x)
			require.NoError(t, err)
			exec, err = builder.Compile([]backends.Op{x, x2}, nil)
			require.NoError(t, err)

			bIn, err := backend.BufferFromFlatData(0, []float64{7, 2, 1}, shapes.Make(dtypes.Float64, 3))
			require.NoError(t, err)
			bOuts, err := exec.Execute([]backends.Buffer{bIn}, []bool{true}, 0)
			require.NoError(t, err)
			out0, out1 := make([]float64, 3), make([]float64, 3)
			err = backend.BufferToFlatData(bOuts[0], out0)
			require.NoError(t, err)
			err = backend.BufferToFlatData(bOuts[1], out1)
			require.NoError(t, err)
			fmt.Printf("output=%v, %v\n", out0, out1)
			for range 10 {
				runtime.GC()
			}
			err = backend.BufferFinalize(bIn)
			require.NoError(t, err)
			err = backend.BufferFinalize(bOuts[0])
			require.NoError(t, err)
			err = backend.BufferFinalize(bOuts[1])
			require.NoError(t, err)
		}
		_ = exec
		backend.Finalize()
	}
	for _, backend := range testBackends {
		backend.Finalize()
	}
}
