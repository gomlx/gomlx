package xla

import (
	"flag"
	"fmt"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"math/rand"
	"runtime"
	"testing"
)

var flagPlugin = flag.String("plugin", "cpu", "Plugin to use for testing for xla backend")

func TestRepeatedClients(t *testing.T) {
	fmt.Println("Creating and destroying 10 backend clients: one at a time")
	for range 10 {
		backend := New(*flagPlugin)
		backend.Finalize()
	}

	fmt.Println("Creating and destroying 10 backend clients: all at once")
	testBackends := make([]backends.Backend, 0, 10)
	for range 10 {
		testBackends = append(testBackends, New(*flagPlugin))
	}
	for _, backend := range testBackends {
		backend.Finalize()
	}

	fmt.Println("Creating and destroying 10 backend and graphs: one at a time")
	for ii := range 100 {
		backend := New(*flagPlugin)
		builder := backend.Builder(fmt.Sprintf("builder_#%d", ii))
		var exec backends.Executable
		{
			x := builder.Parameter("x", shapes.Make(dtypes.Float64, 3))
			for range rand.Intn(10) {
				x = builder.Add(x, x)
			}
			x2 := builder.Mul(x, x)
			exec = builder.Compile(x, x2)

			bIn := backend.BufferFromFlatData(0, []float64{7, 2, 1}, shapes.Make(dtypes.Float64, 3))
			bOuts := exec.Execute([]backends.Buffer{bIn}, []bool{true})
			out0, out1 := make([]float64, 3), make([]float64, 3)
			backend.BufferToFlatData(bOuts[0], out0)
			backend.BufferToFlatData(bOuts[1], out1)
			fmt.Printf("output=%v, %v\n", out0, out1)
			for range 10 {
				runtime.GC()
			}
			backend.BufferFinalize(bIn)
			backend.BufferFinalize(bOuts[0])
			backend.BufferFinalize(bOuts[1])
		}
		_ = exec
		backend.Finalize()
	}
	for _, backend := range testBackends {
		backend.Finalize()
	}
}
