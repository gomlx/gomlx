/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package context

import (
	"fmt"
	ml "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/gomlx/gomlx/xla"
	"github.com/stretchr/testify/require"
	"testing"
)

// How to get a memory profile using the "gperftools":
//
// $ go test -c && env HEAPCHECK=strict ./computation.test -test.run TestMemoryLeak
//
// After this open the profile file created with google-pprof:
//
// $ google-pprof ./computation.test "/tmp/computation.test.10193._main_-end.heap" --inuse_objects --lines --heapcheck  --edgefraction=1e-10 --nodefraction=1e-10 --web
//
// Notice the filename of the heap is printed on the output of the test -- at least in my version
// it is not observing the environment variable "HEAPPROFILE", and is creating its own file path.
//
// TODO: xla::Client::Compile() leaks, the returned ExecutionHandle can never be freed. We need
//	to change to use xla::LocalClient::Compile(), but it is more complex since one needs to handle
//	partition.

// LeakThreshold for MemoryLeakTest: there are always small leaks, but gproftools doesn't seem to
// catch these (maybe growth on underlying vector capacities that are never returned), so we don't care
// about it either.
const LeakThreshold int64 = 300000

// TestMemoryLeaksCtxExec creates and executes a simple computation graph with Context at different shape sizes.
// It then destroys the returning tensor.Device objects (xla.OnDeviceBuffer), and checks that memory doesn't get
// out of control.
func TestMemoryLeaksCtxExec(t *testing.T) {
	manager, err := ml.BuildManager().Done()
	if err != nil {
		t.Fatalf("Failed to create Manager: %+v", err)
	}
	initialValue := slices.SliceWithValue(100, 0.0)

	graphFn := func(ctx *Context, x *Node) []*Node {
		g := x.Graph()
		_ = g
		sliceVar := ctx.In("memory_leak_test").Checked(false).VariableWithValue("slice",
			initialValue)
		_ = sliceVar
		reducedX := ml.ReduceAllSum(x)
		added := ml.Add(sliceVar.ValueGraph(g), reducedX)
		total := ml.ReduceAllSum(added)
		var results []*Node
		results = append(results, total)
		ctx.EnumerateVariables(func(v *Variable) {
			results = append(results, ml.Add(v.ValueGraph(g), ml.Const(g, float64(1))))
		})
		return results
	}

	fn := func() {
		{
			exec := NewExec(manager, nil, graphFn)
			exec.SetMaxCache(100)

			// Tests for various parameters.
			count := 0
			for xV := 0.0; xV < 100.0; xV += 1 {
				for size := 1; size <= 100; size++ {
					inputT := tensor.FromValue(slices.SliceWithValue(size, xV))
					results, err := exec.Call(inputT)
					require.NoErrorf(t, err, "Failed to execute computation: xV=%g, size=%d", xV, size)
					totalT := results[0]
					varUpdates := results[1:]
					if totalT.Error() != nil {
						t.Fatalf("Failed to execute computation: %+v", totalT.Error())
					}
					totalL := totalT.Local()
					if totalL.Error() != nil {
						t.Fatalf("Failed total: %+v", totalL.Error())
					}
					total := totalT.Local().Value().(float64)
					want := 100.0 * (float64(count) + float64(size)*xV)
					//want := float64(float32(100.0) * (float32(10) + float32(size)*xV))
					//want := 2 * (1000.0 + float64(size)*float64(xV))
					//want := float64(float32(size) * xV)
					if total != want {
						totalD := totalT.Device(manager, 0)
						literal, err := xla.FromOnDeviceBuffer(totalD.ShapedBuffer())
						if err != nil {
							t.Fatalf("Failed to convert to local tensor: %+v", err)
						}
						data := literal.Data().([]float64)
						fmt.Printf("\n*****\nre-read value=%g\n*******\n\n", data[0])
						fmt.Printf("Variables:\n")
						exec.Context().EnumerateVariables(func(v *Variable) {
							fmt.Printf("\t%s: %v\n", v.name, v.Value().Local().Value().([]float32))
						})
						t.Errorf("Unexpected total %s, wanted %f for xV=%f, size=%d, count=%d", totalT.Local().GoStr(), want,
							xV, size, count)
						return
					}
					count++
					totalT.FinalizeAll()
					inputT.FinalizeAll()

					_ = varUpdates
					update := varUpdates[0].Local().Value().([]float64)
					for ii := range update {
						if update[ii] != float64(count) {
							t.Fatalf("Failed for count %d: got update=%v", count, update)
						}
					}
					ii := 0
					exec.Context().EnumerateVariables(func(v *Variable) {
						v.SetValue(varUpdates[ii])
						ii++
					})
				}
			}
		}
		initializers.Finalize()
		xla.GarbageCollectXLAObjects(true)
		fmt.Printf("TestMemoryLeaksCtxExec: L=%d S=%d\n", xla.LiteralsCount(), xla.OnDeviceBufferCount())
	}

	fn() // Warm-up call: static allocations happen here.

	msg := "Executing computation graph with Context for various tensor shapes"
	used := xla.MemoryUsedByFn(fn, msg)
	fmt.Printf("\tMemory increase=%s\n", xla.HumanBytes(used))
	fmt.Printf("\tLiteralsCount=%d\n", xla.LiteralsCount())
	fmt.Printf("\tShapedBufferCount=%d\n", xla.OnDeviceBufferCount())
	fmt.Printf("\tOpsCount=%d\n", xla.OpsCount)
	xla.NoGlobalLeaks() // Saves memory profile, if enabled with HEAPPROFILE
	//fmt.Printf("MemoryStats:\n%s\n", xla.MemoryStats())

	if used > LeakThreshold || xla.LiteralsCount() != 0 {
		t.Errorf("Still %d xla.Literal objects alive", xla.LiteralsCount())
		t.Errorf("Still %d xla.OnDeviceBuffer objects alive", xla.OnDeviceBufferCount())
		t.Fatalf("Repeated %s increased memory usage by %d", msg, used)
	}
}
