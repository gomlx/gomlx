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

package graph

import (
	"fmt"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/gomlx/gomlx/xla"
	"testing"
	"time"
)

// How to get a memory profile using the "gperftools":
//
// $ go test -c && env HEAPCHECK=strict ./computation.test -test.run TestMemoryLeak
//
// After this open the profile file created with google-pprof:
//
// $ google-pprof ./computation.test "/tmp/computation.test.10193._main_-end.heap"  --inuse_objects --lines --heapcheck  --edgefraction=1e-10 --nodefraction=1e-10 --web
//
// Notice the filename of the heap is printed on the output of the test -- at least in my version
// it is not observing the environment variable "HEAPPROFILE", and is creating its own file path.
//
// TODO: xla::Client::Compile() leaks, the returned ExecutionHandle can never be freed. We need
//
//	to change to use xla::LocalClient::Compile(), but it is more complex since one needs to xlaHandle
//	partition.

// LeakThreshold for MemoryLeakTest: there are always small leaks, but gproftools doesn't seem to
// catch these (maybe growth on underlying vector capacities that are never returned), so we don't care
// about it either.
const LeakThreshold int64 = 300000

// TestMemoryLeaksLiteral creates and destroys tensor.Local objects (xla.Literal), and checks that memory
// doesn't get out of control.
func TestMemoryLeaksLiteral(t *testing.T) {
	fn := func() {
		// Tests for various parameters.
		for xV := float64(0); xV < 100; xV += 1 {
			for yV := float64(0); yV < 100; yV += 1 {
				l := tensor.MakeLocalTupleAny(slices.SliceWithValue(100, xV), slices.SliceWithValue(10, yV))
				l.Finalize()
			}
		}
		fmt.Printf("TestMemoryLeaksLiteral: L=%d S=%d\n", xla.LiteralsCount(), xla.OnDeviceBufferCount())
		xla.GarbageCollectXLAObjects(true)
		time.Sleep(time.Second)
		xla.GarbageCollectXLAObjects(true)
	}

	fn()
	msg := "Creating xla.Literal tuples (tensor.Local)"
	used := xla.MemoryUsedByFn(fn, msg)
	fmt.Printf("\tMemory increase=%s\n", xla.HumanBytes(used))
	fmt.Printf("\tLiteralsCount=%d\n", xla.LiteralsCount())
	fmt.Printf("\tShapedBufferCount=%d\n", xla.OnDeviceBufferCount())
	fmt.Printf("\tOpsCount=%d\n", xla.OpsCount)
	xla.NoGlobalLeaks() // Saves memory profile, if enabled with HEAPPROFILE
	//fmt.Printf("MemoryStats:\n%s\n", xla.MemoryStats())

	if used > LeakThreshold || xla.LiteralsCount() != 0 {
		t.Errorf("Still %d xla.Literal objects alive", xla.LiteralsCount())
		t.Fatalf("Repeated %s increased memory usage by %d", msg, used)
	}
}

// TestMemoryLeaksShapedBuffer creates and destroys tensor.Device objects (xla.OnDeviceBuffer), and checks that memory
// doesn't get out of control.
func TestMemoryLeaksShapedBuffer(t *testing.T) {
	manager, err := BuildManager().Done()
	if err != nil {
		t.Fatalf("Failed to create Manager: %+v", err)
	}

	fn := func() {
		// Tests for various parameters.
		for xV := float64(0); xV < 100; xV += 1 {
			for yV := float64(0); yV < 100; yV += 1 {
				l := tensor.MakeLocalTupleAny(slices.SliceWithValue(100, xV), slices.SliceWithValue(10, yV))
				sb, err := l.Literal().ToOnDeviceBuffer(manager.Client(), manager.DefaultDeviceNum())
				if err != nil {
					t.Fatalf("Failed to convert xla.Literal to xla.OnDeviceBuffer: %+v", err)
				}
				l.Finalize()
				d := tensor.FromShapedBuffer(sb)
				d.Finalize()
			}
		}
		fmt.Printf("TestMemoryLeaksShapedBuffer: L=%d S=%d\n", xla.LiteralsCount(), xla.OnDeviceBufferCount())
		xla.GarbageCollectXLAObjects(true)
		time.Sleep(time.Second)
		xla.GarbageCollectXLAObjects(true)
	}

	fn()
	msg := "Creating xla.OnDeviceBuffer tuples (tensor.Device)"
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

// TestMemoryLeaksExec creates and executes a simple computation graph at different shape sizes. It then destroys
// the returning tensor.Device objects (xla.OnDeviceBuffer), and checks that memory doesn't get out of control.
func TestMemoryLeaksExec(t *testing.T) {
	manager, err := BuildManager().Done()
	if err != nil {
		t.Fatalf("Failed to create Manager: %+v", err)
	}

	graphFn := func(x, y *Node) (*Node, *Node) {
		added := Add(x, y)
		return added, ReduceAllSum(added)
	}

	fn := func() {
		{
			exec := NewExec(manager, graphFn)
			exec.SetMaxCache(100)

			// Tests for various parameters.
			for xV := float64(0); xV < 100; xV += 1 {
				for size := 1; size <= 100; size++ {
					results := exec.Call(slices.SliceWithValue(size, xV), float64(size))
					addedT, reducedT := results[0], results[1]
					if addedT.Error() != nil {
						t.Fatalf("Failed to execute computation: %+v", addedT.Error())
					}
					if addedT.Rank() != 1 && addedT.Shape().Dimensions[0] != size {
						t.Errorf("Unexpected shape %s for size %d, value %f", addedT.Shape(), size, xV)
					}
					reduced := tensor.ToScalar[float64](reducedT.Local())
					want := float64(size) * (xV + float64(size))
					if reduced != want {
						t.Errorf("Unexpected reduced sum %s, wanted %f", reducedT.Local().GoStr(), want)
					}
					addedT.FinalizeAll()
					reducedT.FinalizeAll()
				}
			}
		}
		xla.GarbageCollectXLAObjects(true)
		fmt.Printf("TestMemoryLeaksExec: L=%d S=%d\n", xla.LiteralsCount(), xla.OnDeviceBufferCount())
	}

	fn()
	msg := "Executing computation graph for various tensor shapes"
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
