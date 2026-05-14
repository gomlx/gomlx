// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package tensors_test

import (
	"bytes"
	"encoding/gob"
	"flag"
	"fmt"
	"runtime"
	"sync"
	"testing"

	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	_ "github.com/gomlx/gomlx/backends/default" // Use xla backend.
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/testutil"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2"
)

var flagBackend = flag.String("backend", "xla:cpu", "backend to use, this is overwritten by GOMLX_BACKEND if it is set")

func init() {
	klog.InitFlags(nil)
}

func must(err error) {
	if err != nil {
		klog.Errorf("Failed with error: %+v", err)
		panic(err)
	}
}

func must1[T any](value T, err error) T {
	must(err)
	return value
}

var (
	backend       compute.Backend
	setupTestOnce sync.Once
)

func setupTest(t *testing.T) {
	// setupTest is also called from benchmarks, make sure it only executes once though.
	setupTestOnce.Do(func() {
		compute.DefaultConfig = *flagBackend
		if t != nil {
			require.NotPanics(t, func() {
				backend = compute.MustNew()
			})
		} else {
			backend = compute.MustNew()
		}
	})
}

func testOnDeviceInputOutputImpl[T dtypes.Number](t *testing.T, backend compute.Backend) {
	dtype := dtypes.FromGenericsType[T]()
	deviceNum := compute.DeviceNum(0)

	t.Run(dtype.String(), func(t *testing.T) {
		// Create trivial f(x)=x^2 program using plain XlaBuilder
		if !backend.Capabilities().DTypes[dtype] {
			t.Skipf("Backend %s does not support dtype %s", backend.Name(), dtype)
		}

		dims := []int{3, 2}
		builder := backend.Builder(fmt.Sprintf("%s_%s", t.Name(), dtype))
		mainFn := builder.Main()
		x, err := mainFn.Parameter("x", shapes.Make(dtype, dims...), nil)
		require.NoError(t, err)
		x2, err := mainFn.Mul(x, x)
		require.NoError(t, err)
		err = mainFn.Return([]compute.Value{x2}, nil)
		require.NoError(t, err)
		exec, err := builder.Compile()
		require.NoError(t, err)

		// Create local Tensor input.
		values := []T{0, 1, 2, 3, 4, 11}
		var tensor *tensors.Tensor
		require.NotPanics(t, func() { tensor = tensors.FromFlatDataAndDimensions(values, dims...) })

		buffer, err := tensor.Buffer(backend, deviceNum)
		require.NoError(t, err)
		if backend.HasSharedBuffers() {
			// Input tensor must have become shared during conversion to "on-device".
			// Check that the shared buffer got loaded with the right values:
			require.True(t, tensor.IsShared())
			tensors.MustConstFlatData(tensor, func(flat []T) {
				require.Equal(t, []T{0, 1, 2, 3, 4, 11}, flat)
			})
		}

		var outputs []compute.Buffer
		outputs, err = exec.Execute([]compute.Buffer{buffer}, nil, 0)
		require.NoError(t, err)

		// Convert the buffer to a tensor: the converted tensor should not be shared, since the buffer comes from the output
		// of a backend execution.
		outputTensor, err := tensors.FromBuffer(outputs[0])
		require.NoError(t, err)
		require.False(t, outputTensor.IsShared())
		fmt.Printf("\tf(x) = x^2, f(%s) = %s\n", tensor.GoStr(), outputTensor.GoStr())
		require.NoErrorf(t, outputTensor.Shape().Check(dtype, 3, 2),
			"Output tensor for dtype %s got shape %s", dtype, outputTensor.Shape())
		want := []T{0, 1, 4, 9, 16, 121}
		outputTensor.MustConstFlatData(func(flatAny any) {
			flat := flatAny.([]T)
			require.Equal(t, want, flat) //  "Output tensor value was %s", outputTensor.GoStr())
		})
	})
}

func TestOnDeviceInputOutput(t *testing.T) {
	setupTest(t)

	testOnDeviceInputOutputImpl[float32](t, backend)

	testOnDeviceInputOutputImpl[int8](t, backend)
	testOnDeviceInputOutputImpl[int16](t, backend)
	testOnDeviceInputOutputImpl[int32](t, backend)
	testOnDeviceInputOutputImpl[int64](t, backend)

	testOnDeviceInputOutputImpl[uint8](t, backend)
	testOnDeviceInputOutputImpl[uint16](t, backend)
	testOnDeviceInputOutputImpl[uint32](t, backend)
	testOnDeviceInputOutputImpl[uint64](t, backend)

	testOnDeviceInputOutputImpl[float32](t, backend)
	testOnDeviceInputOutputImpl[float64](t, backend)

	testOnDeviceInputOutputImpl[complex64](t, backend)
	testOnDeviceInputOutputImpl[complex128](t, backend)
}

var testShapes = []shapes.Shape{
	shapes.Make(dtypes.Float32, 1, 1),
	shapes.Make(dtypes.Float32, 10, 10),
	shapes.Make(dtypes.Float32, 100, 100),
	shapes.Make(dtypes.Float32, 1000, 1000),
}

// BenchmarkHostToDevice benchmarks for various sizes of transfer from host to device.
//
// Results on cpu (xla):
//
//	cpu: 12th Gen Intel(R) Core(TM) i9-12900K
//	BenchmarkHostToDevice/(Float32)[1_1]-24                   827562              1486 ns/op
//	BenchmarkHostToDevice/(Float32)[10_10]-24                 761961              1519 ns/op
//	BenchmarkHostToDevice/(Float32)[100_100]-24               444972              2317 ns/op
//	BenchmarkHostToDevice/(Float32)[1000_1000]-24               9177            133306 ns/op
func BenchmarkHostToDevice(b *testing.B) {
	setupTest(nil)
	deviceNum := compute.DeviceNum(0)

	// Pre-allocate tensors.
	numShapes := len(testShapes)
	inputTensors := make([]*tensors.Tensor, numShapes)
	for shapeIdx, s := range testShapes {
		inputTensors[shapeIdx] = tensors.FromShape(s)
		must(tensors.MutableFlatData(inputTensors[shapeIdx], func(flat []float32) {
			for ii := range flat {
				flat[ii] = 0 // float32(ii)
			}
		}))
	}

	// Run test for a shape
	benchShape := func(_ float32, shapeIdx int) {
		// Set input to the value of v.
		x := inputTensors[shapeIdx]
		must(x.MaterializeOnDevice(backend, false, deviceNum))
		must(x.InvalidateOnDevice())
	}

	// Warmup for each shape.
	for shapeIdx := range testShapes {
		for i := range 10 {
			benchShape(float32(i), shapeIdx)
		}
	}

	// Reset timer and start actual benchmark
	b.ResetTimer()

	// Test each shape.
	for shapeIdx, s := range testShapes {
		b.Run(s.String(), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				benchShape(float32(i), shapeIdx)
			}
		})
	}
}

// Benchmark local copy of tensors, using various sizes.
//
// Results on cpu:
//
//	BenchmarkCopyFromLocal/(Float32)[1_1]-24                29717851                37.48 ns/op
//	BenchmarkCopyFromLocal/(Float32)[10_10]-24              29925506                39.23 ns/op
//	BenchmarkCopyFromLocal/(Float32)[100_100]-24             1992057               613.4 ns/op
//	BenchmarkCopyFromLocal/(Float32)[1000_1000]-24             10000            114128 ns/op
func BenchmarkCopyFromLocal(b *testing.B) {
	setupTest(nil)

	// Pre-allocate tensors.
	numShapes := len(testShapes)
	inputTensors := make([]*tensors.Tensor, numShapes)
	outputTensors := make([]*tensors.Tensor, numShapes)
	for shapeIdx, s := range testShapes {
		inputTensors[shapeIdx] = tensors.FromShape(s)
		must(tensors.MutableFlatData(inputTensors[shapeIdx], func(flat []float32) {
			for ii := range flat {
				flat[ii] = float32(ii)
			}
		}))
		outputTensors[shapeIdx] = tensors.FromShape(s)
	}

	// Run test for a shape
	benchShape := func(_ float32, shapeIdx int) {
		must(outputTensors[shapeIdx].CopyFrom(inputTensors[shapeIdx]))
	}

	// Warmup for each shape.
	for shapeIdx := range testShapes {
		for i := range 10 {
			benchShape(float32(i), shapeIdx)
		}
	}

	// Reset timer and start actual benchmark
	b.ResetTimer()

	// Test each shape.
	for shapeIdx, s := range testShapes {
		b.Run(s.String(), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				benchShape(float32(i), shapeIdx)
			}
		})
	}
}

// BenchmarkCopyFromDevice benchmarks the time to transfer from device to local.
//
// Results on CPU:
//
//	BenchmarkCopyFromDevice/(Float32)[1_1]-24                 465709              2498 ns/op
//	BenchmarkCopyFromDevice/(Float32)[10_10]-24               479907              2538 ns/op
//	BenchmarkCopyFromDevice/(Float32)[100_100]-24             198081              6144 ns/op
//	BenchmarkCopyFromDevice/(Float32)[1000_1000]-24             8956            133465 ns/op
func BenchmarkCopyFromDevice(b *testing.B) {
	setupTest(nil)
	deviceNum := compute.DeviceNum(0)

	// Pre-allocate tensors.
	numShapes := len(testShapes)
	inputTensors := make([]*tensors.Tensor, numShapes)
	outputTensors := make([]*tensors.Tensor, numShapes)
	for shapeIdx, s := range testShapes {
		inputTensors[shapeIdx] = tensors.FromShape(s)
		tensors.MustMutableFlatData(inputTensors[shapeIdx], func(flat []float32) {
			for ii := range flat {
				flat[ii] = float32(ii)
			}
		})
		// Don't use shared buffers for benchmark:
		must(inputTensors[shapeIdx].MaterializeOnDevice(backend, false, deviceNum))
		inputTensors[shapeIdx].FinalizeLocal()
		outputTensors[shapeIdx] = tensors.FromShape(s)
	}

	// Run test for a shape
	benchShape := func(_ float32, shapeIdx int) {
		must(outputTensors[shapeIdx].CopyFrom(inputTensors[shapeIdx]))
	}

	// Warmup for each shape.
	for shapeIdx := range testShapes {
		for i := range 10 {
			benchShape(float32(i), shapeIdx)
		}
	}

	// Reset timer and start actual benchmark
	b.ResetTimer()

	// Test each shape.
	for shapeIdx, s := range testShapes {
		b.Run(s.String(), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				benchShape(float32(i), shapeIdx)
			}
		})
	}
}

func TestClones(t *testing.T) {
	setupTest(t)
	deviceNum := compute.DeviceNum(0)
	refValues := []int32{1, 3, 5, 7, 11}
	for cloneType := range 3 {
		for fromLocation := range 2 {
			originalTensor := tensors.FromValue(refValues)
			if fromLocation == 1 {
				// originalTensor is on device.
				must(originalTensor.MaterializeOnDevice(backend, false, deviceNum))
				originalTensor.FinalizeLocal()
			}

			// Create clone.
			var cloneTensor *tensors.Tensor
			var err error
			switch cloneType {
			case 0:
				cloneTensor, err = originalTensor.Clone()
			case 1:
				cloneTensor, err = originalTensor.LocalClone()
			case 2:
				cloneTensor, err = originalTensor.OnDeviceClone(backend, deviceNum)
			}
			require.NoError(t, err)
			require.NotNil(t, cloneTensor)

			// MustFinalize original tensor and make sure it is garbage collected.
			originalTensor.MustFinalizeAll()
			for range 3 {
				runtime.GC()
			}

			// Check that the cloned tensor has the shape and values we started with.
			cloneTensor.Shape().AssertDims(5)
			require.Equal(t, refValues, tensors.MustCopyFlatData[int32](cloneTensor))
		}
	}
}

func TestToLocal(t *testing.T) {
	setupTest(t)
	deviceNum := compute.DeviceNum(0)
	refValues := []int32{1, 3, 5, 7, 11}

	for _, shared := range []bool{false, true} {
		t.Run(fmt.Sprintf("Shared=%v", shared), func(t *testing.T) {
			tensor := tensors.FromValue(refValues)
			require.NoError(t, tensor.MaterializeOnDevice(backend, shared, deviceNum))
			b2, err := tensor.Backend()
			require.NoError(t, err)
			require.Equal(t, backend, b2)

			// Move to local: there should be no on-device storage or backend associated,
			// and the contents should still be the same.
			require.NoError(t, tensor.ToLocal())
			_, err = tensor.Backend()
			require.Error(t, err)
			require.Equal(t, refValues, tensors.MustCopyFlatData[int32](tensor))
		})
	}
}

func TestOnDeviceSerialization(t *testing.T) {
	deviceNum := compute.DeviceNum(0)
	// Test reading directly to a device.
	setupTest(t)
	if backend == nil {
		panic("Backend not set!?")
	}
	{
		values := [][]int64{{2}, {3}, {5}, {7}, {11}}
		var tensor *tensors.Tensor
		require.NotPanics(t, func() { tensor = tensors.FromValue(values) })
		buf := &bytes.Buffer{}
		enc := gob.NewEncoder(buf)

		// Serialized repeats times:
		repeats := 10
		for range repeats {
			require.NoError(t, tensor.GobSerialize(enc))
		}
		fmt.Printf("\t%#v serialized %d times to %d bytes\n", values, repeats, buf.Len())

		// Deserialize repeats times:
		dec := gob.NewDecoder(buf)
		for range repeats {
			var err error
			tensor, err = tensors.GobDeserializeToDevice(dec, backend, deviceNum)
			require.NoError(t, err)
			require.Equal(t, values, tensor.Value().([][]int64))
			tensor.MustFinalizeAll()
		}
	}
}

func testFromRaw(t *testing.T, backend compute.Backend) {
	deviceNum := compute.DeviceNum(0)

	// Float32 test
	t.Run("float32", func(t *testing.T) {
		f32s := []float32{1.0, 2.0, 3.0, 4.0}
		data := unsafe.Slice((*byte)(unsafe.Pointer(&f32s[0])), len(f32s)*4)
		tensor, err := tensors.FromRaw(backend, deviceNum, shapes.Make(dtypes.Float32, 2, 2), data)
		require.NoError(t, err)
		tensor.MustConstFlatData(func(flatAny any) {
			flat := flatAny.([]float32)
			require.Equal(t, f32s, flat)
		})
	})

	// Int32 test
	t.Run("int32", func(t *testing.T) {
		i32s := []int32{10, 20, 30, 40}
		data := unsafe.Slice((*byte)(unsafe.Pointer(&i32s[0])), len(i32s)*4)
		tensor, err := tensors.FromRaw(backend, deviceNum, shapes.Make(dtypes.Int32, 2, 2), data)
		require.NoError(t, err)
		tensor.MustConstFlatData(func(flatAny any) {
			flat := flatAny.([]int32)
			require.Equal(t, i32s, flat)
		})
	})

	// Uint4 test (packed sub-byte)
	t.Run("uint4", func(t *testing.T) {
		if backend.Name() == "xla" {
			t.Skipf("Backend %s does not fully support FromRaw for packed sub-byte types yet", backend.Name())
		}
		// uint4 packed: 4 elements -> 2 bytes.
		u4s := []uint8{0x12, 0x34} // Represents 4 uint4 values
		data := u4s
		tensor, err := tensors.FromRaw(backend, deviceNum, shapes.Make(dtypes.Uint4, 4), data)
		require.NoError(t, err)
		tensor.MustConstFlatData(func(flatAny any) {
			flat := flatAny.([]uint8)
			require.Equal(t, u4s, flat)
		})
	})

	// Zero-shaped test
	t.Run("zero-shaped", func(t *testing.T) {
		data := []byte{}
		tensor, err := tensors.FromRaw(backend, deviceNum, shapes.Make(dtypes.Float32, 0), data)
		require.NoError(t, err)
		tensor.MustConstFlatData(func(flatAny any) {
			flat := flatAny.([]float32)
			require.Empty(t, flat)
		})
	})
}

func TestFromRaw(t *testing.T) {
	testutil.TestOfficialBackends(t, testFromRaw)
}
