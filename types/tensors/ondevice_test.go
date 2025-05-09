package tensors

import (
	"flag"
	"fmt"
	"runtime"
	"sync"
	"testing"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/xla" // Use xla backend.
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2"
)

var flagBackend = flag.String("backend", "xla:cpu", "backend to use, this is overwritten by GOMLX_BACKEND if it is set")

func init() {
	klog.InitFlags(nil)
}

var (
	backend backends.Backend
)

func setupTest(t *testing.T) {
	// setupTest is also called from benchmarks, make sure it only executes once though.
	sync.OnceFunc(func() {
		backends.DefaultConfig = *flagBackend
		if t != nil {
			require.NotPanics(t, func() {
				backend = backends.New()
			})
		} else {
			backend = backends.New()
		}
	})()
}

func testOnDeviceInputOutputImpl[T dtypes.Number](t *testing.T, backend backends.Backend) {
	// Create trivial f(x)=x^2 program using plain XlaBuilder
	dtype := dtypes.FromGenericsType[T]()
	dims := []int{3, 2}
	builder := backend.Builder(fmt.Sprintf("%s_%s", t.Name(), dtype))
	x, err := builder.Parameter("x", shapes.Make(dtype, dims...))
	require.NoError(t, err)
	x2, err := builder.Mul(x, x)
	require.NoError(t, err)
	exec, err := builder.Compile(x2)
	require.NoError(t, err)

	// Create local Tensor input.
	values := []T{0, 1, 2, 3, 4, 11}
	var tensor *Tensor
	require.NotPanics(t, func() { tensor = FromFlatDataAndDimensions(values, dims...) })

	var buffer backends.Buffer
	require.NotPanics(t, func() {
		buffer = tensor.Buffer(backend)
	})
	if backend.HasSharedBuffers() {
		// Input tensor must have become shared during conversion to "on-device".
		// Check that the shared buffer got loaded with the right values:
		require.True(t, tensor.IsShared())
		ConstFlatData(tensor, func(flat []T) {
			require.Equal(t, []T{0, 1, 2, 3, 4, 11}, flat)
		})
	}

	var outputs []backends.Buffer
	outputs, err = exec.Execute([]backends.Buffer{buffer}, nil)
	require.NoError(t, err)

	// Convert the buffer to a tensor: the converted tensor should not be shared, since the buffer comes from the output
	// of a backend execution.
	outputTensor := FromBuffer(backend, outputs[0])
	require.False(t, outputTensor.isShared)
	fmt.Printf("\tf(x) = x^2, f(%s) = %s\n", tensor.GoStr(), outputTensor.GoStr())
	require.NoErrorf(t, outputTensor.Shape().Check(dtype, 3, 2), "Output tensor for dtype %s got shape %s", dtype, outputTensor.Shape())
	want := []T{0, 1, 4, 9, 16, 121}
	outputTensor.ConstFlatData(func(flatAny any) {
		flat := flatAny.([]T)
		require.Equal(t, want, flat) //  "Output tensor value was %s", outputTensor.GoStr())
	})
}

func TestOnDeviceInputOutput(t *testing.T) {
	setupTest(t)
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
// Results on cpu:
//
//	cpu: 12th Gen Intel(R) Core(TM) i9-12900K
//	BenchmarkHostToDevice/(Float32)[1_1]-24                   827562              1486 ns/op
//	BenchmarkHostToDevice/(Float32)[10_10]-24                 761961              1519 ns/op
//	BenchmarkHostToDevice/(Float32)[100_100]-24               444972              2317 ns/op
//	BenchmarkHostToDevice/(Float32)[1000_1000]-24               9177            133306 ns/op
func BenchmarkHostToDevice(b *testing.B) {
	setupTest(nil)

	// Pre-allocate tensors.
	numShapes := len(testShapes)
	inputTensors := make([]*Tensor, numShapes)
	for shapeIdx, s := range testShapes {
		inputTensors[shapeIdx] = FromShape(s)
		MutableFlatData(inputTensors[shapeIdx], func(flat []float32) {
			for ii := range flat {
				flat[ii] = 0 // float32(ii)
			}
		})
	}

	// Run test for a shape
	benchShape := func(_ float32, shapeIdx int) {
		// Set input to value of v.
		x := inputTensors[shapeIdx]
		x.MaterializeOnDevices(backend, false)
		x.InvalidateOnDevice()
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
	inputTensors := make([]*Tensor, numShapes)
	outputTensors := make([]*Tensor, numShapes)
	for shapeIdx, s := range testShapes {
		inputTensors[shapeIdx] = FromShape(s)
		MutableFlatData(inputTensors[shapeIdx], func(flat []float32) {
			for ii := range flat {
				flat[ii] = float32(ii)
			}
		})
		outputTensors[shapeIdx] = FromShape(s)
	}

	// Run test for a shape
	benchShape := func(_ float32, shapeIdx int) {
		outputTensors[shapeIdx].CopyFrom(inputTensors[shapeIdx])
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

	// Pre-allocate tensors.
	numShapes := len(testShapes)
	inputTensors := make([]*Tensor, numShapes)
	outputTensors := make([]*Tensor, numShapes)
	for shapeIdx, s := range testShapes {
		inputTensors[shapeIdx] = FromShape(s)
		MutableFlatData(inputTensors[shapeIdx], func(flat []float32) {
			for ii := range flat {
				flat[ii] = float32(ii)
			}
		})
		inputTensors[shapeIdx].MaterializeOnDevices(backend, false) // Don't use shared buffers for benchmark
		inputTensors[shapeIdx].FinalizeLocal()
		outputTensors[shapeIdx] = FromShape(s)
	}

	// Run test for a shape
	benchShape := func(_ float32, shapeIdx int) {
		outputTensors[shapeIdx].CopyFrom(inputTensors[shapeIdx])
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
	refValues := []int32{1, 3, 5, 7, 11}
	for cloneType := range 3 {
		for fromLocation := range 2 {
			originalTensor := FromValue(refValues)
			if fromLocation == 1 {
				// originalTensor is on device.
				originalTensor.MaterializeOnDevices(backend, false)
				originalTensor.FinalizeLocal()
			}

			// Create clone.
			var cloneTensor *Tensor
			switch cloneType {
			case 0:
				cloneTensor = originalTensor.Clone()
			case 1:
				cloneTensor = originalTensor.LocalClone()
			case 2:
				cloneTensor = originalTensor.OnDeviceClone(backend)
			}

			// Finalize original tensor, and make sure it is garbage collected.
			originalTensor.FinalizeAll()
			for range 3 {
				runtime.GC()
			}

			// Check that the cloned tensor has the shape and values we started with.
			cloneTensor.Shape().AssertDims(5)
			require.Equal(t, refValues, CopyFlatData[int32](cloneTensor))
		}
	}
}
