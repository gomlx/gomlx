package tensors

import (
	"flag"
	"fmt"
	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/xla" // Use xla backend.
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2"
	"sync"
	"testing"
)

var flagBackend = flag.String("backend", "xla:cpu", "backend to use, this is overwritten by GOMLX_BACKEND if it is set")

// backend is set up by setupTest.
var backend backends.Backend

func init() {
	klog.InitFlags(nil)
}

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
	x := builder.Parameter("x", shapes.Make(dtype, dims...))
	x2 := builder.Mul(x, x)
	exec := builder.Compile(x2)

	// Create local Tensor input.
	values := []T{0, 1, 2, 3, 4, 11}
	var tensor *Tensor
	require.NotPanics(t, func() { tensor = FromFlatDataAndDimensions(values, dims...) })

	var buffer backends.Buffer
	require.NotPanics(t, func() {
		buffer = tensor.Buffer(backend)
	})
	var outputs []backends.Buffer
	require.NotPanics(t, func() {
		outputs = exec.Execute([]backends.Buffer{buffer}, nil)
	})

	// Convert buffer to tensor.
	outputTensor := FromBuffer(backend, outputs[0])
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
//	BenchmarkHostToDevice/(Float32)[1_1]-24                   720470              1629 ns/op
//	BenchmarkHostToDevice/(Float32)[10_10]-24                 720577              1770 ns/op
//	BenchmarkHostToDevice/(Float32)[100_100]-24               464420              2532 ns/op
//	BenchmarkHostToDevice/(Float32)[1000_1000]-24               8936            133507 ns/op
func BenchmarkHostToDevice(b *testing.B) {
	setupTest(nil)

	// Pre-allocate tensors.
	numShapes := len(testShapes)
	inputTensors := make([]*Tensor, numShapes)
	for shapeIdx, s := range testShapes {
		inputTensors[shapeIdx] = FromShape(s)
		MutableFlatData[float32](inputTensors[shapeIdx], func(flat []float32) {
			for ii := range flat {
				flat[ii] = 0 // float32(ii)
			}
		})
	}

	// Run test for a shape
	benchShape := func(v float32, shapeIdx int) {
		// Set input to value of v.
		x := inputTensors[shapeIdx]
		x.MaterializeOnDevices(backend)
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

// Benchmark of local copy of tensors of various sizes.
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
		MutableFlatData[float32](inputTensors[shapeIdx], func(flat []float32) {
			for ii := range flat {
				flat[ii] = float32(ii)
			}
		})
		outputTensors[shapeIdx] = FromShape(s)
	}

	// Run test for a shape
	benchShape := func(v float32, shapeIdx int) {
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
//	BenchmarkCopyFromDevice/(Float32)[1_1]-24                 391078              2948 ns/op
//	BenchmarkCopyFromDevice/(Float32)[10_10]-24               413571              2882 ns/op
//	BenchmarkCopyFromDevice/(Float32)[100_100]-24             217750              5967 ns/op
//	BenchmarkCopyFromDevice/(Float32)[1000_1000]-24             8544            132102 ns/op
func BenchmarkCopyFromDevice(b *testing.B) {
	setupTest(nil)

	// Pre-allocate tensors.
	numShapes := len(testShapes)
	inputTensors := make([]*Tensor, numShapes)
	outputTensors := make([]*Tensor, numShapes)
	for shapeIdx, s := range testShapes {
		inputTensors[shapeIdx] = FromShape(s)
		MutableFlatData[float32](inputTensors[shapeIdx], func(flat []float32) {
			for ii := range flat {
				flat[ii] = float32(ii)
			}
		})
		inputTensors[shapeIdx].MaterializeOnDevices(backend)
		inputTensors[shapeIdx].FinalizeLocal()
		outputTensors[shapeIdx] = FromShape(s)
	}

	// Run test for a shape
	benchShape := func(v float32, shapeIdx int) {
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
