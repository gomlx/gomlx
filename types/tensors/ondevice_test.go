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
	"testing"
)

var flagBackend = flag.String("backend", "xla:cpu", "backend to use, this is overwritten by GOMLX_BACKEND if it is set")

// backend is set up by setupTest.
var backend backends.Backend

func init() {
	klog.InitFlags(nil)
}

type errTester[T any] struct {
	value T
	err   error
}

func setupTest(t *testing.T) {
	backends.DefaultConfig = *flagBackend
	require.NotPanics(t, func() {
		backend = backends.New()
	})
}

// capture is a shortcut to test that there is no error and return the value.
func capture[T any](value T, err error) errTester[T] {
	return errTester[T]{value, err}
}

func (e errTester[T]) Test(t *testing.T) T {
	require.NoError(t, e.err)
	return e.value
}

func testOnDeviceInputOutputImpl[T dtypes.Number](t *testing.T, backend backends.Backend) {
	// Create trivial f(x)=x^2 program using plain XlaBuilder
	dtype := dtypes.FromGenericsType[T]()
	dims := []int{3, 2}
	builder := backend.Builder(fmt.Sprintf("%s_%s", t.Name(), dtype))
	x := builder.Parameter("x", shapes.Make(dtype, dims...))
	exec := builder.Compile(x)

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
		outputs = exec.Execute(buffer)
	})

	// Convert buffer to tensor.
	outputTensor := FromBuffer(backend, outputs[0])
	fmt.Printf("\tf(x) = x^2, f(%s) = %s\n", tensor.GoStr(), outputTensor.GoStr())
	require.NoErrorf(t, outputTensor.Shape().Check(dtype, 3, 2), "Output tensor for dtype %s got shape %s", dtype, outputTensor.Shape())
	want := []T{0, 1, 2, 3, 4, 11}
	//want := []T{0, 1, 4, 9, 16, 121}
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
