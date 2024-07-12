package tensors

import (
	"flag"
	"fmt"
	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/xla" // Use xla backend.
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/pjrt"
	xla "github.com/gomlx/gopjrt/xlabuilder"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2"
	"testing"
)

var flagPluginName = flag.String("plugin", "cpu", "plugin name")

func init() {
	klog.InitFlags(nil)

	// Default test uses XLA/PJRT with "cpu" plugin.
	backends.DefaultConfig = "xla:cpu"
}

type errTester[T any] struct {
	value T
	err   error
}

// capture is a shortcut to test that there is no error and return the value.
func capture[T any](value T, err error) errTester[T] {
	return errTester[T]{value, err}
}

func (e errTester[T]) Test(t *testing.T) T {
	require.NoError(t, e.err)
	return e.value
}

// getPJRTClient loads a PJRT plugin and create a client to run tests on.
// It exits the test if anything goes wrong.
func getPJRTClient(t *testing.T) *pjrt.Client {
	// PJRT plugin and create a client.
	plugin, err := pjrt.GetPlugin(*flagPluginName)
	require.NoError(t, err, "Failed to get plugin %q", *flagPluginName)
	fmt.Printf("Loaded PJRT plugin %s\n", plugin)
	client, err := plugin.NewClient(nil)
	require.NoErrorf(t, err, "Failed to create a client on %s", plugin)
	return client
}

// compile compiles the program and returns the executable that can be used for testing.
// It exits the test if anything goes wrong.
func compile(t *testing.T, client *pjrt.Client, comp pjrt.XlaComputation) (exec *pjrt.LoadedExecutable) {
	var err error
	exec, err = client.Compile().WithComputation(comp).Done()
	require.NoErrorf(t, err, "Failed to compile program")
	return
}

func testOnDeviceInputOutputImpl[T dtypes.Number](t *testing.T, client *pjrt.Client) {
	// Create trivial f(x)=x^2 program using plain XlaBuilder
	dtype := dtypes.FromGenericsType[T]()
	dims := []int{3, 2}
	builder := xla.New(fmt.Sprintf("%s_%s", t.Name(), dtype))
	x := capture(xla.Parameter(builder, "x", 0, xla.MakeShape(dtype, dims...))).Test(t)
	x2 := capture(xla.Mul(x, x)).Test(t)
	comp := capture(builder.Build(x2)).Test(t)
	exec := compile(t, client, comp)

	// Create local Tensor input.
	values := []T{0, 1, 2, 3, 4, 11}
	var tensor *Tensor
	require.NotPanics(t, func() { tensor = FromFlatDataAndDimensions(values, dims...) })

	// Execute PJRT program.
	var buffer *pjrt.Buffer
	require.NotPanics(t, func() {
		buffer = tensor.Buffer(client)
	})
	outputs := capture(exec.Execute(buffer).Done()).Test(t)

	// Convert buffer to tensor.
	outputTensor := FromBuffer(outputs[0])
	fmt.Printf("\tf(x) = x^2, f(%s) = %s\n", tensor.GoStr(), outputTensor.GoStr())
	require.NoErrorf(t, outputTensor.Shape().Check(dtype, 3, 2), "Output tensor for dtype %s got shape %s", dtype, outputTensor.Shape())
	want := []T{0, 1, 4, 9, 16, 121}
	outputTensor.ConstFlatData(func(flatAny any) {
		flat := flatAny.([]T)
		require.Equal(t, want, flat) //  "Output tensor value was %s", outputTensor.GoStr())
	})
}

func TestOnDeviceInputOutput(t *testing.T) {
	client := getPJRTClient(t)
	testOnDeviceInputOutputImpl[int8](t, client)
	testOnDeviceInputOutputImpl[int16](t, client)
	testOnDeviceInputOutputImpl[int32](t, client)
	testOnDeviceInputOutputImpl[int64](t, client)

	testOnDeviceInputOutputImpl[uint8](t, client)
	testOnDeviceInputOutputImpl[uint16](t, client)
	testOnDeviceInputOutputImpl[uint32](t, client)
	testOnDeviceInputOutputImpl[uint64](t, client)

	testOnDeviceInputOutputImpl[float32](t, client)
	testOnDeviceInputOutputImpl[float64](t, client)

	testOnDeviceInputOutputImpl[complex64](t, client)
	testOnDeviceInputOutputImpl[complex128](t, client)
}
