package simplego

import (
	"fmt"
	"os"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2"
)

var backend backends.Backend

func init() {
	klog.InitFlags(nil)
}

func setup() {
	fmt.Printf("Available backends: %q\n", backends.List())
	// Perform your setup logic here
	if os.Getenv(backends.ConfigEnvVar) == "" {
		must.M(os.Setenv(backends.ConfigEnvVar, "go"))
	} else {
		fmt.Printf("\t$%s=%q\n", backends.ConfigEnvVar, os.Getenv(backends.ConfigEnvVar))
	}
	backend = backends.MustNew()
	fmt.Printf("Backend: %s, %s\n", backend.Name(), backend.Description())
}

func teardown() {
	backend.Finalize()
}

func TestMain(m *testing.M) {
	setup()
	code := m.Run() // Run all tests in the file
	teardown()
	os.Exit(code)
}

func TestDuplicatedOutputNodes(t *testing.T) {
	// Create a builder and a node
	builder := backend.Builder("test_duplicated_outputs")
	mainFn := builder.Main()
	node, err := mainFn.Constant([]float32{1.0, 2.0, 3.0}, 3)
	require.NoError(t, err)
	require.NotNil(t, node)

	// Compile with the same node duplicated as outputs
	// This should create Identity nodes for the duplicate
	err = mainFn.Return([]backends.Value{node, node}, nil)
	require.NoError(t, err)
	exec, err := builder.Compile()
	require.NoError(t, err)
	require.NotNil(t, exec)

	// Execute with no inputs (since we're using a constant)
	outputs, err := exec.Execute(nil, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 2)

	// Verify that the two output buffers are different (not the same pointer)
	output0 := outputs[0].(*Buffer)
	output1 := outputs[1].(*Buffer)
	require.NotSame(t, output0, output1, "duplicated output nodes should yield different buffers")

	// Verify that the underlying flat data slices are also different
	// (they may have the same values but should be different slices)
	flat0 := output0.flat.([]float32)
	flat1 := output1.flat.([]float32)
	require.NotSame(t, &flat0[0], &flat1[0], "duplicated output nodes should have different underlying data slices")

	// Verify that the values are correct (both should be [1.0, 2.0, 3.0])
	require.Equal(t, []float32{1.0, 2.0, 3.0}, flat0)
	require.Equal(t, []float32{1.0, 2.0, 3.0}, flat1)

	// Verify shapes are correct
	shape0, err := backend.BufferShape(outputs[0])
	require.NoError(t, err)
	require.True(t, shape0.Equal(shapes.Make(dtypes.Float32, 3)))

	shape1, err := backend.BufferShape(outputs[1])
	require.NoError(t, err)
	require.True(t, shape1.Equal(shapes.Make(dtypes.Float32, 3)))
}
