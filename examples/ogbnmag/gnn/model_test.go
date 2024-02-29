package gnn

// Test will download OGBN-MAG dataset if not yet downloaded, into the directory `~/work/ogbnmag`.

import (
	"flag"
	"fmt"
	"github.com/dustin/go-humanize"
	mag "github.com/gomlx/gomlx/examples/ogbnmag"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

var (
	flagDataDir = flag.String("data", "~/work/ogbnmag", "Directory to cache downloaded and generated dataset files.")
)

func TestModel(t *testing.T) {
	manager := graphtest.BuildTestManager()
	ctx := context.NewContext(manager)
	err := mag.Download(*flagDataDir)
	require.NoError(t, err, "failed to download OGBN-MAG dataset")
	mag.UploadOgbnMagVariables(ctx) // Uploads the Papers frozen embedding table.

	trainDS, _, _, _, err := MakeDatasets(*flagDataDir)
	require.NoError(t, err, "failed to make datasets")

	// Create graph function that will take a sampled sub-graph.
	var spec any
	testGraphFn := func(ctx *context.Context, inputs []*Node) []*Node {
		return MagModelGraph(ctx, spec, inputs)
	}
	testGraphExec := context.NewExec(manager, ctx, testGraphFn)

	var inputs []tensor.Tensor
	spec, inputs, _, err = trainDS.Yield()
	totalSizeBytes := uint64(0)
	for _, input := range inputs {
		totalSizeBytes += uint64(input.Shape().Memory())
	}
	fmt.Printf("One sample (batch) size is %s bytes\n", humanize.Bytes(totalSizeBytes))
	require.NoError(t, err)
	outputs := testGraphExec.Call(inputs)
	for ii, output := range outputs {
		fmt.Printf("output #%d=%s\n", ii, output.Shape())
	}
	assert.NoError(t, outputs[0].Shape().Check(shapes.F32, BatchSize, mag.NumLabels))
	assert.NoError(t, outputs[1].Shape().Check(shapes.Bool, BatchSize))
}

// BenchmarkParallelSampling measures the average time create one sampled subgraph (with `BatchSize` seeds).
//
// With BatchSize=32, I'm getting:
//
//	goos: linux
//	goarch: amd64
//	pkg: github.com/gomlx/gomlx/examples/ogbnmag/gnn
//	cpu: 12th Gen Intel(R) Core(TM) i9-12900K
//	BenchmarkParallelSampling-24              125136             92917 ns/op
func BenchmarkParallelSampling(b *testing.B) {
	err := mag.Download(*flagDataDir)
	require.NoError(b, err)
	ds, _, _, _, err := MakeDatasets(*flagDataDir)
	require.NoError(b, err)

	b.ResetTimer()
	for range b.N {
		_, inputs, _, err := ds.Yield()
		if err != nil {
			b.Fatalf("Failed to sample: %+v", err)
		}
		_ = inputs
	}
}
