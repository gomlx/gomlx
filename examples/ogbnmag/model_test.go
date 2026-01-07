package ogbnmag

// Test will download OGBN-MAG dataset if not yet downloaded, into the directory `~/work/ogbnmag`.

import (
	"flag"
	"fmt"
	"testing"

	"github.com/dustin/go-humanize"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/pbnjay/memory"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

var (
	flagDataDir = flag.String("data", "~/work/ogbnmag", "Directory to cache downloaded and generated dataset files.")
)

func checkMemory(t *testing.T) {
	fmt.Printf("Total memory: %s\n", humanize.Bytes(memory.TotalMemory()))
	if memory.TotalMemory() < 32*1024*1024*1024 {
		t.Skipf("Test requires at least 32GB RAM, found %s", humanize.Bytes(memory.TotalMemory()))
	}
}

func TestModel(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping long-running test.")
	}
	checkMemory(t)

	backend := graphtest.BuildTestBackend()
	ctx := context.New()
	err := Download(*flagDataDir)
	require.NoError(t, err, "failed to download OGBN-MAG dataset")
	UploadOgbnMagVariables(backend, ctx) // Uploads the Papers frozen embedding table.

	trainDS, _, _, _, err := MakeDatasets(*flagDataDir)
	require.NoError(t, err, "failed to make datasets")

	// Create graph function that will take a sampled sub-graph.
	var spec any
	testGraphFn := func(ctx *context.Context, inputs []*Node) []*Node {
		return MagModelGraph(ctx, spec, inputs)
	}
	testGraphExec := context.MustNewExec(backend, ctx, testGraphFn)

	var inputs []*tensors.Tensor
	spec, inputs, _, err = trainDS.Yield()
	totalSizeBytes := uint64(0)
	for _, input := range inputs {
		totalSizeBytes += uint64(input.Shape().Memory())
	}
	fmt.Printf("One sample (batch) size is %s bytes\n", humanize.Bytes(totalSizeBytes))
	require.NoError(t, err)
	outputs := testGraphExec.MustExec(inputs)
	for ii, output := range outputs {
		fmt.Printf("output #%d=%s\n", ii, output.Shape())
	}
	assert.NoError(t, outputs[0].Shape().Check(dtypes.Float32, BatchSize, NumLabels))
	assert.NoError(t, outputs[1].Shape().Check(dtypes.Bool, BatchSize))
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
	err := Download(*flagDataDir)
	require.NoError(b, err)
	ds, _, _, _, err := MakeDatasets(*flagDataDir)
	require.NoError(b, err)

	for b.Loop() {
		_, inputs, _, err := ds.Yield()
		if err != nil {
			b.Fatalf("Failed to sample: %+v", err)
		}
		_ = inputs
	}
}
