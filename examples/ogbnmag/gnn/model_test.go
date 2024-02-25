package gnn

// Test will download OGBN-MAG dataset if not yet downloaded, into the directory `~/work/ogbnmag`.

import (
	"flag"
	"fmt"
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
		return MagModelGraph(ctx, spec, inputs, nil)
	}
	testGraphExec := context.NewExec(manager, ctx, testGraphFn)

	var inputs []tensor.Tensor
	spec, inputs, _, err = trainDS.Yield()
	require.NoError(t, err)
	outputs := testGraphExec.Call(inputs)
	for ii, output := range outputs {
		fmt.Printf("output #%d=%s\n", ii, output.Shape())
	}
	assert.NoError(t, outputs[0].Shape().Check(shapes.F32, BatchSize, mag.NumLabels))
	assert.NoError(t, outputs[1].Shape().Check(shapes.Bool, BatchSize))
	assert.NoError(t, outputs[2].Shape().Check(shapes.I32, BatchSize))
}
