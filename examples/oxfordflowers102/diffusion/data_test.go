package diffusion

import (
	"fmt"
	flowers "github.com/gomlx/gomlx/examples/oxfordflowers102"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/pkg/ml/datasets"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
	"io"
	"testing"
	"time"
)

func benchmarkDataset(ds train.Dataset) {
	var batchSize int
	// Warm up, run 100 ds.Yield().
	for ii := 0; ii < 10; ii++ {
		_, inputs, labels := must.M3(ds.Yield())
		batchSize = inputs[0].Shape().Dimensions[0]
		finalize(inputs)
		finalize(labels)
	}

	// Start benchmark.
	start := time.Now()
	count := 0
	for count < 100 {
		_, inputs, labels, err := ds.Yield()
		if err == io.EOF {
			break
		}
		must.M(err)
		finalize(inputs)
		finalize(labels)
		count++
	}
	elapsed := time.Since(start)
	fmt.Printf("\t%d batches of %d examples read in %s\n", count, batchSize, elapsed)
}

func loopDataset(b *testing.B, ds train.Dataset, n int) {
	for ii := 0; ii < n; ii++ {
		_, inputs, labels, err := ds.Yield()
		if err == io.EOF {
			ds.Reset()
		}
		if err != nil {
			b.Fatalf("Dataset failed with %+v", err)
		}
		finalize(inputs)
		finalize(labels)
	}
}

func BenchmarkDatasets(b *testing.B) {
	config := getTestConfig()
	ds := flowers.NewDataset(dtypes.Float32, config.ImageSize)
	dsBatched := datasets.Batch(config.Backend, ds, config.BatchSize, true, true)
	require.NoError(b, flowers.DownloadAndParse(config.DataDir))

	dsParallel := datasets.Parallel(dsBatched)

	// Warmup.
	loopDataset(b, dsParallel, 100) // Warms up both dsParallel and the underlying dsBatched.

	dsBatched.Reset()
	b.Run("Disk", func(b *testing.B) { loopDataset(b, dsBatched, b.N) })

	dsParallel.Reset()
	b.Run("ParallelDisk", func(b *testing.B) { loopDataset(b, dsParallel, b.N) })

	trainDS, _ := config.CreateInMemoryDatasets()
	trainDS.BatchSize(config.BatchSize, true)
	trainDS.Infinite(true)
	loopDataset(b, trainDS, 100) // Warm up.
	trainDS.Reset()
	b.Run("InMemory", func(b *testing.B) { loopDataset(b, trainDS, b.N) })
}

//
//	fmt.Printf("\nInMemory dataset:\n")
//	trainDS, _ := diffusion.CreateInMemoryDatasets()
//	trainDS.BatchSize(diffusion.BatchSize, true)
//	trainDS.Infinite(true)
//	BenchmarkDataset(trainDS)
//}
