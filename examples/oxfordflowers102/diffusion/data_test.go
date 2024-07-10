package diffusion

import (
	"fmt"
	flowers "github.com/gomlx/gomlx/examples/oxfordflowers102"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/stretchr/testify/require"
	"io"
	"testing"
	"time"
)

func benchmarkDataset(ds train.Dataset) {
	Init()

	// Warm up, run 100 ds.Yield().
	for ii := 0; ii < 10; ii++ {
		_, inputs, labels, err := ds.Yield()
		AssertNoError(err)
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
		AssertNoError(err)
		finalize(inputs)
		finalize(labels)
		count++
	}
	elapsed := time.Since(start)
	fmt.Printf("\t%d batches of %d examples read in %s\n", count, *flagBatchSize, elapsed)
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
	Init()
	ds := flowers.NewDataset(dtypes.Float32, ImageSize)
	dsBatched := data.Batch(manager, ds, BatchSize, true, true)
	require.NoError(b, flowers.DownloadAndParse(DataDir))

	dsParallel := data.Parallel(dsBatched)

	// Warmup.
	loopDataset(b, dsParallel, 100) // Warms up both dsParallel and the underlying dsBatched.

	dsBatched.Reset()
	b.Run("Disk", func(b *testing.B) { loopDataset(b, dsBatched, b.N) })

	dsParallel.Reset()
	b.Run("ParallelDisk", func(b *testing.B) { loopDataset(b, dsParallel, b.N) })

	trainDS, _ := CreateInMemoryDatasets()
	trainDS.BatchSize(BatchSize, true)
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
