// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package diffusion

import (
	"fmt"
	"iter"
	"testing"
	"time"

	"github.com/gomlx/compute/dtypes"
	flowers "github.com/gomlx/gomlx/examples/oxfordflowers102"
	"github.com/gomlx/gomlx/ml/dataset"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/stretchr/testify/require"
)

func benchmarkDataset(ds train.Dataset) {
	var batchSize int
	// Warm up, run 10 ds.Iter() steps.
	next, stop := iter.Pull2(ds.Iter())
	for range 10 {
		batch, err, ok := next()
		if !ok {
			stop()
			next, stop = iter.Pull2(ds.Iter())
			batch, err, ok = next()
			if !ok {
				panic("empty dataset")
			}
		}
		check(err)
		batchSize = batch.Inputs[0].Shape().Dimensions[0]
		_ = batch.Finalize()
	}

	// Start benchmark.
	start := time.Now()
	count := 0
	for count < 100 {
		batch, err, ok := next()
		if !ok {
			break
		}
		check(err)
		_ = batch.Finalize()
		count++
	}
	stop()
	elapsed := time.Since(start)
	fmt.Printf("\t%d batches of %d examples read in %s\n", count, batchSize, elapsed)
}

func loopDataset(b *testing.B, ds train.Dataset, n int) {
	next, stop := iter.Pull2(ds.Iter())
	defer stop()
	for range n {
		batch, err, ok := next()
		if !ok {
			stop()
			next, stop = iter.Pull2(ds.Iter())
			batch, err, ok = next()
			if !ok {
				b.Fatalf("Dataset is empty even after resetting")
			}
		}
		if err != nil {
			b.Fatalf("Dataset failed with %+v", err)
		}
		_ = batch.Finalize()
	}
}

func BenchmarkDatasets(b *testing.B) {
	config := getTestConfig()
	ds := flowers.NewDataset(dtypes.Float32, config.ImageSize)
	dsBatched := dataset.Batch(config.Backend, ds, config.BatchSize, true, true)
	require.NoError(b, flowers.DownloadAndParse(config.DataDir))

	dsParallel := dataset.Buffer(dsBatched)

	// Warmup.
	loopDataset(b, dsParallel, 100) // Warms up both dsParallel and the underlying dsBatched.

	b.Run("Disk", func(b *testing.B) { loopDataset(b, dsBatched, b.N) })

	b.Run("ParallelDisk", func(b *testing.B) { loopDataset(b, dsParallel, b.N) })

	trainDS, _ := config.CreateInMemoryDatasets()
	trainDS.BatchSize(config.BatchSize, true)
	trainDS.Infinite(true)
	loopDataset(b, trainDS, 100) // Warm up.
	b.Run("InMemory", func(b *testing.B) { loopDataset(b, trainDS, b.N) })
}

//
//	fmt.Printf("\nInMemory dataset:\n")
//	trainDS, _ := diffusion.CreateInMemoryDatasets()
//	trainDS.BatchSize(diffusion.BatchSize, true)
//	trainDS.Infinite(true)
//	BenchmarkDataset(trainDS)
//}
