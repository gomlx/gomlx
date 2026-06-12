// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package adult provides a `InMemoryDataset` implementation for
// UCI Adult Census dataset. See attached notebook (using GoNB) for details
// and examples on how to use it.
//
// Mostly one will want to use `LoadAndPreprocessData` to download and preprocess
// the data, the singleton `Flat` to access it, and `NewDataset` to create datasets
// for training and evaluating.
//
// It also provides preprocessing functionality:
//
// - Downloading and caching of the dataset.
// - List of column names organized by types (`AdultFieldNames` and `AdultFieldTypes`)
// - Vocabularies for categorical features.
// - Quantiles for continuous features.
// - Pretty print some stats.
package adult

import (
	"fmt"
	"iter"
	"log"

	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/ml/dataset"
	"github.com/pkg/errors"
)

// NewDataset creates a new `datasets.InMemoryDataset` (can be used for training and evaluation) for the
// MCI Adult dataset.
func NewDataset(backend compute.Backend, rawData *RawData, name string) *dataset.InMemoryDataset {
	tensorData := rawData.CreateTensors(backend)
	ds, err := dataset.InMemoryFromData(backend, name,
		[]any{tensorData.CategoricalTensor, tensorData.ContinuousTensor, tensorData.WeightsTensor},
		[]any{tensorData.LabelsTensor})
	if err != nil {
		panic(errors.WithMessagef(err, "failed to create UCI Adult dataset"))
	}
	return ds
}

// PrintBatchSamples just generate a couple of batches of size 3 and print on the output.
// Just for debugging.
func PrintBatchSamples(backend compute.Backend, data *RawData) {
	sampler := NewDataset(backend, data, "batched sample dataset")
	sampler.BatchSize(3, true)
	next, stop := iter.Pull2(sampler.Iter())
	defer stop()
	for ii := range 2 {
		fmt.Printf("\nSample batch %d:\n", ii)
		batch, err, ok := next()
		if !ok {
			log.Fatalf("Failed to sample batches: dataset finished early")
		}
		if err != nil {
			log.Fatalf("Failed to sample batches: %+v", err)
		}
		fmt.Printf("\tcategorical=%v\n", batch.Inputs[0])
		fmt.Printf("\tcontinuos=%v\n", batch.Inputs[1])
		fmt.Printf("\tweights=%v\n", batch.Inputs[2])
		fmt.Printf("\tlabels=%v\n", batch.Labels)
		_ = batch.Finalize()
	}
}
