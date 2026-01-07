/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

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
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/ml/datasets"
	"github.com/pkg/errors"
	"log"
)

// NewDataset creates a new `datasets.InMemoryDataset` (can be used for training and evaluation) for the
// MCI Adult dataset.
func NewDataset(backend backends.Backend, rawData *RawData, name string) *datasets.InMemoryDataset {
	tensorData := rawData.CreateTensors(backend)
	ds, err := datasets.InMemoryFromData(backend, name,
		[]any{tensorData.CategoricalTensor, tensorData.ContinuousTensor, tensorData.WeightsTensor},
		[]any{tensorData.LabelsTensor})
	if err != nil {
		panic(errors.WithMessagef(err, "failed to create UCI Adult dataset"))
	}
	return ds
}

// PrintBatchSamples just generate a couple of batches of size 3 and print on the output.
// Just for debugging.
func PrintBatchSamples(backend backends.Backend, data *RawData) {
	sampler := NewDataset(backend, data, "batched sample dataset")
	sampler.BatchSize(3, true)
	for ii := range 2 {
		fmt.Printf("\nSample batch %d:\n", ii)
		_, inputs, labels, err := sampler.Yield()
		if err != nil {
			log.Fatalf("Failed to sample batches: %+v", err)
		}
		fmt.Printf("\tcategorical=%v\n", inputs[0])
		fmt.Printf("\tcontinuos=%v\n", inputs[1])
		fmt.Printf("\tweights=%v\n", inputs[2])
		fmt.Printf("\tlabels=%v\n", labels)
	}
}
