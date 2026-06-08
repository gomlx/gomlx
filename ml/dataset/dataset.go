// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package datasets is a collection of utility datasets (train.Dataset) that can be combined for efficient
// preprocessing: `Take`, `InMemory`, `Parallel`, `Map`, `MapOnHost`, `Freeing`.
//
// It also includes normalization tools.
package dataset

import (
	"fmt"
	"iter"

	"github.com/gomlx/gomlx/ml/train"
)

// takeDataset implements a `train.Dataset` that only yields `take` batches.
type takeDataset struct {
	ds   train.Dataset
	take int
}

// Take returns a wrapper to `ds`, a `train.Dataset` that only yields `n` batches.
func Take(ds train.Dataset, n int) train.Dataset {
	return &takeDataset{
		ds:   ds,
		take: n,
	}
}

// Name implements train.Dataset. It returns the dataset name.
func (ds *takeDataset) Name() string {
	return fmt.Sprintf("%s [Take %d]", ds.ds.Name(), ds.take)
}

// Iter implements train.Dataset.
func (ds *takeDataset) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		count := 0
		for batch, err := range ds.ds.Iter() {
			if err != nil {
				yield(batch, err)
				return
			}
			if count >= ds.take {
				return
			}
			count++
			if !yield(batch, nil) {
				return
			}
		}
	}
}
