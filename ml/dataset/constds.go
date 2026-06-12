// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dataset

import (
	"iter"

	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/train"
)

// Const returns a dataset that always yields a clone of the given batch, indefinitely.
//
// Since batch ownership is transferred on yield, a clone of the batch is created
// on each iteration so that the original batch remains unchanged.
func Const(batch train.Batch) train.Dataset {
	return &constDataset{
		batch: batch,
	}
}

// Zero returns a dataset that always yields a batch with 0 scalar as input and labels.
//
// This is useful when training something that generates its own inputs and labels -- like trying to
// approximate a function with another function.
//
// It loops indefinitely.
func Zero() train.Dataset {
	batch := train.Batch{
		Inputs: []*tensors.Tensor{tensors.FromScalar(int32(0))},
		Labels: []*tensors.Tensor{tensors.FromScalar(int32(0))},
	}
	return Const(batch)
}

// constDataset is a dataset that always yields the same batch.
type constDataset struct {
	batch train.Batch
}

var _ train.Dataset = &constDataset{}

func (ds *constDataset) Name() string {
	return "dataset.Const"
}

func (ds *constDataset) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		for {
			cloned, err := ds.batch.Clone()
			if err != nil {
				yield(train.Batch{}, err)
				return
			}
			if !yield(cloned, nil) {
				return
			}
		}
	}
}
