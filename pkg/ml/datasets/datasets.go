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

// Package datasets is a collection of utility datasets (train.Dataset) that can be combined for efficient
// preprocessing: `Take`, `InMemory`, `Parallel`, `MapWithGraphFn`, `Freeing`.
//
// It also includes normalization tools.
package datasets

import (
	"fmt"
	"io"

	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/train"
)

// takeDataset implements a `train.Dataset` that only yields `take` batches.
type takeDataset struct {
	ds          train.Dataset
	count, take int
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

// Reset implements train.Dataset.
func (ds *takeDataset) Reset() {
	ds.ds.Reset()
	ds.count = 0
}

// Yield implements train.Dataset.
func (ds *takeDataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	if ds.count >= ds.take {
		err = io.EOF
		return
	}
	ds.count++
	spec, inputs, labels, err = ds.ds.Yield()
	return
}
