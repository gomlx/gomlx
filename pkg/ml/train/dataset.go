/*
 *	Copyright 2025 Jan Pfeifer
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

package train

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/distributed"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// Dataset for a train.Trainer provides the data, one batch at a time. Flat consists of a slice of *tensors.Tensor
// for `inputs` and for `labels`.
//
// Dataset has to also provide a Dataset.Name() and a dataset `spec`, which usually is the same for
// the whole dataset, but can vary per batch, if the Dataset is yielding different types of data.
//
// For a static Dataset that always provides the exact same data type, the `spec` can simply be nil.
//
// Notice one batch (the unit of data) is a slice of tensors for inputs and one tensor for labels.
//
// The Dataset interface allows for extensions/customizations by defining extra optional interfaces that
// a Dataset optionally can implement.
// See DistributedDataset and DatasetCustomOwnership.
type Dataset interface {
	// Name identifies the dataset. Used for debugging, pretty-printing and plots.
	Name() string

	// Reset restarts the dataset from the beginning. Can be called after io.EOF is reached,
	// for instance when running another evaluation on a test dataset.
	Reset()

	// Yield one "batch" (or whatever is the unit for a training step) or an error.
	// It should return a `spec` for the dataset, a slice of `inputs` and a slice of `labels` tensors
	// (even when there is only one tensor for each of them).
	//
	// In the simplest case `inputs` and `labels` should always have the same number of elements and the
	// same shape (including `dtype`).
	//
	// The `inputs` and `labels` ownership is transferred to the caller (usually a training or evaluation
	// function). It's expected that they will be finalized (Tensor.FinalizeAll) immediately after use.
	// If you don't want that behavior, consider implementing FinalizeYieldsAfterUse() bool, and return false.
	//
	// If the `inputs` or `labels` change shapes during training or evaluation, it will trigger the creation
	// of a new computation graph and new JIT (just-in-time) compilation. There is a finite-sized cache,
	// and this can become inefficient -- it may spend more time JIT compiling than executing code. Consider
	// instead using padding for things that have variable length. And if there are various such elements,
	// consider padding to powers of 2 (or some other base) to limit the number of shapes that will be used.
	//
	// Yield also returns an opaque `spec` object that is normally simply passed to the model function
	// -- it can simply be nil. The `spec` usually is static (always the same) for a dataset. E.g.: the field names
	// and types of a generic CSV file reader.
	//
	// **Important**:
	// 1. For train.Trainer the `spec` is converted to string as a key of a `map[string]` for different computation
	//    graphs, so each time the `spec` changes, the model graph is regenerated and re-compiled. Just like with
	//   `inputs` or `labels` of different shapes.
	// 2. The number of `inputs` and `labels` should not change for the same `spec` -- the train.Trainer will return
	//    an error if they do. Their shape can vary (at the cost of creating a new JIT-compiled graph for each
	//    different combination of shapes). If the number of `inputs` or `labels` needs changing, a new `spec`
	//    needs to be given.
	//
	// If using Loop.RunSteps for training having an infinite dataset stream is ok. But careful
	// not to use Loop.RunEpochs on a dataset configured to loop indefinitely.
	//
	// Optionally, it can return an error. If the error is `io.EOF` the training/evaluation terminates
	// normally, as it indicates end of data for finite datasets -- maybe the end of the epoch.
	//
	// Any other errors should interrupt the training/evaluation and be returned to the user.
	Yield() (spec any, inputs, labels []*tensors.Tensor, err error)
}

// DistributedDataset is different API to a Dataset but yields distributed.Tensors, ready for distributed
// execution.
//
// An important aspect of the dataset is the distributed.ShardingSpec of each input and label.
// They are fed into the trainer to guide the distributed execution, and must remain the same for the same spec.
//
// Note: if you need to control the device assignment, that is done in the Trainer.
//
// A DistributedDataset implementation will usually also implement a Dataset interface, so it can be used with
// the Loop trainer, even if its Yield() method always returns an error.
type DistributedDataset interface {
	// Dataset must also be implemented.
	Dataset

	// Strategy returns the distributed.Strategy to use for this dataset.
	// Usually, distributed.AutoSharding. But distributed.SPMD (experimental) can also be used.
	Strategy() distributed.Strategy

	// DeviceAssignment returns the device assignment for the distributed dataset.
	// The Yield() method will return distributed.Tensor already on the corresponding device.
	DeviceAssignment() []backends.DeviceNum

	// DistributedYield one "batch" (or whatever is the unit for a training step) or an error.
	// Very similar to Dataset.Yield, all the notes there apply here.
	//
	// The returned data (inputs and labels) are given as distributed.Tensor instances, hence already sharded.
	// The ShardingSpec of each input and label must be the same for the same spec.
	// If your training is heterogeneous, it's ok to have different ShardingSpecs for different specs.
	DistributedYield() (spec any, inputs, labels []*distributed.Tensor, err error)
}

// HasShortName allows a dataset to specify a short name (used when displaying a short version of metric names).
// It defaults to the first 3 letters of the dataset name.
//
// It's optional.
type HasShortName interface {
	// ShortName returns the short name of the dataset.
	ShortName() string
}

// DatasetCustomOwnership allows a dataset to specify whether the ownership of the yielded tensors are transferred
// to the caller (the training loop). The training loops can finalize the yielded values after use.
// It defaults to yes.
//
// It applies to all types of datasets.
type DatasetCustomOwnership interface {
	// IsOwnershipTransferred specifies whether caller owns the yielded transfers -- and can finalize (free) them after use.
	// It defaults to true.
	IsOwnershipTransferred() bool
}
