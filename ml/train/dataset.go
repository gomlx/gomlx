// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package train

import (
	"iter"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/distributed"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/core/tensors/dtensor"
	"github.com/pkg/errors"
)

// Dataset for a train.Trainer provides the data, one batch at a time. Flat consists of a slice of *tensors.Tensor
// for `inputs` and for `labels`.
//
// Dataset has to also provide a Dataset.Name() and a dataset `spec`, which usually is the same for
// the whole dataset, but can vary per batch, if the Dataset is yielding different types of data.
//
// For a static Dataset that always provides the exact same data type, the `spec` can simply be nil.
//
// Notice one batch (the unit of data) is represented by the Batch struct.
//
// The Dataset interface allows for extensions/customizations by defining extra optional interfaces that
// a Dataset optionally can implement.
// See DistributedDataset.
type Dataset interface {
	// Name identifies the dataset. Used for debugging, pretty-printing and plots.
	Name() string

	// Iter returns an iterator over the dataset.
	// It can be called multiple times (for example, to start a new epoch).
	// It implicitly resets the dataset.
	Iter() iter.Seq2[Batch, error]
}

// Batch represents one "item" yielded by a Dataset.
type Batch struct {
	// Inputs and Labels of a batch, either are optional. The inputs are passed to the model function during training
	// and evaluation. The labels are passed to the loss function(s) and metrics. The order must be agreed between the
	// dataset and the model function that takes the inputs, and the loss/metric function for the loss(es) and
	// metric(s).
	//
	// The Inputs and Labels ownership is transferred to the caller (usually a training or evaluation function). It's
	// expected that they will be finalized immediately after use. This usually happens automatically if you use it
	// with `train.Trainer` (it takes over the ownership).
	//
	// But if you are using the Batch yourself, remember to finalize the tensors (or call Batch.Finalize, it will
	// do that for you).
	Inputs, Labels []*tensors.Tensor

	// Spec defines the task, configuration, or structural requirements
	// for this specific batch. If this changes, GoMLX will generate
	// and JIT-compile a new computation graph.
	//
	// This is used for multi-task training.
	// If you are not using that, you can ignore it.
	//
	// For train.Trainer the Batch.Spec is converted to string as a key of a map[string] for different computation
	// graphs, so each time the `spec` changes, the model graph is regenerated and re-compiled. Just like with
	// Inputs or Labels of different shapes.
	//
	// Spec is assumed to be immutable, and ownership is not transferred.
	Spec any
}

// Finalize finalizes (frees) all tensors in the batch.
func (b Batch) Finalize() error {
	for _, t := range b.Inputs {
		if err := t.FinalizeAll(); err != nil {
			return err
		}
	}
	for _, t := range b.Labels {
		if err := t.FinalizeAll(); err != nil {
			return err
		}
	}
	return nil
}

// Clone returns a deep copy of the Batch.
// If any tensor clone fails, it returns the error.
func (b Batch) Clone() (Batch, error) {
	cloned := Batch{
		Spec: b.Spec,
	}
	if b.Inputs != nil {
		cloned.Inputs = make([]*tensors.Tensor, len(b.Inputs))
		for i, t := range b.Inputs {
			var err error
			cloned.Inputs[i], err = t.Clone()
			if err != nil {
				return Batch{}, errors.WithMessagef(err, "failed to clone Batch input tensor #%d", i)
			}
		}
	}
	if b.Labels != nil {
		cloned.Labels = make([]*tensors.Tensor, len(b.Labels))
		for i, t := range b.Labels {
			var err error
			cloned.Labels[i], err = t.Clone()
			if err != nil {
				return Batch{}, errors.WithMessagef(err, "failed to clone Batch label tensor #%d", i)
			}
		}
	}
	return cloned, nil
}

// OnDeviceClone creates a clone of the Batch with its tensors placed on the specified backend and device.
// If any tensor clone fails, it returns the error.
func (b Batch) OnDeviceClone(backend compute.Backend, deviceNum compute.DeviceNum) (Batch, error) {
	cloned := Batch{
		Spec: b.Spec,
	}
	if b.Inputs != nil {
		cloned.Inputs = make([]*tensors.Tensor, len(b.Inputs))
		for i, t := range b.Inputs {
			var err error
			cloned.Inputs[i], err = t.OnDeviceClone(backend, deviceNum)
			if err != nil {
				return Batch{}, errors.WithMessagef(err, "failed to clone Batch input tensor #%d to device", i)
			}
		}
	}
	if b.Labels != nil {
		cloned.Labels = make([]*tensors.Tensor, len(b.Labels))
		for i, t := range b.Labels {
			var err error
			cloned.Labels[i], err = t.OnDeviceClone(backend, deviceNum)
			if err != nil {
				return Batch{}, errors.WithMessagef(err, "failed to clone Batch label tensor #%d to device", i)
			}
		}
	}
	return cloned, nil
}

// DistributedBatch represents one sharded batch of tensors yielded by a DistributedDataset.
type DistributedBatch struct {
	// Inputs and Labels of a batch, either are optional. They are sharded dtensor.Tensor instances.
	// The inputs are passed to the model function during training and evaluation. The labels are
	// passed to the loss function(s) and metrics.
	//
	// The Inputs and Labels ownership is transferred to the caller (usually a training or evaluation function).
	// It's expected that they will be finalized immediately after use. This usually happens automatically
	// if you use it with `train.Trainer` (it takes over the ownership).
	//
	// But if you are using the DistributedBatch yourself, remember to finalize the tensors (or call
	// DistributedBatch.Finalize, it will do that for you).
	Inputs, Labels []*dtensor.Tensor

	// Spec defines the task, configuration, or structural requirements
	// for this specific batch. If this changes, GoMLX will generate
	// and JIT-compile a new computation graph.
	Spec any
}

// Finalize finalizes (frees) all tensors in the batch.
func (b DistributedBatch) Finalize() error {
	for _, t := range b.Inputs {
		if err := t.FinalizeAll(); err != nil {
			return err
		}
	}
	for _, t := range b.Labels {
		if err := t.FinalizeAll(); err != nil {
			return err
		}
	}
	return nil
}

// Clone returns a deep copy of the DistributedBatch.
// If any tensor clone fails, it returns the error.
func (b DistributedBatch) Clone() (DistributedBatch, error) {
	cloned := DistributedBatch{
		Spec: b.Spec,
	}
	if b.Inputs != nil {
		cloned.Inputs = make([]*dtensor.Tensor, len(b.Inputs))
		for i, t := range b.Inputs {
			var err error
			cloned.Inputs[i], err = t.Clone()
			if err != nil {
				return DistributedBatch{}, errors.WithMessagef(err, "failed to clone DistributedBatch input tensor #%d", i)
			}
		}
	}
	if b.Labels != nil {
		cloned.Labels = make([]*dtensor.Tensor, len(b.Labels))
		for i, t := range b.Labels {
			var err error
			cloned.Labels[i], err = t.Clone()
			if err != nil {
				return DistributedBatch{}, errors.WithMessagef(err, "failed to clone DistributedBatch label tensor #%d", i)
			}
		}
	}
	return cloned, nil
}

// DistributedDataset is different API to a Dataset but yields dtensor.Tensors, ready for distributed
// execution.
//
// An important aspect of the dataset is the distributed.ShardingSpec of each input and label.
// They are fed into the trainer to guide the distributed execution, and must remain the same for the same spec.
//
// Note: if you need to control the device assignment, that is done in the Trainer.
//
// A DistributedDataset implementation will usually also implement a Dataset interface, so it can be used with
// the Loop trainer.
type DistributedDataset interface {
	// Dataset must also be implemented.
	Dataset

	// Strategy returns the distributed.Strategy to use for this dataset.
	// Usually, distributed.AutoSharding. But distributed.SPMD (experimental) can also be used.
	Strategy() distributed.Strategy

	// DeviceAssignment returns the device assignment for the distributed dataset.
	// The DistributedIter() method will return dtensor.Tensor already on the corresponding device.
	DeviceAssignment() []compute.DeviceNum

	// DistributedIter returns an iterator over the distributed dataset.
	// It can be called multiple times (for example, to start a new epoch).
	// It implicitly resets the dataset.
	DistributedIter() iter.Seq2[DistributedBatch, error]
}

// HasShortName allows a dataset to specify a short name (used when displaying a short version of metric names).
// It defaults to the first 3 letters of the dataset name.
//
// It's optional.
type HasShortName interface {
	// ShortName returns the short name of the dataset.
	ShortName() string
}
