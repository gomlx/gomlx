package datasets

import (
	"io"

	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/pkg/errors"
)

// freeingDataset implements a `train.Dataset` that frees inputs and labels GPU memory after each use.
type freeingDataset struct {
	name                   string
	ds                     train.Dataset
	prevInputs, prevLabels []*tensors.Tensor
}

// Freeing implements a sequential dataset (it should not to be parallelized) that immediately releases the yielded
// inputs and labels in between each `Yield` call, not waiting for garbage collection.
//
// This is needed for datasets with large inputs, to prevent more than one input to be alive in the accelerator's (GPU)
// memory at the same time, in case garbage collection hasn't run yet. Notice Go's garbage collection has no notion
// of associated resource usage in GPU and is not able to respond to that by itself.
//
// It works by keeping a reference to previously yielded values and freeing them before yielding the next one.
//
// While you can wrap a parallelized (with [Parallel]) dataset with [Freeing], the other way around will break:
// [Freeing] will free the yielded tensor before they are actually used.
func Freeing(ds train.Dataset) *freeingDataset {
	return &freeingDataset{
		ds:   ds,
		name: ds.Name(),
	}
}

// freePreviousYield finalize the tensors returned by the previous [Yield] call.
func (ds *freeingDataset) freePreviousYield() {
	for _, t := range ds.prevInputs {
		t.MustFinalizeAll()
	}
	ds.prevInputs = nil
	for _, t := range ds.prevLabels {
		t.MustFinalizeAll()
	}
	ds.prevLabels = nil
}

// Name implements train.Dataset.
func (ds *freeingDataset) Name() string { return ds.name }

// Yield implements train.Dataset.
func (ds *freeingDataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	ds.freePreviousYield()
	spec, inputs, labels, err = ds.ds.Yield()
	ds.prevInputs = inputs
	ds.prevLabels = labels
	if err != nil {
		if err == io.EOF {
			return
		}
		err = errors.WithMessage(err, "dataset failure: notice it is being run with `data.Freeing`, which "+
			"frees the yielded inputs/labels in between uses")
		inputs = nil // Just in case.
		labels = nil
		ds.freePreviousYield()
		return
	}
	return
}

// Reset implements train.Dataset.
func (ds *freeingDataset) Reset() {
	ds.freePreviousYield()
	ds.ds.Reset()
}
