// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dataset

import (
	"fmt"
	"iter"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/pkg/errors"
)

type batchElement struct {
	inputs, labels []*tensors.Tensor
	spec           any
}

// FinalizeAll calls FinalizeAll on all inputs and labels tensors.
func (e *batchElement) FinalizeAll() {
	for _, t := range e.inputs {
		t.MustFinalizeAll()
	}
	for _, t := range e.labels {
		t.MustFinalizeAll()
	}
}

// batchedDataset implements train.Dataset and batches results from the underlying
// BaseDataset.
//
// See details in Batch, the function used to create it.
type batchedDataset struct {
	backend compute.Backend
	ds      train.Dataset // Source Dataset.

	batchSize                              int
	createLeadingAxis, dropIncompleteBatch bool

	batchExec *graph.Exec // Batch tensors.
}

// Batch creates dataset that batches `ds` into batches of size `batchSize`.
//
// It uses GoMLX to batch the tensors themselves, so it takes a graph.Backend as its first
// parameter. Also, it means that it yields examples already stored "on device" -- whichever
// the platform Backend was configured with.
//
// Typically, Batch can benefit from ReadAhead, so while training or evaluation of batch
// is happening, the next batch is being built. Consider using ReadAhead on the Batch dataset,
// even with a buffer of only 1.
//
// Args:
//   - `backend`: will be used to create the graph that actually does the batching.
//   - `ds`: the dataset to be batched.
//   - `batch_size`: size of each batch, except when there are no more examples, in which
//     case batches can be smaller (except if `dropIncompleteBatch` was selected).
//   - `createLeadingAxis`: usually set to true, it will create a new leading
//     axis that becomes the batch dimension. Otherwise, it simply concatenates the individual
//     results at the axis 0 -- this can be used for instance to increase the size of a batch.
//   - `dropIncompleteBatch`: at the end of an epoch, if there are not enough examples to fill a
//     batch, and this is set to true, the last batch is dropped. Otherwise, it returns only
//     a partial batch -- with a different shape this may trigger the re-compilation of a graph.
//     Usually desirable for evaluation, but not desirable for training.
//
// Returns a `train.Dataset` that yields batched examples.
func Batch(
	backend compute.Backend,
	ds train.Dataset,
	batchSize int,
	createLeadingAxis, dropIncompleteBatch bool,
) train.Dataset {
	batched := &batchedDataset{
		backend:             backend,
		ds:                  ds,
		batchSize:           batchSize,
		createLeadingAxis:   createLeadingAxis,
		dropIncompleteBatch: dropIncompleteBatch,
	}
	batched.batchExec = graph.MustNewExec(backend, batched.batchTensorsGraph)
	return batched
}

// Name implements train.Dataset. It returns the dataset name.
func (ds *batchedDataset) Name() string {
	return fmt.Sprintf("%s [Batch]", ds.ds.Name())
}

// Iter implements train.Dataset.
func (ds *batchedDataset) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		var buffer []batchElement

		defer func() {
			for _, element := range buffer {
				element.FinalizeAll()
			}
		}()

		for batch, err := range ds.ds.Iter() {
			if err != nil {
				yield(train.Batch{}, err)
				return
			}

			e := batchElement{
				inputs: batch.Inputs,
				spec:   batch.Spec,
				labels: batch.Labels,
			}
			buffer = append(buffer, e)

			if len(buffer) >= ds.batchSize {
				batched, err := ds.batchBuffer(buffer)
				for _, element := range buffer {
					element.FinalizeAll()
				}
				buffer = buffer[0:0]

				if err != nil {
					yield(train.Batch{}, err)
					return
				}

				yieldedBatch := train.Batch{
					Spec:   batched.spec,
					Inputs: batched.inputs,
					Labels: batched.labels,
				}
				if !yield(yieldedBatch, nil) {
					return
				}
			}
		}

		if len(buffer) > 0 {
			if ds.dropIncompleteBatch {
				for _, element := range buffer {
					element.FinalizeAll()
				}
				buffer = buffer[0:0]
				return
			}

			batched, err := ds.batchBuffer(buffer)
			for _, element := range buffer {
				element.FinalizeAll()
			}
			buffer = buffer[0:0]

			if err != nil {
				yield(train.Batch{}, err)
				return
			}

			yieldedBatch := train.Batch{
				Spec:   batched.spec,
				Inputs: batched.inputs,
				Labels: batched.labels,
			}
			if !yield(yieldedBatch, nil) {
				return
			}
		}
	}
}

// batchBuffer batches each element of inputs and labels, and take the first `spec` value.
func (ds *batchedDataset) batchBuffer(buffer []batchElement) (batched batchElement, err error) {
	// Extract shapes from first element of the buffer.
	if len(buffer) == 0 {
		err = errors.Errorf("trying to batch a zero elements in the buffer!?")
		return
	}
	e := buffer[0]
	batched.spec = e.spec
	var inputsShapes, labelsShapes []shapes.Shape
	if len(e.inputs) > 0 {
		inputsShapes = make([]shapes.Shape, 0, len(e.inputs))
		for _, t := range e.inputs {
			inputsShapes = append(inputsShapes, t.Shape())
		}
	}
	if len(e.labels) > 0 {
		labelsShapes = make([]shapes.Shape, 0, len(e.labels))
		for _, t := range e.labels {
			labelsShapes = append(labelsShapes, t.Shape())
		}
	}

	// Check that the other elements of the buffer have the same shape.
	for ii := 1; ii < len(buffer); ii++ {
		e = buffer[ii]
		if len(e.inputs) != len(inputsShapes) {
			err = errors.Errorf("inputs to be batched don't have all the same number of elements: seen one Yield() "+
				"returns %d elements and another returns %d elements", len(inputsShapes), len(e.inputs))
			return
		}
		for ii, input := range e.inputs {
			if !inputsShapes[ii].Equal(input.Shape()) {
				err = errors.Errorf("input #%d returned by Yield has varying shapes (seen %s and %s)",
					ii, inputsShapes[ii], input.Shape())
				return
			}
		}
		if len(e.labels) != len(labelsShapes) {
			err = errors.Errorf("labels to be batched don't have all the same number of elements: seen one Yield() "+
				"returns %d elements and another returns %d elements", len(labelsShapes), len(e.labels))
			return
		}
		for ii, label := range e.labels {
			if !labelsShapes[ii].Equal(label.Shape()) {
				err = errors.Errorf("label #%d returned by Yield has varying shapes (seen %s and %s)",
					ii, labelsShapes[ii], label.Shape())
				return
			}
		}
	}

	allInputs := make([][]*tensors.Tensor, 0, len(buffer))
	allLabels := make([][]*tensors.Tensor, 0, len(buffer))
	for _, e := range buffer {
		allInputs = append(allInputs, e.inputs)
		allLabels = append(allLabels, e.labels)
	}
	batched.inputs, err = ds.batchTensorsList(allInputs)
	if err != nil {
		return
	}
	batched.labels, err = ds.batchTensorsList(allLabels)
	return
}

// batchTensorsList receives a list of inputs or labels collections, and concatenate them
// into a batch. Returns the list of the concatenated tensors.
//
// The batching happens on the tensors on the first axis of the `inputs` slice.
func (ds *batchedDataset) batchTensorsList(
	inputs [][]*tensors.Tensor,
) (batchedTensors []*tensors.Tensor, err error) {
	numBatchedTensors := len(inputs[0])
	numParts := len(inputs)
	batchedTensors = make([]*tensors.Tensor, 0, numBatchedTensors)
	parts := make([]*tensors.Tensor, numParts)
	for batchedTensorIdx := range numBatchedTensors {
		for ii, inputTensors := range inputs {
			parts[ii] = inputTensors[batchedTensorIdx]
		}
		var batchedTensor *tensors.Tensor
		batchedTensor, err = ds.batchTensor(parts)
		if err != nil {
			return
		}
		batchedTensors = append(batchedTensors, batchedTensor)
	}
	return
}

// batchTensor batches the given tensor list, all should already have the same shape.
func (ds *batchedDataset) batchTensor(parts []*tensors.Tensor) (batched *tensors.Tensor, err error) {
	partsAny := make([]any, 0, len(parts))
	for _, part := range parts {
		partsAny = append(partsAny, part)
	}
	batched, err = ds.batchExec.Call1(partsAny...)
	if err != nil {
		return nil, err
	}
	return batched, nil
}

// batchTensorsGraph builds the computational graph that batches a collection of Nodes
// (that will hold the tensors).
func (ds *batchedDataset) batchTensorsGraph(inputs []*graph.Node) *graph.Node {
	if ds.createLeadingAxis {
		newInputs := make([]*graph.Node, 0, len(inputs))
		for _, input := range inputs {
			newInputs = append(newInputs, graph.InsertAxes(input, 0))
		}
		inputs = newInputs
	}
	// Batch dimension is by convention the very first.
	if len(inputs) == 1 {
		return inputs[0]
	}
	return graph.Concatenate(inputs, 0)
}

// ReadAhead returns a Dataset that reads bufferSize elements of the given `ds`
// so that when Yield is called, the results are immediate.
//
// Deprecated: please use Buffer() instead.
func ReadAhead(ds train.Dataset, bufferSize int) train.Dataset {
	if bufferSize <= 0 {
		return ds
	}
	return Buffer(ds, bufferSize)
}
