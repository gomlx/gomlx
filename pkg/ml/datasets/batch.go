package datasets

import (
	"fmt"
	"io"
	"sync"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/train"
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
	backend backends.Backend
	ds      train.Dataset // Source Dataset.

	batchSize                              int
	createLeadingAxis, dropIncompleteBatch bool

	buffer []batchElement
	mu     sync.Mutex // Protects buffer.

	batchExec *Exec // Batch tensors.
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
	backend backends.Backend,
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
	batched.batchExec = MustNewExec(backend, batched.batchTensorsGraph)
	return batched
}

// Name implements train.Dataset. It returns the dataset name.
func (ds *batchedDataset) Name() string {
	return fmt.Sprintf("%s [Batch]", ds.ds.Name())
}

// Reset implements train.Dataset.
func (ds *batchedDataset) Reset() {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	ds.lockedFreeBuffer()
	ds.ds.Reset()
}

// lockedFreeBuffer finalizes all intermediary tensors. It must be called with `ds.mu` locked.
func (ds *batchedDataset) lockedFreeBuffer() {
	for _, element := range ds.buffer {
		element.FinalizeAll()
	}
	ds.buffer = ds.buffer[0:0]
}

// Yield implements train.Dataset.
func (ds *batchedDataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	defer ds.mu.Unlock()
	for {
		var eSpec any
		var eInputs, eLabels []*tensors.Tensor
		eSpec, eInputs, eLabels, err = ds.ds.Yield()
		ds.mu.Lock()
		if err == io.EOF {
			if ds.dropIncompleteBatch || len(ds.buffer) == 0 {
				ds.lockedFreeBuffer()
				return
			}
			// Else returns incomplete batch.
			break
		}
		if err != nil {
			return
		}

		e := batchElement{
			inputs: eInputs,
			spec:   eSpec,
			labels: eLabels,
		}
		ds.buffer = append(ds.buffer, e)
		if len(ds.buffer) >= ds.batchSize {
			// buffer full.
			break
		}
		ds.mu.Unlock()
	}

	// Return the batch -- in case this is the last one, and dropIncompleteBatch == false, it
	// may be a partial batch.
	// Future work: extract the buffer, and do the batching without locking, to allow more
	// parallelization.
	batched, err := ds.lockedBatchBuffer()
	ds.lockedFreeBuffer()
	if err != nil {
		return
	}
	spec, inputs, labels = batched.spec, batched.inputs, batched.labels
	return
}

// lockedBatchBuffer batches each element of inputs and labels, and take the first `spec` value.
// It assumes `ds.mu` is locked.
func (ds *batchedDataset) lockedBatchBuffer() (batched batchElement, err error) {
	// Extract shapes from first element of the buffer.
	if len(ds.buffer) == 0 {
		err = errors.Errorf("trying to batch a zero elements in the buffer!?")
		return
	}
	var inputsShapes, labelsShapes []shapes.Shape
	e := ds.buffer[0]
	batched.spec = e.spec
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
	for ii := 1; ii < len(ds.buffer); ii++ {
		e = ds.buffer[ii]
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

	allInputs := make([][]*tensors.Tensor, 0, len(ds.buffer))
	allLabels := make([][]*tensors.Tensor, 0, len(ds.buffer))
	for _, e := range ds.buffer {
		allInputs = append(allInputs, e.inputs)
		allLabels = append(allLabels, e.labels)
	}
	batched.inputs, err = ds.lockedBatchTensorsList(allInputs)
	if err != nil {
		return
	}
	batched.labels, err = ds.lockedBatchTensorsList(allLabels)
	return
}

// lockedBatchTensorsList receives a list of inputs or labels collections, and concatenate them
// into a batch. Returns the list of the concatenated tensors.
//
// The batching happens on the tensors on the first axis of the `inputs` slice.
func (ds *batchedDataset) lockedBatchTensorsList(
	inputs [][]*tensors.Tensor,
) (batchedTensors []*tensors.Tensor, err error) {
	numBatchedTensors := len(inputs[0])
	numParts := len(inputs)
	batchedTensors = make([]*tensors.Tensor, 0, numBatchedTensors)
	parts := make([]*tensors.Tensor, numParts)
	for batchedTensorIdx := 0; batchedTensorIdx < numBatchedTensors; batchedTensorIdx++ {
		for ii, inputTensors := range inputs {
			parts[ii] = inputTensors[batchedTensorIdx]
		}
		var batchedTensor *tensors.Tensor
		batchedTensor, err = ds.lockedBatchTensor(parts)
		if err != nil {
			return
		}
		batchedTensors = append(batchedTensors, batchedTensor)
	}
	return
}

// lockedBatchTensor batches the given tensor list, all should already have the same shape.
func (ds *batchedDataset) lockedBatchTensor(parts []*tensors.Tensor) (batched *tensors.Tensor, err error) {
	partsAny := make([]any, 0, len(parts))
	for _, part := range parts {
		partsAny = append(partsAny, part)
	}
	err = TryCatch[error](func() { batched = ds.batchExec.MustExec(partsAny...)[0] })
	if err != nil {
		return
	}
	return
}

// batchTensorsGraph builds the computational graph that batches a collection of Nodes
// (that will hold the tensors).
func (ds *batchedDataset) batchTensorsGraph(inputs []*Node) *Node {
	if ds.createLeadingAxis {
		newInputs := make([]*Node, 0, len(inputs))
		for _, input := range inputs {
			newInputs = append(newInputs, InsertAxes(input, 0))
		}
		inputs = newInputs
	}
	// Batch dimension is by convention the very first.
	if len(inputs) == 1 {
		return inputs[0]
	}
	return Concatenate(inputs, 0)
}

// ReadAhead returns a Dataset that reads bufferSize elements of the given `ds`
// so that when Yield is called, the results are immediate.
//
// It uses ParallelDataset to implement it.
func ReadAhead(ds train.Dataset, bufferSize int) train.Dataset {
	if bufferSize <= 0 {
		return ds
	}
	return CustomParallel(ds).Parallelism(1).Buffer(bufferSize - 1).Start()
}
