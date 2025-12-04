package datasets

import (
	"slices"
	"sync"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// onDeviceBatch represents a batch that has been uploaded to the target device.
type onDeviceBatch struct {
	spec   any
	inputs []*tensors.Tensor
	labels []*tensors.Tensor
	err    error // io.EOF or other error
}

// OnDevice is a wrapper around a train.Dataset that yields tensors already on the target device.
//
// It reads from the source dataset in parallel (if configured) and uploads the data to the target device,
// keeping batches buffered in a channel so they're ready when Yield() is called.
//
// The dataset uses a single channel for synchronization and buffering. The background reader goroutine
// reads from the source dataset and uploads tensors to the target device before placing them in the buffer.
//
// The order of the yields is not preserved for bufferSize > 1: the queuing process may yield results in slighly
// different order.
type OnDevice struct {
	backend backends.Backend

	// Source dataset to read from
	source                train.Dataset
	sourceConcurrencySafe bool
	sourceMu              sync.Mutex // Serializes source.Yield call, if sourceConcurrencySafe is false.

	// Configuration
	name, shortName string
	targetDevice    backends.DeviceNum
	bufferSize      int

	// Channel for buffering batches that are already on device (size 1 by default)
	nextBatch chan *onDeviceBatch
}

var _ train.Dataset = &OnDevice{}

// NewOnDevice creates a new OnDeviceDataset that wraps the source dataset and uploads batches to the target device.
//
// - backend: the backend to use for device operations
// - source: the source dataset to read from.
// - concurrent: whether the source dataset Yield method can be called in parallel.
// - bufferSize: size of the buffer channel (defaults to 1 if <= 0)
// - targetDevice: 0
//
// Use SetTargetDevice() and SetCanCallInParallel() to configure before first use.
func NewOnDevice(backend backends.Backend, source train.Dataset, concurrent bool,
	bufferSize int, targetDevice backends.DeviceNum) (*OnDevice, error) {
	if backend == nil {
		return nil, errors.New("backend cannot be nil")
	}
	if source == nil {
		return nil, errors.New("source dataset cannot be nil")
	}
	if bufferSize <= 0 {
		bufferSize = 1
	}

	ds := &OnDevice{
		backend:               backend,
		source:                source,
		sourceConcurrencySafe: concurrent,
		name:                  source.Name(),
		targetDevice:          targetDevice,
		bufferSize:            bufferSize,
		nextBatch:             make(chan *onDeviceBatch, bufferSize),
	}
	if sn, ok := source.(train.HasShortName); ok {
		ds.shortName = sn.ShortName()
	} else {
		ds.shortName = ds.name[:3]
	}

	// Start the bufferSize readers.
	ds.startReader()
	return ds, nil
}

// Name implements train.Dataset.
func (ds *OnDevice) Name() string {
	return ds.name
}

// ShortName implements train.HasShortName.
func (ds *OnDevice) ShortName() string {
	return ds.shortName
}

// Reset implements train.Dataset.
func (ds *OnDevice) Reset() {
	// Drain the channel: this guarantees there are no more readers running.
	for range ds.bufferSize {
		batch := <-ds.nextBatch
		if batch != nil {
			// Finalize any tensors in the batch: we discard the results and errors.
			for _, slice := range [][]*tensors.Tensor{batch.inputs, batch.labels} {
				for _, t := range slice {
					err := t.FinalizeAll()
					if err != nil {
						klog.Errorf("failed to finalize tensor in dataset %q, in Reset() method: %v", ds.name, err)
					}
				}
			}
		}
	}

	// Reset the source dataset and start new readers to populate the buffer channel.
	ds.source.Reset()
	ds.startReader()
}

// Yield implements train.Dataset.
func (ds *OnDevice) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	// Read the next prepared batch from channel, and prepare the next reader to prepare the next batch in parallel.
	batch := <-ds.nextBatch
	go ds.reader() // Start next reader to repopulate the batch we are consuming.
	if batch.err != nil {
		return nil, nil, nil, batch.err
	}
	return batch.spec, batch.inputs, batch.labels, nil
}

// startReader starts a background reader goroutines to fill the buffer channel.
func (ds *OnDevice) startReader() {
	for range ds.bufferSize {
		go ds.reader()
	}
}

// reader reads the source, transfers the tensors to the target device an enqueue it in the buffered channel.
//
// It is meant to be called on its own goroutine.
func (ds *OnDevice) reader() {
	// Read from source dataset
	spec, inputs, labels, err := ds.safeYield()
	if err != nil {
		// Send error batch
		ds.nextBatch <- &onDeviceBatch{err: err}
		return
	}

	// Upload inputs and labels to target device
	for _, slice := range [][]*tensors.Tensor{inputs, labels} {
		for _, t := range slice {
			err := t.MaterializeOnDevice(ds.backend, true, ds.targetDevice)
			if err != nil {
				ds.nextBatch <- &onDeviceBatch{
					err: errors.WithMessagef(err, "dataset %q failed to upload tensor to device %d",
						ds.name, ds.targetDevice)}
				return
			}
			t.FinalizeLocal()
		}
	}

	// Send ready batch to channel (blocking if channel is full)
	ds.nextBatch <- &onDeviceBatch{
		spec:   spec,
		inputs: inputs,
		labels: labels,
	}
}

// safeYield handles the serialization of calls to source.Yield, if required.
func (ds *OnDevice) safeYield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	if !ds.sourceConcurrencySafe {
		ds.sourceMu.Lock()
		defer ds.sourceMu.Unlock()
	}
	spec, inputs, labels, err = ds.source.Yield()
	if err != nil {
		return
	}
	inputs = slices.Clone(inputs)
	labels = slices.Clone(labels)
	return
}
