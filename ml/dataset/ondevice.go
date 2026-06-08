// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dataset

import (
	"iter"
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/pkg/errors"
)

// onDeviceBatch represents a batch that has been uploaded to the target device.
type onDeviceBatch struct {
	spec   any
	inputs []*tensors.Tensor
	labels []*tensors.Tensor
	err    error
}

// OnDevice is a wrapper around a train.Dataset that yields tensors already on the target device.
//
// It reads from the source dataset concurrently and uploads the data to the target device,
// keeping batches buffered in a channel so they're ready when Iter() is called.
type OnDevice struct {
	backend compute.Backend

	// Source dataset to read from
	source train.Dataset

	// Configuration
	name, shortName string
	targetDevice    compute.DeviceNum
	bufferSize      int
}

var _ train.Dataset = &OnDevice{}

// NewOnDevice creates a new OnDevice wrapper that wraps the source dataset and uploads batches to the target device.
//
// - backend: the backend to use for device operations
// - source: the source dataset to read from.
// - bufferSize: size of the buffer channel (defaults to 1 if <= 0)
// - targetDevice: the target device number
func NewOnDevice(backend compute.Backend, source train.Dataset,
	bufferSize int, targetDevice compute.DeviceNum) (*OnDevice, error) {
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
		backend:      backend,
		source:       source,
		name:         source.Name(),
		targetDevice: targetDevice,
		bufferSize:   bufferSize,
	}
	if sn, ok := source.(train.HasShortName); ok {
		ds.shortName = sn.ShortName()
	} else {
		ds.shortName = ds.name[:3]
	}

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

// Iter implements train.Dataset.
func (ds *OnDevice) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		nextBatch := make(chan *onDeviceBatch, ds.bufferSize)

		// 1. Defer draining nextBatch (runs last of the defers).
		// Since the worker is guaranteed to have finished and closed nextBatch before this runs,
		// ranging over nextBatch is completely safe and won't block.
		defer func() {
			for b := range nextBatch {
				if b.inputs != nil || b.labels != nil {
					_ = train.Batch{Inputs: b.inputs, Labels: b.labels}.Finalize()
				}
			}
		}()

		var wg sync.WaitGroup
		// 2. Defer wg.Wait() (runs after close(stopChan) and stop()).
		defer wg.Wait()

		stopChan := make(chan struct{})
		// 3. Defer close(stopChan) (runs after stop()).
		defer close(stopChan)

		next, stop := iter.Pull2(ds.source.Iter())
		// 4. Defer stop() (runs first).
		defer stop()

		// Start worker.
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer close(nextBatch)

			for {
				select {
				case <-stopChan:
					return
				default:
				}

				batch, err, ok := next()
				if !ok {
					return
				}
				if err != nil {
					_ = batch.Finalize()
					select {
					case nextBatch <- &onDeviceBatch{err: err}:
					case <-stopChan:
					}
					return
				}

				// Upload inputs and labels to target device
				for _, slice := range [][]*tensors.Tensor{batch.Inputs, batch.Labels} {
					for _, t := range slice {
						err := t.MaterializeOnDevice(ds.backend, true, ds.targetDevice)
						if err != nil {
							_ = batch.Finalize()
							select {
							case nextBatch <- &onDeviceBatch{
								err: errors.WithMessagef(err, "dataset %q failed to upload tensor to device %d",
									ds.name, ds.targetDevice),
							}:
							case <-stopChan:
							}
							return
						}
						t.FinalizeLocal()
					}
				}

				select {
				case nextBatch <- &onDeviceBatch{
					spec:   batch.Spec,
					inputs: batch.Inputs,
					labels: batch.Labels,
				}:
				case <-stopChan:
					_ = batch.Finalize()
					return
				}
			}
		}()

		// Read batches converted concurrently.
		for b := range nextBatch {
			if b.err != nil {
				yield(train.Batch{}, b.err)
				return
			}
			batch := train.Batch{
				Spec:   b.spec,
				Inputs: b.inputs,
				Labels: b.labels,
			}
			if !yield(batch, nil) {
				return
			}
		}
	}
}
