// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dataset

import (
	"iter"
	"sync"

	"github.com/gomlx/gomlx/ml/train"
)

// BufferDataset is a wrapper around a `train.Dataset` that buffers (pre-fetches) batches in a background goroutine.
// This overlaps CPU data preparation with GPU/accelerator training.
type BufferDataset struct {
	ds         train.Dataset
	bufferSize int
}

var _ train.Dataset = (*BufferDataset)(nil)

// Buffer wraps a train.Dataset in a BufferDataset to prefetch yielded batches in a background goroutine.
//
// If bufferSize is not specified, it defaults to 1.
//
// The order of the batches returned by the underlying dataset is preserved.
func Buffer(ds train.Dataset, bufferSize ...int) *BufferDataset {
	size := 1
	if len(bufferSize) > 0 {
		size = bufferSize[0]
	}
	return &BufferDataset{
		ds:         ds,
		bufferSize: size,
	}
}

// Name implements train.Dataset.
func (bd *BufferDataset) Name() string { return bd.ds.Name() }

// ShortName returns a short version of the dataset name, it implements train.HasShortName.
func (bd *BufferDataset) ShortName() string {
	if sn, ok := bd.ds.(train.HasShortName); ok {
		return sn.ShortName()
	}
	name := bd.ds.Name()
	if len(name) > 3 {
		return name[:3]
	}
	return name
}

// WithBufferSize configures the size of the prefetch queue (buffered channel) and returns the updated dataset.
func (bd *BufferDataset) WithBufferSize(size int) *BufferDataset {
	bd.bufferSize = size
	return bd
}

// Iter implements train.Dataset.
func (bd *BufferDataset) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		type result struct {
			batch train.Batch
			err   error
		}

		done := make(chan struct{})
		var wg sync.WaitGroup
		defer func() {
			close(done)
			wg.Wait()
		}()

		buffer := make(chan result, bd.bufferSize)

		next, stop := iter.Pull2(bd.ds.Iter())
		defer stop()

		wg.Go(func() {
			for {
				select {
				case <-done:
					return
				default:
				}

				batch, err, ok := next()
				if !ok {
					close(buffer)
					return
				}
				if err != nil {
					select {
					case <-done:
					case buffer <- result{err: err}:
					}
					close(buffer)
					return
				}

				select {
				case <-done:
					return
				case buffer <- result{batch: batch}:
				}
			}
		})

		for {
			select {
			case <-done:
				return
			case res, ok := <-buffer:
				if !ok {
					return
				}
				if res.err != nil {
					yield(train.Batch{}, res.err)
					return
				}
				if !yield(res.batch, nil) {
					return
				}
			}
		}
	}
}
