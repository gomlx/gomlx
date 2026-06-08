// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dataset

import (
	"io"
	"iter"
	"log"
	"runtime"
	"sync"

	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/support/xsync"
	"k8s.io/klog/v2"
)

// ParallelDataset is a wrapper around a `train.Dataset` that parallelizes calls to Iter().
// See details in CustomParallel.
type ParallelDataset struct {
	Dataset train.Dataset

	// name is set by default to the underlying dataset name.
	name, shortName string

	// parallelism is the number of goroutines started generating examples.
	parallelism int

	// extraBufferSize is the size of the buffer of pre-generated batches.
	extraBufferSize int

	// set to true once Start is called
	started bool
}

// parallelDatasetImpl separates the implementation of ParallelDataset.
type parallelDatasetImpl struct {
	config ParallelDataset // A copy of the configuration.

	err   error
	muErr sync.Mutex

	pullMu sync.Mutex

	buffer                                chan train.Batch
	epochFinished, stopEpoch, stopDataset chan struct{}
	done                                  *xsync.Latch
}

var _ train.Dataset = (*ParallelDataset)(nil)

// Parallel parallelizes yield calls of any train.Dataset.
//
// It uses CustomParallel and automatically starts it with the default
// parameters.
//
// The order of the yields is not preserved -- the parallelization may yield results in different order, and in some
// exceptional circumstance may create an order bias (faster results to generate being yield first).
//
// Example:
//
//	var ds train.Dataset
//	ds = NewMyDataset(...)
//	ds = dataset.Parallel(ds)
//	MyTrainFunc(ds)
func Parallel(ds train.Dataset) *ParallelDataset {
	pds := CustomParallel(ds)
	return pds.Buffer(pds.parallelism).Start()
}

// CustomParallel builds a ParallelDataset that can be used to parallelize any
// train.Dataset.
//
// ParallelDataset can be further configured (see SetParallelism and Buffer),
// and then one has to call Start before actually using the Dataset.
//
// The order of the yields is not preserved -- the parallelization may yield results in different order, and in some
// exceptional circumstance may create an order bias (faster results to generate being yield first).
//
// Example:
//
//		var ds train.Dataset
//		ds = NewMyDataset(...)
//	 	ds = dataset.CustomParallel(ds).Buffer(10).Start()
//	 	MyTrainFunc(ds)
func CustomParallel(ds train.Dataset) *ParallelDataset {
	pd := &ParallelDataset{
		name:    ds.Name(),
		Dataset: ds,
	}
	if sn, ok := ds.(train.HasShortName); ok {
		pd.shortName = sn.ShortName()
	} else {
		pd.shortName = pd.name[:3]
	}
	pd.Parallelism(0) // 0 here means it will take the number of cores available.
	return pd
}

// Parallelism is the number of goroutines to start, each pulling from the underlying dataset's iterator.
// If set to 0 (the default), and it will use the number of cores in the system plus 1.
//
// This must be called before a call to Start.
//
// It returns the updated ParallelDataset, so calls can be cascaded.
func (pd *ParallelDataset) Parallelism(n int) *ParallelDataset {
	if pd.started {
		log.Printf("ParallelDataset invalid configuration change after Start has been called.")
		return nil
	}
	if n == 0 {
		n = runtime.NumCPU() + 1
	}
	pd.parallelism = n
	return pd
}

// WithName sets the name of the parallel dataset, and optionally its short name.
// It defaults to the original dataset name.
//
// It returns the updated ParallelDataset, so calls can be cascaded.
func (pd *ParallelDataset) WithName(name string, shortName ...string) *ParallelDataset {
	pd.name = name
	if len(shortName) > 0 {
		pd.shortName = shortName[0]
	}
	return pd
}

// Buffer reserved in the channel that collects the parallel yields.
// Notice there is already an intrinsic buffering that happens in the goroutines sampling
// in parallel.
//
// This must be called before a call to Start.
//
// It returns the updated ParallelDataset, so calls can be cascaded.
func (pd *ParallelDataset) Buffer(n int) *ParallelDataset {
	if pd.started {
		log.Printf("ParallelDataset invalid configuration change after Start has been called.")
		return nil
	}
	pd.extraBufferSize = n
	return pd
}

// Start indicates that the dataset is finished to be configured, and starts
// being a valid Dataset.
//
// After Start its configuration can no longer be changed.
//
// It returns the updated ParallelDataset, so calls can be cascaded.
func (pd *ParallelDataset) Start() *ParallelDataset {
	pd.started = true
	return pd
}

// Done is a no-op in the new iterator-based ParallelDataset as goroutines are automatically cleaned up when the iterator finishes.
func (pd *ParallelDataset) Done() {}

// Name implements train.Dataset.
func (pd *ParallelDataset) Name() string {
	return pd.name
}

// ShortName returns a short version of the dataset name, it implements train.HasShortName.
func (pd *ParallelDataset) ShortName() string {
	return pd.shortName
}

// Iter implements train.Dataset.
func (pd *ParallelDataset) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		if !pd.started {
			pd.Start()
		}
		impl := &parallelDatasetImpl{
			buffer:      make(chan train.Batch, pd.extraBufferSize),
			stopDataset: make(chan struct{}),
			config:      *pd,
			done:        xsync.NewLatch(),
		}
		impl.startGoRoutines()
		defer impl.stopIteration()

		for {
			batch, err := impl.nextBatch()
			if err != nil {
				if err == io.EOF {
					return
				}
				yield(train.Batch{}, err)
				return
			}
			if !yield(batch, nil) {
				return
			}
		}
	}
}

func (impl *parallelDatasetImpl) stopIteration() {
	close(impl.stopDataset)
	impl.done.Wait()
}

func (impl *parallelDatasetImpl) nextBatch() (batch train.Batch, err error) {
	select {
	case <-impl.stopDataset:
		impl.muErr.Lock()
		err = impl.err
		impl.muErr.Unlock()
		return
	case batch = <-impl.buffer:
		// We got a new batch
	case <-impl.epochFinished:
		// No more records being produced, but we still need to exhaust the buffer.
		select {
		case batch = <-impl.buffer:
			// We got a new batch, simply continue.
		default:
			// Generation exhausted, and no more records in buffer.
			err = io.EOF
			return
		}
	}
	return
}

func (impl *parallelDatasetImpl) startGoRoutines() {
	impl.epochFinished = make(chan struct{})
	impl.stopEpoch = make(chan struct{})
	var wg sync.WaitGroup

	next, stop := iter.Pull2(impl.config.Dataset.Iter())

	for ii := 0; ii < impl.config.parallelism; ii++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-impl.stopEpoch:
					return
				case <-impl.stopDataset:
					return
				default:
					// Move forward and generate the next batch.
				}

				var batch train.Batch
				var err error
				var ok bool

				impl.pullMu.Lock()
				batch, err, ok = next()
				impl.pullMu.Unlock()

				if !ok {
					return
				}
				if err != nil {
					klog.Errorf("Error: %+v", err)
					impl.muErr.Lock()
					if impl.err == nil {
						impl.err = err
					}
					select {
					case <-impl.stopEpoch:
					default:
						close(impl.stopEpoch)
					}
					select {
					case <-impl.stopDataset:
					default:
						close(impl.stopDataset)
					}
					impl.muErr.Unlock()
					return
				}

				select {
				case <-impl.stopEpoch:
					return
				case <-impl.stopDataset:
					return
				case impl.buffer <- batch:
					// Batch generated and cached, move to next.
					continue
				}
			}
		}()
	}

	// Start the controller job.
	go func() {
		wg.Wait()
		stop()
		impl.muErr.Lock()
		defer impl.muErr.Unlock()
		select {
		case <-impl.stopDataset:
			impl.done.Trigger()
			return
		default:
			//
		}
		close(impl.epochFinished)
		impl.done.Trigger()
	}()
}
