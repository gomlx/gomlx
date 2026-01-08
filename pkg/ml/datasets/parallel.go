// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package datasets

import (
	"io"
	"log"
	"runtime"
	"sync"

	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/support/xsync"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// ParallelDataset is a wrapper around a `train.Dataset` that parallelize calls to Yield.
// See details in CustomParallel.
type ParallelDataset struct {
	Dataset train.Dataset

	// name is set by default to the underlying dataset name.
	name, shortName string

	// parallelism is the number of goroutines started generating examples.
	parallelism int

	// extraBufferSize is the size of the buffer of pre-generated batches.
	extraBufferSize int

	// impl is the actual implementation.
	impl *parallelDatasetImpl

	// keepAlive is used only to keep ParallelDataset alive in the middle of long calls.
	keepAlive int64

	// set to true once controller is started
	started bool
}

type yieldUnit struct {
	spec   any
	inputs []*tensors.Tensor
	labels []*tensors.Tensor
}

// parallelDatasetImpl separates the implementation of ParallelDataset. It's important
// that it doesn't point back to the original ParallelDataset, so garbage collecting
// will also stop the goroutines.
type parallelDatasetImpl struct {
	config ParallelDataset // A copy of ht

	err   error
	muErr sync.Mutex

	buffer                                chan yieldUnit
	epochFinished, stopEpoch, stopDataset chan struct{}
	done                                  *xsync.Latch
}

// Parallel parallelizes yield calls of any tread-safe train.Dataset.
//
// It uses CustomParallel and automatically starts it with the default
// parameters.
//
// To avoid leaking goroutines, call ParallelDataset.Cancel when exiting.
//
// The order of the yields is not preserved -- the parallelization may yield results in different order, and in some
// exceptional circumstance may create an order bias (faster results to generate being yield first).
//
// Example:
//
//	var ds train.Dataset
//	ds = NewMyDataset(...)
//	ds = data.Parallel(ds)
//	MyTrainFunc(ds)
func Parallel(ds train.Dataset) *ParallelDataset {
	pds := CustomParallel(ds)
	return pds.Buffer(pds.parallelism).Start()
}

// CustomParallel builds a ParallelDataset that can be used to parallelize any
// train.Dataset, as long as the underlying dataset ds is thread-safe.
//
// ParallelDataset can be further configured (see SetParallelism and Buffer),
// and then one has to call Start before actually using the Dataset.
//
// To avoid leaking goroutines, call ParallelDataset.Cancel when exiting.
//
// The order of the yields is not preserved -- the parallelization may yield results in different order, and in some
// exceptional circumstance may create an order bias (faster results to generate being yield first).
//
// Example:
//
//		var ds train.Dataset
//		ds = NewMyDataset(...)
//	 	ds = data.CustomParallel(ds).Buffer(10).Start()
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

// Parallelism is the number of goroutines to start, each calling `ds.Yield()` in parallel
// to accelerate the generation of batches. If set to 0 (the default), and it will use the
// number of cores in the system plus 1.
//
// It also allocates a buffer (in a Go channel) for each goroutine.
//
// This must be called before a call to Start.
//
// It returns the updated ParallelDataset, so calls can be cascaded.
func (pd *ParallelDataset) Parallelism(n int) *ParallelDataset {
	if pd.impl != nil {
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
	if pd.impl != nil {
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
	if pd.impl != nil {
		log.Printf("ParallelDataset.Start called more than once!?")
		return nil
	}
	impl := &parallelDatasetImpl{
		buffer:      make(chan yieldUnit, pd.extraBufferSize),
		stopDataset: make(chan struct{}),
		config:      *pd, // Copy.
		done:        xsync.NewLatch(),
	}
	pd.impl = impl
	// If the ParallelDataset is garbage collected, stop all parallel goroutines.
	runtime.SetFinalizer(pd, func(pd *ParallelDataset) {
		if pd.impl != nil {
			close(pd.impl.stopDataset)
			pd.impl = nil
		}
	})

	// Start goroutines
	impl.startGoRoutines()
	return pd
}

func (impl *parallelDatasetImpl) startGoRoutines() {
	impl.epochFinished = make(chan struct{})
	impl.stopEpoch = make(chan struct{})
	var wg sync.WaitGroup
	for ii := 0; ii < impl.config.parallelism; ii++ {
		// Start all goroutines.
		wg.Add(1)
		go func(impl *parallelDatasetImpl) {
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
				var unit yieldUnit
				var err error
				unit.spec, unit.inputs, unit.labels, err = impl.config.Dataset.Yield()
				if err == io.EOF {
					return
				}
				if err != nil {
					klog.Errorf("Error: %+v", err)
					// Fatal error, stop everything.
					impl.muErr.Lock()
					if impl.err != nil {
						impl.err = err
					}
					close(impl.stopEpoch)
					close(impl.stopDataset)
					impl.muErr.Unlock()
					return
				}
				select {
				case <-impl.stopEpoch:
					return
				case <-impl.stopDataset:
					return
				case impl.buffer <- unit:
					// Batch generated and cached, move to next.
					continue
				}
			}
		}(impl)
	}

	// Start the controller job.
	go func() {
		wg.Wait()
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
	}()
}

// Name implements train.Dataset.
func (pd *ParallelDataset) Name() string {
	return pd.name
}

// ShortName returns a short version of the dataset name, it implements train.HasShortName.
func (pd *ParallelDataset) ShortName() string {
	return pd.shortName
}

// Done stops all the parallel dataset and wait them to finish.
func (pd *ParallelDataset) Done() {
	if pd.impl != nil {
		impl := pd.impl
		close(impl.stopDataset)
		pd.impl = nil
		impl.done.Wait()
	}
}

// Reset implements train.Dataset.
func (pd *ParallelDataset) Reset() {
	impl := pd.impl
	if impl == nil {
		klog.Warningf("ParallelDataset.Reset was called before it was started with ParallelDataset.Start or after ParallelDataset.Done")
		return
	}

	// Indicate to readers to stop generating data, and drain whatever is still in the buffer.
	impl.muErr.Lock()
	close(impl.stopEpoch) // Indicate to goroutines to stop generating batches.
	impl.muErr.Unlock()
drainDataset: //
	for {
		select {
		case <-impl.stopDataset:
			// Return immediately, do nothing.
			return
		case <-impl.epochFinished:
			// All finished, we can move on.
			break drainDataset
		case <-impl.buffer:
			// Discard remaining entries that were in the buffer.
		}
	}

	// Reset underlying dataset and start again.
	impl.config.Dataset.Reset()
	impl.startGoRoutines()

	// This no-op prevents `pd` from being garbage collected and the goroutines killed in the middle
	// of the Reset operation. Leave this at the end.
	pd.keepAlive++
}

// Yield implements train.Dataset.
func (pd *ParallelDataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	impl := pd.impl
	if impl == nil {
		err = errors.Errorf("ParallelDataset.Yield was called before it was started with ParallelDataset.Start or after it was stopped with ParallelDataset.Done")
		return
	}
	var unit yieldUnit
	select {
	case <-impl.stopDataset:
		// An error occurred, dataset is closed.
		impl.muErr.Lock()
		err = impl.err
		impl.muErr.Unlock()
		return
	case unit = <-impl.buffer:
		// We got a new batch
	case <-impl.epochFinished:
		// No more records being produced (until Reset() is called), but we still need to exhaust the buffer.
		select {
		case unit = <-impl.buffer:
			// We got a new batch, simply continue.
		default:
			// Generation exhausted, and no more records in buffer.
			err = io.EOF
			return
		}
	}
	spec, inputs, labels = unit.spec, unit.inputs, unit.labels

	// This no-op prevents `pd` from being garbage collected and the goroutines killed in the middle
	// of the Yield operation. Leave this at the end.
	pd.keepAlive++
	return
}
