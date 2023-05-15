package data

import (
	"fmt"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/pkg/errors"
	"io"
	"log"
	"runtime"
	"sync"
)

// ParallelDataset is a wrapper around a `train.Dataset` that parallelize calls to Yield.
// See details in CustomParallel.
type ParallelDataset struct {
	Dataset train.Dataset

	// parallelism is the number of goroutines started generating examples.
	parallelism int

	// extraBufferSize is the size of the cache of pre-generated batches.
	extraBufferSize int

	// impl is the actual implementation.
	impl *parallelDatasetImpl

	// keepAlive is used only to keep ParallelDataset alive in the middle of long calls.
	keepAlive int64
}

type yieldUnit struct {
	spec   any
	inputs []tensor.Tensor
	labels []tensor.Tensor
}

// parallelDatasetImpl separates the implementation of ParallelDataset. It's important
// that it doesn't point back to the original ParallelDataset, so garbage collecting
// will also stop the goroutines.
type parallelDatasetImpl struct {
	config ParallelDataset // A copy of ht

	err   error
	muErr sync.Mutex

	cache                                 chan yieldUnit
	epochFinished, stopEpoch, stopDataset chan struct{}
}

// Parallel parallelizes any tread-safe train.Dataset.
//
// It uses CustomParallel and automatically starts it with the default
// parameters.
//
// To avoid leaking goroutines, call ParallelDataset.Cancel when exiting.
//
// Example:
//
//		var ds train.Dataset
//		ds = NewMyDataset(...)
//	 	ds = data.Parallel(ds)
//	    MyTrainFunc(ds)
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
// Example:
//
//		var ds train.Dataset
//		ds = NewMyDataset(...)
//	 	ds = data.CustomParallel(ds).Buffer(10).Start()
//	 	MyTrainFunc(ds)
func CustomParallel(ds train.Dataset) *ParallelDataset {
	pd := &ParallelDataset{
		Dataset: ds,
	}
	pd.Parallelism(0)
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

// Buffer reserved in the channel that collects the parallel yields.
// Notice there is already a intrinsic buffering that happens
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
		cache:       make(chan yieldUnit, pd.extraBufferSize),
		stopDataset: make(chan struct{}),
		config:      *pd, // Copy.
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
					log.Printf("Error: %+v", err)
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
				case impl.cache <- unit:
					// Batch generated and cached, move to next.
					continue
				}
			}
		}(impl)
	}

	// Start controller job.
	go func() {
		wg.Wait()
		impl.muErr.Lock()
		defer impl.muErr.Unlock()
		select {
		case <-impl.stopDataset:
			return
		default:
			//
		}
		close(impl.epochFinished)
	}()
}

// Name implements train.Dataset.
func (pd *ParallelDataset) Name() string {
	return fmt.Sprintf("%s [CustomParallel]", pd.Dataset.Name())
}

// Reset implements train.Dataset.
func (pd *ParallelDataset) Reset() {
	impl := pd.impl
	if impl == nil {
		log.Printf("ParallelDataset.Reset was called before it was started with ParallelDataset.Start")
		return
	}
	impl.muErr.Lock()
	close(impl.stopEpoch) // Indicate to goroutines to stop generating batches.
	impl.muErr.Unlock()
	select {
	case <-impl.stopDataset:
		// Return immediately, do nothing.
		return
	case <-impl.cache:
		// Discard remaining entries in cache.
	case <-impl.epochFinished:
		// All finished, we can move on.
	}

	// Reset underlying dataset and start again.
	impl.config.Dataset.Reset()
	impl.startGoRoutines()

	// This no-op prevents `pd` from being garbage collected and the goroutines killed in the middle
	// of the Reset operation. Leave this at the end.
	pd.keepAlive++
}

// Yield implements train.Dataset.
func (pd *ParallelDataset) Yield() (spec any, inputs []tensor.Tensor, labels []tensor.Tensor, err error) {
	impl := pd.impl
	if impl == nil {
		err = errors.Errorf("ParallelDataset.Yield was called before it was started with ParallelDataset.Start")
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
	case unit = <-impl.cache:
		// We got a new batch
	case <-impl.epochFinished:
		// No more records being produced (until Reset() is called), but we still need to exhaust the cache.
		select {
		case unit = <-impl.cache:
			// We got a new batch, simply continue.
		default:
			// Generation exhausted, and no more records in cache.
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
