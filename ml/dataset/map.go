// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dataset

import (
	"fmt"
	"iter"
	"runtime"
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/pkg/errors"
)

// GraphBatch represents a batch of nodes in the graph building phase.
type GraphBatch struct {
	Inputs, Labels []*graph.Node
	Spec           any
}

// MapFn is a graph building function that transforms a GraphBatch.
type MapFn func(graphBatch GraphBatch) GraphBatch

// ModelMapFn is a graph building function that transforms a GraphBatch using a model scope.
type ModelMapFn func(scope *model.Scope, graphBatch GraphBatch) GraphBatch

// mapDatasetExecutorKey is the caching key for map executors.
type mapDatasetExecutorKey struct {
	numInputs, numLabels int
	specString           string
}

// mapDatasetExecutor holds a model.Exec along with output lengths.
type mapDatasetExecutor struct {
	exec            *model.Exec
	numMappedInputs int
	numMappedLabels int
}

// mapDataset implements a `train.Dataset` that maps a graph building function to a wrapped dataset.
type mapDataset struct {
	backend   compute.Backend
	store     *model.Store
	ds        train.Dataset
	mapFn     ModelMapFn
	executors map[mapDatasetExecutorKey]*mapDatasetExecutor
}

var _ train.Dataset = (*mapDataset)(nil)

// Map returns a `train.Dataset` with the result of applying (mapping) the batches yielded by the provided
// `dataset` by the graph function `mapFn`.
// The function is executed by the `backend` given.
func Map(backend compute.Backend, dataset train.Dataset, mapFn MapFn) train.Dataset {
	return ModelMap(backend, nil, dataset, func(_ *model.Scope, graphBatch GraphBatch) GraphBatch {
		return mapFn(graphBatch)
	})
}

// ModelMap returns a `train.Dataset` with the result of applying (mapping) the batches yielded by the provided
// `dataset` by the graph function `mapFn` that can use trainable variables in the model store.
// The function is executed by the `backend` given.
// If `store` is nil, a new one is created.
func ModelMap(backend compute.Backend, store *model.Store, dataset train.Dataset, mapFn ModelMapFn) train.Dataset {
	if store == nil {
		store = model.NewStore()
	}
	mapDS := &mapDataset{
		backend:   backend,
		store:     store,
		ds:        dataset,
		mapFn:     mapFn,
		executors: make(map[mapDatasetExecutorKey]*mapDatasetExecutor),
	}
	return mapDS
}

// Name implements train.Dataset.
func (mapDS *mapDataset) Name() string {
	return mapDS.ds.Name()
}

func specToString(spec any) string {
	if spec == nil {
		return ""
	}
	if stringer, ok := spec.(fmt.Stringer); ok {
		return stringer.String()
	}
	return fmt.Sprintf("%+v", spec)
}

// Iter implements train.Dataset.
func (mapDS *mapDataset) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		for batch, err := range mapDS.ds.Iter() {
			if err != nil {
				yield(batch, err)
				return
			}

			inputs := batch.Inputs
			labels := batch.Labels

			key := mapDatasetExecutorKey{
				numInputs:  len(inputs),
				numLabels:  len(labels),
				specString: specToString(batch.Spec),
			}

			executor, found := mapDS.executors[key]
			if !found {
				executor = &mapDatasetExecutor{}
				executor.exec = model.MustNewExec(mapDS.backend, mapDS.store,
					func(scope *model.Scope, inputsAndLabels []*graph.Node) []*graph.Node {
						var inputs, labels []*graph.Node
						if key.numInputs > 0 {
							inputs = inputsAndLabels[:key.numInputs]
						}
						if key.numLabels > 0 {
							labels = inputsAndLabels[key.numInputs:]
						}

						graphBatch := GraphBatch{
							Inputs: inputs,
							Labels: labels,
							Spec:   batch.Spec,
						}
						mappedBatch := mapDS.mapFn(scope, graphBatch)
						executor.numMappedInputs = len(mappedBatch.Inputs)
						executor.numMappedLabels = len(mappedBatch.Labels)
						return append(mappedBatch.Inputs, mappedBatch.Labels...)
					})
				mapDS.executors[key] = executor
			}

			inputsAndLabels := append(inputs, labels...)
			inputsAndLabelsAny := make([]any, len(inputsAndLabels))
			var tensorsToFinalize []*tensors.Tensor
			for idx, t := range inputsAndLabels {
				if t.IsShared() {
					inputsAndLabelsAny[idx] = t
					tensorsToFinalize = append(tensorsToFinalize, t)
				} else {
					donated, err := graph.DonateTensorBuffer(t, mapDS.backend, 0)
					if err != nil {
						inputsAndLabelsAny[idx] = t
						tensorsToFinalize = append(tensorsToFinalize, t)
					} else {
						inputsAndLabelsAny[idx] = donated
					}
				}
			}

			var mappedInputsAndLabels []*tensors.Tensor
			mappedInputsAndLabels, err = executor.exec.Call(inputsAndLabelsAny...)
			if err != nil {
				err = errors.WithMessagef(err, "while executing MapFn provided for dataset.Map()")
				yield(train.Batch{}, err)
				return
			}

			// Finalize original tensors that were not donated.
			for _, t := range tensorsToFinalize {
				t.MustFinalizeAll()
			}

			var mappedInputs, mappedLabels []*tensors.Tensor
			if executor.numMappedInputs > 0 {
				mappedInputs = mappedInputsAndLabels[:executor.numMappedInputs]
			}
			if executor.numMappedLabels > 0 {
				mappedLabels = mappedInputsAndLabels[executor.numMappedInputs:]
			}

			mappedBatch := train.Batch{
				Spec:   batch.Spec,
				Inputs: mappedInputs,
				Labels: mappedLabels,
			}
			if !yield(mappedBatch, nil) {
				return
			}
		}
	}
}

// MapOnHostFn is a normal Go function that applies a transformation to the batch of a dataset on the host CPU.
//
// Note: MapOnHostFn takes ownership of the incoming batch.
// Please manually finalize any input tensors that aren't reused in the output batch.
// Go's garbage collector will eventually free them, but since the GC doesn't track
// accelerator memory pressure, manual cleanup helps prevent out-of-memory errors.
type MapOnHostFn func(batch train.Batch) train.Batch

// mapOnHostDataset implements a `train.Dataset` that maps a function executed on the host CPU to a wrapped dataset.
type mapOnHostDataset struct {
	ds    train.Dataset
	mapFn MapOnHostFn
}

var _ train.Dataset = (*mapOnHostDataset)(nil)

// MapOnHost maps a dataset through a transformation with a (normal Go) function that runs in the host CPU.
func MapOnHost(ds train.Dataset, mapFn MapOnHostFn) train.Dataset {
	return &mapOnHostDataset{
		ds:    ds,
		mapFn: mapFn,
	}
}

// Name implements train.Dataset.
func (ds *mapOnHostDataset) Name() string { return ds.ds.Name() }

// Iter implements train.Dataset.
func (ds *mapOnHostDataset) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		for batch, err := range ds.ds.Iter() {
			if err != nil {
				yield(batch, err)
				return
			}
			mappedBatch := ds.mapFn(batch)
			if err, ok := mappedBatch.Spec.(error); ok {
				yield(train.Batch{}, err)
				return
			}
			if !yield(mappedBatch, nil) {
				return
			}
		}
	}
}

// ParallelMapOnHostDataset implements a `train.Dataset` that maps a function executed on the host CPU
// to a wrapped dataset in parallel.
type ParallelMapOnHostDataset struct {
	ds            train.Dataset
	parallelism   int
	mapFn         func(train.Batch) (train.Batch, error)
	preserveOrder bool
}

var _ train.Dataset = (*ParallelMapOnHostDataset)(nil)

// ParallelMapOnHost maps a dataset through a transformation with a (normal Go) function that runs in the host CPU,
// executing the mapping function in parallel using the specified number of worker goroutines.
//
// The default parallelism is the number of CPUs (runtime.NumCPU()), but it can be changed with WithParallelism(n).
//
// The order of the batches returned by the underlying dataset is preserved by default,
// unless WithPreserveOrder(false) is called on the result.
func ParallelMapOnHost(ds train.Dataset, mapFn MapOnHostFn) *ParallelMapOnHostDataset {
	wrappedMapFn := func(b train.Batch) (train.Batch, error) {
		mapped := mapFn(b)
		if err, ok := mapped.Spec.(error); ok {
			return train.Batch{}, err
		}
		return mapped, nil
	}
	return &ParallelMapOnHostDataset{
		ds:            ds,
		parallelism:   runtime.NumCPU(),
		mapFn:         wrappedMapFn,
		preserveOrder: true,
	}
}

// Name implements train.Dataset.
func (ds *ParallelMapOnHostDataset) Name() string { return ds.ds.Name() }

// ShortName returns a short version of the dataset name, it implements train.HasShortName.
func (ds *ParallelMapOnHostDataset) ShortName() string {
	if sn, ok := ds.ds.(train.HasShortName); ok {
		return sn.ShortName()
	}
	name := ds.ds.Name()
	if len(name) > 3 {
		return name[:3]
	}
	return name
}

// WithPreserveOrder configures whether to preserve the original order of the batches.
// By default, it is enabled (true). Setting it to false allows out-of-order execution,
// which can improve performance/latency by avoiding re-ordering synchronization.
func (ds *ParallelMapOnHostDataset) WithPreserveOrder(preserveOrder bool) *ParallelMapOnHostDataset {
	ds.preserveOrder = preserveOrder
	return ds
}

// WithParallelism configures the number of parallel worker goroutines.
func (ds *ParallelMapOnHostDataset) WithParallelism(parallelism int) *ParallelMapOnHostDataset {
	if parallelism <= 0 {
		parallelism = runtime.NumCPU()
	}
	ds.parallelism = parallelism
	return ds
}

// Iter implements train.Dataset.
func (ds *ParallelMapOnHostDataset) Iter() iter.Seq2[train.Batch, error] {
	if !ds.preserveOrder {
		return parallelMapUnorderedImpl(ds.ds.Iter(), ds.parallelism, ds.mapFn)
	}
	return parallelMapImpl(ds.ds.Iter(), ds.parallelism, ds.mapFn)
}

// ParallelMapDataset implements a `train.Dataset` that maps a graph building function
// to a wrapped dataset in parallel.
type ParallelMapDataset struct {
	backend       compute.Backend
	store         *model.Store
	ds            train.Dataset
	mapFn         ModelMapFn
	parallelism   int
	preserveOrder bool

	mu        sync.RWMutex
	executors map[mapDatasetExecutorKey]*mapDatasetExecutor
}

var _ train.Dataset = (*ParallelMapDataset)(nil)

// ParallelMap returns a `train.Dataset` with the result of applying (mapping) the batches yielded by the provided
// `dataset` by the graph function `mapFn` in parallel.
//
// The default parallelism is the number of CPUs (runtime.NumCPU()), but it can be changed with WithParallelism(n).
//
// The order of the batches returned by the underlying dataset is preserved by default,
// unless WithPreserveOrder(false) is called on the result.
func ParallelMap(backend compute.Backend, dataset train.Dataset, mapFn MapFn) *ParallelMapDataset {
	return ParallelModelMap(backend, nil, dataset, func(_ *model.Scope, graphBatch GraphBatch) GraphBatch {
		return mapFn(graphBatch)
	})
}

// ParallelModelMap returns a `train.Dataset` with the result of applying (mapping) the batches yielded by the provided
// `dataset` by the graph function `mapFn` that can use trainable variables in the model store, executed in parallel.
//
// The default parallelism is the number of CPUs (runtime.NumCPU()), but it can be changed with WithParallelism(n).
//
// The order of the batches returned by the underlying dataset is preserved by default,
// unless WithPreserveOrder(false) is called on the result.
//
// If `store` is nil, a new one is created.
func ParallelModelMap(backend compute.Backend, store *model.Store, dataset train.Dataset, mapFn ModelMapFn) *ParallelMapDataset {
	if store == nil {
		store = model.NewStore()
	}
	return &ParallelMapDataset{
		backend:       backend,
		store:         store,
		ds:            dataset,
		mapFn:         mapFn,
		parallelism:   runtime.NumCPU(),
		preserveOrder: true,
		executors:     make(map[mapDatasetExecutorKey]*mapDatasetExecutor),
	}
}

// Name implements train.Dataset.
func (ds *ParallelMapDataset) Name() string { return ds.ds.Name() }

// ShortName returns a short version of the dataset name, it implements train.HasShortName.
func (ds *ParallelMapDataset) ShortName() string {
	if sn, ok := ds.ds.(train.HasShortName); ok {
		return sn.ShortName()
	}
	name := ds.ds.Name()
	if len(name) > 3 {
		return name[:3]
	}
	return name
}

// WithPreserveOrder configures whether to preserve the original order of the batches.
// By default, it is enabled (true). Setting it to false allows out-of-order execution,
// which can improve performance/latency by avoiding re-ordering synchronization.
func (ds *ParallelMapDataset) WithPreserveOrder(preserveOrder bool) *ParallelMapDataset {
	ds.preserveOrder = preserveOrder
	return ds
}

// WithParallelism configures the number of parallel worker goroutines.
func (ds *ParallelMapDataset) WithParallelism(parallelism int) *ParallelMapDataset {
	if parallelism <= 0 {
		parallelism = runtime.NumCPU()
	}
	ds.parallelism = parallelism
	return ds
}

// Iter implements train.Dataset.
func (ds *ParallelMapDataset) Iter() iter.Seq2[train.Batch, error] {
	mapBatch := func(batch train.Batch) (train.Batch, error) {
		inputs := batch.Inputs
		labels := batch.Labels

		key := mapDatasetExecutorKey{
			numInputs:  len(inputs),
			numLabels:  len(labels),
			specString: specToString(batch.Spec),
		}

		ds.mu.RLock()
		executor, found := ds.executors[key]
		ds.mu.RUnlock()

		if !found {
			ds.mu.Lock()
			// Double-check under write lock
			executor, found = ds.executors[key]
			if !found {
				executor = &mapDatasetExecutor{}
				executor.exec = model.MustNewExec(ds.backend, ds.store,
					func(scope *model.Scope, inputsAndLabels []*graph.Node) []*graph.Node {
						var inputs, labels []*graph.Node
						if key.numInputs > 0 {
							inputs = inputsAndLabels[:key.numInputs]
						}
						if key.numLabels > 0 {
							labels = inputsAndLabels[key.numInputs:]
						}

						graphBatch := GraphBatch{
							Inputs: inputs,
							Labels: labels,
							Spec:   batch.Spec,
						}
						mappedBatch := ds.mapFn(scope, graphBatch)
						executor.numMappedInputs = len(mappedBatch.Inputs)
						executor.numMappedLabels = len(mappedBatch.Labels)
						return append(mappedBatch.Inputs, mappedBatch.Labels...)
					})
				ds.executors[key] = executor
			}
			ds.mu.Unlock()
		}

		inputsAndLabels := append(inputs, labels...)
		inputsAndLabelsAny := make([]any, len(inputsAndLabels))
		var tensorsToFinalize []*tensors.Tensor
		for idx, t := range inputsAndLabels {
			if t.IsShared() {
				inputsAndLabelsAny[idx] = t
				tensorsToFinalize = append(tensorsToFinalize, t)
			} else {
				donated, err := graph.DonateTensorBuffer(t, ds.backend, 0)
				if err != nil {
					inputsAndLabelsAny[idx] = t
					tensorsToFinalize = append(tensorsToFinalize, t)
				} else {
					inputsAndLabelsAny[idx] = donated
				}
			}
		}

		var mappedInputsAndLabels []*tensors.Tensor
		var err error
		mappedInputsAndLabels, err = executor.exec.Call(inputsAndLabelsAny...)
		if err != nil {
			return train.Batch{}, errors.WithMessagef(err, "while executing MapFn provided for dataset.ParallelMap()")
		}

		// Finalize original tensors that were not donated.
		for _, t := range tensorsToFinalize {
			t.MustFinalizeAll()
		}

		var mappedInputs, mappedLabels []*tensors.Tensor
		if executor.numMappedInputs > 0 {
			mappedInputs = mappedInputsAndLabels[:executor.numMappedInputs]
		}
		if executor.numMappedLabels > 0 {
			mappedLabels = mappedInputsAndLabels[executor.numMappedInputs:]
		}

		mappedBatch := train.Batch{
			Spec:   batch.Spec,
			Inputs: mappedInputs,
			Labels: mappedLabels,
		}
		return mappedBatch, nil
	}

	if !ds.preserveOrder {
		return parallelMapUnorderedImpl(ds.ds.Iter(), ds.parallelism, mapBatch)
	}
	return parallelMapImpl(ds.ds.Iter(), ds.parallelism, mapBatch)
}

// parallelMapImpl maps an iterator `seq` of type `In` to type `Out` in parallel using the specified number
// of worker goroutines.
//
// The order of the items returned by the underlying iterator is preserved.
func parallelMapImpl[In, Out any](seq iter.Seq2[In, error], parallelism int, mapFn func(In) (Out, error)) iter.Seq2[Out, error] {
	if parallelism <= 0 {
		parallelism = runtime.NumCPU()
	}
	return func(yield func(Out, error) bool) {
		type result struct {
			val Out
			err error
		}
		type task struct {
			val     In
			resChan chan result
		}
		type pendingTask struct {
			resChan chan result
		}

		done := make(chan struct{})
		var wg sync.WaitGroup
		defer func() {
			close(done)
			wg.Wait()
		}()

		tasksChan := make(chan task, parallelism)
		pendingChan := make(chan pendingTask, parallelism*2)

		for range parallelism {
			wg.Go(func() {
				for {
					select {
					case <-done:
						return
					case t, ok := <-tasksChan:
						if !ok {
							return
						}
						select {
						case <-done:
							return
						default:
						}
						outVal, err := mapFn(t.val)
						res := result{val: outVal, err: err}
						select {
						case t.resChan <- res:
						case <-done:
							return
						}
					}
				}
			})
		}

		// Feed goroutine
		wg.Go(func() {
			defer close(tasksChan)
			defer close(pendingChan)
			for val, err := range seq {
				if err != nil {
					resChan := make(chan result, 1)
					resChan <- result{err: err}
					select {
					case pendingChan <- pendingTask{resChan: resChan}:
					case <-done:
					}
					return
				}
				resChan := make(chan result, 1)
				select {
				case tasksChan <- task{val: val, resChan: resChan}:
				case <-done:
					return
				}
				select {
				case pendingChan <- pendingTask{resChan: resChan}:
				case <-done:
					return
				}
			}
		})

		// Yield results in order
		for {
			select {
			case <-done:
				return
			case pt, ok := <-pendingChan:
				if !ok {
					return
				}
				select {
				case <-done:
					return
				case res := <-pt.resChan:
					if res.err != nil {
						var zero Out
						yield(zero, res.err)
						return
					}
					if !yield(res.val, nil) {
						return
					}
				}
			}
		}
	}
}

// parallelMapUnorderedImpl maps an iterator `seq` of type `In` to type `Out` in parallel using the specified number
// of worker goroutines, without preserving the original order of items.
func parallelMapUnorderedImpl[In, Out any](seq iter.Seq2[In, error], parallelism int, mapFn func(In) (Out, error)) iter.Seq2[Out, error] {
	if parallelism <= 0 {
		parallelism = runtime.NumCPU()
	}
	return func(yield func(Out, error) bool) {
		type result struct {
			val Out
			err error
		}

		done := make(chan struct{})
		var wg sync.WaitGroup
		defer func() {
			close(done)
			wg.Wait()
		}()

		tasksChan := make(chan In, parallelism)
		resultsChan := make(chan result, parallelism*2)

		var workersWg sync.WaitGroup
		for range parallelism {
			workersWg.Add(1)
			wg.Go(func() {
				defer workersWg.Done()
				for {
					select {
					case <-done:
						return
					case val, ok := <-tasksChan:
						if !ok {
							return
						}
						select {
						case <-done:
							return
						default:
						}
						outVal, err := mapFn(val)
						select {
						case <-done:
							return
						case resultsChan <- result{val: outVal, err: err}:
						}
					}
				}
			})
		}

		// Feed goroutine
		wg.Go(func() {
			defer close(tasksChan)
			for val, err := range seq {
				if err != nil {
					select {
					case <-done:
					case resultsChan <- result{err: err}:
					}
					return
				}
				select {
				case <-done:
					return
				case tasksChan <- val:
				}
			}
		})

		// Close resultsChan when all workers are done
		wg.Go(func() {
			workersWg.Wait()
			close(resultsChan)
		})

		// Yield results as they arrive
		for {
			select {
			case <-done:
				return
			case res, ok := <-resultsChan:
				if !ok {
					return
				}
				if res.err != nil {
					var zero Out
					yield(zero, res.err)
					return
				}
				if !yield(res.val, nil) {
					return
				}
			}
		}
	}
}
