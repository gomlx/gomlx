// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dataset

import (
	"fmt"
	"iter"

	"github.com/gomlx/compute"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/pkg/errors"
)

// GraphBatch represents a batch of nodes in the graph building phase.
type GraphBatch struct {
	Inputs, Labels []*Node
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
					func(scope *model.Scope, inputsAndLabels []*Node) []*Node {
						var inputs, labels []*Node
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
					donated, err := DonateTensorBuffer(t, mapDS.backend, 0)
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
			if !yield(mappedBatch, nil) {
				return
			}
		}
	}
}
