package data

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/pkg/errors"
)

// MapGraphFn if a graph building function that transforms inputs and labels.
type MapGraphFn func(ctx *context.Context, inputs, labels []*Node) (mappedInputs, mappedLabels []*Node)

// mapGraphFnDataset implements a `train.Dataset` that maps a graph building function to a wrapped dataset.
// See [MapWithGraphFn] on how to use it.
type mapGraphFnDataset struct {
	backend                          backends.Backend
	ctx                              *context.Context
	ds                               train.Dataset
	mapGraphFn                       MapGraphFn
	mapGraphFnExec                   *context.Exec
	numInputs, numLabels             int
	numMappedInputs, numMappedLabels int
}

// MapWithGraphFn returns a `train.Dataset` with the result of applying (mapping) the batches yielded by the provided
// `dataset` by the graph function `graphFn`.
// The function is executed by the `backend` given.
// If `ctx` is nil, a new one is created.
//
// The graph building function `graphFn` can return a different number of `inputs` or `labels` than what it was given,
// but these numbers should never change -- always return the same number of inputs and labels.
func MapWithGraphFn(backend backends.Backend, ctx *context.Context, dataset train.Dataset, graphFn MapGraphFn) train.Dataset {
	mapDS := &mapGraphFnDataset{
		backend:    backend,
		ctx:        ctx,
		ds:         dataset,
		mapGraphFn: graphFn,
	}
	if mapDS.ctx == nil {
		mapDS.ctx = context.New()
	}
	return mapDS
}

// Reset implements train.Dataset.
func (mapDS *mapGraphFnDataset) Reset() {
	mapDS.ds.Reset()
	mapDS.ds.Name()
}

// Name implements train.Dataset.
func (mapDS *mapGraphFnDataset) Name() string {
	return mapDS.ds.Name()
}

// Yield implements train.Dataset.
func (mapDS *mapGraphFnDataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	spec, inputs, labels, err = mapDS.ds.Yield()
	if err != nil {
		return
	}

	if len(inputs) != mapDS.numInputs || len(labels) != mapDS.numLabels || mapDS.mapGraphFnExec == nil {
		// Build execution of MapGraphFn
		mapDS.numInputs = len(inputs)
		mapDS.numLabels = len(labels)
		mapDS.mapGraphFnExec = context.MustNewExec(mapDS.backend, mapDS.ctx,
			func(ctx *context.Context, inputsAndLabels []*Node) []*Node {
				var inputs, labels []*Node
				if mapDS.numInputs > 0 {
					inputs = inputsAndLabels[:mapDS.numInputs]
				}
				if mapDS.numLabels > 0 {
					labels = inputsAndLabels[mapDS.numInputs:]
				}
				mappedInputs, mappedLabels := mapDS.mapGraphFn(ctx, inputs, labels)
				mapDS.numMappedInputs = len(mappedInputs)
				mapDS.numMappedLabels = len(mappedLabels)
				return append(mappedInputs, mappedLabels...)
			})
	}

	inputsAndLabels := append(inputs, labels...)
	err = exceptions.TryCatch[error](func() {
		inputsAndLabels = mapDS.mapGraphFnExec.Call(
			// We have to map inputsAndLabels to an `[]any` slice.
			xslices.Map(inputsAndLabels, func(e *tensors.Tensor) any { return e })...)
	})
	if err != nil {
		err = errors.WithMessagef(err, "while executing MapGraphFn provided for data.MapWithGraphFn()")
		return
	}
	inputs, labels = nil, nil
	if mapDS.numMappedInputs > 0 {
		inputs = inputsAndLabels[:mapDS.numMappedInputs]
	}
	if mapDS.numMappedLabels > 0 {
		labels = inputsAndLabels[mapDS.numMappedInputs:]
	}
	return
}

// MapExampleFn if normal Go function that applies a transformation to the inputs/labels of a dataset.
type MapExampleFn func(inputs, labels []*tensors.Tensor) (mappedInputs, mappedLabels []*tensors.Tensor)

// mapDataset implements a `train.Dataset` that maps a function executed on the host to a wrapped dataset.
type mapDataset struct {
	ds    train.Dataset
	mapFn MapExampleFn
}

// Check that mapGraphFnDataset implements train.Dataset.
var _ train.Dataset = (*mapDataset)(nil)

// Map maps a dataset through a transformation with a (normal Go) function that runs in the host cpu.
//
// See [MapWithGraphFn] for a function that runs on the accelerator, with a graph building function.
func Map(ds train.Dataset, mapFn MapExampleFn) train.Dataset {
	return &mapDataset{
		ds:    ds,
		mapFn: mapFn,
	}
}

// Name implements train.Dataset.
func (ds *mapDataset) Name() string { return ds.ds.Name() }

// Yield implements train.Dataset.
func (ds *mapDataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	spec, inputs, labels, err = ds.ds.Yield()
	if err != nil {
		return
	}
	inputs, labels = ds.mapFn(inputs, labels)
	return
}

// Reset implements train.Dataset.
func (ds *mapDataset) Reset() {
	ds.ds.Reset()
}
