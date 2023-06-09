package data

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/pkg/errors"
)

// MapGraphFn if a graph building function that transforms inputs and labels.
type MapGraphFn func(ctx *context.Context, inputs, labels []*Node) (mappedInputs, mappedLabels []*Node)

// mapDataset implements a `train.Dataset` that maps a function to a wrapped dataset.
type mapDataset struct {
	manager                          *Manager
	ctx                              *context.Context
	ds                               train.Dataset
	mapGraphFn                       MapGraphFn
	mapGraphFnExec                   *context.Exec
	numInputs, numLabels             int
	numMappedInputs, numMappedLabels int
}

// Map returns a `train.Dataset` with the result of applying (mapping) the batches yielded by the provided `dataset`
// by the graph function `fn`.
// The function is executed by the `manager` given.
// If `ctx` is nil, a new one is created.
//
// The graph building function `fn` can return a different number of `inputs` or `labels` than what it was given.
// But these numbers should never change -- always return the same number of inputs and labels.
func Map(manager *Manager, ctx *context.Context, dataset train.Dataset, fn MapGraphFn) train.Dataset {
	mapDS := &mapDataset{
		manager:    manager,
		ctx:        ctx,
		ds:         dataset,
		mapGraphFn: fn,
	}
	if mapDS.ctx == nil {
		mapDS.ctx = context.NewContext(manager)
	}
	return mapDS
}

// Reset implements train.Dataset.
func (mapDS *mapDataset) Reset() {
	mapDS.ds.Reset()
	mapDS.ds.Name()
}

// Name implements train.Dataset.
func (mapDS *mapDataset) Name() string {
	return mapDS.ds.Name() + " [Map]"
}

// Yield implements train.Dataset.
func (mapDS *mapDataset) Yield() (spec any, inputs []tensor.Tensor, labels []tensor.Tensor, err error) {
	spec, inputs, labels, err = mapDS.ds.Yield()
	if err != nil {
		return
	}

	if len(inputs) != mapDS.numInputs || len(labels) != mapDS.numLabels || mapDS.mapGraphFnExec == nil {
		// Build execution of MapGraphFn
		mapDS.numInputs = len(inputs)
		mapDS.numLabels = len(labels)
		mapDS.mapGraphFnExec = context.NewExec(mapDS.manager, mapDS.ctx,
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
	inputsAndLabels, err = mapDS.mapGraphFnExec.Call(
		// We have to map inputsAndLabels to an `[]any` slice.
		slices.Map(inputsAndLabels, func(e tensor.Tensor) any { return e })...)
	if err != nil {
		err = errors.WithMessagef(err, "while executing MapGraphFn provided for data.Map()")
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
