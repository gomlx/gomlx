// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package datasets

import (
	"io"

	"github.com/gomlx/compute"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/initializer"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/support/exceptions"
	"github.com/pkg/errors"
)

// Normalization calculates the normalization parameters `mean` and `stddev` for the `inputsIndex`-th input
// from the given dataset.
//
// These values can later be used for normalization by simply applying `Div(Sub(x, mean), stddev)`.
// To use as side inputs in an ML model, set them to variables.
//
// The parameter `independentAxes` list axes that should not be normalized together.
// A typical value is -1, the feature axis (last axis), so that each feature gets its own normalization.
//
// Notice for any feature that happens to be constant, the `stddev` will be 0. If trying to normalize (divide)
// by that will result in error. Use ReplaceZerosByOnes below to avoid the numeric issues.
func Normalization(backend compute.Backend, ds train.Dataset, inputsIndex int, independentAxes ...int) (mean, stddev *tensors.Tensor, err error) {
	store := model.NewStore()
	updateValuesWithInput := model.MustNewExec(backend, store, func(scope *model.Scope, batch *Node) {
		g := batch.Graph()
		scope = scope.WithInitializer(initializer.Zero)

		// Find axes to reduce from the input.
		mapIndependentAxes := make([]bool, batch.Rank())
		for _, axis := range independentAxes {
			adjustedAxis := MustAdjustAxis(axis, batch)
			mapIndependentAxes[adjustedAxis] = true
		}
		reduceAxes := make([]int, 0, batch.Rank()-len(independentAxes))
		for axis, independent := range mapIndependentAxes {
			if !independent {
				reduceAxes = append(reduceAxes, axis)
			}
		}

		// Parameters of batch.
		batchSum := ReduceAndKeep(batch, ReduceSum, reduceAxes...)
		reducedCount := batch.Shape().Size() / batchSum.Shape().Size()

		countVar := scope.VariableWithValue("count", 0.0)
		countVar.SetNodeValue(AddScalar(countVar.NodeValue(g), float64(reducedCount)))

		sumVar := scope.VariableWithShape("sum", batchSum.Shape())
		sumVar.SetNodeValue(Add(sumVar.NodeValue(g), batchSum))

		batchSum2 := ReduceAndKeep(Square(batch), ReduceSum, reduceAxes...)
		sumSquareVar := scope.VariableWithShape("sum^2", batchSum2.Shape())
		sumSquareVar.SetNodeValue(Add(sumSquareVar.NodeValue(g), batchSum2))
	})

	// Read through dataset updating measurements.
	batchNum := 0
	var inputs []*tensors.Tensor
	for {
		_, inputs, _, err = ds.Yield()
		if err == io.EOF {
			break
		}
		if err != nil {
			err = errors.WithMessagef(err, "while reading batch #%d of the dataset", batchNum)
			return
		}
		if inputsIndex >= len(inputs) {
			err = errors.Errorf("asked for inputsIndex=%d, but inputs has only %d elements",
				inputsIndex, len(inputs))
			return
		}
		batch := inputs[inputsIndex]
		if !batch.DType().IsFloat() {
			err = errors.Errorf("dataset input %d has invalid dtype (shape=%s): Normalization() only accepts float values.",
				inputsIndex, batch.Shape())
			return
		}
		err = exceptions.TryCatch[error](func() { updateValuesWithInput.MustExec(batch) })
		if err != nil {
			err = errors.WithMessagef(err, "while processing batch #%d of the dataset", batchNum)
			return
		}
	}

	// Calculate mean and stddev, using a graph.
	var results []*tensors.Tensor
	err = exceptions.TryCatch[error](func() {
		results = model.MustNewExec(backend, store, func(scope *model.Scope, g *Graph) []*Node {
			countVar := scope.GetVariable("count")
			count := countVar.NodeValue(g)

			sumVar := scope.GetVariable("sum")
			sum := sumVar.NodeValue(g)

			sumSquareVar := scope.GetVariable("sum^2")
			sumSquare := sumSquareVar.NodeValue(g)

			count = ConvertDType(count, sum.DType())
			mean := Div(sum, count)
			variance := Sub(
				Div(sumSquare, count),
				Square(mean))
			stddev := Sqrt(variance)
			return []*Node{mean, stddev}
		}).MustExec()
	})
	if err != nil {
		err = errors.WithMessagef(err, "while calculating the final mean/stddev from accumulated batch statistics")
		return
	}
	mean = results[0]
	stddev = results[1]
	return
}

// ReplaceZerosByOnes replaces any zero values in x by one.
// This is useful if normalizing a value with a standard deviation
// (`stddev`) that has zeros.
func ReplaceZerosByOnes(x *Node) *Node {
	g := x.Graph()
	return Where(
		Equal(x, ScalarZero(g, x.DType())),
		OnesLike(x),
		x)
}
