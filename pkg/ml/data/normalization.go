/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package data

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/initializers"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/pkg/errors"
	"io"
)

// Normalization calculates the normalization parameters `mean` and `stddev` for the `inputsIndex`-th input
// from the given dataset.
//
// These values can later be used for normalization by simply applying `Div(Sub(x, mean), stddev)`. To use
// as side inputs in an ML model, just set them to variables.
//
// The parameter `independentAxes` list axes that should not be normalized together.
// A typical value is -1, the feature axis (last axis), so that each feature gets its own normalization.
//
// Notice for any feature that happens to be constant, the `stddev` will be 0. If trying to normalize (divide)
// by that will result in error. Use ReplaceZerosByOnes below to avoid the numeric issues.
func Normalization(backend backends.Backend, ds train.Dataset, inputsIndex int, independentAxes ...int) (mean, stddev *tensors.Tensor, err error) {
	ctx := context.New()
	updateValuesWithInput := context.MustNewExec(backend, ctx, func(ctx *context.Context, batch *Node) {
		g := batch.Graph()
		ctx = ctx.WithInitializer(initializers.Zero)

		// Find axes to reduce from the input.
		mapIndependentAxes := make([]bool, batch.Rank())
		for _, axis := range independentAxes {
			adjustedAxis := AdjustAxisToOperandRank(batch, axis)
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

		countVar := ctx.VariableWithValue("count", 0)
		countVar.SetValueGraph(AddScalar(countVar.ValueGraph(g), float64(reducedCount)))

		sumVar := ctx.VariableWithShape("sum", batchSum.Shape())
		sumVar.SetValueGraph(Add(sumVar.ValueGraph(g), batchSum))

		batchSum2 := ReduceAndKeep(Square(batch), ReduceSum, reduceAxes...)
		sumSquareVar := ctx.VariableWithShape("sum^2", batchSum2.Shape())
		sumSquareVar.SetValueGraph(Add(sumSquareVar.ValueGraph(g), batchSum2))
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
		results = context.MustNewExec(backend, ctx, func(ctx *context.Context, g *Graph) []*Node {
			countVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "count")
			count := countVar.ValueGraph(g)

			sumVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "sum")
			sum := sumVar.ValueGraph(g)

			sumSquareVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "sum^2")
			sumSquare := sumSquareVar.ValueGraph(g)

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
		return
	}
	mean, stddev = results[0], results[1]
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
