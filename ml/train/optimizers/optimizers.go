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

// Package optimizers implements a collection of ML optimizers, that can be used by train.Trainer,
// or by themselves. They all implement optimizers.Interface.
package optimizers

import (
	. "github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"golang.org/x/exp/maps"
)

// Interface implemented by optimizer implementations.
type Interface interface {
	// UpdateGraph is the function called during computation graph building, it
	// calculates the updates to the variables (weights) of the model needed for one
	// training step.
	// It should return these updates.
	//
	// Variable values can be updated in graph building time (inside UpdateGraph) using Variable.SetValueGraph,
	// and the trainer (train.Trainer) will make sure these values are returned from the graph execution
	// and the materialized values used to update the variables (Variable.SetValue).
	//
	// ctx holds the variables to train (marked as trainable), the hyperparameters
	// used by the optimizer (in `ctx.Params`) and non-trainable variables
	// that the optimizer itself may create. One should scope it (context.Context.In("<some scope name>"))
	// to avoid naming conflicts on the variables created -- notice that
	// some complex training scheduling scheme may have more than one optimizer
	// on the same Context object.
	//
	// loss must be a scalar value.
	UpdateGraph(ctx *context.Context, g *Graph, loss *Node)

	// Clear deletes all temporary variables used by the optimizer.
	// This may be used for a model to be used by inference to save space, or if the training should be reset
	// for some other reason.
	Clear(ctx *context.Context)
}

var (
	// KnownOptimizers is a map of known optimizers by name to their default constructors.
	// This provides an easy quick start point. One can hyperparameter-tune the optimizers
	// for usually slightly better results.
	KnownOptimizers = map[string]func(ctx *context.Context) Interface{
		"sgd":    func(ctx *context.Context) Interface { return StochasticGradientDescent() },
		"adam":   func(ctx *context.Context) Interface { return Adam().FromContext(ctx).Done() },
		"adamax": func(ctx *context.Context) Interface { return Adam().Adamax().FromContext(ctx).Done() },
		"adamw":  func(ctx *context.Context) Interface { return Adam().WeightDecay(0.004).FromContext(ctx).Done() },
	}

	// ParamOptimizer is the context parameter with the name of the optimizer.
	// The default value is "adamw", and the valid values are "sgd", "adam", "adamw" and "adamax".
	ParamOptimizer = "optimizer"

	// ParamLearningRate is the context parameter name for the default value of learning rate.
	// It is used by most (all?) optimizers.
	ParamLearningRate = "learning_rate"

	// LearningRateKey is an alias to ParamLearningRate
	//
	// Deprecated: use ParamLearningRate instead.
	LearningRateKey = ParamLearningRate

	// ParamClipStepByValue is a clip scalar value for each individual value of the gradient step, after
	// being scaled by the learning rate and optimizer.
	// The step applied will be `ClipScalar(step, -clip_step_by_value, +clip_step_by_value)`.
	// Defaults to no clipping, and values are expected to be float64.
	ParamClipStepByValue = "clip_step_by_value"

	// ParamClipNaN will drop any updates to variables that will lead to NaN.
	// This is a double-edged option: it keeps training running, but probably will replace NaN by bad training.
	//
	// Default  is false.
	ParamClipNaN = "clip_nan"
)

const (
	// GlobalStepVariableName as stored in context.Context, usually in the root scope -- but depends on the
	// caller.
	GlobalStepVariableName = "global_step"

	// Scope reserved for optimizers.
	Scope = "optimizers"
)

// FromContext creates an optimizer from context hyperparameters.
// See [ParamOptimizer]. The default is "adamw".
func FromContext(ctx *context.Context) Interface {
	optName := context.GetParamOr(ctx, ParamOptimizer, "adamw")
	return ByName(ctx, optName)
}

// ByName returns an optimizer given the name, or panics if one does not exist.
// It uses  KnownOptimizers -- in case one wants to better handle invalid values.
//
// Some optimizers (e.g.: Adam) uses optional hyperparameters set in the context for configuration.
//
// See also FromContext.
//
// Example usage:
//
// ```
// var flagOptimizer = flag.String("optimizer", "adamw", fmt.Sprintf("Optimizer, options: %q", maps.Keys(optimizers.KnownOptimizers)))
//
// ...
//
//	trainer := train.NewTrainer(manager, ctx, ModelGraph,
//	   losses.SomeLoss,
//	   optimizers.ByName(ctx, *flagOptimizer),
//	   []metrics.Interface{someMetric},    // trainMetrics
//	   []metrics.Interface{otherMetric})   // evalMetrics
//
// ```
func ByName(ctx *context.Context, optName string) Interface {
	optBuilder, found := KnownOptimizers[optName]
	if !found {
		Panicf("Unknown optimizer %q, valid values are %v.", optName, maps.Keys(KnownOptimizers))
	}
	return optBuilder(ctx)
}

// GetGlobalStepVar returns the global step counter, a dtypes.Int64 variable.
// It creates it (initialized with 0) if not already there.
// This can be used in graph building or directly.
func GetGlobalStepVar(ctx *context.Context) *context.Variable {
	return ctx.Checked(false).VariableWithValue(GlobalStepVariableName, int64(0)).SetTrainable(false)
}

// GetGlobalStep returns the current global step value.
// It creates the global step variable if it does not yet exist.
func GetGlobalStep(ctx *context.Context) int64 {
	vAny := GetGlobalStepVar(ctx).Value().Value()
	v, ok := vAny.(int64)
	if !ok {
		Panicf("Context(scope=%q)[%q]=%#v, and cannot be converted to int64", ctx.Scope(), GlobalStepVariableName, vAny)
	}
	return v
}

// DeleteGlobalStep in case one wants to reset the model state, or hide how many steps were taken.
func DeleteGlobalStep(ctx *context.Context) {
	ctx.DeleteVariable(ctx.Scope(), GlobalStepVariableName)
}

// IncrementGlobalStepGraph creates (if not there yet) a global step counter, and
// returns it incremented -- its first returned value will be 1.
//
// It only builds the computation graph, no actual values are generated.
//
// Typically, this is called by the optimizers UpdateGraph method.
//
// GlobalStep is always stored as dtypes.Int64, but it is converted to the given DType
// before being returned.
func IncrementGlobalStepGraph(ctx *context.Context, g *Graph, dtype dtypes.DType) *Node {
	globalStepVar := GetGlobalStepVar(ctx)
	globalStep := globalStepVar.ValueGraph(g)
	globalStep = Add(globalStep, OnesLike(globalStep))
	globalStepVar.SetValueGraph(globalStep)
	if dtype != dtypes.Int64 {
		globalStep = ConvertDType(globalStep, dtype)
	}
	return globalStep
}

// LearningRateVar returns the learning rate variable -- a scalar value of the given dtype.
//
// If variable doesn't exist yet, it will be created using the parameter ParamLearningRate, if it
// is set, or the provided defaultValue (must be a scalar convertible to dtype) if not.
func LearningRateVar(ctx *context.Context, dtype dtypes.DType, defaultValue float64) *context.Variable {
	lrValue := context.GetParamOr(ctx, ParamLearningRate, defaultValue)
	return LearningRateVarWithValue(ctx, dtype, lrValue)
}

// LearningRateVarWithValue creates (or reuses) variable for learning rate with the given value.
func LearningRateVarWithValue(ctx *context.Context, dtype dtypes.DType, value float64) *context.Variable {
	ctx = ctx.Checked(false).In(Scope)
	return ctx.VariableWithValue(ParamLearningRate, shapes.CastAsDType(value, dtype)).SetTrainable(false)
}

// ClipStepByValue applies the [ParamClipStepByValue] hyperparameter if it is not 0.0 (the default).
func ClipStepByValue(ctx *context.Context, step *Node) *Node {
	clipByValue := context.GetParamOr(ctx, ParamClipStepByValue, 0.0)
	if clipByValue == 0 {
		return step
	}
	return ClipScalar(step, -clipByValue, clipByValue)
}

// ClipNaNsInGradients will replace the gradient tensor by zeros if there are any NaNs or +/-Inf values.
// It is only enabled if ParamClipNaN is set to true.
//
// See also ClipNaNsInUpdates.
func ClipNaNsInGradients(ctx *context.Context, gradients *Node) *Node {
	if !context.GetParamOr(ctx, ParamClipNaN, false) {
		return gradients
	}
	return Where(LogicalAll(IsFinite(gradients)), gradients, ZerosLike(gradients))
}

// ClipNaNsInUpdates will replace original values into updates, where updates have NaN (or +/-Inf) values,
// if the ParamClipNaN is set to true.
//
// See also ClipNaNsInGradients.
func ClipNaNsInUpdates(ctx *context.Context, original, updates *Node) *Node {
	if !context.GetParamOr(ctx, ParamClipNaN, false) {
		return updates
	}
	return Where(IsFinite(updates), updates, original)
}

// sgd is an empty struct that implements Interface for SGD.
type sgd struct{}

// SgdDefaultLearningRate is the default learning rate used by the StochasticGradientDescent optimizer.
const SgdDefaultLearningRate = 0.1

// StochasticGradientDescent creates an optimizer that performs SGD.
// It looks for "learning_rate" in Context.Params for the initial
// learning rate, otherwise it defaults to SgdDefaultLearningRate.
//
// It has a decay of learning rate given by: `learning_rate = initial_learning_rate / Sqrt(global_step)`
func StochasticGradientDescent() Interface {
	return &sgd{}
}

// UpdateGraph builds the graph to update the weights for one training step.
// It implements optimizers.Interface.
func (sgd *sgd) UpdateGraph(ctx *context.Context, g *Graph, loss *Node) {
	if !loss.Shape().IsScalar() {
		Panicf("optimizer requires a scalar loss to optimize, got loss.shape=%s instead", loss.Shape())
	}
	dtype := loss.DType()
	lrVar := LearningRateVar(ctx, dtype, SgdDefaultLearningRate)
	learningRate := lrVar.ValueGraph(g)
	globalStep := IncrementGlobalStepGraph(ctx, g, dtype)
	learningRate = Div(learningRate, Sqrt(globalStep)) // Factor global_step into the learning rate.
	addGradientsToVariablesGraph(ctx, loss, learningRate, globalStep)
	return
}

// Clear all optimizer variables.
// There are none for SGD, so this is a non-op.
// It implements optimizers.Interface.
func (sgd *sgd) Clear(_ *context.Context) {}

// addGradientsToVariablesGraph takes the output of Context.BuildTrainableVariablesGradientsGraph,
// multiply by (-learningRate) and add to the current value of the variablesMap.
//
// It replaces NaNs with zero.
func addGradientsToVariablesGraph(ctx *context.Context, loss, learningRate, globalStep *Node) {
	g := loss.Graph()
	if !learningRate.Shape().IsScalar() {
		Panicf("Context.addGradientsToVariablesGraph require scalar learningRate, instead got %s", learningRate.Shape())
	}
	grads := ctx.BuildTrainableVariablesGradientsGraph(loss)
	if len(grads) == 0 {
		return
	}
	numTrainable := len(grads)
	ii := 0
	ctx.EnumerateVariables(func(v *context.Variable) {
		if !v.Trainable || !v.InUseByGraph(g) {
			// Not interested in this variable.
			return
		}
		lrCast := learningRate
		if lrCast.DType() != grads[ii].DType() {
			// Some variables may not be of the same DType as the learning rate (which has the same DType
			// as the loss), so we need to cast it.
			// Two common reasons: variables can have different resolution (to save space); variables could be
			// complex.
			lrCast = ConvertDType(learningRate, grads[ii].DType())
		}
		scaledGradient := Mul(grads[ii], lrCast)
		scaledGradient = ClipStepByValue(ctx, scaledGradient)
		vNode := v.ValueGraph(g)
		updatedValue := Sub(vNode, scaledGradient)
		updatedValue = ClipNaNsInUpdates(ctx, vNode, updatedValue)
		v.SetValueGraph(updatedValue)
		ii++
	})
	if ii != numTrainable {
		Panicf("number of trainable variables for BuildTrainableVariablesGradientsGraph (%d) and addGradientsToVariablesGraph (%d) "+
			"are different -- did new trainable variables were created or variables `.Trainable` property "+
			"changed in between?", numTrainable, ii)
	}
	return
}

// MonotonicProjection transforms the input into a monotonic sequence on the given axis that respects the
// minimum margin between consecutive points.
//
// Here we call "viable" solution, one that respects the given margin between consecutive points. And the goal
// is to find the viable solution that is L2-closest to the original input -- we don't achieve that, but some
// approximate that is hopefully good enough for most algorithms.
//
// This is not a trivial problem, as adjustments to one point may break the monotonicity of the next, and so on.
// A close to optimal approximate solution can be achieved using lagrange multipliers (and Dykstra alternate
// projections), see implementation in TensorFlow Lattice:
// https://github.com/tensorflow/lattice/blob/master/tensorflow_lattice/python/pwl_calibration_lib.py#L472
//
// Unfortunately, GoMLX doesn't support "while" loops in the computation graph yet, so instead we make
// a coarse but simple projection to the viable space using a simple algorithm -- see code.
//
// The usual way to use this is inside a call to train.AddPerStepUpdateGraphFn, making the projection happen after
// the gradient step.
func MonotonicProjection(input *Node, margin *Node, axis int) *Node {
	adjustedAxis := AdjustAxisToOperandRank(input, axis)
	axisDim := input.Shape().Dim(axis)
	if axisDim < 2 {
		Panicf("MonotonicProjection of input shaped %s at axis %d is not valid: it requires axis to have dimension >= 2",
			input.Shape(), axis)
	}

	const numIter = 3
	// Fix to the right: increasing values.
	diffRight := ConsecutiveDifference(input, adjustedAxis, false)
	// For a fixed number of times try to prevent everything to be pushed if possible.
	if axisDim > 2 {
		for range numIter {
			adjustedDiff := Max(diffRight, margin) // Pushes everything to the right, whenever monotonicity is broken.
			adjustment := Sub(diffRight, adjustedDiff)
			fixedAdjustment := ShiftWithScalar(adjustment, adjustedAxis, ShiftDirRight, 1, 0.0)
			diffRight = Add(adjustedDiff, fixedAdjustment)
		}
	}
	diffRight = Max(diffRight, margin) // Make sure its valid, if numIter wasn't enough.
	leftMostInput := SliceAxis(input, adjustedAxis, AxisElem(0))
	diffRight = Concatenate([]*Node{leftMostInput, diffRight}, adjustedAxis)
	fixRight := CumSum(diffRight, adjustedAxis)

	// Fix to the left: increasing values.
	diffLeft := ConsecutiveDifference(input, adjustedAxis, false)
	initialTotalDiff := ReduceAndKeep(diffLeft, ReduceSum, adjustedAxis)

	// For a fixed number of times try to prevent everything to be pushed if possible.
	if axisDim > 2 {
		for range numIter {
			adjustedDiff := Max(diffLeft, margin) // Pushes everything to the left, whenever monotonicity is broken.
			adjustment := Sub(diffLeft, adjustedDiff)
			fixedAdjustment := ShiftWithScalar(adjustment, adjustedAxis, ShiftDirLeft, 1, 0.0)
			diffLeft = Add(adjustedDiff, fixedAdjustment)
		}
	}
	diffLeft = Max(diffLeft, margin) // Make sure its valid, if numIter wasn't enough.
	finalTotalDiff := ReduceAndKeep(diffLeft, ReduceSum, adjustedAxis)

	leftMostInput = SliceAxis(input, adjustedAxis, AxisElem(0))
	leftMostInput = Sub(leftMostInput,
		Sub(finalTotalDiff, initialTotalDiff))

	diffLeft = Concatenate([]*Node{leftMostInput, diffLeft}, adjustedAxis)
	fixLeft := CumSum(diffLeft, adjustedAxis)

	// Reconstruct value.
	return DivScalar(Add(fixRight, fixLeft), 2)
}
