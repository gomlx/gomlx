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
	"log"

	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	. "github.com/gomlx/gomlx/types/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
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
}

var (
	// KnownOptimizers is a map of known optimizers by name to their default constructors.
	// This provides an easy quick start point. One can hyperparameter-tune the optimizers
	// for usually slightly better results.
	KnownOptimizers = map[string]func() Interface{
		"sgd":    StochasticGradientDescent,
		"adam":   func() Interface { return Adam().Done() },
		"adamax": func() Interface { return Adam().Adamax().Done() },
		"adamw":  func() Interface { return Adam().WeightDecay(0.004).Done() },
	}
)

// GlobalStepVariableName as stored in context.Context, usually in the root scope -- but depends on the
// caller.
const GlobalStepVariableName = "global_step"

// MustOptimizerByName returns an optimizer given the name, or log.Fatal if one does not exist. It uses
// KnownOptimizers -- in case one wants to better handle invalid values.
//
// Example usage:
//
// ```
// var flagOptimizer = flag.String("optimizer", "adamw", fmt.Sprintf("Optimizer, options: %q", types.SortedKeys(optimizers.KnownOptimizers)))
//
// ...
//
//	trainer := train.NewTrainer(manager, ctx, ModelGraph,
//	   losses.SomeLoss,
//	   optimizers.MustOptimizerByName(*flagOptimizer),
//	   []metrics.Interface{someMetric},    // trainMetrics
//	   []metrics.Interface{otherMetric})   // evalMetrics
//
// ```
func MustOptimizerByName(optName string) Interface {
	optBuilder, found := KnownOptimizers[optName]
	if !found {
		log.Fatalf("Unknown optimizer %q, valid values are %v.", optName, slices.Keys(KnownOptimizers))
	}
	return optBuilder()
}

// GetGlobalStepVar returns the global step counter.
// It creates it (initialized with 0) if not already there.
// This can be used in graph building or directly.
func GetGlobalStepVar(ctx *context.Context) *context.Variable {
	return ctx.Checked(false).VariableWithValue(GlobalStepVariableName, 0).SetTrainable(false)
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

// IncrementGlobalStepGraph creates (if not there yet) a global step counter, and
// returns it incremented -- its first returned value will be 1.
//
// It only builds the computation graph, no actual values are generated.
//
// Typically, this is called by the optimizers UpdateGraph method.
//
// GlobalStep is always stored as shapes.Int64, but it is converted to the given DType
// before being returned.
func IncrementGlobalStepGraph(ctx *context.Context, g *Graph, dtype shapes.DType) *Node {
	globalStepVar := GetGlobalStepVar(ctx)
	globalStep := globalStepVar.ValueGraph(g)
	globalStep = Add(globalStep, OnesLike(globalStep))
	globalStepVar.SetValueGraph(globalStep)
	if dtype != shapes.I64 {
		globalStep = ConvertType(globalStep, dtype)
	}
	return globalStep
}

// LearningRateKey is the string key for learning rate in Context.Params.
const LearningRateKey = "learning_rate"

// LearningRateVar returns the learning rate variable -- a scalar value of the given dtype.
//
// If variable doesn't exist yet, it will be created using the parameter LearningRateKey, if it
// is set, or the provided defaultValue (must be a scalar convertible to dtype) if not.
func LearningRateVar(ctx *context.Context, dtype shapes.DType, defaultValue float64) *context.Variable {
	lrValue := context.GetParam(ctx, LearningRateKey, defaultValue)
	return LearningRateVarWithValue(ctx, dtype, lrValue)
}

// LearningRateVarWithValue creates (or reuses) variable for learning rate with the given value.
func LearningRateVarWithValue(ctx *context.Context, dtype shapes.DType, value float64) *context.Variable {
	ctx = ctx.Checked(false).In("optimizers")
	return ctx.VariableWithValue(LearningRateKey, shapes.CastAsDType(value, dtype))
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

// addGradientsToVariablesGraph takes the output of Context.BuildTrainableVariablesGradientsGraph,
// multiply by (-learningRate) and add to the current value of the variablesMap.
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
			lrCast = ConvertType(learningRate, grads[ii].DType())
		}
		updatedValue := Sub(v.ValueGraph(g), Mul(grads[ii], lrCast))
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
