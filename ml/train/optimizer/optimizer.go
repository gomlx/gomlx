// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package optimizer implements a collection of ML optimizers that can be used by train.Trainer,
// or by themselves. They all implement optimizer.Interface.
package optimizer

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/initializer"
	. "github.com/gomlx/gomlx/support/exceptions"
)

// Interface implemented by optimizer implementations.
//
// Optionally, an optimizer may also implement the interface trainer.OptimizerWithGradients to allow for
// updates from accumulated gradients.
type Interface interface {
	// UpdateGraph is the function called during computation graph building, it
	// calculates the updates to the variables (weights) of the model needed for one
	// training step.
	// It should return these updates.
	//
	// Variable values can be updated in graph building time (inside UpdateGraph) using Variable.SetNodeValue,
	// and the trainer (train.Trainer) will make sure these values are returned from the graph execution
	// and the materialized values used to update the variables (Variable.SetValue).
	//
	// The scope holds the variables to train (marked as trainable), the hyperparameters
	// used by the optimizer (in `scope.Store().GetParam()`) and non-trainable variables
	// that the optimizer itself may create. One should scope it (model.Scope.In("<some scope name>"))
	// to avoid naming conflicts on the variables created -- notice that
	// some complex training schedule may have more than one optimizer on the same Store object.
	//
	// The loss must be a scalar value.
	//
	// This is a graph building function and panics on error.
	UpdateGraph(scope *model.Scope, g *Graph, theLoss *Node)

	// Clear deletes all temporary variables used by the optimizer.
	// This may be used for a model to be used by inference to save space, or if the training should be reset
	// for some other reason.
	Clear(scope *model.Scope) error
}

var (
	// KnownOptimizers is a map of known optimizers by name to their default constructors.
	// This provides an easy quick start point. One can hyperparameter-tune the optimizers
	// for usually slightly better results.
	KnownOptimizers = map[string]func(scope *model.Scope) Interface{
		"sgd":     func(scope *model.Scope) Interface { return StochasticGradientDescent() },
		"adam":    func(scope *model.Scope) Interface { return Adam().FromScope(scope).Done() },
		"adamax":  func(scope *model.Scope) Interface { return Adam().Adamax().FromScope(scope).Done() },
		"adamw":   func(scope *model.Scope) Interface { return Adam().WeightDecay(0.004).FromScope(scope).Done() },
		"rmsprop": func(scope *model.Scope) Interface { return RMSProp().FromScope(scope).Done() },
	}

	// ParamOptimizer is the parameter name (in the [model.Store]) for the name of the optimizer to use.
	// The default value is "adamw", and the valid values are "sgd", "adam", "adamw" and "adamax".
	ParamOptimizer = "optimizer"

	// ParamLearningRate is the parameter name for the default value of learning rate.
	// It is used by most (all?) optimizer.
	ParamLearningRate = "learning_rate"

	// LearningRateKey is an alias to ParamLearningRate
	//
	// Deprecated: use ParamLearningRate instead.
	LearningRateKey = ParamLearningRate

	// ParamClipStepByValue is a scalar value used to clip each value of the gradient step, after
	// being scaled by the learning rate and the optimizer.
	// The step applied will be `ClipScalar(step, -clip_step_by_value, +clip_step_by_value)`.
	// Defaults to no clipping, and values are expected to be float64.
	ParamClipStepByValue = "clip_step_by_value"

	// ParamClipNaN will drop any updates with NaNs.
	// This is a double-edged option: it keeps training running, but probably it will replace NaNs with bad training results.
	// It works well to handle spurious results.
	//
	// See also ParamNanLogger to help debug it.
	//
	// The default is false.
	ParamClipNaN = "clip_nan"

	// ParamNanLogger configures a nanlogger to use to report NaNs in gradients updates for example. See TraceNaNInGradients.
	// This value is not saved in a checkpoint.Handler.
	// It should be set to a Tracer (which a *nanlogger.NanLogger is).
	//
	// Typical use:
	//
	//	var nanLogger *nanlogger.NanLogger
	//	if debugNaNs {
	//		nanLogger = nanlogger.New()
	//		scope.SetParam(optimizer.ParamNanLogger, nanLogger)
	//	}
	//	trainer := train.NewTrainer(…)
	//	nanLogger.AttachToTrainer(trainer)
	ParamNanLogger = "nanlogger"
)

const (
	// GlobalStepVariableName as stored in model.Store, usually in the root scope -- but depends on the
	// caller.
	GlobalStepVariableName = "global_step"

	// Scope reserved for optimizer.
	Scope = "optimizers"
)

// FromScope creates an optimizer from the scope (and its [model.Store]) hyperparameters.
// See [ParamOptimizer]. The default is "adamw".
func FromScope(scope *model.Scope) Interface {
	optName := model.GetParamOr(scope, ParamOptimizer, "adamw")
	return ByName(scope, optName)
}

// ByName returns an optimizer given the name, or panics if one does not exist.
// It uses KnownOptimizers -- in case one wants to better handle invalid values.
//
// Some optimizers (e.g.: Adam) uses optional hyperparameters set in the [model.Store] for configuration.
//
// See also FromScope.
//
// Example usage:
//
// ```
// var flagOptimizer = flag.String("optimizer", "adamw", fmt.Sprintf("Optimizer, options: %q", maps.Keys(optimizer.KnownOptimizers)))
//
// ...
//
//	trainer := train.NewTrainer(manager, store, ModelGraph,
//	   losses.SomeLoss,
//	   optimizer.ByName(scope, *flagOptimizer),
//	   []metrics.Interface{someMetric},    // trainMetrics
//	   []metrics.Interface{otherMetric})   // evalMetrics
//
// ```
func ByName(scope *model.Scope, optName string) Interface {
	optBuilder, found := KnownOptimizers[optName]
	if !found {
		Panicf("Unknown optimizer %q, valid values are %v.", optName, xslices.Keys(KnownOptimizers))
	}
	return optBuilder(scope)
}

// GetGlobalStepVar returns the global step counter, a dtypes.Int64 variable.
// It creates it (initialized with 0) if not already there.
// This can be used in graph building or directly.
func GetGlobalStepVar(scopeOrStore model.StoreProvider) *model.Variable {
	return scopeOrStore.Store().VariableWithValue(GlobalStepVariableName, int64(0)).SetTrainable(false)
}

// GetGlobalStep returns the current global step value.
// It creates the global step variable if it does not yet exist.
func GetGlobalStep(scopeOrStore model.StoreProvider) int64 {
	vAny := GetGlobalStepVar(scopeOrStore).MustValue().Value()
	v, ok := vAny.(int64)
	if !ok {
		Panicf("Variable %q=%#v, and cannot be converted to int64", GlobalStepVariableName, vAny)
	}
	return v
}

// DeleteGlobalStep in case one wants to reset the model state, or hide how many steps were taken.
func DeleteGlobalStep(scopeOrStore model.StoreProvider) error {
	scope := scopeOrStore.Store().RootScope()
	return scope.DeleteVariable(GlobalStepVariableName)
}

// IncrementGlobalStep creates (if not there yet) a global step counter (stored at the root scope
// of the [model.Store]), and returns it incremented -- its first returned value will be 1.
//
// It only builds the computation graph, no actual values are generated.
//
// Typically, this is called by the optimizers UpdateGraph method.
//
// GlobalStep is always stored as dtypes.Int64, but it is converted to the given DType
// before being returned.
func IncrementGlobalStep(g *Graph, dtype dtypes.DType) *Node {
	store := model.GetStore(g)
	if store == nil {
		Panicf("the current graph does not have an associated model.Store; to get one, you need to create the graph using a [model.Exec] (see [model.NewExec])")
	}
	globalStepVar := GetGlobalStepVar(store)
	globalStep := globalStepVar.NodeValue(g)
	globalStep = Add(globalStep, OnesLike(globalStep))
	globalStepVar.SetNodeValue(globalStep)
	if dtype != dtypes.Int64 {
		globalStep = ConvertDType(globalStep, dtype)
	}
	return globalStep
}

// IncrementCounter creates a counter in the given scope if not there yet,
// and increments it. Its first returned value will be 1.
func IncrementCounter(scope *model.Scope, g *Graph, counterName string, dtype dtypes.DType) *Node {
	counterVar := scope.WithInitializer(initializer.Zero).VariableWithShape(counterName, shapes.Make(dtype))
	counterVar.SetTrainable(false)
	counter := AddScalar(counterVar.NodeValue(g), 1)
	counterVar.SetNodeValue(counter)
	return counter
}

// LearningRateVar returns the learning rate variable -- a scalar value of the given dtype.
//
// If the variable doesn't exist yet, it is initialized with initialValue.
//
// Consider reading the initialValue from model.GetParamOr(scope, ParamLearningRate, SGDDefaultLearningRate).
func LearningRateVar(scope *model.Scope, dtype dtypes.DType, initialValue float64) *model.Variable {
	return LearningRateVarWithValue(scope, dtype, initialValue)
}

// LearningRateVarWithValue creates (or reuses) variable for learning rate with the given value.
func LearningRateVarWithValue(scope *model.Scope, dtype dtypes.DType, value float64) *model.Variable {
	scope = scope.At(Scope)
	return scope.VariableWithValue(ParamLearningRate, shapes.CastAsDType(value, dtype)).SetTrainable(false)
}

// ClipStepByValue applies the [ParamClipStepByValue] hyperparameter if it is not 0.0 (the default).
func ClipStepByValue(scope *model.Scope, step *Node) *Node {
	clipByValue := model.GetParamOr(scope, ParamClipStepByValue, 0.0)
	if clipByValue == 0 {
		return step
	}
	return ClipScalar(step, -clipByValue, clipByValue)
}

// Tracer can trace a node with a scope. Used to represent a nanlogger.NanLogger.
type Tracer interface {
	Trace(node *Node, scopes ...string)
}

// TraceNaNInGradients will report a NaN/Inf value in a gradient for the given variable, is a "Tracer" (typically a nanlogger.NanLogger)
// has been configured in the model.
func TraceNaNInGradients(scope *model.Scope, variable *model.Variable, gradients *Node) {
	lAny, found := scope.GetParam(ParamNanLogger)
	if !found {
		return
	}
	l, ok := lAny.(Tracer)
	if !ok {
		return
	}
	l.Trace(gradients, "Gradients", variable.Path())
}

// ClipNaNsInGradients will replace the gradient tensor by zeros if there are any NaNs or +/-Inf values.
// It is only enabled if ParamClipNaN is set to true.
//
// See also ClipNaNsInUpdates.
func ClipNaNsInGradients(scope *model.Scope, gradients *Node) *Node {
	if !model.GetParamOr(scope, ParamClipNaN, false) {
		return gradients
	}
	return Where(LogicalAll(IsFinite(gradients)), gradients, ZerosLike(gradients))
}

// ClipNaNsInUpdates will replace original values into updates, where updates have NaN (or +/-Inf) values,
// if the ParamClipNaN is set to true.
//
// See also ClipNaNsInGradients.
func ClipNaNsInUpdates(scope *model.Scope, original, updates *Node) *Node {
	if !model.GetParamOr(scope, ParamClipNaN, false) {
		return updates
	}
	return Where(IsFinite(updates), updates, original)
}

// SGDConfig implements a Stochastic Gradient Descent optimizer.
type SGDConfig struct {
	initialLearningRate float64

	// Whether to decay the learning rate with the global step.
	useDecay bool
}

// SGDDefaultLearningRate is the default learning rate used by the StochasticGradientDescent optimizer.
const SGDDefaultLearningRate = 0.1

// StochasticGradientDescent creates an optimizer that performs SGD.
// It looks for "learning_rate" in the [model.Store] parameters for the initial
// learning rate, otherwise it defaults to SGDDefaultLearningRate.
//
// By default, it has a learning rate decay given by: `learning_rate = initial_learning_rate / Sqrt(global_step)`
func StochasticGradientDescent() *SGDConfig {
	return &SGDConfig{
		initialLearningRate: -1, // -1 means not set.
		useDecay:            true,
	}
}

// WithDecay sets whether to use a learning rate decay with the global step.
//
// It is enabled by default, but tests may want to disable it.
//
// It returns itself to allow chaining.
func (sgd *SGDConfig) WithDecay(enabled bool) *SGDConfig {
	sgd.useDecay = enabled
	return sgd
}

// WithLearningRate sets the initial learning rate. The default value is SGDDefaultLearningRate.
//
// It returns itself to allow chaining.
func (sgd *SGDConfig) WithLearningRate(initialLearningRate float64) *SGDConfig {
	sgd.initialLearningRate = initialLearningRate
	return sgd
}

// Done returns an optimizer.Interface.
// It's a no-op since SGDConfig is itself implements optimizer.Interface, but it keeps it consistent with
// the builder pattern, and the returned Interface is no longer configurable.
func (sgd *SGDConfig) Done() Interface {
	return sgd
}

// UpdateGraph builds the graph to update the weights for one training step.
// It implements optimizer.Interface.
func (sgd *SGDConfig) UpdateGraph(scope *model.Scope, g *Graph, theLoss *Node) {
	_ = g
	if !theLoss.Shape().IsScalar() {
		Panicf("optimizer requires a scalar loss to optimize, got loss.shape=%s instead", theLoss.Shape())
	}
	grads := scope.BuildTrainableVariablesGradientsGraph(theLoss)
	sgd.UpdateGraphWithGradients(scope, grads, theLoss.DType())
}

func (sgd *SGDConfig) UpdateGraphWithGradients(scope *model.Scope, grads []*Node, lossDType dtypes.DType) {
	if len(grads) == 0 {
		return
	}
	dtype := lossDType
	g := grads[0].Graph()

	initialLearningRate := sgd.initialLearningRate
	if initialLearningRate <= 0 {
		// If the value was not set, read it from the model.
		initialLearningRate = model.GetParamOr(scope, ParamLearningRate, SGDDefaultLearningRate)
	}

	lrVar := LearningRateVar(scope, dtype, initialLearningRate)
	learningRate := lrVar.NodeValue(g)
	globalStep := IncrementGlobalStep(g, dtype)
	if sgd.useDecay {
		learningRate = Div(learningRate, Sqrt(globalStep)) // Factor global_step into the learning rate.
	}
	addGradientsToVariablesGraph(scope, grads, learningRate)
}

// Clear all optimizer variables.
// There are none for sgd, so this is a non-op.
// It implements optimizer.Interface.
func (sgd *SGDConfig) Clear(_ *model.Scope) error {
	return nil
}

// addGradientsToVariablesGraph takes the output of scope.BuildTrainableVariablesGradientsGraph,
// multiply by (-learningRate) and add to the current value of the variablesMap.
//
// It replaces NaNs with zero.
func addGradientsToVariablesGraph(scope *model.Scope, grads []*Node, learningRate *Node) {
	g := learningRate.Graph()
	if !learningRate.Shape().IsScalar() {
		Panicf("addGradientsToVariablesGraph require scalar learningRate, instead got %s", learningRate.Shape())
	}
	numTrainable := len(grads)
	ii := 0
	for v := range scope.IterVariables() {
		if !v.Trainable || !v.InUseByGraph(g) {
			// Not interested in this variable.
			continue
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
		scaledGradient = ClipStepByValue(scope, scaledGradient)
		TraceNaNInGradients(scope, v, scaledGradient)

		vNode := v.NodeValue(g)
		updatedValue := Sub(vNode, scaledGradient)
		updatedValue = ClipNaNsInUpdates(scope, vNode, updatedValue)
		v.SetNodeValue(updatedValue)
		ii++
	}
	if ii != numTrainable {
		Panicf(
			"number of trainable variables for BuildTrainableVariablesGradientsGraph (%d) and addGradientsToVariablesGraph (%d) "+
				"are different -- did new trainable variables were created or variables `.Trainable` property "+
				"changed in between?",
			numTrainable,
			ii,
		)
	}
}

// MonotonicProjection transforms the input into a monotonic sequence on the given axis that respects the
// minimum margin between consecutive points.
//
// Here we call "viable solution" one that respects the given margin between consecutive points. And the goal
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
	adjustedAxis := MustAdjustAxis(axis, input)
	axisDim := input.Shape().Dim(axis)
	if axisDim < 2 {
		Panicf(
			"MonotonicProjection of input shaped %s at axis %d is not valid: it requires axis to have dimension >= 2",
			input.Shape(),
			axis,
		)
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
	diffLeft = Max(diffLeft, margin) // Make sure it's valid if numIter wasn't enough.
	finalTotalDiff := ReduceAndKeep(diffLeft, ReduceSum, adjustedAxis)

	leftMostInput = SliceAxis(input, adjustedAxis, AxisElem(0))
	leftMostInput = Sub(leftMostInput,
		Sub(finalTotalDiff, initialTotalDiff))

	diffLeft = Concatenate([]*Node{leftMostInput, diffLeft}, adjustedAxis)
	fixLeft := CumSum(diffLeft, adjustedAxis)

	// Reconstruct value.
	return DivScalar(Add(fixRight, fixLeft), 2)
}
