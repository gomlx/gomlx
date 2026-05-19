// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package cosineschedule implements a cosine annealing schedule for the learning rate.
// See New for details and example of usage, and original paper description in [1]
//
// [1] https://paperswithcode.com/method/cosine-annealing
package cosineschedule

import (
	"math"

	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/gomlx/gomlx/support/exceptions"
)

var (
	// ParamPeriodSteps enables cosine annealing (cosine schedule) for the learning rate.
	// The cosine schedule will have a cosine-like decrease of the learning rate during the given number of steps,
	// and then restart -- there is a discontinuity in the learning rate at every new period.
	//
	// If left at 0, the default, it is disabled.
	//
	// ParamPeriodSteps and ParamPeriodCycles are mutually exclusive.
	//
	//  Requires calling `New().FromContext().Done()` at the start of your model.
	//
	//  This only affects training; there is no effect during inference or evaluation.
	ParamPeriodSteps = "cosine_schedule_steps"

	// ParamCycles enables cosine annealing (cosine schedule) for the learning rate.
	// The cosine schedule will split the train steps into ParamCycles cycles of with a cosine-like decrease of
	// the learning rate. There is a discontinuity in the learning rate at every new cycle.
	//
	// If left at 0, the default, it is disabled.
	//
	// ParamPeriodSteps and ParamCycles are mutually exclusive.
	//
	//  Requires calling `New().FromContext().Done()` at the start of your model.
	//
	//  This only affects training; there is no effect during inference or evaluation.
	ParamCycles = "cosine_schedule_cycles"

	// ParamWarmUpSteps is the number of warmup steps: during these initial steps the learning rate
	// linearly increases from 0 to the learning rate defined by ParamLearningRate.
	// Only after the warmup steps the cosine annealing schedule starts.
	// The default is 0, which means no warmup.
	// Only values > 0 are allowed.
	//
	// If ParamPeriodSteps and ParamCycles are both set to 0, cosine annealing schedule is disabled
	// and this value is ignored.
	ParamWarmUpSteps = "cosine_schedule_warmup_steps"

	// ParamMinLearningRate is the minimum value of the learning rate during the
	// cosine annealing schedule.
	// Defaults to 0.0.
	ParamMinLearningRate = "cosine_schedule_min_learning_rate"
)

// Config of the cosine annealing schedule strategy.
// New creates it and once configured, call Config.Done to add it into the computation graph.
type Config struct {
	graph                         *Graph
	scope                         *model.Scope
	dtype                         dtypes.DType
	learningRate, minLearningRate float64
	periodNumSteps, numCycles     int
	warmUpSteps                   int
}

// New creates a configuration to apply a cosine annealing schedule for the learning rate.
// See details in [1].
// The cosine schedule only affects training; there is no effect during inference or evaluation.
//
// It returns a Config that can be configured. When finished configuring, call
// `Done` and it will generate the computation graph that updates the learning rate at every
// training step.
//
// If you don't set ParamPeriodSteps or ParamCycles, the cosine annealing schedule is disabled.
//
// (*) The paper describes a cosine schedule with a warmup, but this is not implemented here.
//
//	The warmup is implemented in the context, see ParamWarmUpSteps.
//
// Example with only one cycle, and a warmup of 1000 steps.
// We assume that the learning rate is set in the context as the parameter "learning_rate"
// (== optimizer.ParamLearningRate):
//
//	func MyModelGraph(cxt *model.Scope, inputs []*Node) *Node {
//		...
//		g := inputs[0].Graph()
//		cosineschedule.New(ctx, g, dtypes.Float32).
//			MinLearningRate(0.001).
//			WarmUpSteps(1000).
//			NumCycles(1).Done()
//	}
//
// Or more simply, pass the hyperparameters in the context (see ParamPeriodSteps, ParamMinLearningRate, and
// ParamWarmUpSteps):
//
//	func modelGraph(cxt *model.Scope, inputs []*Node) *Node {
//		...
//		g := inputs[0].Graph()
//		cosineschedule.New(ctx, g, dtypes.Float32).FromContext().Done()
//	}
//
// [1] https://paperswithcode.com/method/cosine-annealing.
func New(scope *model.Scope, graph *Graph, dtype dtypes.DType) *Config {
	return &Config{
		scope: scope,
		graph: graph,
		dtype: dtype,
	}
}

// FromScope configures the cosine annealing from the scope (and the model.Store), using the keys
// [ParamPeriodSteps], [ParamMinLearningRate], [ParamCycles], [optimizer.ParamLearningRate], and [ParamWarmUpSteps]
func (opt *Config) FromScope() *Config {
	opt.PeriodSteps(model.GetParamOr(opt.scope, ParamPeriodSteps, 0))
	opt.NumCycles(model.GetParamOr(opt.scope, ParamCycles, 0))
	opt.LearningRate(model.GetParamOr(opt.scope, optimizer.ParamLearningRate, 0.0))
	opt.MinLearningRate(model.GetParamOr(opt.scope, ParamMinLearningRate, 0.0))
	opt.WarmUpSteps(model.GetParamOr(opt.scope, ParamWarmUpSteps, 0))
	return opt
}

// PeriodSteps sets the number of steps for one period of the cosine schedule. The effective
// learning rate decreases over the given period of training steps and then is restarted at
// each new period.
//
// If left at 0, the default, it is disabled.
//
// PeriodSteps and NumCycles are mutually exclusive.
func (opt *Config) PeriodSteps(steps int) *Config {
	if steps < 0 {
		exceptions.Panicf("PeriodSteps requires steps >= 0, but got %d", steps)
	}
	opt.periodNumSteps = steps
	return opt
}

// PeriodInSteps is an alias to PeriodSteps.
// Deprecated: use PeriodSteps instead.
func (opt *Config) PeriodInSteps(steps int) *Config {
	return opt.PeriodSteps(steps)
}

// NumCycles sets the number of cycles for the cosine schedule.
// The effective learning rate decreases over the given number of cycles and then is restarted at each new cycle.
//
// If left at 0, the default, it is disabled.
//
// PeriodSteps and NumCycles are mutually exclusive.
//
// Note: it depends on the train.Loop object to report how many steps the model is going to be trained for.
// This works fine if one is using Loop.RunSteps, but if one is using Loop.RunEpochs, the number of steps
// is not known until the end of the first epoch.
func (opt *Config) NumCycles(numCycles int) *Config {
	if numCycles < 0 {
		exceptions.Panicf("numCycles must be >= 0, but got %d", numCycles)
	}
	opt.numCycles = numCycles
	return opt
}

// MinLearningRate at the end of the cosine cycle. Defaults to 0.0.
func (opt *Config) MinLearningRate(minLearningRate float64) *Config {
	if minLearningRate < 0 {
		exceptions.Panicf("minLearningRate must be >= 0, but got %g", minLearningRate)
	}
	opt.minLearningRate = minLearningRate
	return opt
}

// WarmUpSteps sets the number of steps to linearly increase the learning rate from 0 to the
// learning rate defined by ParamLearningRate.
//
// The default is 0, which means no warmup.
func (opt *Config) WarmUpSteps(warmUpSteps int) *Config {
	if warmUpSteps < 0 {
		exceptions.Panicf("warmUpSteps must be >= 0, but got %d", warmUpSteps)
	}
	opt.warmUpSteps = warmUpSteps
	return opt
}

// LearningRate at the start of the cosine cycle.
// If not given, it will try to read from the context params (keyed by ParamLearningRate).
// If neither is set, it will fail and return an error in the context and graph.
func (opt *Config) LearningRate(learningRate float64) *Config {
	if learningRate < 0 {
		exceptions.Panicf("learningRate must be >= 0, but got %g", learningRate)
	}
	opt.learningRate = learningRate
	return opt
}

const (
	Scope = "cosine_schedule"

	// DefaultLastStep is the default value for the last step of the training while one is not yet known.
	DefaultLastStep = 1_000_000_000
)

// Done finalizes the configuration of New and generates the computation
// graph code to implement it.
//
// If invalid options are given, an error is raised in the Graph.
func (opt *Config) Done() {
	scope := opt.scope
	graph := opt.graph

	if !scope.IsTraining(opt.graph) || (opt.periodNumSteps <= 0 && opt.numCycles <= 0) {
		// Nothing to do.
		return
	}
	if opt.periodNumSteps > 0 && opt.numCycles > 0 {
		exceptions.Panicf("PeriodSteps (%d) and NumCycles (%d) are mutually exclusive, but both were set",
			opt.periodNumSteps, opt.numCycles)
		return
	}

	lrValue := opt.learningRate
	if lrValue <= 0 {
		lrValue = model.GetParamOr(opt.scope, optimizer.ParamLearningRate, 0.0)
		if lrValue == 0 {
			exceptions.Panicf("learning rate not configured for New and also "+
				"not set in the context as parameter %q", optimizer.ParamLearningRate)
			return
		}
	}
	lrMinValue := max(opt.minLearningRate, 0.0)

	// Current training step: cosine schedule keeps its own step counter,
	// stored in an absolute scope.
	cosineScope := scope.Store().Scope(model.JoinPath(optimizer.Scope, Scope))
	cosineStep := optimizer.IncrementCounter(cosineScope, graph, "step", opt.dtype)
	cosineStep = MinusOne(cosineStep) // The value before the increment.
	adjustedCosineStep := cosineStep
	if opt.warmUpSteps > 0 {
		// Shift the cosineStep by the warmUp steps.
		adjustedCosineStep = SubScalar(adjustedCosineStep, opt.warmUpSteps)
	}

	// Calculate the fraction of the cycle we are in.
	var cycle *Node
	if opt.periodNumSteps > 0 {
		cycle = DivScalar(adjustedCosineStep, float64(opt.periodNumSteps))
	} else {
		// opt.numCyles > 0
		lastStep := train.GetTrainLastStepVar(scope).NodeValue(graph)
		lastStep = Where(IsNegative(lastStep), Const(graph, DefaultLastStep), lastStep)
		if opt.warmUpSteps > 0 {
			lastStep = SubScalar(lastStep, opt.warmUpSteps)
		}
		periodNumSteps := DivScalar(ConvertDType(lastStep, opt.dtype), opt.numCycles)
		cycle = Div(adjustedCosineStep, periodNumSteps)
	}

	// A cycle represents the fraction of a half-circle (180 degrees, or pi radians).
	cycle = Sub(cycle, Floor(cycle)) // Take only the fractional part: so always in the range `[0.0, 1.0)`.

	// Calculate cosine schedule.
	cosine := Cos(MulScalar(cycle, math.Pi))                      // from -1.0 to 1.0
	lr := DivScalar(OnePlus(cosine), 2)                           // (Cos()+1.0)/2.0 -> from 0.0 to 1.0
	lr = AddScalar(MulScalar(lr, lrValue-lrMinValue), lrMinValue) // Now from lrMin to lrMax

	// Calculate and merge warmup schedule.
	if opt.warmUpSteps > 0 {
		ratio := DivScalar(cosineStep, opt.warmUpSteps)
		ratio = MinScalar(ratio, 1.0)
		warmUpLR := AddScalar(MulScalar(ratio, lrValue-lrMinValue), lrMinValue)
		// Apply the learning rate only if in the warmup period.
		lr = Where(IsNegative(adjustedCosineStep), warmUpLR, lr)
	}

	// Update learning rate.
	lrVar := optimizer.LearningRateVarWithValue(scope.Store().Scope(scope.Scope()), opt.dtype, lrValue)
	lrVar.SetNodeValue(lr)
}
