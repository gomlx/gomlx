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

// Package cosineschedule implements a cosine annealing schedule for the learning rate.
// See New for details and example of usage, and original paper description in [1]
//
// [1] https://paperswithcode.com/method/cosine-annealing.
package cosineschedule

import (
	"math"

	. "github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gopjrt/dtypes"
)

var (
	// ParamPeriodSteps enables cosine annealing (cosine schedule) for the learning rate.
	//
	// This parameter defines the number of steps in a cosine annealing period.
	//
	//  * 0: Disables cosine annealing (default).
	//  * Positive value: Sets the period to the specified number of steps.
	//  * Negative value: Sets the period to a fraction of the total training steps.
	//      * -1: Period equals the total number of training steps (common setting).
	//      * -2: Period equals half the total number of training steps, and so on.
	//
	//  Requires calling `New().FromContext().Done()` at the start of your model.
	//
	//  Only affects training; no effect during inference or evaluation.
	ParamPeriodSteps = "cosine_schedule_steps"

	// ParamWarmUpSteps is the number of warmup steps: during these initial steps the learning rate
	// linearly increases from 0 to the learning rate defined by ParamLearningRate.
	// Only after the warmup steps the cosine annealing schedule starts.
	// The default is 0, which means no warmup.
	// Only values > 0 are allowed.
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
	ctx                           *context.Context
	dtype                         dtypes.DType
	learningRate, minLearningRate float64
	periodNumSteps                int
	warmUpSteps                   int
}

// New creates a configuration to apply a cosine annealing schedule for the learning rate.
// See details https://paperswithcode.com/method/cosine-annealing. (*)
//
// It returns a Config that can be configured. When finished configuring, call
// `Done` and it will generate the computation graph that updates the learning rate at every
// training step.
//
// Example with only one cycle, and a warmup of 1000 steps. We assume *flagNumSteps is the number of training steps,
// and that the learning rate is set in the context as the parameter "learning_rate" (== optimizers.ParamLearningRate).
//
//	func MyModelGraph(cxt *context.Context, inputs []*Node) *Node {
//		...
//		g := inputs[0].Graph()
//		cosineschedule.New(ctx, g, dtypes.Float32).
//			MinLearningRate(0.001).
//			WarmUpSteps(1000).
//			PeriodInSteps(*flagNumSteps).Done()
//	}
//
// Or more simply, pass the hyperparameters in the context (see ParamPeriodSteps, ParamMinLearningRate, and
// ParamWarmUpSteps):
//
//	func modelGraph(cxt *context.Context, inputs []*Node) *Node {
//		...
//		g := inputs[0].Graph()
//		cosineschedule.New(ctx, g, dtypes.Float32).FromContext().Done()
//	}
func New(ctx *context.Context, graph *Graph, dtype dtypes.DType) *Config {
	return &Config{
		ctx:   ctx,
		graph: graph,
		dtype: dtype,
	}
}

// FromContext configures the cosine annealing from the context, using the keys
// [ParamPeriodSteps] and [ParamMinLearningRate].
func (opt *Config) FromContext() *Config {
	opt.periodNumSteps = context.GetParamOr(opt.ctx, ParamPeriodSteps, 0)
	opt.learningRate = context.GetParamOr(opt.ctx, optimizers.ParamLearningRate, 0.0)
	opt.minLearningRate = context.GetParamOr(opt.ctx, ParamMinLearningRate, 0.0)
	opt.warmUpSteps = context.GetParamOr(opt.ctx, ParamWarmUpSteps, 0)
	return opt
}

// PeriodInSteps sets the number of steps for one period of the cosine schedule. The effective
// learning rate decreases over the given period of training steps and then is restarted at
// each new period.
//
// It's common to use only one period (so no annealing, just a cosine schedule), in which case
// set to the number of steps that will be used for training.
//
// The default is -1, which will trigger an exception when building the graph, so it must be
// defined. If set to 0, the cosine annealing schedule is silently disabled.
func (opt *Config) PeriodInSteps(periodSteps int) *Config {
	opt.periodNumSteps = periodSteps
	return opt
}

// MinLearningRate at the end of the cosine cycle. Defaults to 0.0.
func (opt *Config) MinLearningRate(minLearningRate float64) *Config {
	opt.minLearningRate = minLearningRate
	return opt
}

// WarmUpSteps sets the number of steps to linearly increase the learning rate from 0 to the
// learning rate defined by ParamLearningRate.
//
// The default is 0, which means no warmup.
func (opt *Config) WarmUpSteps(warmUpSteps int) *Config {
	opt.warmUpSteps = warmUpSteps
	return opt
}

// LearningRate at the start of the cosine cycle.
// If not given, it will try to read from the context params (keyed by ParamLearningRate).
// If neither is set, it will fail and return an error in the context and graph.
func (opt *Config) LearningRate(learningRate float64) *Config {
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
	ctx := opt.ctx.Checked(false)
	graph := opt.graph

	if !ctx.IsTraining(opt.graph) || opt.periodNumSteps == 0 {
		return
	}

	lrValue := opt.learningRate
	if lrValue == 0 {
		lrValue = context.GetParamOr(opt.ctx, optimizers.ParamLearningRate, 0.0)
		if lrValue == 0 {
			Panicf("learning rate not configured for New and also "+
				"not set in the context as parameter %q", optimizers.ParamLearningRate)
			return
		}
	}
	lrMinValue := opt.minLearningRate

	// Current training step: cosine schedule keeps its own "global step" counter.
	cosineStep := optimizers.IncrementGlobalStepGraph(ctx.In(optimizers.Scope).In(Scope), graph, opt.dtype)
	cosineStep = MinusOne(cosineStep) // Since the count starts at 1.
	if opt.warmUpSteps > 0 {
		// Shift the cosineStep by the warmUp steps.
		cosineStep = AddScalar(cosineStep, -opt.warmUpSteps)
	}

	// Calculate the fraction of the cycle we are in.
	var cycle *Node
	if opt.periodNumSteps > 0 {
		cycle = DivScalar(cosineStep, float64(opt.periodNumSteps))
	} else {
		// If opt.periodNumSteps < 0, the actual period is calculated as a fraction of the total number of steps
		// to be trained (train.GetTrainLastStepVar).
		lastStep := train.GetTrainLastStepVar(ctx).ValueGraph(graph)
		lastStep = Where(IsNegative(lastStep), Const(graph, DefaultLastStep), lastStep)
		periodNumSteps := DivScalar(ConvertDType(lastStep, opt.dtype), -opt.periodNumSteps)
		cycle = Div(cosineStep, periodNumSteps)

		// Since if using RunEpoch() the last step may not be known for a while (and set to -1), we have to check for
		// that case.
		cycle = MaxScalar(cycle, 0)
	}
	// A cycle represents the fraction of a half-circle (180 degrees, or pi radians).
	cycle = Sub(cycle, Floor(cycle)) // Take only the fractional part: so always in the range `[0.0, 1.0)`.

	// Calculate cosine schedule.
	cosine := Cos(MulScalar(cycle, math.Pi))                      // from -1.0 to 1.0
	lr := DivScalar(OnePlus(cosine), 2)                           // (Cos()+1.0)/2.0 -> from 0.0 to 1.0
	lr = AddScalar(MulScalar(lr, lrValue-lrMinValue), lrMinValue) // Now from lrMin to lrMax

	// Update learning rate.
	lrVar := optimizers.LearningRateVarWithValue(ctx, opt.dtype, lrValue)
	lrVar.SetValueGraph(lr)
}
