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

package optimizers

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	. "github.com/gomlx/gomlx/types/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"math"
)

// This file implements learning rate schedules.

var (
	// ParamCosineScheduleSteps will enable cosine annealing (aka. "cosine schedule")
	// of the learning rate, if set to a value > 0. It defines the number of steps of the
	// period of the cosine annealing schedule.
	// It is very commonly to use the same value as the number of steps being trained.
	//
	// It requires that the model call `CosineAnnealingSchedule().FromContext().Done()`
	// in the start of the model.
	//
	// It only impacts training, and it is a no-op during inference and evaluation.
	ParamCosineScheduleSteps = "cosine_schedule_steps"

	// ParamCosineScheduleMinLearningRate is the minimum value of the learning rate, during
	// cosine annealing schedule.
	// Defaults to 10^-3 * initial learning rate.
	ParamCosineScheduleMinLearningRate = "cosine_annealing_min_learning_rate"
)

// CosineScheduleOptions is returned by CosineAnnealingSchedule to configure the cosine annealing schedule
// strategy. When finished to configure, call `Done`.
type CosineScheduleOptions struct {
	graph                         *Graph
	ctx                           *context.Context
	dtype                         shapes.DType
	learningRate, minLearningRate float64
	periodNumSteps                int
}

// CosineAnnealingSchedule allows one to set up a cosine annealing schedule for the learning
// rate. See details https://paperswithcode.com/method/cosine-annealing.
//
// This is slightly different in the sense that $T_i$ is fixed to what here is called [PeriodInSteps].
//
// It returns a CosineScheduleOptions that can be configured. When finished configuring call
// `Done` and it will generate the computation graph that updates the learning rate at every
// training step.
//
// Example with only one cycle (assuming `*flagNumSteps` is the number of training steps):
//
//	func modelGraph(cxt *context.Context, inputs []*Node) *Node {
//		...
//		g := inputs[0].Graph()
//		optimizers.CosineAnnealingSchedule(ctx, g, shapes.Float32).PeriodInSteps(*flagNumSteps).Done()
//	}
//
// Or more simply, just pass the hyperparameters in the context (see [ParamCosineScheduleSteps]):
//
//	func modelGraph(cxt *context.Context, inputs []*Node) *Node {
//		...
//		g := inputs[0].Graph()
//		optimizers.CosineAnnealingSchedule(ctx, g, shapes.Float32).FromContext().Done()
//	}
func CosineAnnealingSchedule(ctx *context.Context, graph *Graph, dtype shapes.DType) *CosineScheduleOptions {
	return &CosineScheduleOptions{
		ctx:   ctx,
		graph: graph,
		dtype: dtype,
	}
}

// FromContext configures the cosine annealing from the context, using the keys
// [ParamCosineScheduleSteps] and [ParamCosineScheduleMinLearningRate].
func (opt *CosineScheduleOptions) FromContext() *CosineScheduleOptions {
	opt.periodNumSteps = context.GetParamOr(opt.ctx, ParamCosineScheduleSteps, 0)
	opt.learningRate = context.GetParamOr(opt.ctx, ParamLearningRate, 0.0)
	opt.minLearningRate = context.GetParamOr(opt.ctx, ParamCosineScheduleMinLearningRate, 0.0)
	return opt
}

// PeriodInSteps sets the number of steps for one period of the cosine schedule. The effective
// learning rate decreases over the given period of training steps, and then is restarted at
// each new period.
//
// It's common to use only one period (so no annealing, just a cosine schedule), in which case
// just set to the number of steps that will be used for training.
//
// The default is -1, which will trigger an exception when building the graph, so it must be
// defined. If set to 0, the cosine annealing schedule is silently disabled.
func (opt *CosineScheduleOptions) PeriodInSteps(periodSteps int) *CosineScheduleOptions {
	opt.periodNumSteps = periodSteps
	return opt
}

// MinLearningRate at the end of the cosine cycle. Defaults to 10^-3 * initial learning rate.
func (opt *CosineScheduleOptions) MinLearningRate(minLearningRate float64) *CosineScheduleOptions {
	opt.minLearningRate = minLearningRate
	return opt
}

// LearningRate at the start of the cosine cycle. If not given, it will try to read from the context
// params (keyed by ParamLearningRate). If neither are set, it will fail and return an error in the
// context and graph.
func (opt *CosineScheduleOptions) LearningRate(learningRate float64) *CosineScheduleOptions {
	opt.learningRate = learningRate
	return opt
}

const CosineScheduleScope = "cosine_schedule"

// Done finalizes the configuration of CosineAnnealingSchedule and generates the computation
// graph code to implement it.
//
// If invalid options are given, an error is raised in the Graph.
func (opt *CosineScheduleOptions) Done() {
	ctx := opt.ctx.Checked(false)
	graph := opt.graph
	if !ctx.IsTraining(opt.graph) || opt.periodNumSteps == 0 {
		return
	}
	if opt.periodNumSteps < 0 {
		Panicf("period of the CosineAnnealingSchedule in number of steps was not set, or set to < 0")
	}

	lrValue := opt.learningRate
	if lrValue == 0 {
		lrValue = context.GetParamOr(opt.ctx, ParamLearningRate, 0.0)
		if lrValue == 0 {
			Panicf("learning rate not configured for CosineAnnealingSchedule and also "+
				"not set in the context as parameter %q", ParamLearningRate)
			return
		}
	}
	lrMinValue := opt.minLearningRate
	if lrMinValue == 0 {
		lrMinValue = lrValue * 1e-3
	}

	// Current training step: cosine schedule keeps its own "global step" counter.
	cosineStep := IncrementGlobalStepGraph(ctx.In(Scope).In(CosineScheduleScope), graph, opt.dtype)
	cosineStep = MinusOne(cosineStep) // Since the count starts at 1.

	// Calculate
	cycle := Div(cosineStep, Const(graph, shapes.CastAsDType(opt.periodNumSteps, opt.dtype)))
	cycle = Sub(cycle, Floor(cycle)) // Take only the fractional part: so always in range `[0.0, 1.0)`.
	cosine := Cos(MulScalar(cycle, math.Pi))
	lr := MulScalar(OnePlus(cosine), 0.5)                           // (Cos()+1.0)/2.0
	lr = AddScalar(MulScalar(lr, (lrValue-lrMinValue)), lrMinValue) // Now from lrMin to lrMax

	lrVar := LearningRateVarWithValue(ctx, opt.dtype, lrValue)
	lrVar.SetValueGraph(lr)
}
