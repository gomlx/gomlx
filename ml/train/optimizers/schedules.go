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
	"github.com/gomlx/gomlx/types/shapes"
	"math"
)

// This file implements learning rate schedules.

// CosineAnnealingOptions is returned by CosineAnnealingSchedule to configure the cosine annealing schedule
// strategy. When finished to configure, call `IsNil`.
type CosineAnnealingOptions struct {
	graph                         *Graph
	ctx                           *context.Context
	dtype                         shapes.DType
	learningRate, minLearningRate float64
	periodNumSteps                int
}

// CosineAnnealingSchedule allows one to set up a cosine annealing schedule for the learning
// rate. See details https://paperswithcode.com/method/cosine-annealing.
//
// It returns a CosineAnnealingOptions that can be configured. When finished configuring call
// `IsNil` and it will generate the computation graph that updates the learning rate at every
// training step.
//
// Example with only one cycle (assuming `*flagNumSteps` is the number of training steps):
//
// ```
//
//		func modelGraph(cxt *context.Context, inputs []*Node) *Node {
//	     graph := inputs[0].Graph()
//			if *flagUseCosineSchedule {
//				optimizers.CosineAnnealingSchedule(ctx, graph, types.Float32).PeriodInSteps(*flagNumSteps).IsNil()
//			}
//		}
//
// ```
func CosineAnnealingSchedule(ctx *context.Context, graph *Graph, dtype shapes.DType) *CosineAnnealingOptions {
	return &CosineAnnealingOptions{
		ctx:   ctx,
		graph: graph,
		dtype: dtype,
	}
}

// PeriodInSteps sets the number of steps for one period of the cosine schedule. The effective
// learning rate decreases over the given period of training steps, and then is restarted at
// each new period.
//
// It's common to use only one period (so no annealing, just a cosine schedule), in which case
// just set to the number of steps that will be used for training.
//
// There is no default yet, this value must be given, or an error will be issued in the graph
// and context.
func (opt *CosineAnnealingOptions) PeriodInSteps(periodSteps int) *CosineAnnealingOptions {
	opt.periodNumSteps = periodSteps
	return opt
}

// MinLearningRate at the end of the cosine cycle. Defaults to 10^-3 * initial learning rate.
func (opt *CosineAnnealingOptions) MinLearningRate(minLearningRate float64) *CosineAnnealingOptions {
	opt.minLearningRate = minLearningRate
	return opt
}

// LearningRate at the start of the cosine cycle. If not given, it will try to read from the context
// params (keyed by LearningRateKey). If neither are set, it will fail and return an error in the
// context and graph.
func (opt *CosineAnnealingOptions) LearningRate(learningRate float64) *CosineAnnealingOptions {
	opt.learningRate = learningRate
	return opt
}

// Done finalizes the configuration of CosineAnnealingSchedule and generates the computation
// graph code to implment it.
//
// If invalid options are given, an error is raised in the Graph.
func (opt *CosineAnnealingOptions) Done() {
	ctx := opt.ctx.Checked(false)
	graph := opt.graph
	if !ctx.Ok() || !graph.Ok() {
		return
	}
	if opt.periodNumSteps <= 0 {
		graph.SetErrorf("period of the CosineAnnealingSchedule in number of steps was not set, or set to <= 0")
		return
	}

	lrValue := opt.learningRate
	if lrValue == 0 {
		lrValue = context.GetParam(opt.ctx, LearningRateKey, 0.0)
		if lrValue == 0 {
			graph.SetErrorf("learning rate not configured for CosineAnnealingSchedule and also "+
				"not set in the context as parameter %q", LearningRateKey)
			return
		}
	}
	lrMinValue := opt.minLearningRate
	if lrMinValue == 0 {
		lrMinValue = lrValue / 1000.0
	}

	lrVar := LearningRateVarWithValue(ctx, opt.dtype, lrValue)
	lrMax := Const(graph, shapes.CastAsDType(lrValue, opt.dtype))
	lrMin := Const(graph, shapes.CastAsDType(lrMinValue, opt.dtype))
	cosineStep := IncrementGlobalStepGraph(ctx.In("optimizers").In("cosine"), graph, opt.dtype)
	cosineStep = MinusOne(cosineStep) // Since LoopStep starts at 1.

	cycle := Div(cosineStep, Const(graph, shapes.CastAsDType(opt.periodNumSteps, opt.dtype)))
	cycle = Sub(cycle, Floor(cycle)) // Take only the fractional part.
	pi := Const(graph, shapes.CastAsDType(math.Pi, opt.dtype))
	cosine := Cos(Mul(cycle, pi))
	lr := Div(OnePlus(cosine), Scalar(graph, opt.dtype, 2)) // From 0 to 1
	lr = Add(Mul(lr, Sub(lrMax, lrMin)), lrMin)             // Now from lrMin to lrMax
	lrVar.SetValueGraph(lr)
}
