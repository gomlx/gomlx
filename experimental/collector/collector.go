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

// Package collector implements DataCollector, which attaches itself to a computation graph executor
// and collects values of any selected computation graph node. This  data can then be used for data
// analysis.
package collector

import (
	"fmt"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/google/uuid"
	"github.com/pkg/errors"
	"strings"
)

// These are Context.Params keys used to hold DataCollector for specialized uses. It's merely a convention,
// not every implementation may adhere to.
const (
	// OptimizerGradsL2CollectorKey is a key to `Context.Params`. If set to a DataCollector, most optimizers
	// will use it to collect the L2 norm of the gradients at each step.
	OptimizerGradsL2CollectorKey = "optimizer_grads_l2_collector"

	// OptimizerGradsL2Series is the name of the series used when collecting the l2 norm of gradients in the
	// optimizers, if a collector is defined with OptimizerGradsL2CollectorKey.
	OptimizerGradsL2Series = "optimizer_grads_l2norm"

	// OptimizerGlobalStepSeries is the name of the series where the global step is stored by optimizers, if a collector
	// is defined with OptimizerGradsL2CollectorKey.
	OptimizerGlobalStepSeries = "optimizer_global_step"

	// TrainerLogStepSizeL2CollectorKey is a key to `Context.Params`. If set to a DataCollector, the train.Trainer will
	// collect the Log of the L2 norm of the updates to all trainable variables. This is a measure of how much the weights
	// of the models actually moved. It's usually a combination of the gradient and the learning rate calculated
	// by the optimizer. But it can also be affected by other projections.
	TrainerLogStepSizeL2CollectorKey = "trainer_log_step_size_l2_collector"

	// TrainerLogStepSizeL2Series are the step size L2 norm series name with the values collected by the train.Trainer,
	// if a collector is defined with TrainerLogStepSizeL2CollectorKey.
	TrainerLogStepSizeL2Series = "trainer_log_step_size_l2norm"
)

// DataCollector attaches itself to a computation graph executor -- graph.Exec or context.Exec -- and
// listens to the graph.Node that are set to be logged. Those selected for collection are intercepted
// and collected here. These can then be used for plotting.
//
// It only works for scalar nodes (it could be extended to collect tensors of different shapes).
//
// Example: collect the L2 norm of the last layer and logits layer.
//
// ```
//
//	 	var collector := NewDataCollector(...)
//
//			func modelGraph(ctx *context.Context, inputs []*Node) (logits *Node) {
//			    // ... build model ...
//			    lastLayer := layers.Dense(...)
//			    collector.Collect(L2Norm(lastLayer), "lastLayer")
//			    logits = layers.Dense(layers.Relu(lastLayer),...)
//			    collector.Collect(L2Norm(logits), "logits")
//			    return logits
//			}
//
//			func train(...) {
//			    ...
//			    trainer := train.NewTrainer(manager, modelGraph, ...)
//			    ...
//			    loop := train.NewLoop(trainer)
//			    collector.AttachToLoop(loop)
//			    ...
//			    _, err = loop.RunSteps(dsTrain, *flagNumSteps)
//			    ...
//			    lastLayerL2 := collector.GetSeriesValues("lastLayer")  // []float64
//			    logitsL2 := collector.GetSeriesValues("logits")  // []float64
//			}
//
// ```
type DataCollector struct {
	tag  string
	data map[string]seriesData

	previousLoggerFn graph.LoggerFn

	// How often to collect data points. One of them need to be set
	everyNSteps int
	keepNPoints int
	collectAll  bool
}

type seriesData struct {
	slice           []float64
	idx             int // Index of the next value collected -- which may or may not be stored.
	keepNPointsSkip int
}

// HasNodeLogger is any object that uses a graph.LoggerFn to log values. The standard implementations
// being the executors graph.Exec and context.Exec.
type HasNodeLogger interface {
	SetNodeLogger(loggerFn graph.LoggerFn)
	GetNodeLogger() graph.LoggerFn
}

// NewDataCollector returns a new DataCollector. Before it's ready to use one needs
// to complete 2 steps:
//
//  1. Select how often to collect data with one of EveryNSteps, KeepNPoints or CollectAll.
//  2. Attach to an executor (graph.Exec or context.Exec) with AttachToExecutor. Notice a train.Trainer
//     uses an executor to train, it can be access by train.Trainer.GetTrainExec (or GetEvalExec).
//
// Once set up, you can mark the graph.Node to collect in your graph building function with Collect.
// Each time the graph is executed, the data points are collected. When ready (after training), one
// can get the collected data with Flat.
func NewDataCollector() *DataCollector {
	return &DataCollector{
		tag:  fmt.Sprintf("<DataCollector id=%s>", uuid.NewString()),
		data: make(map[string]seriesData),
	}
}

// EveryNSteps configures the DataCollector to collect data only every N steps.
//
// One and only one of CollectAll, KeepNPoints or EveryNSteps must be set.
//
// It returns itself, so methods calling can be cascaded.
func (c *DataCollector) EveryNSteps(n int) *DataCollector {
	c.everyNSteps = n
	return c
}

// KeepNPoints configures the DataCollector to collect at most N data points. It starts
// collecting every point and whenever the buffer is full, it halves the frequency of
// collecting points. In the end it will have collected anywhere between N/2 and (N-1) points.
//
// One and only one of CollectAll, KeepNPoints or EveryNSteps must be set.
//
// It returns itself, so methods calling can be cascaded.
func (c *DataCollector) KeepNPoints(n int) *DataCollector {
	c.keepNPoints = n
	return c
}

// CollectAll sets the DataCollector to collect every data point. Memory consumption grow
// unbounded.
//
// One and only one of CollectAll, KeepNPoints or EveryNSteps must be set.
//
// It returns itself, so methods calling can be cascaded.
func (c *DataCollector) CollectAll() *DataCollector {
	c.collectAll = true
	return c
}

// check whether DataCollector was correctly initialized.
func (c *DataCollector) check() error {
	configured := 0
	if c.collectAll {
		configured++
	}
	if c.keepNPoints > 0 {
		configured++
	}
	if c.everyNSteps > 0 {
		configured++
	}
	if configured == 0 {
		return errors.New("DataCollector requires that exactly one of CollectAll, KeepNPoints or EveryNSteps is configured")
	}
	if configured > 1 {
		return errors.Errorf("DataCollector requires that only one of CollectAll, KeepNPoints or EveryNSteps is configured, %d were configured", configured)
	}
	return nil
}

// AttachToLoop attaches the DataCollector to the trainer of the given Loop variable.
//
// It actually calls AttachToExecutor to the executor associated with the Loop / Trainer.
//
// A DataCollector can only be installed in one Loop / Trainer (and underlying executor), it will return
// an error if installed somewhere else.
func (c *DataCollector) AttachToLoop(loop *train.Loop) error {
	_ = loop.Trainer
	//...	return c.AttachToExecutor(trainer.GetTrainExec())
	return errors.Errorf("Not implemented")
}

// AttachToExecutor attach the DataCollector to the executor (or anything that support HasNodeLogger interface)
// and start listening to any logged tensors for those marked for the DataCollector.
//
// This is an alternative to the AttachToLoop function, if not using the standard Trainer/Loop objects.
//
// Any logged tensor that was not marked for the DataCollector (see Collect method) are passed through
// to the previous LoggerFn registered. That means that multiple DataCollector's can be active at the
// same time.
//
// A DataCollector can only be installed in one executor, it will return an error if installed somewhere
// else.
func (c *DataCollector) AttachToExecutor(exec HasNodeLogger) error {
	if c.previousLoggerFn != nil {
		return errors.Errorf("DataCollector already installed, it can only be installed in one location")
	}
	if err := c.check(); err != nil {
		return err
	}
	c.previousLoggerFn = exec.GetNodeLogger()
	exec.SetNodeLogger(c.collectMessagesAndValues)
	return nil
}

const dataCollectorTagClose = "</DataCollector>"

// Collect indicates that the graph node value should be "collected" (saved) at every execution
// and stored in the named series.
//
// Note that a node marked for collection can't be logged -- they use the same mechanism.
func (c *DataCollector) Collect(node *graph.Node, series string) {
	g := node.Graph()
	if !g.Ok() {
		return
	}
	if !node.Shape().IsScalar() {
		g.SetErrorf("DataCollector can only be used for scalar values, but got node with shape %s", node.Shape())
		return
	}
	if !node.DType().IsFloat() && !node.DType().IsInt() {
		g.SetErrorf("DataCollector can only be used for numeric values (ints or floats), but got node with shape %s", node.Shape())
		return
	}
	node.SetLogged(fmt.Sprintf("%s%s%s", c.tag, series, dataCollectorTagClose))
}

// collectMessagesAndValues implements graph.LoggerFn. It filters the messages/values for this DataCollector,
// and pass the remaining ones to c.previousLoggerFn.
func (c *DataCollector) collectMessagesAndValues(g *graph.Graph, messages []string, values []tensor.Tensor, nodes []graph.NodeId) {
	passThroughIdx := 0
	for ii, msg := range messages {
		if strings.HasPrefix(msg, c.tag) && strings.HasSuffix(msg, dataCollectorTagClose) {
			series := msg[len(c.tag) : len(msg)-len(dataCollectorTagClose)]
			value := shapes.CastAsDType(values[ii].Local().Value(), shapes.Float64).(float64)
			c.storeValue(series, value)
		} else {
			// Keep message/value.
			messages[passThroughIdx] = messages[ii]
			values[passThroughIdx] = values[ii]
			passThroughIdx++
		}
	}

	// Call previous logging function for values not consumed.
	c.previousLoggerFn(g, messages[:passThroughIdx], values[:passThroughIdx], nodes)
}

// storeValue manages the storage of the values collected.
func (c *DataCollector) storeValue(series string, value float64) {
	data := c.data[series]
	if c.collectAll {
		// Store everything.
		data.slice = append(data.slice, value)

	} else if c.keepNPoints > 0 {
		// Store a limit number of elements, dynamically splitting the frequency of
		// storage by two if space limit is reached.
		if len(data.slice) == 0 {
			data.slice = make([]float64, 0, c.keepNPoints)
			data.keepNPointsSkip = 1
		}
		if data.idx%data.keepNPointsSkip == 0 {
			data.slice = append(data.slice, value)
			if len(data.slice) == c.keepNPoints {
				// Reserved area full, half the frequency of storage.
				for ii := 2; ii < c.keepNPoints; ii += 2 {
					data.slice[ii/2] = data.slice[ii]
				}
				data.slice = data.slice[:c.keepNPoints/2]
				data.keepNPointsSkip *= 2
			}
		}

	} else {
		// Store every N steps.
		if data.idx%c.everyNSteps == 0 {
			data.slice = append(data.slice, value)
		}
	}

	data.idx++
	c.data[series] = data
}

// GetAllSeriesNames returns the list of series names used so far.
func (c *DataCollector) GetAllSeriesNames() []string {
	return slices.Keys(c.data)
}

// GetSeriesValues returns the values stored for a given series name.
func (c *DataCollector) GetSeriesValues(series string) []float64 {
	data := c.data[series]
	return data.slice
}

// CollectGradL2 will calculate and collect the L2 norm of the gradients, if a DataCollector is configured
// in the current context. See `collector.OptimizerGradsL2CollectorKey`.
//
// It also collects the globalStep (useful for an x-axis in a plot), if it is not nil.
//
// This is useful to debug/monitor different optimizers.
//
// TODO: Add hook in optimziers tto clal this.
func CollectGradL2(ctx *context.Context, g *graph.Graph, grads []*graph.Node, globalStep *graph.Node) {
	if !g.Ok() {
		return
	}
	dc := context.GetParam[*DataCollector](ctx, OptimizerGradsL2CollectorKey, nil)
	if dc == nil {
		return
	}
	var total2 *graph.Node
	for _, grad := range grads {
		term := graph.ReduceAllSum(graph.Mul(grad, grad))
		if total2 == nil {
			total2 = term
		} else {
			total2 = graph.Add(total2, term)
		}
	}
	l2norm := graph.Sqrt(total2)
	dc.Collect(l2norm, OptimizerGradsL2Series)
	if globalStep != nil {
		noop := graph.Add(globalStep, graph.ZerosLike(globalStep)) // Separate node, so it doesn't interfere with global_step already being logged.
		dc.Collect(noop, OptimizerGlobalStepSeries)
	}
}

// CollectLogStepSizeL2 will calculate and collect the log of the L2 norm of the "step sizes" (how much the
// trainable variables changed), if a DataCollector is configured in the current context.
// See `collector.TrainerLogStepSizeL2CollectorKey`.
//
// It also collects the globalStep (useful for an x-axis in a plot), if it is not nil.
//
// This is useful to debug/monitor different optimizers.
//
// TODO: Add hook to trainer to call this.
func CollectLogStepSizeL2(ctx *context.Context, g *graph.Graph) {
	if !g.Ok() {
		return
	}
	dc := context.GetParam[*DataCollector](ctx, OptimizerGradsL2CollectorKey, nil)
	if dc == nil {
		return
	}
	var total2 *graph.Node
	ctx.EnumerateVariables(func(v *context.Variable) {
		if v.Trainable && v.InUseByGraph(g) {
			// stepSize is the difference between the initial value of the variable (`v.ParamNode`)
			// and the value it is going to be updated with (`v.ValueGraph`).
			stepSize := graph.Sub(v.ValueGraph(g), v.ParamNode(g))
			term2 := graph.ReduceAllSum(graph.Mul(stepSize, stepSize))
			if total2 == nil {
				total2 = term2
			} else {
				total2 = graph.Add(total2, term2)
			}
		}
	})
	l2norm := graph.Sqrt(total2)
	dc.Collect(graph.Log(l2norm), TrainerLogStepSizeL2Series)
}
