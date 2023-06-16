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

// Package metrics holds a library of metrics and defines
package metrics

import (
	"fmt"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/google/uuid"
	"github.com/pkg/errors"
)

// Interface for a Metric.
type Interface interface {
	// Name of the metric.
	Name() string

	// ShortName is a shortened version of the name (preferably a few characters) to display in progress bars or
	// similar UIs.
	ShortName() string

	// ScopeName used to store state: a combination of name and something unique.
	ScopeName() string

	// MetricType is a key for metrics that share the same quantity or semantics. Eg.:
	// "Moving-Average-Accuracy" and "Batch-Accuracy" would both have the same
	// "accuracy" metric type, and for instance, can be displayed on the same plot, sharing
	// the Y-axis.
	MetricType() string

	// UpdateGraph builds a graph that takes as input the predictions (or logits) and labels and
	// outputs the resulting metric (a scalar).
	UpdateGraph(ctx *context.Context, labels, predictions []*Node) (metric *Node)

	// PrettyPrint is used to pretty-print a metric value, usually in a short form.
	PrettyPrint(value tensor.Tensor) string

	// Reset metrics internal counters, when starting a new evaluation. Notice this may be called
	// before UpdateGraph, the metric should handle this without errors.
	Reset(ctx *context.Context) error
}

const (
	LossMetricType     = "loss"
	AccuracyMetricType = "accuracy"
)

// BaseMetricGraph is a graph building function of any metric that can be calculated stateless, without the need for
// any context. It should return a scalar, the mean for the given batch.
type BaseMetricGraph func(ctx *context.Context, labels, predictions []*Node) *Node

// PrettyPrintFn is a function to convert a metric value to a string.
type PrettyPrintFn func(value tensor.Tensor) string

// baseMetric implements a stateless metric.Interface.
type baseMetric struct {
	name, shortName, metricType, scopeName string
	metricFn                               BaseMetricGraph
	pPrintFn                               PrettyPrintFn // if nil will display default.
}

func (m *baseMetric) Name() string {
	return m.name
}

func (m *baseMetric) ShortName() string {
	return m.shortName
}

func (m *baseMetric) MetricType() string {
	return m.metricType
}

func (m *baseMetric) ScopeName() string {
	if m.scopeName == "" {
		m.scopeName = context.EscapeScopeName(fmt.Sprintf("%s_uuid_%s", m.Name(), uuid.NewString()))
	}
	return m.scopeName
}

func (m *baseMetric) UpdateGraph(ctx *context.Context, labels, predictions []*Node) (metric *Node) {
	g := predictions[0].Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}
	result := m.metricFn(ctx, labels, predictions)
	if !result.Shape().IsScalar() {
		g.SetErrorf("metric %q should return a scalar, instead got shape %s", m.Name(), result.Shape())
		return g.InvalidNode()
	}
	return result
}

func (m *baseMetric) PrettyPrint(value tensor.Tensor) string {
	if m.pPrintFn == nil {
		return fmt.Sprintf("%.3f", value.Value())
	}
	return m.pPrintFn(value)
}

func (m *baseMetric) Reset(_ *context.Context) error {
	return nil
}

// NewBaseMetric creates a stateless metric from any BaseMetricGraph function, it will return the metric
// calculated solely on the last batch.
// pPrintFn can be left as nil, and a default will be used.
func NewBaseMetric(name, shortName, metricType string, metricFn BaseMetricGraph, pPrintFn PrettyPrintFn) Interface {
	return &baseMetric{
		name: name, shortName: shortName, metricType: metricType,
		metricFn: metricFn, pPrintFn: pPrintFn}
}

// meanMetric implements a metric that keeps the mean of a metric.
type meanMetric struct {
	baseMetric
}

// NewMeanMetric creates a metric from any BaseMetricGraph function. It assumes the batch size (to weight the
// mean with each new result) is given by the first dimension of the labels' node.
// pPrintFn can be left as nil, and a default will be used.
func NewMeanMetric(name, shortName, metricType string, metricFn BaseMetricGraph, pPrintFn PrettyPrintFn) Interface {
	return &meanMetric{baseMetric: baseMetric{name: name, shortName: shortName, metricType: metricType, metricFn: metricFn, pPrintFn: pPrintFn}}
}

// BatchSize returns the batch size (assumed first dimension) of the data node, casting
// it to the same dtype as data.
func BatchSize(data *Node) *Node {
	g := data.Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}
	var batchSizeInt int
	if data.Shape().IsScalar() {
		// So it works for a single example at a time.
		batchSizeInt = 1
	} else {
		// Assumption is that first dimension of labels will be weight.
		batchSizeInt = data.Shape().Dimensions[0]
	}
	return Const(g, shapes.CastAsDType(batchSizeInt, data.DType()))
}

func (m *meanMetric) UpdateGraph(ctx *context.Context, labels, predictions []*Node) (metric *Node) {
	g := predictions[0].Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}
	result := m.metricFn(ctx, labels, predictions)
	if !result.Shape().IsScalar() {
		g.SetErrorf("metric %q should return a scalar, instead got shape %s", m.Name(), result.Shape())
		return g.InvalidNode()
	}

	// Create scope in context for metrics state, and mark it as unchecked -- model variables
	// may be set for reuse, but metrics variables are not.
	ctx = ctx.Checked(false).In(m.ScopeName())
	if !ctx.Ok() {
		g.SetError(errors.WithMessagef(ctx.Error(), "failed building computation graph for mean metric %q", m.Name()))
		return g.InvalidNode()
	}
	dtype := result.DType()
	zero := shapes.CastAsDType(0, dtype)
	totalVar := ctx.VariableWithValue("total", zero).SetTrainable(false)
	if totalVar == nil {
		g.SetError(errors.Errorf("variable nil building computation graph for mean metric %q", m.Name()))
		return g.InvalidNode()
	}
	weightVar := ctx.VariableWithValue("weight", zero).SetTrainable(false)
	if weightVar == nil {
		g.SetError(errors.Errorf("variable nil building computation graph for mean metric %q", m.Name()))
		return g.InvalidNode()
	}

	total := totalVar.ValueGraph(g)
	previousWeight := weightVar.ValueGraph(g)
	resultWeight := BatchSize(predictions[0])
	total = Add(total, Mul(result, resultWeight))
	weight := Add(previousWeight, resultWeight)
	mean := Div(total, weight)

	// Update variable values.
	weightVar.SetValueGraph(weight)
	totalVar.SetValueGraph(total)

	return mean
}

func (m *meanMetric) Reset(ctx *context.Context) error {
	if !ctx.Ok() {
		return ctx.Error()
	}
	ctx = ctx.Reuse().In(m.ScopeName())
	if !ctx.Ok() {
		return ctx.Error()
	}
	totalVar := ctx.InspectVariable(ctx.Scope(), "total")
	if totalVar == nil {
		// Assume this was called before the graph was first built, so there is nothing to reset yet.
		return nil
	}
	totalVar.SetValue(tensor.FromAnyValue(shapes.CastAsDType(0, totalVar.Value().DType())))
	weightVar := ctx.InspectVariable(ctx.Scope(), "weight")
	if weightVar != nil {
		weightVar.SetValue(tensor.FromAnyValue(shapes.CastAsDType(0, weightVar.Value().DType())))
	} else {
		return fmt.Errorf("can't find variable \"weight\" in scope %q", ctx.Scope())
	}
	return nil
}

// movingAverageMetric implements a metric that keeps the mean of a metric.
//
// It behaves just like a meanMetric, but each new batch has weight of newExampleWeight, and
// the stored weight is capped at (1-newExampleWeight).
type movingAverageMetric struct {
	meanMetric
	newExampleWeight float64
}

// NewExponentialMovingAverageMetric creates a metric from any BaseMetricGraph function. It takes new examples with
// the given weight (newExampleWeight), and decays the reset to 1-newExampleWeight.
//
// A typical value of newExampleWeight is 0.01, the smaller the value, the slower the moving average moves.
// pPrintFn can be left as nil, and a default will be used.
//
// This doesn't have a set prior, it will start being a normal average until there are enough terms, and it becomes
// an exponential moving average.
func NewExponentialMovingAverageMetric(name, shortName, metricType string, metricFn BaseMetricGraph, pPrintFn PrettyPrintFn, newExampleWeight float64) Interface {
	return &movingAverageMetric{meanMetric: meanMetric{baseMetric: baseMetric{
		name: name, shortName: shortName, metricType: metricType,
		metricFn: metricFn, pPrintFn: pPrintFn}}, newExampleWeight: newExampleWeight}
}

// UpdateGraph implements metrics.Interface.
func (m *movingAverageMetric) UpdateGraph(ctx *context.Context, labels, predictions []*Node) (metric *Node) {
	g := predictions[0].Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}
	result := m.metricFn(ctx, labels, predictions)
	if !g.Ok() {
		return g.InvalidNode()
	}
	if !result.Shape().IsScalar() {
		g.SetErrorf("metric %q should return a scalar, instead got shape %s", m.Name(), result.Shape())
		return g.InvalidNode()
	}

	// Create scope in context for metrics state, and mark it as unchecked -- model variables
	// may be set for reuse, but metrics variables are not.
	ctx = ctx.Checked(false).In(m.ScopeName())
	if !ctx.Ok() {
		g.SetError(errors.WithMessagef(ctx.Error(), "failed building computation graph for mean metric %q", m.Name()))
		return g.InvalidNode()
	}
	if !g.Ok() {
		return g.InvalidNode()
	}
	dtype := result.DType()
	zero := shapes.CastAsDType(0, dtype)

	meanVar := ctx.VariableWithValue("mean", zero).SetTrainable(false)
	if meanVar == nil {
		g.SetError(errors.Errorf("variable nil building computation graph for mean metric %q", m.Name()))
		return g.InvalidNode()
	}
	mean := meanVar.ValueGraph(g)

	countVar := ctx.VariableWithValue("count", zero).SetTrainable(false)
	count := countVar.ValueGraph(g)
	count = Add(count, OnesLike(count))
	countVar.SetValueGraph(count)

	weight := Max(Const(g, shapes.CastAsDType(m.newExampleWeight, dtype)), Inverse(count))
	mean = Add(Mul(mean, OneMinus(weight)), Mul(result, weight)) // total are the values multiplied by weights, and then summed.
	meanVar.SetValueGraph(mean)

	return mean
}

// BinaryAccuracyGraph can be used in combination with New*Metric functions to build metrics for binary accuracy.
// It assumes predictions are probabilities, that labels are `{0, 1}`, and those predictions and labels have
// the same shape and dtype.
func BinaryAccuracyGraph(_ *context.Context, labels, predictions []*Node) *Node {
	prediction := predictions[0]
	g := prediction.Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}
	if len(labels) != 1 {
		g.SetErrorf("BinaryAccuracy requires one labels tensor, got (%d) instead", len(labels))
		return g.InvalidNode()
	}
	label := labels[0]
	if !prediction.Shape().Eq(label.Shape()) {
		g.SetErrorf("prediction (%s) and label (%s) have different shapes, can't calculate binary accuracy",
			prediction.Shape(), label.Shape())
		return g.InvalidNode()
	}
	diff := Abs(Sub(label, prediction))
	// Accuracy is true if diff < 0.5. Notice this will take predictions of 0.5 to be false independent of label
	// (assuming labels are 1 or 0).
	dtype := prediction.DType()
	correctExamples := OneMinus(PositiveIndicator(Sub(diff, Const(g, shapes.CastAsDType(0.5, dtype)))))
	countExamples := Const(g, shapes.CastAsDType(correctExamples.Shape().Size(), correctExamples.DType()))
	return Div(ReduceAllSum(correctExamples), countExamples)
}

func accuracyPPrint(value tensor.Tensor) string {
	return fmt.Sprintf("%.2f%%", shapes.ConvertTo[float64](value.Value())*100.0)
}

// NewMeanBinaryAccuracy returns a new binary accuracy metric with the given names.
func NewMeanBinaryAccuracy(name, shortName string) Interface {
	return NewMeanMetric(name, shortName, AccuracyMetricType, BinaryAccuracyGraph, accuracyPPrint)
}

// NewMovingAverageBinaryAccuracy returns a new binary accuracy metric with the given names.
// A typical value of newExampleWeight is 0.01, the smaller the value, the slower the moving average moves.
func NewMovingAverageBinaryAccuracy(name, shortName string, newExampleWeight float64) Interface {
	return NewExponentialMovingAverageMetric(name, shortName, AccuracyMetricType, BinaryAccuracyGraph, accuracyPPrint, newExampleWeight)
}

// BinaryLogitsAccuracyGraph can be used in combination with New*Metric functions to build metrics for binary accuracy for logit.
// Notice 0s are considered a miss.
// It assumes predictions are logits, that labels are {0, 1} and that predictions and labels have the same size and dtype.
// The shape may be different (e.g.: `[batch_size, 1]` and `[batch_size]`), they will be reshaped to the
// logits shape before the accuracy is calculated.
func BinaryLogitsAccuracyGraph(_ *context.Context, labels, logits []*Node) *Node {
	logits0 := logits[0]
	g := logits0.Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}
	if len(labels) != 1 {
		g.SetErrorf("BinaryLogitsAccuracyGraph requires one labels tensor, got (%d) instead", len(labels))
		return g.InvalidNode()
	}
	labels0 := labels[0]
	if logits0.DType() != labels0.DType() {
		g.SetErrorf("logits0 (%s) and labels0 (%s) have different dtypes, can't calculate binary accuracy",
			logits0.DType(), labels0.DType())
		return g.InvalidNode()

	}
	if logits0.Shape().Size() != labels0.Shape().Size() {
		g.SetErrorf("logits0 (%s) and labels0 (%s) have different shapes (different total sizes), can't calculate binary accuracy",
			logits0.Shape(), labels0.Shape())
		return g.InvalidNode()
	}
	if !logits0.Shape().Eq(labels0.Shape()) {
		// They are the same size, so we assume the labels0 can simply be re-shaped.
		// Not strictly true, depending on how they are organized, but generally yes, and
		// this is very convenient functionality.
		labels0 = Reshape(labels0, logits0.Shape().Dimensions...)
	}
	if !g.Ok() {
		return g.InvalidNode()
	}

	dtype := logits0.DType()
	labels0 = Sub(labels0, Const(g, shapes.CastAsDType(0.5, dtype)))    // Labels: -0.5 for false, +0.5 for true.
	correctExamples := StrictlyPositiveIndicator(Mul(logits0, labels0)) // 0s are considered a miss.
	countExamples := Const(g, shapes.CastAsDType(correctExamples.Shape().Size(), correctExamples.DType()))
	mean := Div(ReduceAllSum(correctExamples), countExamples)
	if !g.Ok() {
		return g.InvalidNode()
	}
	return mean
}

// NewMeanBinaryLogitsAccuracy returns a new binary accuracy metric with the given names.
func NewMeanBinaryLogitsAccuracy(name, shortName string) Interface {
	return NewMeanMetric(name, shortName, AccuracyMetricType, BinaryLogitsAccuracyGraph, accuracyPPrint)
}

// NewMovingAverageBinaryLogitsAccuracy returns a new binary accuracy metric with the given names.
// A typical value of newExampleWeight is 0.01, the smaller the value, the slower the moving average moves.
func NewMovingAverageBinaryLogitsAccuracy(name, shortName string, newExampleWeight float64) Interface {
	return NewExponentialMovingAverageMetric(name, shortName, AccuracyMetricType, BinaryLogitsAccuracyGraph, accuracyPPrint, newExampleWeight)
}

// SparseCategoricalAccuracyGraph returns the accuracy -- fraction of times argmax(logits)
// is the true label. It works for both probabilities or logits. Ties are considered misses.
// Labels is expected to be some integer type. And the returned dtype is the same as logits.
func SparseCategoricalAccuracyGraph(_ *context.Context, labels, logits []*Node) *Node {
	logits0 := logits[0]
	g := logits0.Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}
	if len(labels) != 1 {
		g.SetErrorf("SparseCategoricalAccuracyGraph requires one labels tensor, got (%d) instead", len(labels))
		return g.InvalidNode()
	}
	labels0 := labels[0]
	labelsShape := labels0.Shape()
	labelsRank := labelsShape.Rank()
	logitsShape := logits0.Shape()
	logitsRank := logitsShape.Rank()
	if !labelsShape.DType.IsInt() {
		g.SetErrorf("labels0 indices dtype (%s), it must be integer", labelsShape.DType)
		return g.InvalidNode()
	}
	if labelsRank != logitsRank {
		g.SetErrorf("labels0(%s) and logits0(%s) must have the same rank", labelsShape, logitsShape)
		return g.InvalidNode()
	}
	if labelsShape.Dimensions[labelsRank-1] != 1 {
		g.SetErrorf("labels0(%s) are expected to have the last dimension == 1, with the true/labeled category", labelsShape)
		return g.InvalidNode()
	}

	// TODO: implement with argmax

	// Convert logits0 such that only those with the maximum value become 1, the rest 0.
	logitsMax := ReduceAndKeep(logits0, ReduceMax, -1)
	logitsMaxIndicator := PositiveIndicator(Sub(logits0, logitsMax)) // Notice 0s become 1s.

	// Now the problem is that ties will have more than one indicator, we eliminate those by subtracting
	// the sum. Any row with a sum != 1, will become zero.
	logitsSum := ReduceAndKeep(logitsMaxIndicator, ReduceMax, -1)
	logitsMaxIndicator = PositiveIndicator(Sub(logitsMaxIndicator, logitsSum))

	// Convert labels0 to one hot encoding.
	// Remove last dimension, it will be re-added by OneHot
	reducedLabels := Reshape(labels0, labels0.Shape().Dimensions[:labelsRank-1]...)
	labelsValues := OneHot(reducedLabels, logitsShape.Dimensions[logitsRank-1], logitsShape.DType)

	// correctExamples will be those where labelsValues and logitsMaxIndicator are 1.
	correctExamples := Mul(logitsMaxIndicator, labelsValues)
	countExamples := Const(g, shapes.CastAsDType(labels0.Shape().Size(), correctExamples.DType()))
	return Div(ReduceAllSum(correctExamples), countExamples)
}

// NewSparseCategoricalAccuracy returns a new sparse categorical accuracy metric with the given names.
// The accuracy is defined as the fraction of times argmax(logits) is the true label.
// It works for both probabilities or logits. Ties are considered misses.
// Labels is expected to be some integer type. And the returned dtype is the same as logits.
func NewSparseCategoricalAccuracy(name, shortName string) Interface {
	return NewMeanMetric(name, shortName, AccuracyMetricType, SparseCategoricalAccuracyGraph, accuracyPPrint)
}

// NewMovingAverageSparseCategoricalAccuracy returns a new sparse categorical accuracy metric with the given names.
// The accuracy is defined as the fraction of times argmax(logits) is the true label.
// It works for both probabilities or logits. Ties are considered misses.
// Labels is expected to be some integer type. And the returned dtype is the same as logits.
// A typical value of newExampleWeight is 0.01, the smaller the value, the slower the moving average moves.
func NewMovingAverageSparseCategoricalAccuracy(name, shortName string, newExampleWeight float64) Interface {
	return NewExponentialMovingAverageMetric(name, shortName, AccuracyMetricType, SparseCategoricalAccuracyGraph, accuracyPPrint, newExampleWeight)
}
