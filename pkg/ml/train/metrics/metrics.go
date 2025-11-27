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

	. "github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/google/uuid"
	"github.com/pkg/errors"
	"github.com/x448/float16"
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
	//
	// Weights and masks are by convention given as extra `*Node` values in the labels slice,
	// when supported by the metrics.
	UpdateGraph(ctx *context.Context, labels, predictions []*Node) (metric *Node)

	// PrettyPrint is used to pretty-print a metric value, usually in a short form.
	PrettyPrint(value *tensors.Tensor) string

	// Reset metrics internal counters when starting a new evaluation.
	// Notice this may be called before UpdateGraph, and the metric should handle this without errors.
	Reset(ctx *context.Context)
}

// UpdateGo interface can be implemented by metrics that prefer to update their values during
// evaluation of a dataset in Go (as opposed to using a computation graph).
//
// These may be easier for calculating things like running quantiles where it's simpler
// to handle one element at a time, as opposed to process them as vectors (tensors).
type UpdateGo interface {
	Interface

	// UpdateGo is called for each batch with the resulting metric returned by UpdateGraph()
	UpdateGo(value *tensors.Tensor)

	// ReadGo can be called whenever one wants the current value of the target metric.
	// Typically it is called after UpdateGo is called for all batches of the dataset.
	ReadGo() *tensors.Tensor
}

const (
	// LossMetricType is the type of loss metrics.
	// Used to aggregate metrics of the same  type in the same plot.
	LossMetricType = "loss"

	// AccuracyMetricType is the type of loss metrics.
	// Used to aggregate metrics of the same  type in the same plot.
	AccuracyMetricType = "accuracy"

	// Scope used to store metrics helper variables (e.g.: running averages).
	Scope = "metrics"
)

// BaseMetricGraph is a graph building function of any metric that can be calculated stateless, without the need for
// any context. It should return a scalar, the mean for the given batch.
type BaseMetricGraph func(ctx *context.Context, labels, predictions []*Node) *Node

// PrettyPrintFn is a function to convert a metric value to a string.
type PrettyPrintFn func(value *tensors.Tensor) string

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
	result := m.metricFn(ctx, labels, predictions)
	if !result.Shape().IsScalar() {
		Panicf("metric %q should return a scalar, instead got shape %s", m.Name(), result.Shape())
	}
	return result
}

func (m *baseMetric) PrettyPrint(value *tensors.Tensor) string {
	if m.pPrintFn == nil {
		dtype := value.DType()
		isScalar := value.Shape().IsScalar()
		v := value.Value()
		if dtype.IsFloat() && isScalar {
			if dtype == dtypes.Float16 {
				v = v.(float16.Float16).Float32()
			} else if dtype == dtypes.BFloat16 {
				v = v.(bfloat16.BFloat16).Float32()
			}
			return fmt.Sprintf("%.3g", v)
		}
		return fmt.Sprintf("%s", v)
	}
	return m.pPrintFn(value)
}

func (m *baseMetric) Reset(_ *context.Context) {}

// NewBaseMetric creates a stateless metric from any BaseMetricGraph function, it will return the metric
// calculated solely on the last batch.
// pPrintFn can be left as nil, and a default will be used.
func NewBaseMetric(name, shortName, metricType string, metricFn BaseMetricGraph, pPrintFn PrettyPrintFn) Interface {
	return &baseMetric{
		name: name, shortName: shortName, metricType: metricType,
		metricFn: metricFn, pPrintFn: pPrintFn}
}

// MeanMetric implements a metric that keeps the mean of a metric.
type MeanMetric struct {
	baseMetric
	dynamicBatch bool
}

// NewMeanMetric creates a metric from any BaseMetricGraph function.
//
// It assumes the batch size (to weight the mean with each new result) is given by the first dimension of the labels' node.
// If you want all batches to count the same, sed WithDynamicBatch(false).
//
// `prettyPrintFn` can be left as nil, and a default will be used.
func NewMeanMetric(
	name, shortName, metricType string,
	metricFn BaseMetricGraph,
	prettyPrintFn PrettyPrintFn,
) *MeanMetric {
	return &MeanMetric{
		baseMetric: baseMetric{
			name:       name,
			shortName:  shortName,
			metricType: metricType,
			metricFn:   metricFn,
			pPrintFn:   prettyPrintFn,
		},
		dynamicBatch: true,
	}
}

// WithDynamicBatch sets whether the mean should attempt to weight each batch by its size, defined as the dimension
// of the first axis. Default is true.
//
// If set to false, each batch counts as 1, independent of its shape.
func (m *MeanMetric) WithDynamicBatch(dynamicBatch bool) *MeanMetric {
	m.dynamicBatch = dynamicBatch
	return m
}

// BatchSize returns the batch size (assumed first dimension) of the data node, casting
// it to the same dtype as data.
func BatchSize(data *Node) *Node {
	g := data.Graph()
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

// upPrecision promotes the precision of `x` if it is float16, to float32.
func upPrecision(x *Node) *Node {
	if x.DType() == dtypes.Float16 || x.DType() == dtypes.BFloat16 {
		x = ConvertDType(x, dtypes.Float32)
	}
	return x
}

func (m *MeanMetric) UpdateGraph(ctx *context.Context, labels, predictions []*Node) (metric *Node) {
	g := predictions[0].Graph()
	var result *Node
	err := TryCatch[error](func() { result = m.metricFn(ctx, labels, predictions) })
	if err != nil {
		panic(errors.WithMessagef(err, "failed building computation graph for mean metric %q", m.Name()))
	}
	if !result.Shape().IsScalar() {
		Panicf("metric %q should return a scalar, instead got shape %s", m.Name(), result.Shape())
	}

	// Up the precision for float16/bfloat16, often not enough.
	result = upPrecision(result)

	// Create scope in context for metrics state, and mark it as unchecked -- model variables
	// may be set for reuse, but metrics variables are not.
	ctx = ctx.Checked(false).In(Scope).In(m.ScopeName())
	dtype := result.DType()
	zero := shapes.CastAsDType(0, dtype)
	totalVar := ctx.VariableWithValue("total", zero).SetTrainable(false)
	if totalVar == nil {
		Panicf("variable nil building computation graph for mean metric %q", m.Name())
	}
	weightVar := ctx.VariableWithValue("weight", zero).SetTrainable(false)
	if weightVar == nil {
		Panicf("variable nil building computation graph for mean metric %q", m.Name())
	}

	total := totalVar.ValueGraph(g)
	previousWeight := weightVar.ValueGraph(g)
	var resultWeight *Node
	if m.dynamicBatch {
		resultWeight = BatchSize(predictions[0])
	} else {
		resultWeight = ScalarOne(g, weightVar.Shape().DType)
	}
	resultWeight = upPrecision(resultWeight)

	total = Add(total, Mul(result, resultWeight))
	weight := Add(previousWeight, resultWeight)
	mean := Div(total, weight)

	// Update variable values.
	weightVar.SetValueGraph(weight)
	totalVar.SetValueGraph(total)
	return mean
}

func (m *MeanMetric) Reset(ctx *context.Context) {
	ctx = ctx.Reuse().In(Scope).In(m.ScopeName())
	totalVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "total")
	if totalVar == nil {
		// Assume this was called before the graph was first built, so there is nothing to reset yet.
		return
	}
	totalVar.MustSetValue(tensors.FromAnyValue(shapes.CastAsDType(0, totalVar.MustValue().DType())))
	weightVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "weight")
	if weightVar != nil {
		weightVar.MustSetValue(tensors.FromAnyValue(shapes.CastAsDType(0, weightVar.MustValue().DType())))
	} else {
		Panicf("can't find variable \"weight\" in scope %q", ctx.Scope())
	}
}

// movingAverageMetric implements a metric that keeps the mean of a metric.
//
// It behaves just like a MeanMetric, but each new batch has weight of newExampleWeight, and
// the stored weight is capped at (1-newExampleWeight).
type movingAverageMetric struct {
	MeanMetric
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
func NewExponentialMovingAverageMetric(
	name, shortName, metricType string,
	metricFn BaseMetricGraph,
	pPrintFn PrettyPrintFn,
	newExampleWeight float64,
) Interface {
	return &movingAverageMetric{MeanMetric: MeanMetric{baseMetric: baseMetric{
		name: name, shortName: shortName, metricType: metricType,
		metricFn: metricFn, pPrintFn: pPrintFn}}, newExampleWeight: newExampleWeight}
}

// UpdateGraph implements metrics.Interface.
func (m *movingAverageMetric) UpdateGraph(ctx *context.Context, labels, predictions []*Node) (metric *Node) {
	g := predictions[0].Graph()
	var result *Node
	err := TryCatch[error](func() { result = m.metricFn(ctx, labels, predictions) })
	if err != nil {
		panic(errors.WithMessagef(err, "failed building computation graph for mean metric %q", m.Name()))
	}
	if !result.Shape().IsScalar() {
		Panicf("metric %q should return a scalar, instead got shape %s", m.Name(), result.Shape())
	}
	result = upPrecision(result)

	// Create scope in context for metrics state, and mark it as unchecked -- model variables
	// may be set for reuse, but metrics variables are not.
	ctx = ctx.Checked(false).In(Scope).In(m.ScopeName())
	dtype := result.DType()
	zero := shapes.CastAsDType(0, dtype)

	meanVar := ctx.VariableWithValue("mean", zero).SetTrainable(false)
	if meanVar == nil {
		Panicf("variable nil building computation graph for mean metric %q", m.Name())
	}
	mean := meanVar.ValueGraph(g)

	countVar := ctx.VariableWithValue("count", zero).SetTrainable(false)
	count := countVar.ValueGraph(g)
	count = Add(count, OnesLike(count))
	countVar.SetValueGraph(count)

	weight := Max(Const(g, shapes.CastAsDType(m.newExampleWeight, dtype)), Reciprocal(count))
	mean = Add(
		Mul(mean, OneMinus(weight)),
		Mul(result, weight),
	) // total are the values multiplied by weights, and then summed.
	meanVar.SetValueGraph(mean)

	return mean
}

// BinaryAccuracyGraph can be used in combination with New*Metric functions to build metrics for binary accuracy.
// It assumes predictions are probabilities, that labels are `{0, 1}`, and those predictions and labels have
// the same shape and dtype.
func BinaryAccuracyGraph(_ *context.Context, labels, predictions []*Node) *Node {
	prediction := predictions[0]
	g := prediction.Graph()
	if len(labels) != 1 {
		Panicf("BinaryAccuracy requires one labels tensor, got (%d) instead", len(labels))
	}
	label := ConvertDType(labels[0], prediction.DType())
	if !prediction.Shape().Equal(label.Shape()) {
		Panicf("prediction (%s) and label (%s) have different shapes, can't calculate binary accuracy",
			prediction.Shape(), label.Shape())
	}
	diff := Abs(Sub(label, prediction))
	// Accuracy is true if diff < 0.5. Notice this will take predictions of 0.5 to be false independent of label
	// (assuming labels are 1 or 0).
	dtype := prediction.DType()
	correctExamples := OneMinus(NonNegativeIndicator(Sub(diff, Const(g, shapes.CastAsDType(0.5, dtype)))))
	countExamples := Const(g, shapes.CastAsDType(correctExamples.Shape().Size(), correctExamples.DType()))
	return Div(ReduceAllSum(correctExamples), countExamples)
}

func accuracyPPrint(value *tensors.Tensor) string {
	return fmt.Sprintf("%.2f%%", shapes.ConvertTo[float64](value.Value())*100.0)
}

// NewMeanBinaryAccuracy returns a new binary accuracy metric with the given names.
func NewMeanBinaryAccuracy(name, shortName string) *MeanMetric {
	return NewMeanMetric(name, shortName, AccuracyMetricType, BinaryAccuracyGraph, accuracyPPrint)
}

// NewMovingAverageBinaryAccuracy returns a new binary accuracy metric with the given names.
// A typical value of newExampleWeight is 0.01, the smaller the value, the slower the moving average moves.
func NewMovingAverageBinaryAccuracy(name, shortName string, newExampleWeight float64) Interface {
	return NewExponentialMovingAverageMetric(
		name,
		shortName,
		AccuracyMetricType,
		BinaryAccuracyGraph,
		accuracyPPrint,
		newExampleWeight,
	)
}

// BinaryLogitsAccuracyGraph can be used in combination with New*Metric functions to build metrics for binary accuracy for logit.
// Notice 0s are considered a miss.
// It assumes predictions are logits, that labels are {0, 1} and that predictions and labels have the same size and dtype.
// The shape may be different (e.g.: `[batch_size, 1]` and `[batch_size]`), they will be reshaped to the
// logits shape before the accuracy is calculated.
//
// labels is converted to predictions dtype, and it's expected to convert to 1.0 (for true) or 0.0 for false.
// So booleans should work, as an int type that is 0 or 1.
func BinaryLogitsAccuracyGraph(_ *context.Context, labels, logits []*Node) *Node {
	logits0 := logits[0]
	g := logits0.Graph()
	if len(labels) != 1 {
		Panicf("BinaryLogitsAccuracyGraph requires one labels tensor, got (%d) instead", len(labels))
	}
	labels0 := ConvertDType(labels[0], logits0.DType())
	if logits0.DType() != labels0.DType() {
		Panicf("logits0 (%s) and labels0 (%s) have different dtypes, can't calculate binary accuracy",
			logits0.DType(), labels0.DType())

	}
	if logits0.Shape().Size() != labels0.Shape().Size() {
		Panicf(
			"logits0 (%s) and labels0 (%s) have different shapes (different total sizes), can't calculate binary accuracy",
			logits0.Shape(),
			labels0.Shape(),
		)
	}
	if !logits0.Shape().Equal(labels0.Shape()) {
		// They are the same size, so we assume the labels0 can simply be re-shaped.
		// Not strictly true, depending on how they are organized, but generally yes, and
		// this is very convenient functionality.
		labels0 = Reshape(labels0, logits0.Shape().Dimensions...)
	}

	dtype := logits0.DType()
	labels0 = Sub(labels0, Const(g, shapes.CastAsDType(0.5, dtype))) // Labels: -0.5 for false, +0.5 for true.
	correctExamples := PositiveIndicator(Mul(logits0, labels0))      // 0s are considered a miss.
	countExamples := Const(g, shapes.CastAsDType(correctExamples.Shape().Size(), correctExamples.DType()))
	mean := Div(ReduceAllSum(correctExamples), countExamples)
	return mean
}

// NewMeanBinaryLogitsAccuracy returns a new binary accuracy metric with the given names.
func NewMeanBinaryLogitsAccuracy(name, shortName string) *MeanMetric {
	return NewMeanMetric(name, shortName, AccuracyMetricType, BinaryLogitsAccuracyGraph, accuracyPPrint)
}

// NewMovingAverageBinaryLogitsAccuracy returns a new binary accuracy metric with the given names.
// A typical value of newExampleWeight is 0.01, the smaller the value, the slower the moving average moves.
func NewMovingAverageBinaryLogitsAccuracy(name, shortName string, newExampleWeight float64) Interface {
	return NewExponentialMovingAverageMetric(
		name,
		shortName,
		AccuracyMetricType,
		BinaryLogitsAccuracyGraph,
		accuracyPPrint,
		newExampleWeight,
	)
}

// SparseCategoricalAccuracyGraph returns the accuracy -- fraction of times argmax(logits)
// is the true label. It works for both probabilities or logits. Ties are considered misses.
// Labels is expected to be some integer type. And the returned dtype is the same as logits.
//
// Weights and mask can be given in the `labels` slice, following the labels themselves and they
// will be accounted for.
func SparseCategoricalAccuracyGraph(_ *context.Context, labels, logits []*Node) *Node {
	logits0 := logits[0]
	g := logits0.Graph()
	labels0 := labels[0]
	labelsShape := labels0.Shape()
	labelsDType := labels0.DType()
	labelsRank := labelsShape.Rank()
	logitsShape := logits0.Shape()
	logitsRank := logitsShape.Rank()
	logitsDType := logits0.DType()
	if !labelsShape.DType.IsInt() {
		Panicf("labels0 indices dtype (%s), it must be integer", labelsShape.DType)
	}
	if labelsRank != logitsRank {
		Panicf("labels0(%s) and logits0(%s) must have the same rank", labelsShape, logitsShape)
	}
	if labelsShape.Dimensions[labelsRank-1] != 1 {
		Panicf("labels0(%s) are expected to have the last dimension == 1, with the true/labeled category", labelsShape)
	}

	// Weights and masks: checks whether either are defined.
	weightsShape := shapes.Make(logitsDType, logits0.Shape().Dimensions[:logits0.Rank()-1]...)
	weights, mask := losses.CheckExtraLabelsForWeightsAndMask(weightsShape, labels[1:])
	modelChoices := ArgMax(logits0, -1, labelsDType)
	correctExamples := ConvertDType(
		Equal(modelChoices, Squeeze(labels0, -1)),
		logitsDType,
	) // correctExamples -> 0/1 per example.

	// Apply mask.
	if mask != nil {
		correctExamples = Where(mask, correctExamples, ZerosLike(correctExamples))
	}
	if weights != nil {
		correctExamples = Mul(weights, correctExamples)
	}

	var totalWeight *Node
	if weights == nil && mask == nil {
		// Simple count of examples.
		totalWeight = Scalar(g, logitsDType, float64(correctExamples.Shape().Size()))
	} else if weights == nil {
		// Count of # of elements in the mask set to true.
		totalWeight = ReduceAllSum(ConvertDType(mask, logitsDType))
	} else {
		// Since if mask != nil the corresponding weights will be set to zero, we just need to sum the
		// remaining weights.
		totalWeight = ReduceAllSum(weights)
	}
	return Div(ReduceAllSum(correctExamples), totalWeight)
}

// NewSparseCategoricalAccuracy returns a new sparse categorical accuracy metric with the given names.
// The accuracy is defined as the fraction of times argmax(logits) is the true label.
// It works for both probabilities or logits. Ties are considered misses.
// Labels is expected to be some integer type. And the returned dtype is the same as logits.
func NewSparseCategoricalAccuracy(name, shortName string) *MeanMetric {
	return NewMeanMetric(name, shortName, AccuracyMetricType, SparseCategoricalAccuracyGraph, accuracyPPrint)
}

// NewMovingAverageSparseCategoricalAccuracy returns a new sparse categorical accuracy metric with the given names.
// The accuracy is defined as the fraction of times argmax(logits) is the true label.
// It works for both probabilities or logits. Ties are considered misses.
// Labels is expected to be some integer type. And the returned dtype is the same as logits.
// A typical value of newExampleWeight is 0.01, the smaller the value, the slower the moving average moves.
func NewMovingAverageSparseCategoricalAccuracy(name, shortName string, newExampleWeight float64) Interface {
	return NewExponentialMovingAverageMetric(
		name,
		shortName,
		AccuracyMetricType,
		SparseCategoricalAccuracyGraph,
		accuracyPPrint,
		newExampleWeight,
	)
}
