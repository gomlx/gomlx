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

// Package losses have several standard losses that implement train.LossFn interface. They can also
// be called separately by custom losses.
//
// They all have the same signature that can be used by train.Trainer.
package losses

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/shapes"
)

const (
	Epsilon16 = 1e-4
	Epsilon32 = 1e-7
	Epsilon64 = 1e-8
)

func epsilonForDType(g *Graph, dtype dtypes.DType) *Node {
	var epsilon float64
	switch dtype {
	case dtypes.Float64:
		epsilon = Epsilon64
	case dtypes.Float32:
		epsilon = Epsilon32
	case dtypes.Float16:
		epsilon = Epsilon16
	default:
		Panicf("Unknown epsilon value for dtype %s", dtype)
	}
	return Const(g, shapes.CastAsDType(epsilon, dtype))
}

// MeanSquaredError returns the mean squared error between labels and predictions.
//
// labels and predictions must have the same shape.
//
// If there is an extra `labels` `*Node` with the shape of the `labels[0]` (usually simply `[bath_size]`),
// it is assumed to be weights tensor to be applied to the losses.
// If there is an extra `labels` `*Node` with booleans and the same dimensions as `labels[0]` (usually simply `batch_size`),
// it assumed to be a mask tensor to be applied to the losses.
func MeanSquaredError(labels, predictions []*Node) (loss *Node) {
	predictions0 := predictions[0]
	labels0 := labels[0]
	if !labels0.Shape().Equal(predictions0.Shape()) {
		Panicf("labels[0] (%s) and predictions[0] (%s) must have same shape", labels0.Shape(), predictions0.Shape())
	}
	weights, mask := CheckLabelsForWeightsAndMask(labels0.Shape(), labels)
	loss = Sub(labels0, predictions0)
	loss = Mul(loss, loss)

	if weights != nil {
		loss = Mul(loss, weights)
	}
	if mask != nil {
		loss = Where(mask, loss, ZerosLike(loss))
	}
	loss = ReduceAllMean(loss)
	return loss
}

// CheckLabelsForWeightsAndMask in the labels slice of tensors -- it is assumed that labels[0] are the actual labels, so
// they are not considered.
//
// `weightsShape` is the expected shape for weights (if present) and the dimensions for a mask (if present), although
// a mask is assumed to be of dtype `Bool`.
//
// If there is an extra `labels` `*Node` with the shape of `weightsShape`, it is assumed to be weights.
// If there is an extra `labels` `*Node` with booleans with the same dimension as `weightsShape`, it is assumed to be a mask.
func CheckLabelsForWeightsAndMask(weightsShape shapes.Shape, labels []*Node) (weights, mask *Node) {
	maskShape := shapes.Make(dtypes.Bool, weightsShape.Dimensions...)
	// We skip labels[0] because that contains the actual labels.
	for ii, extra := range labels[1:] {
		if weights == nil && extra.Shape().Equal(weightsShape) {
			weights = extra
		} else if mask == nil && extra.Shape().Equal(maskShape) {
			mask = extra
		} else {
			Panicf("labels ([]*Node) provided by the dataset to the loss function has extra tensors whose use is unknown: labels[%d].shape=%s "+
				"-- label weights shape would be %s, labels mask shape would be %s", ii+1, extra.Shape(), weightsShape, maskShape)
		}
	}
	return
}

// MeanAbsoluteError returns the mean absolute error between labels and predictions.
// It uses only the first element of each.
//
// labels and predictions must have the same shape.
//
// If there is an extra `labels` `*Node` with the shape of the `labels[0]` (usually simply `[bath_size]`),
// it is assumed to be weights tensor to be applied to the losses.
// If there is an extra `labels` `*Node` with booleans and the same dimensions as `labels[0]` (usually simply `batch_size`),
// it assumed to be a mask tensor to be applied to the losses.
func MeanAbsoluteError(labels, predictions []*Node) (loss *Node) {
	predictions0 := predictions[0]
	labels0 := labels[0]
	if !labels0.Shape().Equal(predictions0.Shape()) {
		Panicf("labels[0] (%s) and predictions[0] (%s) must have same shape", labels0.Shape(), predictions0.Shape())
	}

	loss = Abs(Sub(labels0, predictions0))

	weights, mask := CheckLabelsForWeightsAndMask(labels0.Shape(), labels)
	if weights != nil {
		loss = Mul(loss, weights)
	}
	if mask != nil {
		loss = Where(mask, loss, ZerosLike(loss))
	}
	loss = ReduceAllMean(loss)
	return
}

// BinaryCrossentropy returns the cross-entropy loss between labels and predictions,
// for binary classification tasks.
//
// labels and predictions must have the same shape.
//
// It *does not* reduce-mean the losses, they are returned individually for each element of the batch and need
// to be ReduceAllMean (usually the mean, but it could be the sum also) before used for training.
//
// If there is an extra `labels` `*Node` with the shape of the `labels[0]` (usually simply `[bath_size]`),
// it is assumed to be weights tensor to be applied to the losses.
// If there is an extra `labels` `*Node` with booleans and the same dimensions as `labels[0]` (usually simply `batch_size`),
// it assumed to be a mask tensor to be applied to the losses.
func BinaryCrossentropy(labels, predictions []*Node) *Node {
	predictions0 := predictions[0]
	labels0 := labels[0]
	if !labels0.Shape().Equal(predictions0.Shape()) {
		Panicf("labels[0] (%s) and predictions[0] (%s) must have same shape", labels0.Shape(), predictions0.Shape())
	}
	losses := Neg(Add(
		Mul(labels0, Log(predictions0)),
		Mul(OneMinus(labels0), Log(OneMinus(predictions0)))))

	weights, mask := CheckLabelsForWeightsAndMask(labels0.Shape(), labels)
	if weights != nil {
		losses = Mul(losses, weights)
	}
	if mask != nil {
		losses = Where(mask, losses, ZerosLike(losses))
	}
	return losses
}

// BinaryCrossentropyLogits returns the cross-entropy loss between labels and `sigmoid(logits)`,
// for binary classification tasks. It assumes the predictions are given by `sigmoid(logits)`.
// This is a more numerically stable and faster implementation than actually taking the sigmoid of
// the logits and using the equivalent BinaryCrossentropy.
// labels and logits must have the same shape.
//
// It *does not* reduce-mean the losses, they are returned individually for each element of the batch and need
// to be ReduceAllMean (usually the mean, but it could be the sum also) before used for training.
//
// See mathematical derivation of the stable solution in
// https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
//
// If there is an extra `labels` `*Node` with the shape of the `labels[0]` (usually simply `[bath_size]`),
// it is assumed to be weights tensor to be applied to the losses.
// If there is an extra `labels` `*Node` with booleans and the same dimensions as `labels[0]` (usually simply `batch_size`),
// it assumed to be a mask tensor to be applied to the losses.
func BinaryCrossentropyLogits(labels, logits []*Node) *Node {
	logits0 := logits[0]
	labels0 := labels[0]
	if logits0.Shape().Size() != labels0.Shape().Size() {
		Panicf("labels[0] (%s) and logits[0] (%s) have incompatible shapes", labels0.Shape(), logits0.Shape())
	}
	if logits0.Rank() != labels0.Rank() {
		labels0 = Reshape(labels0, logits0.Shape().Dimensions...)
	}
	logPart := Log1P(Exp(Neg(Abs(logits0))))
	prodPart := Mul(logits0, labels0)
	maxPart := Max(logits0, ZerosLike(logits0))
	losses := Add(Sub(maxPart, prodPart), logPart)

	weights, mask := CheckLabelsForWeightsAndMask(labels0.Shape(), labels)
	if weights != nil {
		losses = Mul(losses, weights)
	}
	if mask != nil {
		losses = Where(mask, losses, ZerosLike(losses))
	}
	return losses
}

// SparseCategoricalCrossEntropyLogits returns the cross-entropy loss of the logits, given the labels.
// The labels are provided in "sparse" format, that is, integer numbers from 0 to logits dimension-1.
// labels and logits must have the same rank, and labels last dimension must be 1.
//
// It *does not* reduce-mean the losses, they are returned individually for each element of the batch and need
// to be ReduceAllMean (usually the mean, but it could be the sum also) before used for training.
//
// If there is an extra `labels` `*Node` with the shape of logits without the last axis, it assumed to be weights to the losses.
// If there is an extra `labels` `*Node` with booleans with the same dimensions as logits without the last axis, it assumed to be a mask.
func SparseCategoricalCrossEntropyLogits(labels, logits []*Node) *Node {
	logits0 := logits[0]
	labels0 := labels[0]
	labelsShape := labels0.Shape()
	labelsRank := labelsShape.Rank()
	logitsShape := logits0.Shape()
	logitsRank := logitsShape.Rank()
	if !labelsShape.DType.IsInt() {
		Panicf("labels0 indices dtype (%s), it must be integer", labelsShape.DType)
	}
	if labelsRank != logitsRank {
		Panicf("labels0(%s) and logits0(%s) must have the same rank", labelsShape, logitsShape)
	}
	if labelsShape.Dimensions[labelsRank-1] != 1 {
		Panicf("labels0(%s) are expected to have the last dimension == 1, with the true/labeled category", labelsShape)
	}
	weightsShape := shapes.Make(logits0.DType(), labelsShape.Dimensions[:labelsRank-1]...)
	weights, mask := CheckLabelsForWeightsAndMask(weightsShape, labels)

	// Remove last dimension, it will be re-added by OneHot
	reducedLabels := Reshape(labels0, labels0.Shape().Dimensions[:labelsRank-1]...)
	labelsValues := OneHot(reducedLabels, logitsShape.Dimensions[logitsRank-1], logitsShape.DType)
	return categoricalCrossEntropyLogitsImpl(labelsValues, logits0, weights, mask)
}

// CategoricalCrossEntropyLogits returns the cross-entropy loss of the logits, given the labels.
// The labels are provided in "dense" format, they should have the exact same shape as logits, and be set 1 for
// the true (labeled) category, and 0 for the others -- or any other distribution that sum to 1.
//
// It *does not* reduce-mean the losses, they are returned individually for each element of the batch and need
// to be ReduceAllMean (usually the mean, but it could be the sum also) before used for training.
//
// If there is an extra `labels` `*Node` with the shape of logits without the last axis (usually simply `[bath_size]`),
// it assumed to be weights to the losses.
// If there is an extra `labels` `*Node` with booleans with the same dimensions as logits without the last axis
// (usually simply `batch_size`), it assumed to be a mask.
//
// TODO: implement faster version with logits, see https://github.com/tensorflow/tensorflow/blob/359c3cdfc5fabac82b3c70b3b6de2b0a8c16874f/tensorflow/python/ops/nn_ops.py#L4051
func CategoricalCrossEntropyLogits(labels, logits []*Node) *Node {
	logits0 := logits[0]
	labels0 := labels[0]
	weightsShape := shapes.Make(logits0.DType(), labels0.Shape().Dimensions[:labels0.Rank()-1]...)
	weights, mask := CheckLabelsForWeightsAndMask(weightsShape, labels)
	return categoricalCrossEntropyLogitsImpl(labels0, logits0, weights, mask)
}

// categoricalCrossEntropyLogitsImpl implements CategoricalCrossEntropyLogits taking as input the
// nodes only (as opposed to slices).
func categoricalCrossEntropyLogitsImpl(labels, logits, weights, mask *Node) *Node {
	shape := labels.Shape()
	if !shape.Equal(logits.Shape()) {
		Panicf("labels(%s) and logits(%s) must different shapes", shape, logits.Shape())
	}
	predictions := Softmax(logits)
	return categoricalCrossEntropyImpl(labels, predictions, weights, mask)
}

// CategoricalCrossEntropy returns the cross-entropy loss of the predictions, given the labels.
// The labels are provided in "dense" format, they should have the exact same shape as predictions, and be set 1 for
// the true (labeled) category, and 0 for the others -- or any other distribution that sum to 1.
// predictions should hold probabilities that must sum to 1.0.
//
// It *does not* reduce-mean the losses, they are returned individually for each element of the batch and need
// to be ReduceAllMean (usually the mean, but it could be the sum also) before used for training.
//
// If there is an extra `labels` `*Node` with the shape of logits without the last axis (usually simply `[bath_size]`),
// it assumed to be weights to the losses.
// If there is an extra `labels` `*Node` with booleans with the same dimensions as logits without the last axis
// (usually simply `batch_size`), it assumed to be a mask.
func CategoricalCrossEntropy(labels, predictions []*Node) *Node {
	weightsShape := shapes.Make(predictions[0].DType(), labels[0].Shape().Dimensions[:labels[0].Rank()-1]...)
	weights, mask := CheckLabelsForWeightsAndMask(weightsShape, labels)
	return categoricalCrossEntropyImpl(labels[0], predictions[0], weights, mask)
}

// categoricalCrossEntropyImpl implements CategoricalCrossEntropy.
func categoricalCrossEntropyImpl(labels, predictions, weights, mask *Node) *Node {
	g := predictions.Graph()
	shape := labels.Shape()
	dtype := labels.DType()
	if !shape.Equal(predictions.Shape()) {
		Panicf("labels(%s) and predictions(%s) must different shapes", shape, predictions.Shape())
	}
	epsilon := epsilonForDType(g, dtype)
	predictions = Clip(predictions, epsilon, Sub(ScalarOne(g, dtype), epsilon))
	losses := ReduceSum(Neg(Mul(labels, Log(predictions))), -1)
	// Losses will usually be shaped `[batch_size]` now, ready to apply weights multiplication and/or a mask.
	if weights != nil {
		losses = Mul(losses, weights)
	}
	if mask != nil {
		losses = Where(mask, losses, ZerosLike(losses))
	}
	return losses
}
