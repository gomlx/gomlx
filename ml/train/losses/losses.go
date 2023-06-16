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

const Epsilon = 1e-7

// MeanSquaredError returns the mean squared error between labels and predictions.
// It uses only the first element of each.
//
// labels and predictions must have the same shape.
func MeanSquaredError(labels, predictions []*Node) (loss *Node) {
	predictions0 := predictions[0]
	labels0 := labels[0]
	g := labels0.Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}
	if !labels0.Shape().Eq(predictions0.Shape()) {
		g.SetErrorf("labels0 (%s) and predictions0 (%s) must have same shape", labels0.Shape(), predictions0.Shape())
		return g.InvalidNode()
	}
	loss = Sub(labels0, predictions0)
	loss = Mul(loss, loss)
	loss = ReduceAllMean(loss)
	return
}

// MeanAbsoluteError returns the mean absolute error between labels and predictions.
// It uses only the first element of each.
//
// labels and predictions must have the same shape.
func MeanAbsoluteError(labels, predictions []*Node) (loss *Node) {
	predictions0 := predictions[0]
	labels0 := labels[0]
	g := labels0.Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}
	if !labels0.Shape().Eq(predictions0.Shape()) {
		g.SetErrorf("labels0 (%s) and predictions0 (%s) must have same shape", labels0.Shape(), predictions0.Shape())
		return g.InvalidNode()
	}
	return ReduceAllMean(Abs(Sub(labels0, predictions0)))
}

// BinaryCrossentropy returns the cross-entropy loss between labels and predictions,
// for binary classification tasks.
//
// labels and predictions must have the same shape.
//
// It *does not* reduce-mean the losses, they are returned individually for each element of the batch and need
// to be ReduceAllMean (usually the mean, but it could be the sum also) before used for training.
func BinaryCrossentropy(labels, predictions []*Node) *Node {
	predictions0 := predictions[0]
	labels0 := labels[0]
	g := labels0.Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}
	if !labels0.Shape().Eq(predictions0.Shape()) {
		g.SetErrorf("labels0 (%s) and predictions0 (%s) must have same shape", labels0.Shape(), predictions0.Shape())
		return g.InvalidNode()
	}
	loss := Neg(Add(
		Mul(labels0, Log(predictions0)),
		Mul(OneMinus(labels0), Log(OneMinus(predictions0)))))
	return loss
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
func BinaryCrossentropyLogits(labels, logits []*Node) *Node {
	logits0 := logits[0]
	labels0 := labels[0]
	g := logits0.Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}
	if logits0.Shape().Size() != labels0.Shape().Size() {
		g.SetErrorf("labels0 (%s) and logits0 (%s) have incompatible shapes", labels0.Shape(), logits0.Shape())
	}
	if logits0.Rank() != labels0.Rank() {
		labels0 = Reshape(labels0, logits0.Shape().Dimensions...)
	}
	logPart := Log1p(Exp(Neg(Abs(logits0))))
	prodPart := Mul(logits0, labels0)
	maxPart := Max(logits0, ZerosLike(logits0))
	loss := Add(Sub(maxPart, prodPart), logPart)
	return loss
}

// SparseCategoricalCrossEntropyLogits returns the cross-entropy loss of the logits, given the labels.
// The labels are provided in "sparse" format, that is, integer numbers from 0 to logits dimension-1.
// labels and logits must have the same rank, and labels last dimension must be 1.
//
// It *does not* reduce-mean the losses, they are returned individually for each element of the batch and need
// to be ReduceAllMean (usually the mean, but it could be the sum also) before used for training.
func SparseCategoricalCrossEntropyLogits(labels, logits []*Node) *Node {
	logits0 := logits[0]
	labels0 := labels[0]
	g := logits0.Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}
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

	// Remove last dimension, it will be re-added by OneHot
	reducedLabels := Reshape(labels0, labels0.Shape().Dimensions[:labelsRank-1]...)
	labelsValues := OneHot(reducedLabels, logitsShape.Dimensions[logitsRank-1], logitsShape.DType)
	return categoricalCrossEntropyLogitsImpl(labelsValues, logits0)
}

// CategoricalCrossEntropyLogits returns the cross-entropy loss of the logits, given the labels.
// The labels are provided in "dense" format, they should have the exact same shape as logits, and be set 1 for
// the true (labeled) category, and 0 for the others -- or any other distribution that sum to 1.
//
// It *does not* reduce-mean the losses, they are returned individually for each element of the batch and need
// to be ReduceAllMean (usually the mean, but it could be the sum also) before used for training.
//
// TODO: implement faster version with logits, see https://github.com/tensorflow/tensorflow/blob/359c3cdfc5fabac82b3c70b3b6de2b0a8c16874f/tensorflow/python/ops/nn_ops.py#L4051
func CategoricalCrossEntropyLogits(labels, logits []*Node) *Node {
	logits0 := logits[0]
	labels0 := labels[0]
	return categoricalCrossEntropyLogitsImpl(labels0, logits0)
}

// categoricalCrossEntropyLogitsImpl implements CategoricalCrossEntropyLogits taking as input the
// nodes only (as opposed to slices).
func categoricalCrossEntropyLogitsImpl(labels, logits *Node) *Node {
	g := logits.Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}
	shape := labels.Shape()
	if !shape.Eq(logits.Shape()) {
		g.SetErrorf("labels(%s) and logits(%s) must different shapes", shape, logits.Shape())
		return g.InvalidNode()
	}
	predictions := Softmax(logits)
	return categoricalCrossEntropyImpl(labels, predictions)
}

// CategoricalCrossEntropy returns the cross-entropy loss of the predictions, given the labels.
// The labels are provided in "dense" format, they should have the exact same shape as predictions, and be set 1 for
// the true (labeled) category, and 0 for the others -- or any other distribution that sum to 1.
// predictions should hold probabilities that must sum to 1.0.
//
// It *does not* reduce-mean the losses, they are returned individually for each element of the batch and need
// to be ReduceAllMean (usually the mean, but it could be the sum also) before used for training.
func CategoricalCrossEntropy(labels, predictions []*Node) *Node {
	return categoricalCrossEntropyImpl(labels[0], predictions[0])
}

// categoricalCrossEntropyImpl implements CategoricalCrossEntropy.
func categoricalCrossEntropyImpl(labels, predictions *Node) *Node {
	g := predictions.Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}
	shape := labels.Shape()
	dtype := labels.DType()
	if !shape.Eq(predictions.Shape()) {
		g.SetErrorf("labels(%s) and predictions(%s) must different shapes", shape, predictions.Shape())
		return g.InvalidNode()
	}
	epsilon := Const(g, shapes.CastAsDType(Epsilon, dtype))
	predictions = Clip(predictions, epsilon, Sub(ScalarOne(g, dtype), epsilon))
	return Neg(ReduceSum(Mul(labels, Log(predictions)), -1))
}
