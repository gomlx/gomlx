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
	. "github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
)

// LossFn is the interface used bye train.Trainer to train models.
//
// It takes as inputs the labels and predictions:
//   - labels comes from the dataset.
//   - predictions comes from the model.
//   - the returned loss will be graph.ReduceAllMean by train.Trainer to a scalar, before being used for gradient descent.
//     That means that the loss function is free to return a loss per example or an already reduced scalar loss.
//
// Most of the predefined losses in package `gomlx/ml/train/losses` assume labels and predictions are
// both of length one. For multi-head models, it's very easy to write a small custom LossFn that splits
// the slice and send each label/prediction pair to a predefined loss.
type LossFn func(labels, predictions []*Node) (loss *Node)

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
// If there is an extra element in the input labels with the shape of the labels[0] (usually simply `[bath_size]`),
// it is assumed to be weights tensor to be applied to the losses.
// If there is an extra element in the input labels  with booleans and the same dimensions as `labels[0]` (usually
// simply `batch_size`), it assumed to be a mask tensor to be applied to the losses.
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
// If weights and masks are present, weights are converted to zero for masked out values (where mask is false).
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
	if weights != nil && mask != nil {
		weights = Where(mask, weights, ZerosLike(weights))
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
// labels is converted to predictions dtype, and it's expected to convert to 1.0 (for true) or 0.0 for false.
// So booleans should work, as an int type that is 0 or 1.
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
	labels0 := ConvertDType(labels[0], predictions0.DType())
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
// labels is converted to predictions dtype, and it's expected to convert to 1.0 (for true) or 0.0 for false.
// So booleans should work, as an int type that is 0 or 1.
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
	labels0 := ConvertDType(labels[0], logits0.DType())
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

// categoricalCrossEntropyLogitsImpl implements CategoricalCrossEntropyLogits.
func categoricalCrossEntropyLogitsImpl(labels, logits, weights, mask *Node) *Node {
	shape := labels.Shape()
	if !shape.Equal(logits.Shape()) {
		Panicf("labels(%s) and logits(%s) must different shapes", shape, logits.Shape())
	}
	var expandedMask *Node
	if mask != nil {
		expandedMask = BroadcastToShape(InsertAxes(mask, -1), logits.Shape())
		logits = Where(expandedMask, logits, ZerosLike(logits))
	}
	logPredictions := LogSoftmax(logits)
	losses := ReduceSum(Neg(Mul(labels, logPredictions)), -1)
	// Losses will usually be shaped `[batch_size]` now.
	if weights != nil {
		losses = Mul(losses, weights)
	}
	if mask != nil {
		losses = Where(mask, losses, ZerosLike(losses))
	}
	return losses
}

// CategoricalCrossEntropy returns the cross-entropy loss of the predictions, given the labels.
// The labels are provided in "dense" format, they should have the exact same shape as predictions, and be set 1 for
// the true (labeled) category, and 0 for the others (one-hot encoding) -- or any other distribution that sums to 1.
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
	predictions = Clip(predictions, epsilon, OneMinus(epsilon))
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

var (
	// ParamHuberLossDelta is the name of the hyperparameter that defines the Huber loss delta.
	// See HuberLossBuilder.
	// It defaults to 1.0
	ParamHuberLossDelta = "huber_loss_delta"
)

// MakeHuberLoss returns a Huber loss function: it's similar to an L2 (MeanSquaredLoss) close to the target,
// and it becomes L1 (linear) away from the target.
//
// The delta parameter configures the range where the loss behaves as L2: if the prediction is further than
// delta it becomes linear. It also defines the slope. A good default value is 1.0.
//
// For the returned loss function:
//   - If there is an extra element in the input labels with the shape of the labels[0] (usually simply `[bath_size]`),
//     it is assumed to be weights tensor to be applied to the losses.
//   - If there is an extra element in the input labels  with booleans and the same dimensions as `labels[0]` (usually
//     simply `batch_size`), it assumed to be a mask tensor to be applied to the losses.
//   - The loss is returned per element, and not automatically reduced. train.Trainer will by default take the
//     mean of it.
//
// See https://en.wikipedia.org/wiki/Huber_loss
func MakeHuberLoss(delta float64) LossFn {
	if delta <= 0.0 {
		Panicf("MakeHuberLoss requires delta > 0 (1.0 being a good default), delta=%f given", delta)
	}
	return func(labels, predictions []*Node) (loss *Node) {
		predictions0 := predictions[0]
		g := predictions0.Graph()
		dtype := predictions0.DType()
		labels0 := labels[0]
		if !labels0.Shape().Equal(predictions0.Shape()) {
			Panicf("labels[0] (%s) and predictions[0] (%s) must have same shape", labels0.Shape(), predictions0.Shape())
		}
		weights, mask := CheckLabelsForWeightsAndMask(labels0.Shape(), labels)

		// Calculate Huber loss.
		deltaConst := Scalar(g, dtype, delta)
		absErrors := Abs(Sub(labels0, predictions0))
		quadratic := Min(absErrors, deltaConst)
		// Same as max(absErrors - deltaConst, 0) but avoids potentially doubling gradient. (From Jax implementation)
		linear := Sub(absErrors, quadratic)
		loss = Add(
			MulScalar(Square(quadratic), 0.5),
			Mul(deltaConst, linear),
		)

		// Apply weights and mask.
		if weights != nil {
			loss = Mul(loss, weights)
		}
		if mask != nil {
			loss = Where(mask, loss, ZerosLike(loss))
		}
		return loss
	}
}

// Computes the pairwise distance matrix with numerical stability.

//     output[i, j] = || embeddings[i, :] - embeddings[j, :] ||_2

//     Args:
//       embeddings: 2-D Tensor of size `[number of data, embeddings dimension]`.
//       squared: Boolean, whether or not to square the pairwise distances.

// Returns:
//
//	pairwise distance: 2-D Tensor of size `[number of data, number of data]`.
func pairwiseDistance(embeddings *Node, squared bool) *Node {
	// Distances ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
	pairwiseDistancesSquared := Add(
		Add(
			L2NormSquare(embeddings, 1),
			L2NormSquare(Transpose(embeddings, 0, 1), 0),
		),
		MulScalar(MatMul(embeddings, Transpose(embeddings, 0, 1)), -2.0),
	)
	// Deal with numerical inaccuracies. Set small negatives to zero.
	pairwiseDistancesSquaredZeros := Max(pairwiseDistancesSquared, ZerosLike(pairwiseDistancesSquared))
	// Optionally take the sqrt.
	pairwiseDistances := pairwiseDistancesSquaredZeros
	if !squared {
		// Get the mask where the zero distances are at.
		zeros := ZerosLike(pairwiseDistancesSquaredZeros)
		errorMask := LessOrEqual(pairwiseDistancesSquaredZeros, zeros)
		epsilon := epsilonForDType(pairwiseDistancesSquaredZeros.Graph(), pairwiseDistancesSquaredZeros.DType())
		pairwiseDistancesSqrt := Sqrt(Where(errorMask, epsilon, pairwiseDistancesSquaredZeros))
		//  Undo conditionally adding epsilon
		pairwiseDistances = Where(errorMask, zeros, pairwiseDistancesSqrt)
	}

	//  Explicitly set diagonals to zero.
	return Where(
		Diagonal(pairwiseDistances.Graph(), pairwiseDistances.Shape().Dim(0)),
		ZerosLike(pairwiseDistances),
		pairwiseDistances,
	)
}

// Computes the angular distance matrix.

//     output[i, j] = 1 - cosine_similarity(feature[i, :], feature[j, :])

//     Args:
//       feature: 2-D Tensor of size `[number of data, feature dimension]`.

// Returns:
//
//	angular_distances: 2-D Tensor of size `[number of data, number of data]`.
func angularDistance(embeddings *Node) *Node {
	// normalize input
	embeddingsN := L2Normalize(embeddings, 1)
	// create adjacent matrix of cosine similarity
	angularDistances := Add(OnesLike(embeddingsN), MulScalar(MatMul(embeddingsN, Transpose(embeddingsN, 0, 1)), -1.0))
	// ensure all distances > 1e-16
	angularDistancesZeros := MaxScalar(angularDistances, 0.0)
	return angularDistancesZeros
}
