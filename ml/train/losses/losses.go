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
	"strings"

	. "github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
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

var (
	// ParamLoss defines the loss to use (the value of the hyperparameter is a string),
	// when using LossFromContext.
	//
	// See enumeration Type for accepted loss types.
	//
	// Some losses may have extra parameters, also read from the context hyperparameters -- e.g.:
	// MakeHuberLossFromContext and MakeAdaptivePowerLossFromContext.
	ParamLoss = "loss"
)

// Type of loss, an enumeration of losses supported by
type Type int

//go:generate enumer -type=Type -trimprefix=Type -transform=snake -values -text -json -yaml losses.go

const (
	// TypeMAE represent the MeanAbsoluteError loss.
	TypeMAE Type = iota

	// TypeMSE represents the MeanSquaredError loss.
	TypeMSE

	// TypeHuber represents the Huber loss, see MakeHuberLoss.
	TypeHuber

	// TypeAPL represents the Adaptive-Power-Loss, see MakeAdaptivePowerLoss.
	TypeAPL

	// TypeBinCross represents BinaryCrossentropy.
	TypeBinCross

	// TypeBinCrossLogits represents BinaryCrossentropyLogits.
	TypeBinCrossLogits

	// TypeSparseCross represents CategoricalCrossEntropy.
	TypeCategoricalCross

	// TypeBinCrossLogits represents CategoricalCrossEntropyLogits.
	TypeCategoricalCrossLogits

	// TypeSparseCrossLogits represents SparseCategoricalCrossEntropyLogits
	TypeSparseCrossLogits

	// TypeTriplet
	TypeTriplet
)

// LossFromContext takes the value from the ParamLoss hyperparameter as a string and
// returns or creates the corresponding loss. It defaults to "mae".
//
// Useful for projects where more than one loss matches the problem underlying optimization goal.
//
// It returns an error if the configured loss is unknown.
func LossFromContext(ctx *context.Context) (LossFn, error) {
	lossName := context.GetParamOr(ctx, ParamLoss, "mae")
	lossType, err := TypeString(lossName)
	if err != nil {
		err = errors.Wrapf(err, "invalid value %q for hyperparameter %q, known losses are: \"%s\"",
			lossName, ParamLoss, strings.Join(TypeStrings(), "\", \""))
		return nil, err
	}
	switch lossType {
	case TypeMAE:
		return MeanAbsoluteError, nil
	case TypeMSE:
		return MeanSquaredError, nil
	case TypeAPL:
		return MakeAdaptivePowerLossFromContext(ctx), nil
	case TypeHuber:
		return MakeHuberLossFromContext(ctx), nil
	case TypeBinCross:
		return BinaryCrossentropy, nil
	case TypeBinCrossLogits:
		return BinaryCrossentropyLogits, nil
	case TypeCategoricalCross:
		return CategoricalCrossEntropy, nil
	case TypeCategoricalCrossLogits:
		return CategoricalCrossEntropyLogits, nil
	case TypeSparseCrossLogits:
		return SparseCategoricalCrossEntropyLogits, nil
	case TypeTriplet:
		return MakeTripletLossFromContext(ctx), nil
	default:
		return nil, errors.Errorf("Unknown loss type %q set for hyperparameter %q, known losses are \"%s\"",
			lossType, ParamLoss, strings.Join(TypeStrings(), "\", \""))
	}
}

// CheckExtraLabelsForWeightsAndMask takes the remainder slice of labels tensor (so without the actual labels values),
// and separates a mask (bool) and weights (float), which can be provided in any order.
//
// `weightsShape` is the expected shape for weights (if present) and the dimensions for a mask (if present), although
// a mask is assumed to be of dtype `Bool`.
//
// If weights and masks are present, weights are converted to zero for masked out values (where mask is false).
//
// It raises an exception (panic) if there are more or unknown shaped labels.
//
// This function are used by loss implementations to help handle mask and weights.
func CheckExtraLabelsForWeightsAndMask(weightsShape shapes.Shape, labels []*Node) (weights, mask *Node) {
	maskShape := shapes.Make(dtypes.Bool, weightsShape.Dimensions...)
	for ii, extra := range labels {
		if mask == nil && extra.Shape().Equal(maskShape) {
			mask = extra
		} else if weights == nil && extra.Shape().Equal(weightsShape) {
			weights = extra
		} else {
			Panicf("extra labels ([]*Node) provided by the dataset to the loss function has extra tensors whose use is unknown: labels[%d].shape=%s "+
				"-- label weights shape would be %s, labels mask shape would be %s", ii+1, extra.Shape(), weightsShape, maskShape)
		}
	}
	if weights != nil && mask != nil {
		weights = Where(mask, weights, ZerosLike(weights))
	}
	return
}

// MeanSquaredError returns the mean squared error between labels and predictions.
//
// labels and predictions must have the same shape.
//
// Labels can have 2 optional extra values (in any order):
//
//   - mask: a boolean mask of shape [batchSize] set to true for values to be used, and false for those to be ignored.
//     Typically used for padding. The returned mean loss takes in consideration the mask.
//   - weights: a float value of shape [batchSize] with the relative weights to be applied to each example.
func MeanSquaredError(labels, predictions []*Node) (loss *Node) {
	predictions0 := predictions[0]
	labels0 := labels[0]
	if !labels0.Shape().Equal(predictions0.Shape()) {
		Panicf("labels[0] (%s) and predictions[0] (%s) must have same shape", labels0.Shape(), predictions0.Shape())
	}
	loss = Sub(labels0, predictions0)
	loss = Mul(loss, loss)

	// Factor in weights and mask.
	weights, mask := CheckExtraLabelsForWeightsAndMask(labels0.Shape(), labels[1:])
	if weights != nil {
		loss = Mul(loss, weights)
	}
	if mask != nil {
		loss = Where(mask, loss, ZerosLike(loss))
		if !loss.IsScalar() {
			loss = MaskedReduceAllMean(loss, mask)
		}
	} else if !loss.IsScalar() {
		loss = ReduceAllMean(loss)
	}
	return loss
}

// MeanAbsoluteError returns the mean absolute error between labels and predictions.
// It uses only the first element of each. labels and predictions must have the same shape.
//
// Labels can have 2 optional extra values (in any order):
//
//   - mask: a boolean mask of shape [batchSize] set to true for values to be used, and false for those to be ignored.
//     Typically used for padding. The returned mean loss takes in consideration the mask.
//   - weights: a float value of shape [batchSize] with the relative weights to be applied to each example.
func MeanAbsoluteError(labels, predictions []*Node) (loss *Node) {
	predictions0 := predictions[0]
	labels0 := labels[0]
	if !labels0.Shape().Equal(predictions0.Shape()) {
		Panicf("labels[0] (%s) and predictions[0] (%s) must have same shape", labels0.Shape(), predictions0.Shape())
	}
	loss = Abs(Sub(labels0, predictions0))

	// Factor in weights and mask.
	weights, mask := CheckExtraLabelsForWeightsAndMask(labels0.Shape(), labels[1:])
	if weights != nil {
		loss = Mul(loss, weights)
	}
	if mask != nil {
		loss = Where(mask, loss, ZerosLike(loss))
		if !loss.IsScalar() {
			loss = MaskedReduceAllMean(loss, mask)
		}
	} else if !loss.IsScalar() {
		loss = ReduceAllMean(loss)
	}
	return loss
}

// BinaryCrossentropy returns the cross-entropy loss between labels and predictions,
// for binary classification tasks.
//
// labels and predictions must have the same dimensions.
//
// labels is converted to predictions dtype, and it's expected to convert to 1.0 (for true) or 0.0 for false.
// So labels as booleans should work, as well as labels as int type that is 0 or 1.
//
// Labels can have 2 optional extra values (in any order):
//
//   - mask: a boolean mask of shape [batchSize] set to true for values to be used, and false for those to be ignored.
//     Typically used for padding. The returned mean loss takes in consideration the mask.
//   - weights: a float value of shape [batchSize] with the relative weights to be applied to each example.
func BinaryCrossentropy(labels, predictions []*Node) *Node {
	predictions0 := predictions[0]
	labels0 := ConvertDType(labels[0], predictions0.DType())
	if !labels0.Shape().Equal(predictions0.Shape()) {
		Panicf("labels[0] (%s) and predictions[0] (%s) must have same shape", labels0.Shape(), predictions0.Shape())
	}
	loss := Neg(Add(
		Mul(labels0, Log(predictions0)),
		Mul(OneMinus(labels0), Log(OneMinus(predictions0)))))

	// Factor in weights and mask.
	weights, mask := CheckExtraLabelsForWeightsAndMask(labels0.Shape(), labels[1:])
	if weights != nil {
		loss = Mul(loss, weights)
	}
	if mask != nil {
		loss = Where(mask, loss, ZerosLike(loss))
		if !loss.IsScalar() {
			loss = MaskedReduceAllMean(loss, mask)
		}
	} else if !loss.IsScalar() {
		loss = ReduceAllMean(loss)
	}
	return loss
}

// BinaryCrossentropyLogits returns the cross-entropy loss between labels and `sigmoid(logits)`,
// for binary classification tasks. It assumes the predictions are given by `sigmoid(logits)`.
// This is a more numerically stable and faster implementation than actually taking the sigmoid of
// the logits and using the equivalent BinaryCrossentropy.
// labels and logits must have the same shape.
//
// labels is converted to predictions dtype, and it's expected to convert to 1.0 (for true) or 0.0 for false.
// So booleans should work, as an int type that is 0 or 1.
//
// See mathematical derivation of the stable solution in
// https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
//
// Labels can have 2 optional extra values (in any order):
//
//   - mask: a boolean mask of shape [batchSize] set to true for values to be used, and false for those to be ignored.
//     Typically used for padding. The returned mean loss takes in consideration the mask.
//   - weights: a float value of shape [batchSize] with the relative weights to be applied to each example.
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
	loss := Add(Sub(maxPart, prodPart), logPart)

	// Factor in weights and mask.
	weights, mask := CheckExtraLabelsForWeightsAndMask(labels0.Shape(), labels[1:])
	if weights != nil {
		loss = Mul(loss, weights)
	}
	if mask != nil {
		loss = Where(mask, loss, ZerosLike(loss))
		if !loss.IsScalar() {
			loss = MaskedReduceAllMean(loss, mask)
		}
	} else if !loss.IsScalar() {
		loss = ReduceAllMean(loss)
	}
	return loss
}

// SparseCategoricalCrossEntropyLogits returns the cross-entropy loss of the logits, given the labels.
// The labels are provided in "sparse" format, that is, integer numbers from 0 to logits dimension-1.
// labels and logits must have the same rank, and labels last dimension must be 1.
//
// Labels can have 2 optional extra values (in any order):
//
//   - mask: a boolean mask of shape [batchSize] set to true for values to be used, and false for those to be ignored.
//     Typically used for padding. The returned mean loss takes in consideration the mask.
//   - weights: a float value of shape [batchSize] with the relative weights to be applied to each example.
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

	// Remove last dimension, it will be re-added by OneHot
	reducedLabels := Reshape(labels0, labels0.Shape().Dimensions[:labelsRank-1]...)
	labelsValues := OneHot(reducedLabels, logitsShape.Dimensions[logitsRank-1], logitsShape.DType)
	return categoricalCrossEntropyLogitsImpl(labelsValues, logits0, labels[1:])
}

// CategoricalCrossEntropyLogits returns the cross-entropy loss of the logits, given the labels.
// The labels are provided in "dense" format, they should have the exact same shape as logits, and be set 1 for
// the true (labeled) category, and 0 for the others -- or any other distribution that sum to 1.
//
// It *does not* reduce-mean the losses, they are returned individually for each element of the batch and need
// to be ReduceAllMean (usually the mean, but it could be the sum also) before used for training.
//
// Labels can have 2 optional extra values (in any order):
//
//   - mask: a boolean mask of shape [batchSize] set to true for values to be used, and false for those to be ignored.
//     Typically used for padding. The returned mean loss takes in consideration the mask.
//   - weights: a float value of shape [batchSize] with the relative weights to be applied to each example.
//
// TODO: implement faster version with logits, see https://github.com/tensorflow/tensorflow/blob/359c3cdfc5fabac82b3c70b3b6de2b0a8c16874f/tensorflow/python/ops/nn_ops.py#L4051
func CategoricalCrossEntropyLogits(labels, logits []*Node) *Node {
	logits0 := logits[0]
	labels0 := labels[0]
	return categoricalCrossEntropyLogitsImpl(labels0, logits0, labels[1:])
}

// categoricalCrossEntropyLogitsImpl implements CategoricalCrossEntropyLogits.
func categoricalCrossEntropyLogitsImpl(labels, logits *Node, extras []*Node) *Node {
	shape := labels.Shape()
	if !shape.Equal(logits.Shape()) {
		Panicf("labels(%s) and logits(%s) must have the same shapes", shape, logits.Shape())
	}

	weightsShape := shapes.Make(logits.DType(), labels.Shape().Dimensions[:labels.Rank()-1]...)
	weights, mask := CheckExtraLabelsForWeightsAndMask(weightsShape, extras)

	var expandedMask *Node
	if mask != nil {
		expandedMask = BroadcastToShape(InsertAxes(mask, -1), logits.Shape())
		logits = Where(expandedMask, logits, ZerosLike(logits))
	}
	logPredictions := LogSoftmax(logits)
	loss := ReduceSum(Neg(Mul(labels, logPredictions)), -1)
	// loss will usually be shaped `[batchSize]` now.
	if weights != nil {
		loss = Mul(loss, weights)
	}
	if mask != nil {
		loss = Where(mask, loss, ZerosLike(loss))
		if !loss.IsScalar() {
			loss = MaskedReduceAllMean(loss, mask)
		}
	} else if !loss.IsScalar() {
		loss = ReduceAllMean(loss)
	}
	return loss
}

// CategoricalCrossEntropy returns the cross-entropy loss of the predictions (as probabilities), given the labels.
// The labels are provided in "dense" format, they should have the exact same shape as predictions, and be set 1 for
// the true (labeled) category, and 0 for the others (one-hot encoding) -- or any other distribution that sums to 1.
// predictions should hold probabilities that must sum to 1.0.
//
// Notice the CategoricalCrossEntropyLogits is more stable, given the choice choose that instead.
//
// Labels can have 2 optional extra values (in any order):
//
//   - mask: a boolean mask of shape [batchSize] set to true for values to be used, and false for those to be ignored.
//     Typically used for padding. The returned mean loss takes in consideration the mask.
//   - weights: a float value of shape [batchSize] with the relative weights to be applied to each example.
func CategoricalCrossEntropy(labels, predictions []*Node) *Node {
	labels0 := labels[0]
	predictions0 := predictions[0]
	g := predictions0.Graph()
	shape := labels0.Shape()
	dtype := labels0.DType()
	if !shape.Equal(predictions0.Shape()) {
		Panicf("labels(%s) and predictions(%s) must different shapes", shape, predictions0.Shape())
	}
	epsilon := epsilonForDType(g, dtype)
	predictions0 = Clip(predictions0, epsilon, OneMinus(epsilon))
	loss := ReduceSum(Neg(Mul(labels0, Log(predictions0))), -1)

	// Factor in weights and mask.
	weightsShape := shapes.Make(predictions[0].DType(), labels[0].Shape().Dimensions[:labels[0].Rank()-1]...)
	weights, mask := CheckExtraLabelsForWeightsAndMask(weightsShape, labels[1:])
	if weights != nil {
		loss = Mul(loss, weights)
	}
	if mask != nil {
		loss = Where(mask, loss, ZerosLike(loss))
		if !loss.IsScalar() {
			loss = MaskedReduceAllMean(loss, mask)
		}
	} else if !loss.IsScalar() {
		loss = ReduceAllMean(loss)
	}
	return loss
}

// MakeHuberLoss returns a Huber loss function: it's similar to an L2 (MeanSquaredLoss) close to the target,
// and it becomes L1 (linear) away from the target.
//
// The delta parameter configures the range where the loss behaves as L2: if the prediction is further than
// delta it becomes linear. It also defines the slope. A good default value is 1.0.
//
// For the returned loss function, labels can have 2 optional extra values (in any order):
//
//   - mask: a boolean mask of shape [batchSize] set to true for values to be used, and false for those to be ignored.
//     Typically used for padding. The returned mean loss takes in consideration the mask.
//   - weights: a float value of shape [batchSize] with the relative weights to be applied to each example. Weights
//     are automatically normalized to sum to one (preserving their ratio), preserving their ratio.
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

		// Factor in weights and mask.
		weights, mask := CheckExtraLabelsForWeightsAndMask(labels0.Shape(), labels[1:])
		if weights != nil {
			loss = Mul(loss, weights)
		}
		if mask != nil {
			loss = Where(mask, loss, ZerosLike(loss))
			if !loss.IsScalar() {
				loss = MaskedReduceAllMean(loss, mask)
			}
		} else if !loss.IsScalar() {
			loss = ReduceAllMean(loss)
		}
		return loss
	}
}

var (
	// ParamHuberLossDelta is the name of the hyperparameter that defines the Huber loss delta.
	// See HuberLossBuilder.
	// It defaults to 1.0
	ParamHuberLossDelta = "huber_loss_delta"
)

// MakeHuberLossFromContext calls MakeHuberLoss using the delta configured by the hyperparameter
// ParamHuberLossDelta in the context.
func MakeHuberLossFromContext(ctx *context.Context) LossFn {
	delta := context.GetParamOr(ctx, ParamHuberLossDelta, 1.0)
	return MakeHuberLoss(delta)
}

// MakeAdaptivePowerLoss creates an adaptive power loss function.
//
//   - When the labels and predictions are close, it tends to |labels-predictions|^powerNear.
//   - When the labels and predictions are far, it tends to |labels-predictions|^powerFar.
//   - If labels-predictions == middleDelta, it's exactly mid-point loss between powerNear and powerFar,
//     and the loss is |labels-predictions|^(powerFar+powerNear)/2.
//   - sharpness defines how sharp ("sudden") is the transition.
//
// For the returned loss function, labels can have 2 optional extra values (in any order):
//
//   - mask: a boolean mask of shape [batchSize] set to true for values to be used, and false for those to be ignored.
//     Typically used for padding. The returned mean loss takes in consideration the mask.
//   - weights: a float value of shape [batchSize] with the relative weights to be applied to each example. Weights
//     are automatically normalized to sum to one (preserving their ratio), preserving their ratio.
//
// E.g.: setting powerNear to 2, powerFar to 1, this will behave similarly to a HuberLoss.
func MakeAdaptivePowerLoss(powerNear, powerFar, middleDelta, sharpness float64) LossFn {
	return func(labels, predictions []*Node) (loss *Node) {
		predictions0 := predictions[0]
		g := predictions0.Graph()
		dtype := predictions0.DType()
		labels0 := labels[0]
		if !labels0.Shape().Equal(predictions0.Shape()) {
			Panicf("labels[0] (%s) and predictions[0] (%s) must have same shape", labels0.Shape(), predictions0.Shape())
		}

		// Calculate AdaptivePowerLoss
		delta := Abs(Sub(labels0, predictions0))
		if powerNear == powerFar {
			// Easy case, where they are the same.
			loss = Pow(delta, Scalar(g, dtype, powerNear))

		} else {
			// Find power to use for delta.
			normalizedDelta := DivScalar(delta, middleDelta)
			lnDelta := Log(Max(normalizedDelta, epsilonForDType(g, dtype)))
			powerDiffOverSharpness := (powerNear - powerFar) / sharpness
			scaledLnDelta := MulScalar(lnDelta, powerDiffOverSharpness)

			// version1 is stable (not infinite) for positive scaledLnDelta.
			version1 := AddScalar(
				MulScalar(
					Inverse(OnePlus(Exp(Neg(scaledLnDelta)))),
					powerFar-powerNear),
				powerNear)
			// version2 is stable (not infinite) for negative scaledLnDelta)
			version2 := AddScalar(
				MulScalar(
					Inverse(OnePlus(Exp(scaledLnDelta))),
					powerNear-powerFar),
				powerFar)
			power := Where(GreaterThan(scaledLnDelta, ScalarZero(g, dtype)),
				version1, version2)

			// NaNs would filter out through the Where if we allow, so we treat the calculated power as a constant
			// for the purpose of the loss.
			power = StopGradient(power)

			// Now we know the power (exponent) to use:
			loss = Pow(delta, power)
		}

		// Factor in weights and mask.
		weights, mask := CheckExtraLabelsForWeightsAndMask(labels0.Shape(), labels[1:])
		if weights != nil {
			loss = Mul(loss, weights)
		}
		if mask != nil {
			loss = Where(mask, loss, ZerosLike(loss))
			if !loss.IsScalar() {
				loss = MaskedReduceAllMean(loss, mask)
			}
		} else if !loss.IsScalar() {
			loss = ReduceAllMean(loss)
		}
		return loss
	}
}

var (
	// ParamAdaptivePowerLossNear is the name of the hyperparameter that defines the AdaptivePowerLoss.
	// It defaults to 2.0
	//
	// See MakeAdaptivePowerLoss and MakeAdaptivePowerLossFromContext.
	ParamAdaptivePowerLossNear = "adaptive_loss_near"

	// ParamAdaptivePowerLossFar is the name of one of the hyperparameter that defines the AdaptivePowerLoss
	// It defaults to 1.0
	//
	// See MakeAdaptivePowerLoss and MakeAdaptivePowerLossFromContext.
	ParamAdaptivePowerLossFar = "adaptive_loss_far"

	// ParamAdaptivePowerLossMiddleDelta is the name of one of the hyperparameter that defines the AdaptivePowerLoss.
	// It defaults to 1.0
	//
	// See MakeAdaptivePowerLoss and MakeAdaptivePowerLossFromContext.
	ParamAdaptivePowerLossMiddleDelta = "adaptive_loss_middle"

	// ParamAdaptivePowerLossSharpness is the name of one of the hyperparameter that defines the AdaptivePowerLoss.
	// It defaults to 1.0
	//
	// See MakeAdaptivePowerLoss and MakeAdaptivePowerLossFromContext.
	ParamAdaptivePowerLossSharpness = "adaptive_loss_sharpness"
)

// MakeAdaptivePowerLossFromContext calls MakeAdaptivePowerLoss using the delta configured by the hyperparameter
// in the context.
//
// See ParamAdaptivePowerLossNear, ParamAdaptivePowerLossFar, ParamAdaptivePowerLoss
func MakeAdaptivePowerLossFromContext(ctx *context.Context) LossFn {
	powerNear := context.GetParamOr(ctx, ParamAdaptivePowerLossNear, 2.0)
	powerFar := context.GetParamOr(ctx, ParamAdaptivePowerLossFar, 1.0)
	middleDelta := context.GetParamOr(ctx, ParamAdaptivePowerLossMiddleDelta, 1.0)
	sharpness := context.GetParamOr(ctx, ParamAdaptivePowerLossSharpness, 1.0)
	return MakeAdaptivePowerLoss(powerNear, powerFar, middleDelta, sharpness)
}
