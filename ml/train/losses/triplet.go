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

package losses

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
)

//go:generate enumer -type=PairwiseDistanceMetric -trimprefix=PairwiseDistanceMetric -transform=snake -values -text -json -yaml triplet.go
//go:generate enumer -type=TripletMiningStrategy -trimprefix=TripletMiningStrategy -transform=snake -values -text -json -yaml triplet.go

type PairwiseDistanceMetric int

const (
	PairwiseDistanceMetricL2 PairwiseDistanceMetric = iota
	PairwiseDistanceMetricSquaredL2
	PairwiseDistanceMetricCosine
)

type TripletMiningStrategy int

const (
	TripletMiningStrategyAll TripletMiningStrategy = iota
	TripletMiningStrategyHard
	TripletMiningStrategySemiHard
)

// pairwiseDistances Computes the 2-D matrix of distances between all the embeddings.
//
// Parameters:
//   - embeddings *Node 2-D tensor of shape (batch_size, embed_dim)
//   - metric PairwiseDistanceMetric could be one of L2, squared L2 or cosine similarly distance metric
//
// Returns:
//   - *Node 2-D tensor of shape (batch_size, batch_size)
func pairwiseDistances(embeddings *Node, metric PairwiseDistanceMetric) *Node {
	g := embeddings.Graph()
	batchSize := embeddings.Shape().Dim(0)
	dtype := embeddings.DType()
	zero := ScalarZero(g, dtype)
	eps := epsilonForDType(g, dtype)

	// Get the dot product between all embeddings
	dotProduct := MatMul(embeddings, Transpose(embeddings, 0, 1))
	dotProduct.AssertDims(batchSize, batchSize)

	// Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
	// This also provides more numerical stability (the diagonal of the result will be exactly 0).
	squareL2Norm := MaskedReduceSum(dotProduct, Diagonal(g, batchSize), 0)
	squareL2Norm.AssertDims(batchSize)

	var distances *Node
	switch metric {
	case PairwiseDistanceMetricSquaredL2:
		// Compute the pairwise distance matrix as we have:
		// ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
		distances = Add(Add(
			InsertAxes(squareL2Norm, 1),
			MulScalar(dotProduct, -2.0)),
			InsertAxes(squareL2Norm, 0))
	case PairwiseDistanceMetricL2:
		// Compute the pairwise distance matrix
		distances = Add(Add(
			InsertAxes(squareL2Norm, 1),
			MulScalar(dotProduct, -2.0)),
			InsertAxes(squareL2Norm, 0))
		// Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
		// we need to add a small epsilon where distances == 0.0
		mask := LessThan(distances, eps)
		distances = Where(mask, eps, distances)
		distances = Sqrt(distances)
		// Correct the epsilon added: set the distances on the mask to be exactly 0.0
		distances = Where(mask, zero, distances)
	case PairwiseDistanceMetricCosine:
		// normalize input
		embeddings = Div(embeddings, InsertAxes(Sqrt(squareL2Norm), 1))
		// create adjacent matrix of cosine similarity
		distances = OneMinus(MatMul(embeddings, Transpose(embeddings, 0, 1)))
	}

	// Because of computation errors, some distances might be negative so we put everything >= 0.0
	distances = MaxScalar(distances, 0.0)
	distances.AssertDims(batchSize, batchSize)

	return distances
}

// maskedMaximums Computes the maximum values over masked pairwise distances.
//
// Parameters:
//   - distances *Node 2-D  `Tensor` of [batch_size, batch_size] pairwise distances
//   - mask *Node 2-D Mask `Tensor` of [batch_size, batch_size] valid distance size.
//   - dim int The dimension over which to compute the maximum.
//
// Returns:
//   - *Node A Tuple of Tensors containing the maximum distance value
func maskedMaximums(distances *Node, mask *Node, dim int) *Node {
	g := distances.Graph()
	dtype := distances.DType()
	axisMinimums := Sub(ReduceAndKeep(distances, ReduceMin, dim), epsilonForDType(g, dtype))
	maskedMax := Add(ExpandAxes(MaskedReduceMax(Sub(distances, axisMinimums), mask, dim), dim), axisMinimums)
	return maskedMax
}

// maskedMinimums Computes the minimum values over masked pairwise distances.
//
// Parameters:
//   - distances *Node 2-D  `Tensor` of [batch_size, batch_size] pairwise distances
//   - mask *Node 2-D Mask `Tensor` of [batch_size, batch_size] valid distance size.
//   - dim int The dimension over which to compute the maximum.
//
// Returns:
//   - *Node A Tuple of Tensors containing the maximum distance value for each example.
func maskedMinimums(distances *Node, mask *Node, dim int) *Node {
	g := distances.Graph()
	dtype := distances.DType()
	axisMaximums := Sub(ReduceAndKeep(distances, ReduceMax, dim), epsilonForDType(g, dtype))
	maskedMin := Add(ExpandAxes(MaskedReduceMin(Sub(distances, axisMaximums), mask, dim), dim), axisMaximums)
	return maskedMin
}

var (
	// ParamTripletLossPairwiseDistanceMetric is the name of the hyperparameter that defines the TripletLoss.
	// It defaults to 0
	//
	// See MakeTripletLossFromContext.
	ParamTripletLossPairwiseDistanceMetric = "triplet_loss_pairwise_distance_metric"

	// ParamTripletLossMiningStrategy is the name of the hyperparameter that defines the TripletLoss.
	// It defaults to 2
	//
	// See MakeTripletLossFromContext.
	ParamTripletLossMiningStrategy = "triplet_loss_mining_strategy"

	// ParamTripletLossMargin is the name of the hyperparameter that defines the TripletLoss.
	// It defaults to 1.0
	//
	// See MakeTripletLossFromContext.
	ParamTripletLossMargin = "triplet_loss_margin"
)

// TripletLoss Computes the triplet loss for valid triplet with different mining strategies for positives and negatives over a batch of embeddings.
//
// Parameters:
//   - labels *Node labels of the batch, of size (batch_size,)
//   - embeddings *Node 2-D Tensor of shape (batch_size, embed_dim)
//   - miningStrategy TripletMiningStrategy What mining strategy to use to select embedding from the different class. {'all', 'hard', 'semi-hard'}
//   - margin float64 Defines the target margin between positive and negative pairs, e.g., a margin of 1.0 means that the positive and negative distances should be 1.0 apart
//     if margin is negative, a soft margin will apply. It can be beneficial to pull together samples from the same class as much as possible.
//     See the paper for more details https://arxiv.org/pdf/1703.07737.pdf
//   - metric PairwiseDistanceMetric Which metric to use to compute the pairwise distances between embeddings.
//
// References
//
//	[Oliver Moindrot blog](https://omoindrot.github.io/triplet-loss)
//	[FaceNet](https://arxiv.org/abs/1503.03832)
//	[In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737)
//	[Tensorflow similarity triplet loss](https://github.com/tensorflow/similarity/blob/master/tensorflow_similarity/losses/triplet_loss.py)
func TripletLoss(labels, predictions []*Node,
	miningStrategy TripletMiningStrategy,
	margin float64,
	metric PairwiseDistanceMetric) *Node {

	predictions0 := predictions[0]
	labels0 := labels[0]
	weights, mask := CheckLabelsForWeightsAndMask(labels0.Shape(), labels)

	g := predictions0.Graph()
	batchSize := predictions0.Shape().Dim(0)
	dtype := predictions0.DType()
	zero := ScalarZero(g, dtype)

	distances := pairwiseDistances(predictions0, metric)
	distances.AssertDims(batchSize, batchSize)

	// Positive and Negative Masks
	// A negative mask: should have different labels
	// A positive mask: should have equal labels with different indices
	indicesNotEqual := LogicalNot(DiagonalWithValue(Const(g, true), batchSize))
	labelsEqual := Squeeze(Equal(InsertAxes(labels0, 0), InsertAxes(labels0, 1)))
	negativeMask := LogicalNot(labelsEqual)
	positiveMask := LogicalAnd(indicesNotEqual, labelsEqual)

	positiveMask.AssertDims(batchSize, batchSize)
	negativeMask.AssertDims(batchSize, batchSize)

	// Computes Positive and Negative distances with respect to mining strategy
	var positiveDistances *Node
	var negativeDistances *Node
	var validTriplets *Node
	// var fraction *Node
	switch miningStrategy {
	case TripletMiningStrategyAll:
		// take all combination of positives and negatives with same anchor
		positiveDistances = InsertAxes(distances, 2)
		negativeDistances = InsertAxes(distances, 1)

		// Whenever we have three indices i,j,k∈[1,B],
		// if examples i and j have the same label but are distinct,
		// and example k has a different label, we say that (i,j,k) is a valid triplet
		validTriplets = LogicalAnd(InsertAxes(positiveMask, 2), InsertAxes(negativeMask, 1))

		positiveDistances.AssertDims(batchSize, batchSize, 1)
		negativeDistances.AssertDims(batchSize, 1, batchSize)
		validTriplets.AssertDims(batchSize, batchSize, batchSize)

	case TripletMiningStrategyHard:
		// triplets where the negative is closer to the anchor than the positive, i.e. d(a,n)<d(a,p)
		// find the maximal distance between positive labels
		positiveDistances = maskedMaximums(distances, positiveMask, 1)
		// find the *non-zero* minimal distance between negative labels
		negativeDistances = maskedMinimums(distances, negativeMask, 1)

		// Whenever we have three indices i,j,k∈[1,B],
		// if examples i and j have the same label but are distinct,
		// and example k has a different label, we say that (i,j,k) is a valid triplet
		validTriplets = LogicalAnd(LogicalAny(positiveMask, 1), LogicalAny(negativeMask, 1))

		positiveDistances.AssertDims(batchSize, 1)
		negativeDistances.AssertDims(batchSize, 1)
		validTriplets.AssertDims(batchSize)

	case TripletMiningStrategySemiHard:
		// For each anchor, find the negative label with the minimal distance that is greater
		// than the maximal positive distance. If no such negative exists,
		// i.e., max(d(a,n)) < max(d(a,p)), then use the maximal negative
		// distance.

		// find the maximal distance between positive labels
		positiveDistances = maskedMaximums(distances, positiveMask, 1)

		// select the mask for distance above the max positive distance
		greaterDistances := GreaterThan(distances, positiveDistances)

		// Find the negative label with the maximal distance
		negativeEasyDistances := maskedMaximums(distances, negativeMask, 1)

		// keep negative label that is greater than the maximal positive distance, otherwise use with the maximal negative distance
		negativeDistances = Where(greaterDistances, distances, BroadcastToDims(negativeEasyDistances, batchSize, batchSize))

		// find the  minimal distance between negative labels above threshold
		negativeDistances = maskedMinimums(negativeDistances, negativeMask, 1)

		// Whenever we have three indices i,j,k∈[1,B],
		// if examples i and j have the same label but are distinct,
		// and example k has a different label, we say that (i,j,k) is a valid triplet
		validTriplets = LogicalAnd(LogicalAny(positiveMask, 1), LogicalAny(negativeMask, 1))

		positiveDistances.AssertDims(batchSize, 1)
		negativeDistances.AssertDims(batchSize, 1)
		validTriplets.AssertDims(batchSize)
	}

	loss := Sub(positiveDistances, negativeDistances)

	if margin > 0 {
		loss = AddScalar(loss, margin)
		loss = MaxScalar(loss, 0.0)
	} else {
		loss = Log1P(Exp(loss))
	}

	// Apply weights and mask.
	if weights != nil {
		loss = Mul(loss, weights)
	}
	if mask != nil {
		loss = Where(mask, loss, zero)
	}

	return MaskedReduceAllMean(loss, validTriplets)
}

// MakeTripletLossFromContext calls TripletLoss using the configured by the hyperparameter
// in the context.
func MakeTripletLossFromContext(ctx *context.Context) LossFn {
	miningStrategy := context.GetParamOr(ctx, ParamTripletLossMiningStrategy, TripletMiningStrategySemiHard)
	margin := context.GetParamOr(ctx, ParamTripletLossMargin, 1.0)
	metric := context.GetParamOr(ctx, ParamTripletLossPairwiseDistanceMetric, PairwiseDistanceMetricL2)
	return func(labels, predictions []*Node) (loss *Node) {
		return TripletLoss(labels, predictions, miningStrategy, margin, metric)
	}
}
