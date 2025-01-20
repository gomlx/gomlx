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
)

type TripletLossDistance int

const (
	TripletLossDistanceL2 TripletLossDistance = iota
	TripletLossDistanceSquaredL2
	TripletLossDistanceCosineSimilarity
)

// pairwiseL2Distances Compute the 2D matrix of L2/Squared L2 distances between all the embeddings.
//
// Parameters:
//   - embeddings *Node 2-D tensor of shape (batch_size, embed_dim)
//   - squared bool If true, output is the pairwise squared euclidean distance matrix.
//     If false, output is the pairwise euclidean distance matrix.
//
// Returns:
//   - *Node 2-D tensor of shape (batch_size, batch_size)
func pairwiseL2Distances(embeddings *Node, squared bool) *Node {
	g := embeddings.Graph()
	batchSize := embeddings.Shape().Dim(0)
	dtype := embeddings.DType()
	// ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
	// Get the dot product between all embeddings
	// shape (batch_size, batch_size)
	dotProduct := MatMul(embeddings, Transpose(embeddings, 0, 1))

	// Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
	// This also provides more numerical stability (the diagonal of the result will be exactly 0).
	// shape (batch_size,)
	squareNorm := MaskedReduceSum(dotProduct, Diagonal(g, batchSize), 0)

	// Compute the pairwise distance matrix as we have:
	// ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
	// shape (batch_size, batch_size)
	distances := Add(Add(
		ExpandDims(squareNorm, 1),
		MulScalar(dotProduct, -2.0)),
		ExpandDims(squareNorm, 0))

	// Because of computation errors, some distances might be negative so we put everything >= 0.0
	distances = MaxScalar(distances, 0.0)

	if !squared {
		// Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
		// we need to add a small epsilon where distances == 0.0
		zero := Scalar(g, dtype, 0.0)
		eps := epsilonForDType(g, dtype)
		mask := Equal(distances, zero)
		distances = Where(mask, eps, distances)

		distances = Sqrt(distances)

		// Correct the epsilon added: set the distances on the mask to be exactly 0.0
		distances = Where(mask, zero, distances)
	}

	return distances
}

// pairwiseCosineDistances  Compute the 2D matrix of cosine distances between all the embeddings.
//
// Parameters:
//   - embeddings *Node 2-D tensor of shape (batch_size, embed_dim)
//
// Returns:
//   - *Node 2-D tensor of shape (batch_size, batch_size)
func pairwiseCosineDistances(embeddings *Node) *Node {
	// normalize input
	embeddingsL2Normalize := L2Normalize(embeddings, 1)
	// create adjacent matrix of cosine similarity
	distances := OneMinus(MatMul(embeddingsL2Normalize, Transpose(embeddingsL2Normalize, 0, 1)))
	// ensure all distances >= 0.0
	distances = MaxScalar(distances, 0.0)
	return distances
}

// validTripletMask Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid
//
// A triplet (i, j, k) is valid if:
//   - i, j, k are distinct
//   - labels[i] == labels[j] and labels[i] != labels[k]
//
// Parameters:
//   - labels *Node labels of the batch, of size (batch_size,)
//
// Returns:
//   - *Node a mask of valid triplet(a, p, n) with shape (batch_size,batch_size,batch_size,batch_size)
func validTripletMask(labels *Node) *Node {
	g := labels.Graph()
	batchSize := labels.Shape().Dim(0)
	//  Check that i, j and k are distinct
	indices := LogicalNot(DiagonalWithValue(Const(g, true), batchSize))
	iNotEqualj := InsertAxes(indices, 2)
	iNotEqualk := InsertAxes(indices, 1)
	jNotEqualk := InsertAxes(indices, 0)

	distinct := And(And(iNotEqualj, iNotEqualk), jNotEqualk)

	// Check if labels[i] == labels[j] and labels[i] != labels[k]
	equal := Squeeze(Equal(InsertAxes(labels, 0), ExpandDims(labels, 1)))
	iEqualj := InsertAxes(equal, 2)
	iEqualk := InsertAxes(equal, 1)

	valid := And(iEqualj, LogicalNot(iEqualk))

	return And(distinct, valid)
}

// TripletLoss Compute the triplet loss of all the valid triplet over a batch of embeddings.
//
// # We generate all the valid triplets and average the loss over the positive ones.
//
// Parameters:
//   - labels *Node labels of the batch, of size (batch_size,)
//   - embeddings *Node 2-D Tensor of shape (batch_size, embed_dim)
//   - margin float64 margin for triplet loss
//   - distanceMetric TripletLossDistanceMetric Metric(L2, squared L2 or cosine similarly) to compute the distance matrix
//
// References
//
//	[Oliver Moindrot's blog](https://omoindrot.github.io/triplet-loss#batch-all-strategy)
//	[FaceNet](https://arxiv.org/abs/1503.03832)
//	[In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737)
func TripletLoss(labels, predictions []*Node, margin float64, distanceMetric TripletLossDistance) *Node {
	predictions0 := predictions[0]
	labels0 := labels[0]
	weights, mask := CheckLabelsForWeightsAndMask(labels0.Shape(), labels)

	g := predictions0.Graph()
	dtype := predictions0.DType()
	eps := epsilonForDType(g, dtype)
	zero := ScalarZero(g, dtype)

	// shape (batch_size, batch_size)
	var distances *Node
	switch distanceMetric {
	case TripletLossDistanceL2:
		distances = pairwiseL2Distances(predictions0, false)
	case TripletLossDistanceSquaredL2:
		distances = pairwiseL2Distances(predictions0, true)
	case TripletLossDistanceCosineSimilarity:
		distances = pairwiseCosineDistances(predictions0)
	}

	anchorPositiveDistances := InsertAxes(distances, 2)
	anchorNegativeDistances := InsertAxes(distances, 1)

	// Compute a 3D tensor of size (batch_size, batch_size, batch_size)
	// triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
	// Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
	// and the 2nd (batch_size, 1, batch_size)
	// shape (batch_size, batch_size, batch_size)
	tripletLoss := AddScalar(Sub(anchorPositiveDistances, anchorNegativeDistances), margin)

	// Put to zero the invalid triplets
	// (where label(a) != label(p) or label(n) == label(a) or a == p)
	// shape (batch_size,batch_size,batch_size,batch_size)
	valid := validTripletMask(labels0)

	tripletLoss = Where(valid, tripletLoss, zero)

	// Remove negative losses (i.e. the easy triplets)
	tripletLoss = MaxScalar(tripletLoss, 0.0)

	// Count number of positive triplets (where loss > 0)
	numPositive := ReduceAllSum(Where(
		GreaterThan(tripletLoss, eps),
		OnesLike(tripletLoss),
		ZerosLike(tripletLoss)))

	// Get final mean triplet loss over the positive valid triplets
	loss := Div(ReduceAllSum(tripletLoss), Add(numPositive, eps))

	// Apply weights and mask.
	if weights != nil {
		loss = Mul(loss, weights)
	}
	if mask != nil {
		loss = Where(mask, loss, ZerosLike(loss))
	}

	return loss
}

// TripletHardLoss Compute the triplet loss using the hardest positive and the hardest negative for each anchor over a batch of embeddings.
//
//	We select the hardest positive and the hardest negative for each anchor
//
// Parameters:
//   - labels *Node labels of the batch, of size (batch_size,)
//   - embeddings *Node 2-D Tensor of shape (batch_size, embed_dim)
//   - margin float64 margin for triplet loss
//   - distanceMetric TripletLossDistanceMetric Metric(L2, squared L2 or cosine similarly) to compute the distance matrix
//
// References
//
//	[Oliver Moindrot's blog](https://omoindrot.github.io/triplet-loss#batch-all-strategy)
//	[FaceNet](https://arxiv.org/abs/1503.03832)
//	[In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737)
//	[Ahmed Abdou's Blog](https://ahmedabdou.hashnode.dev/introductory-guide-for-triplet-loss-function)
func TripletHardLoss(labels, predictions []*Node, margin float64, soft bool, distanceMetric TripletLossDistance) *Node {
	predictions0 := predictions[0]
	labels0 := labels[0]
	weights, mask := CheckLabelsForWeightsAndMask(labels0.Shape(), labels)

	g := predictions0.Graph()
	dtype := predictions0.DType()
	zero := ScalarZero(g, dtype)
	batchSize := labels0.Shape().Dim(0)

	// shape (batch_size, batch_size)
	var distances *Node
	switch distanceMetric {
	case TripletLossDistanceL2:
		distances = pairwiseL2Distances(predictions0, false)
	case TripletLossDistanceSquaredL2:
		distances = pairwiseL2Distances(predictions0, true)
	case TripletLossDistanceCosineSimilarity:
		distances = pairwiseCosineDistances(predictions0)
	}

	// shape (batch_size, batch_size)
	indicesNotEqual := LogicalNot(DiagonalWithValue(Const(g, true), batchSize))
	// shape (batch_size, batch_size)
	labelsEqual := Squeeze(Equal(InsertAxes(labels0, 0), ExpandDims(labels0, 1)))

	// For each anchor, get the hardest positive
	// First, we need to get a mask for every valid positive (they should have same label)
	// shape (batch_size, batch_size)
	positivesMask := And(indicesNotEqual, labelsEqual)

	// We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
	// shape (batch_size, 1)
	hardestPositiveDist := ReduceMax(Where(positivesMask, distances, zero), 1)

	// For each anchor, get the hardest negative
	// First, we need to get a mask for every valid negative (they should have different labels)
	// shape (batch_size, batch_size)
	negativesMask := LogicalNot(labelsEqual)

	// We add the maximum value in each row to the invalid negatives (label(a) == label(n))
	// shape (batch_size, 1)
	hardestNegativeDist := ReduceMin(Where(negativesMask, distances, Add(distances, ReduceAndKeep(distances, ReduceMax, 1))), 1)

	// Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
	var tripletLoss *Node
	if soft {
		tripletLoss = Log1P(Exp(Sub(hardestPositiveDist, hardestNegativeDist)))
	} else {
		tripletLoss = MaxScalar(AddScalar(Sub(hardestPositiveDist, hardestNegativeDist), margin), 0.0)
	}

	// Get final mean triplet loss
	loss := ReduceAllMean(tripletLoss)

	// Apply weights and mask.
	if weights != nil {
		loss = Mul(loss, weights)
	}
	if mask != nil {
		loss = Where(mask, loss, ZerosLike(loss))
	}

	return loss
}

// TripletSemiHardLoss Compute the triplet loss using the semi-hard of each anchor over a batch of embeddings.
//
//	We select the triplets where the negative is not closer to the anchor than the positive, but which still have positive loss
//
//	This is a slight difference implementation of semi-hard triplet loss, not base on tensorflow addons
//
// Parameters:
//   - labels *Node labels of the batch, of size (batch_size,)
//   - embeddings *Node 2-D Tensor of shape (batch_size, embed_dim)
//   - margin float64 margin for triplet loss
//   - distanceMetric TripletLossDistanceMetric Metric(L2, squared L2 or cosine similarly) to compute the distance matrix
//
// References
//
//	[FaceNet](https://arxiv.org/abs/1503.03832)
func TripletSemiHardLoss(labels, predictions []*Node, margin float64, soft bool, distanceMetric TripletLossDistance) *Node {
	predictions0 := predictions[0]
	labels0 := labels[0]
	weights, mask := CheckLabelsForWeightsAndMask(labels0.Shape(), labels)

	g := predictions0.Graph()
	dtype := predictions0.DType()
	eps := epsilonForDType(g, dtype)
	zero := ScalarZero(g, dtype)
	batchSize := labels0.Shape().Dim(0)

	// shape (batch_size, batch_size)
	var distances *Node
	switch distanceMetric {
	case TripletLossDistanceL2:
		distances = pairwiseL2Distances(predictions0, false)
	case TripletLossDistanceSquaredL2:
		distances = pairwiseL2Distances(predictions0, true)
	case TripletLossDistanceCosineSimilarity:
		distances = pairwiseCosineDistances(predictions0)
	}

	// shape (batch_size, batch_size)
	indicesNotEqual := LogicalNot(DiagonalWithValue(Const(g, true), batchSize))

	// shape (batch_size, batch_size)
	labelsEqual := Squeeze(Equal(InsertAxes(labels0, 0), ExpandDims(labels0, 1)))

	// A mask for every valid negative (they should have different labels)
	// shape (batch_size, batch_size)
	negativesMask := LogicalNot(labelsEqual)

	// First, we need to get a mask for every valid positive (they should have same label)
	// shape (batch_size, batch_size)
	positivesMask := And(indicesNotEqual, labelsEqual)

	// largest d(a, n)
	// shape (batch_size, 1)
	axisMinimums := ReduceAndKeep(distances, ReduceMin, 1)
	negativesInside := Add(ReduceAndKeep(
		Where(negativesMask, Sub(distances, axisMinimums), ZerosLike(distances)),
		ReduceMax, 1), axisMinimums)

	// d(a, n) > d(a, p)
	// axis 1 are the negatives
	// shape (batch_size, batch_size, batch_size)
	outsideMask := Transpose(And(
		InsertAxes(negativesMask, 1),
		GreaterThan(
			InsertAxes(distances, 1),
			InsertAxes(distances, 0),
		)), 0, 1)

	// smallest d(a, n) where d(a, n) > d(a, p)
	distancesIJK := BroadcastPrefix(distances, batchSize)
	axisMaximums := ReduceAndKeep(distancesIJK, ReduceMax, 1)
	negativesOutside := Squeeze(Add(ReduceAndKeep(
		Where(outsideMask, Sub(distancesIJK, axisMaximums), ZerosLike(distancesIJK)),
		ReduceMin, 1), axisMaximums))
	// shape (batch_size, batch_size)]]
	negativesOutside = Transpose(negativesOutside, 0, 1)

	//shape (batch_size, batch_size)
	outsideMaskFinal := Transpose(LogicalAny(outsideMask, 1), 0, 1)
	tripletLoss := Where(outsideMaskFinal, Sub(distances, negativesOutside), Sub(distances, negativesInside))

	if soft {
		tripletLoss = Log1P(Exp(tripletLoss))
	} else {
		tripletLoss = Max(AddScalar(tripletLoss, margin), zero)
	}
	// Count number of positive triplets
	numPositive := ReduceAllSum(ConvertDType(positivesMask, dtype))

	// Get final mean triplet loss over the positive valid triplets
	loss := Div(ReduceAllSum(Where(positivesMask, tripletLoss, zero)), Add(numPositive, eps))

	// Apply weights and mask.
	if weights != nil {
		loss = Mul(loss, weights)
	}
	if mask != nil {
		loss = Where(mask, loss, ZerosLike(loss))
	}

	return loss
}
