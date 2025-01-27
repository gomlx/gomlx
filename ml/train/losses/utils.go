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
//
package losses

import (
	. "github.com/gomlx/gomlx/graph"
)

//go:generate enumer -type=PairwiseDistanceMetric -trimprefix=PairwiseDistanceMetric -transform=snake -values -text -json -yaml util.go
//go:generate enumer -type=TripletMiningStrategy -trimprefix=TripletMiningStrategy -transform=snake -values -text -json -yaml util.go

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
