package graph

import (
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
)

// This file contains operations that operate on "ragged" representations.

// Ragged2D is a 2D ragged representation with the first dimension dense and the second axis
// being ragged. It can be interpreted as an array (fixed size) of variable length lists.
//
// A "ragged" representation is a special type of sparse representation where
// values are only defined on the start of the axis, and the tail of the axis is assumed
// to be irrelevant, or in some cases zero.
//
// For now, if the user has ragged tensors with larger rank, its up to them
// to reshape and transpose around to get to a 2D representation. A generic ragged representation
// using fixed shaped tensors is a TODO.
//
// To store a "ragged" representation we use a flat compact representation of the data (without the
// irrelevant parts) and the RowIDs for each elements: which row they are part of.
// Because GoMLX doesn't (yet) support dynamic shapes, it also takes the static value of the
// first axis dimension Dim0, it must be known in graph compile time.
//
// See TensorFlow's more generic RaggedTensor in https://www.tensorflow.org/guide/ragged_tensor, which
// was used as a source of inspiration -- this is a much simpler implementation, but covers many
// of the use cases.
//
// A note on padding: because of the static shape requirements, it's practice to use the last
// row as a "padding" row, and assign the RowIDs of the padding values to that extra row.
type Ragged2D struct {
	Dim0         int
	Flat, RowIDs *Node
}

// MakeRagged2D creates a new Ragged2D using rowIDs.
//
// The rowIDs _must be sorted_, meaning the flat values must come in row,col order, otherwise
// many of the operations will display undefined behaviour.
//
// Example:
//
//	MakeRagged2D(dim0=4, flat=[1, 2, 3, 4, 5], rowsIds=[0, 0, 0, 1, 3]) represents the following 4x3 ragged 2D tensor :
//
//	{ {1, 2, 3},
//	  {4},
//	  {},
//	  {5} }
func MakeRagged2D(dim0 int, flat, rowIDs *Node) Ragged2D {
	_ = validateBuildingGraphFromInputs(flat, rowIDs)
	if flat.Rank() != 1 || rowIDs.Rank() > 2 || flat.Shape().Size() != rowIDs.Shape().Size() {
		Panicf("Ragged2D must have 1D flat and rowIDs with the exact same dimensions (dtypes "+
			"can be different), got flat.Shape=%s, and rowIDs.Shape=%s", flat.Shape(), rowIDs.Shape())
	}
	if rowIDs.Rank() != 2 {
		// Make sure rowIDs are shaped [numFlat, 1].
		rowIDs = Reshape(rowIDs, -1, 1)
	}
	if !rowIDs.DType().IsInt() {
		Panicf("Ragged2D's rowIDs must be of integer type, got rowIDs.Shape=%s", rowIDs.Shape())
	}
	return Ragged2D{Dim0: dim0, Flat: flat, RowIDs: rowIDs}
}

// DType returns the dtype of the flat values.
func (r Ragged2D) DType() dtypes.DType {
	return r.Flat.DType()
}

// Graph associated with the Flat and RowIDs nodes.
func (r Ragged2D) Graph() *Graph {
	return r.Flat.Graph()
}

// ReduceSumCols returns the sum over the ragged axis (columns).
// It returns a 1D tensor of shape [Ragged2D.Dim0].
func (r Ragged2D) ReduceSumCols() *Node {
	initialValue := Zeros(r.Graph(), shapes.Make(r.DType(), r.Dim0))
	return ScatterSum(initialValue, r.RowIDs, r.Flat, true, false)
}

// ReduceMaxCols returns the max over the ragged axis (columns).
// It returns a 1D tensor of shape [Ragged2D.Dim0] with the values for each rows.
//
// Rows with no ragged values, will have -Inf values.
func (r Ragged2D) ReduceMaxCols() *Node {
	initialValue := BroadcastToDims(Infinity(r.Graph(), r.DType(), -1), r.Dim0)
	return ScatterMax(initialValue, r.RowIDs, r.Flat, true, false)
}

// ReduceMinCols returns the min over the ragged axis (columns).
// It returns a 1D tensor of shape [Ragged2D.Dim0] with the values for each rows.
//
// Rows with no ragged values, will have Inf values.
func (r Ragged2D) ReduceMinCols() *Node {
	initialValue := BroadcastToDims(Infinity(r.Graph(), r.DType(), 1), r.Dim0)
	return ScatterMin(initialValue, r.RowIDs, r.Flat, true, false)
}

// Softmax of the Ragged2D matrix, returns a Ragged2D with the values converted to probabilities.
//
// Notice that values not represented (the tail of each row) does not participate in the Softmax.
// It is as if they were -inf.
func (r Ragged2D) Softmax() Ragged2D {
	if !r.DType().IsFloat() {
		Panicf("invalid Ragged2D dtype %s, it must be float", r.DType())
	}
	normalizingMax := StopGradient(r.ReduceMaxCols())
	normalizingMax = Gather(normalizingMax, r.RowIDs, true)
	normalizedLogits := Sub(r.Flat, normalizingMax)
	numerators := Exp(normalizedLogits)
	denominators := MakeRagged2D(r.Dim0, numerators, r.RowIDs).ReduceSumCols()
	denominators = Gather(denominators, r.RowIDs, true)
	results := Div(numerators, denominators)
	return MakeRagged2D(r.Dim0, results, r.RowIDs)
}
