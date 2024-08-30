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

// Package shapes defines Shape and DType and associated tools.
//
// Shape represents the shape (rank, dimensions and DType) of either a Tensor or the expected
// shape of a node in a computation Graph. DType indicates the type of the unit element of
// a Tensor (or its representation as a node in a computation Graph).
//
// Shape and DType are used both by the concrete tensor values (see tensor package) and when
// working on the computation graph (see graph package).
//
// Go float16 support (commonly used by Nvidia GPUs) uses github.com/x448/float16 implementation,
// and bfloat16 uses a simple implementation in github.com/gomlx/gopjrt/dtypes/bfloat16.
//
// ## Glossary
//
//   - Rank: number of axes (dimensions) of a Tensor.
//   - Axis: is the index of a dimension on a multidimensional Tensor. Sometimes used
//     interchangeably with Dimension, but here we try to refer to a dimension index as "axis"
//     (plural axes), and its size as its dimension.
//   - Dimension: the size of a multi-dimensions Tensor in one of its axes. See example below:
//   - DType: the data type of the unit element in a tensor. Enumeration defined in github.com/gomlx/gopjrt/dtypes
//   - Scalar: is a shape where there are no axes (or dimensions), only a single value
//     of the associated DType.
//
// Example: The multi-dimensional array `[][]int32{{0, 1, 2}, {3, 4, 5}}` if converted to a Tensor
// would have shape `(int32)[2 3]`. We say it has rank 2 (so 2 axes), axis 0 has
// dimension 2, and axis 1 has dimension 3. This shape could be created with
// `shapes.Make(int32, 2, 3)`.
//
// ## Asserts
//
// When coding ML models, one delicate part is keeping tabs on the shape of
// the nodes of the graphs -- unfortunately there is no compile-time checking of values,
// so validation only happens in runtime. To facilitate, and also to serve as code documentation,
// this package provides two variations of _assert_ functionality. Examples:
//
// `AssertRank` and `AssertDims` checks that the rank and dimensions of the given
//
//	object (that has a `Shape` method) match, otherwise it panics. The `-1` means
//	the dimension is unchecked (it can be anything).
//
// ```
//
//	func modelGraph(ctx *context.Context, spec any, inputs []*Node) ([]*Node) {
//	   _ = spec  // Not needed here, we know the dataset.
//	   shapes.AssertRank(inputs, 2)
//	   batchSize := inputs.Shape().Dimensions[0]
//	   logits := layers.Dense(ctx, inputs[0], /* useBias= */ true, /* outputDim= */ 1)
//	   shapes.AssertDims(logits, batchSize, -1)
//	   return []*Node{logits}
//	}
//
// ```
//
// If you don't want to panic, but instead return an error through the `graph.Graph`, you can
// use the `Node.AssertDims()` method. So it would look like `logits.AssertDims(batchSize, -1)`.
package shapes

import (
	"encoding/gob"
	"fmt"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
	"slices"
	"strings"
)

// Shape represents the shape of either a Tensor or the expected shape
// of the value from a computation node.
//
// Use Make to create a new shape. See example in package shapes documentation.
type Shape struct {
	DType       DType
	Dimensions  []int
	TupleShapes []Shape // Shapes of the tuple, if this is a tuple.
}

// Make returns a Shape structure filled with the values given.
// See MakeTuple for tuple shapes.
func Make(dtype DType, dimensions ...int) Shape {
	s := Shape{Dimensions: slices.Clone(dimensions), DType: dtype}
	for _, dim := range dimensions {
		if dim <= 0 {
			exceptions.Panicf("shapes.Make(%s): cannot create a shape with an axis with dimension <= 0", s)
		}
	}
	return s
}

// Scalar returns a scalar Shape for the given type.
func Scalar[T Number]() Shape {
	return Shape{DType: FromGenericsType[T]()}
}

// Invalid returns an invalid shape.
//
// Invalid().IsOk() == false.
func Invalid() Shape {
	return Shape{DType: InvalidDType}
}

// Ok returns whether this is a valid Shape. A "zero" shape, that is just instantiating it with Shape{} will be invalid.
func (s Shape) Ok() bool { return s.DType != InvalidDType || len(s.TupleShapes) > 0 }

// Rank of the shape, that is, the number of dimensions.
func (s Shape) Rank() int { return len(s.Dimensions) }

// IsScalar returns whether the shape represents a scalar, that is there are no dimensions (rank==0).
func (s Shape) IsScalar() bool { return s.Ok() && s.Rank() == 0 }

// Dim returns the dimension of the given axis. axis can take negative numbers, in which
// case it counts as starting from the end -- so axis=-1 refers to the last axis.
// Like with a slice indexing, it panics for an out-of-bound axis.
func (s Shape) Dim(axis int) int {
	adjustedAxis := axis
	if adjustedAxis < 0 {
		adjustedAxis += s.Rank()
	}
	if adjustedAxis < 0 || adjustedAxis > s.Rank() {
		exceptions.Panicf("Shape.Dim(%d) out-of-bounds for rank %d (shape=%s)", axis, s.Rank(), s)
	}
	return s.Dimensions[adjustedAxis]
}

// Shape returns a shallow copy of itself. It implements the HasShape interface.
func (s Shape) Shape() Shape { return s }

// String implements stringer, pretty-prints the shape.
func (s Shape) String() string {
	if s.TupleSize() > 0 {
		parts := make([]string, 0, s.TupleSize())
		for _, tuple := range s.TupleShapes {
			parts = append(parts, tuple.String())
		}
		return fmt.Sprintf("Tuple<%s>", strings.Join(parts, ", "))
	}
	if s.Rank() == 0 {
		return fmt.Sprintf("(%s)", s.DType)
	}
	return fmt.Sprintf("(%s)%v", s.DType, s.Dimensions)
}

// Size returns the number of elements of DType are needed for this shape. It's the product of all dimensions.
func (s Shape) Size() (size int) {
	size = 1
	for _, d := range s.Dimensions {
		size *= d
	}
	return
}

// Memory returns the memory used to store an array of the given shape, the same as the size in bytes.
// Careful, so far all types in Go and on device seem to use the same sizes, but future type this is not guaranteed.
func (s Shape) Memory() uintptr {
	return s.DType.Memory() * uintptr(s.Size())
}

// MakeTuple returns a shape representing a tuple of elements with the given shapes.
func MakeTuple(elements []Shape) Shape {
	return Shape{DType: InvalidDType, Dimensions: nil, TupleShapes: elements}
}

// IsTuple returns whether the shape represents a tuple.
func (s Shape) IsTuple() bool {
	return s.DType == InvalidDType
}

// TupleSize returns the number of elements in the tuple, if it is a tuple.
func (s Shape) TupleSize() int {
	return len(s.TupleShapes)
}

// Equal compares two shapes for equality: dtype and dimensions are compared.
func (s Shape) Equal(s2 Shape) bool {
	if s.DType != s2.DType {
		return false
	}
	if s.IsTuple() {
		if s.TupleSize() != s2.TupleSize() {
			return false
		}
		for ii, element := range s.TupleShapes {
			if !element.Equal(s2.TupleShapes[ii]) {
				return false
			}
		}
		return true
	}
	if s.Rank() != s2.Rank() {
		return false
	}
	if s.IsScalar() {
		return true
	}
	// For normal shapes just compare dimensions.
	return slices.Equal(s.Dimensions, s2.Dimensions)
}

// EqualDimensions compares two shapes for equality of dimensions. Dtypes can be different.
func (s Shape) EqualDimensions(s2 Shape) bool {
	if s.IsTuple() {
		if !s2.IsTuple() {
			return false
		}
		if s.TupleSize() != s2.TupleSize() {
			return false
		}
		for ii, element := range s.TupleShapes {
			if !element.EqualDimensions(s2.TupleShapes[ii]) {
				return false
			}
		}
		return true
	}
	if s.Rank() != s2.Rank() {
		return false
	}
	if s.IsScalar() {
		return true
	}
	// For normal shapes just compare dimensions.
	return slices.Equal(s.Dimensions, s2.Dimensions)
}

// Clone returns a new deep copy of the shape.
func (s Shape) Clone() (s2 Shape) {
	s2.DType = s.DType
	s2.Dimensions = slices.Clone(s.Dimensions)
	if s.TupleSize() > 0 {
		s2.TupleShapes = make([]Shape, 0, len(s.TupleShapes))
		for _, subShape := range s.TupleShapes {
			s2.TupleShapes = append(s2.TupleShapes, subShape.Clone())
		}
	}
	return
}

// GobSerialize shape in binary format.
func (s Shape) GobSerialize(encoder *gob.Encoder) (err error) {
	enc := func(e any) {
		if err != nil {
			return
		}
		err = encoder.Encode(e)
		if err != nil {
			err = errors.Wrapf(err, "failed to serialize Shape %s", s)
		}
	}
	enc(s.DType)
	enc(s.Dimensions)
	enc(len(s.TupleShapes))
	if err != nil {
		return
	}
	for _, subShape := range s.TupleShapes {
		err = subShape.GobSerialize(encoder)
		if err != nil {
			return
		}
	}
	return
}

// GobDeserialize a Shape. Returns new Shape or an error.
func GobDeserialize(decoder *gob.Decoder) (s Shape, err error) {
	dec := func(data any) {
		if err != nil {
			return
		}
		err = decoder.Decode(data)
		if err != nil {
			err = errors.Wrapf(err, "failed to deserialize Shape")
		}
	}
	dec(&s.DType)
	dec(&s.Dimensions)
	var numTuples int
	dec(&numTuples)
	if err != nil {
		return
	}
	s.TupleShapes = make([]Shape, numTuples)
	for ii := range s.TupleShapes {
		s.TupleShapes[ii], err = GobDeserialize(decoder)
		if err != nil {
			return
		}
	}
	return
}

// ConcatenateDimensions of two shapes. The resulting rank is the sum of both ranks. They must
// have the same dtype. If any of them is a scalar, the resulting shape will be a copy of the other.
// It doesn't work for Tuples.
func ConcatenateDimensions(s1, s2 Shape) (shape Shape) {
	if s1.IsTuple() || s2.IsTuple() {
		return
	}
	if s1.DType == InvalidDType || s2.DType == InvalidDType {
		return
	}
	if s1.DType != s2.DType {
		return
	}
	if s1.IsScalar() {
		return s2.Clone()
	} else if s2.IsScalar() {
		return s1.Clone()
	}
	shape.DType = s1.DType
	shape.Dimensions = make([]int, s1.Rank()+s2.Rank())
	copy(shape.Dimensions, s1.Dimensions)
	copy(shape.Dimensions[s1.Rank():], s2.Dimensions)
	return
}
