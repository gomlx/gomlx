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

package xla

// #include <string.h>
// #include "gomlx/client.h"
// #include "gomlx/computation.h"
import "C"

import (
	"errors"
	"fmt"
	"log"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/gomlx/gomlx/types/shapes"
)

// Literal represents a value stored in the local CPU, in C++ heap, to interact
// with XLA.
type Literal struct {
	cLiteralPtr *C.Literal
	shape       shapes.Shape
}

// Finalize implements Finalizer.
func (l *Literal) Finalize() {
	if l == nil || l.cLiteralPtr == nil {
		return
	}
	defer runtime.KeepAlive(l)
	C.DeleteLiteral(l.cLiteralPtr)
	atomic.AddInt64(&LiteralsCountDeallocated, 1)
	l.cLiteralPtr = nil
}

// IsNil returns whether either the Client object is nil or the contained C pointer.
func (l *Literal) IsNil() bool {
	return l == nil || l.cLiteralPtr == nil
}

// CShapeFromShape allocates int the C-heap a new C-struct representing the shape.
func CShapeFromShape(shape shapes.Shape) *C.Shape {
	cShape := Malloc[C.Shape]()
	cShape.dtype = C.int32_t(shape.DType)
	rank := shape.Rank()
	cShape.rank = C.int64_t(rank)
	if rank > 0 {
		cShape.dimensions = MallocArrayAndSet[C.int64_t](rank, func(ii int) C.int64_t { return C.int64_t(shape.Dimensions[ii]) })
	}
	cShape.tuple_size = C.int32_t(shape.TupleSize())
	if shape.TupleSize() > 0 {
		cShape.tuple_shapes = MallocArrayAndSet[*C.Shape](shape.TupleSize(), func(ii int) *C.Shape {
			return CShapeFromShape(shape.TupleShapes[ii])
		})
	}
	return cShape
}

// ShapeFromCShape converts a shape provided in C struct (cShape) into a shapes.Shape. cShape memory is NOT freed.
func ShapeFromCShape(cShape *C.Shape) (shape shapes.Shape) {
	if cShape == nil {
		return
	}
	shape.DType = shapes.DType(cShape.dtype)
	rank := int(cShape.rank)
	if rank > 0 {
		shape.Dimensions = make([]int, cShape.rank)
		dimensions := unsafe.Slice(cShape.dimensions, rank)
		for ii, dim := range dimensions {
			shape.Dimensions[ii] = int(dim)
		}
	}
	if cShape.tuple_size > 0 {
		shape.TupleShapes = make([]shapes.Shape, int(cShape.tuple_size))
		subShapes := unsafe.Slice(cShape.tuple_shapes, cShape.tuple_size)
		for ii, subShape := range subShapes {
			shape.TupleShapes[ii] = ShapeFromCShape(subShape)
		}
	}
	return
}

var muMem sync.Mutex

// NewLiteralFromShape returns a new Literal with the given shape and uninitialized data.
func NewLiteralFromShape(shape shapes.Shape) *Literal {
	cShape := CShapeFromShape(shape)
	cLiteral := C.MakeLiteralFromShape(cShape)
	if cLiteral == nil || int(cLiteral.size) != shape.Size() {
		log.Fatalf("Allocating xla::Literal with shape %s resulted in size %d, wanted size %d", shape, cLiteral.size, shape.Size())
	}
	return newLiteral(cLiteral)
}

// Refresh will re-read the data pointer and size information from the C++ interface. Mostly used for debugging.
func (l *Literal) Refresh() {
	if l.cLiteralPtr == nil {
		return
	}
	C.XlaLiteralRefreshData(l.cLiteralPtr)
}

func NewLiteralWithZeros(shape shapes.Shape) *Literal {
	literal := NewLiteralFromShape(shape)
	C.memset(unsafe.Pointer(literal.cLiteralPtr.data), 0, C.size_t(literal.cLiteralPtr.size_bytes))
	return literal
}

var (
	LiteralsCountDeallocated = int64(0)
	LiteralsCountAllocated   = int64(0)
)

// LiteralsCount returns the number of Literal objects still allocated. Used for profiling and debugging.
func LiteralsCount() int64 {
	return LiteralsCountAllocated - LiteralsCountDeallocated
}

// newLiteral creates the wrapper to the C.Literal pointer, and registers
// it for deallocation when it's garbage collected (or manually finalized).
func newLiteral(literalPtr *C.Literal) *Literal {
	atomic.AddInt64(&LiteralsCountAllocated, 1)
	shape := ShapeFromCShape(literalPtr.shape)
	l := &Literal{
		cLiteralPtr: literalPtr,
		shape:       shape,
	}
	RegisterFinalizer(l)
	return l
}

// NewLiteralTuple creates a tuple from the individual literals. Flat is copied.
func NewLiteralTuple(literals []*Literal) *Literal {
	cLiterals := MallocArrayAndSet[*C.Literal](len(literals),
		func(idx int) *C.Literal { return literals[idx].cLiteralPtr })
	//fmt.Printf("\tlen(literals)=%d, literals=%v\n", len(literals), literals)
	cLiteral := C.MakeLiteralTuple(cLiterals, C.int(len(literals)))
	return newLiteral(cLiteral)
}

// Shape returns the shape of the literal.
func (l *Literal) Shape() (shape shapes.Shape) {
	if l.IsNil() {
		return
	}
	return l.shape
}

// SplitTuple in the Literal into multiple literals. Unlike with GlobalData, this destroys
// the underlying Literal.
func (l *Literal) SplitTuple() []*Literal {
	shape := l.Shape()
	if !shape.IsTuple() {
		return nil
	}
	numElements := shape.TupleSize()
	parts := make([]*Literal, numElements)
	cLiteralsPtr := C.LiteralDecomposeTuple(l.cLiteralPtr)
	cLiterals := unsafe.Slice(cLiteralsPtr, numElements)
	for ii, cLiteral := range cLiterals {
		parts[ii] = newLiteral(cLiteral)
	}
	C.free(unsafe.Pointer(cLiteralsPtr))
	l.Finalize()
	return parts
}

// Data returns a slice of the data for the corresponding DType (see package types).
// The underlying data is not owned/managed by Go, but it is mutable.
//
// Notice if the Literal goes out-of-scope and is finalized, the underlying data gets freed without
// Go knowing about , and the returned slice may become invalid. Careful to make sure Literal doesn't
// get out of scope --
func (l *Literal) Data() any {
	if l.IsNil() {
		return nil
	}
	rawData := unsafe.Pointer(l.cLiteralPtr.data)
	return shapes.UnsafeSliceForDType(l.shape.DType, rawData, int(l.cLiteralPtr.size))
}

// Bytes returns the same memory as Data, but the raw slice of bytes, with the proper size.
func (l *Literal) Bytes() []byte {
	if l.IsNil() {
		return nil
	}
	rawBytes := (*byte)(unsafe.Pointer(l.cLiteralPtr.data))
	dtype := l.shape.DType
	t := shapes.TypeForDType(dtype)
	size := int(l.cLiteralPtr.size) * int(t.Size())
	return unsafe.Slice(rawBytes, size)
}

// DataFromLiteral returns a pointer to the raw data on the literal (without consideration
// of shape). The data can be mutated.
//
// The slices themselves shouldn't be modified -- the underlying storage is not
// owned by Go, and tensor objects are supposed to be immutable. See discussion on data storage
// and exceptions to mutability on the package description if you really need to mutate its values.
func DataFromLiteral[T shapes.Supported](l *Literal) ([]T, error) {
	if l.IsNil() {
		return nil, errors.New("empty literal, not underlying literal reference")
	}
	wantDType := shapes.DTypeGeneric[T]()
	if wantDType != l.shape.DType {
		return nil, fmt.Errorf("failed to retrieve literal: want dtype %s, but literal is of type  %s", wantDType, l.shape.DType)
	}
	data := unsafe.Slice((*T)(unsafe.Pointer(l.cLiteralPtr.data)), l.cLiteralPtr.size)
	return data, nil
}

// DataAndShapeFromLiteral return Literal's shape and data in one call. Returns error if the
// DType is incompatible with the requested type.
func DataAndShapeFromLiteral[T shapes.Number](l *Literal) (data []T, shape shapes.Shape, err error) {
	shape = l.Shape()
	data, err = DataFromLiteral[T](l)
	return
}

// ScalarFromLiteral returns the scalar stored in a Literal. Returns an error if Literal is
// not a scalar or if it is the wrong DType.
func ScalarFromLiteral[T shapes.Number](l *Literal) (t T, err error) {
	if l.IsNil() {
		err = errors.New("empty literal, not underlying literal reference")
		return
	}
	if int(l.shape.Rank()) != 0 {
		return t, fmt.Errorf("failed to get scalar from literal, it has rank %d", int(l.shape.Rank()))
	}
	data, err := DataFromLiteral[T](l)
	if err != nil {
		return t, err
	}
	return data[0], nil
}

// ToOnDeviceBuffer returns a OnDeviceBuffer allocated on the device (the number of the device is given
// by deviceOrdinal).
func (l *Literal) ToOnDeviceBuffer(client *Client, deviceOrdinal int) (*OnDeviceBuffer, error) {
	statusOr := C.LiteralToOnDeviceBuffer(l.cLiteralPtr, client.cClientPtr, C.int(deviceOrdinal))
	cBufferPtr, err := UnsafePointerOrError(statusOr)
	if err != nil {
		return nil, err
	}
	return newOnDeviceBuffer(client, (*C.OnDeviceBuffer)(cBufferPtr)), nil
}

func FromOnDeviceBuffer(buffer *OnDeviceBuffer) (*Literal, error) {
	statusOr := C.OnDeviceBufferToLiteral(buffer.cOnDeviceBufferPtr, buffer.client.cClientPtr)
	cLiteralPtr, err := UnsafePointerOrError(statusOr)
	if err != nil {
		return nil, err
	}
	return newLiteral((*C.Literal)(cLiteralPtr)), nil
}

// ToGlobalData transfers the literal to the global accelerator server. GlobalData is no
// longer used in our execution path, for now this is deprecated.
func (l *Literal) ToGlobalData(client *Client) (*GlobalData, error) {
	statusOr := C.TransferToServer(l.cLiteralPtr, client.cClientPtr)
	gdPtr, err := UnsafePointerOrError(statusOr)
	if err != nil {
		return nil, err
	}
	return newGlobalData(client, (*C.XlaGlobalData)(gdPtr)), nil
}
