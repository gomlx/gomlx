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

// Package tensors implements a `Tensor`, a representation of a multi-dimensional array.
//
// Tensors are multidimensional arrays (from scalar with 0 dimensions, to arbitrarily large dimensions), defined
// by their shape (a data type and its axes dimensions) and their actual content. As a special case, a Tensor can
// also be a tuple of multiple tensors.
//
// The main use of tensors are to be used as input and output of GoMLX computation graphs.
//
// There are various ways to construct a Tensor from local data:
//
//   - FromShape(shape shapes.Shape): creates a tensor with the given shape, and zero values.
//
//   - FromScalarAndDimensions[T shapes.Supported](value T, dimensions ...int): creates a Tensor with the
//     given dimensions, filled with the scalar value given. `T` must be one of the supported types.
//
//   - FromFlatDataAndDimensions[T shapes.Supported](data []T, dimensions ...int): creates a Tensor with the
//     given dimensions, and set the flattened values with the given data. `T` must be one of the supported types.
//     Example:
//
//     t := FromFlatDataAndDimensions([]int8{1, 2, 3, 4}, 2, 2}) // Tensor with [[1,2], [3,4]]
//
//   - FromValue[S MultiDimensionSlice](value S): Generic conversion, works with the scalar supported `DType`s
//     as well as with any arbitrary multidimensional slice of them. Slices of rank > 1 must be regular, that is
//     all the sub-slices must have the same shape. Example:
//
//     t := FromValue([][]float{{1,2}, {3, 5}, {7, 11}})`
//
//   - FromAnyValue(value any): same as FromValue but non-generic, it takes an anonymous type `any`. The exception
//     is if `value` is already a tensor, then it is a no-op and it returns the tensor itself.
//
// Behind the scenes (as much as possible Tensor tries to hide all the details), Tensor is a container that keeps in
// sync different materialization's of value:
//
//   - `local`: a copy of the values stored in CPU, as a Go flat array of the underlying dtype.
//   - `onDevices`: a copy of the values stored in the accelerator device(s) (CPU, GPU, TPU, etc.),
//     a wrapper for a "XLA's PJRT buffer" managed by the lower levels (see github.com/gomlx/gopjrt).
//     There can be multiple `Device` backing of a tensor, if there are multiple devices (like a multi-GPU set up).
//   - And "on-device" Tensor can also be "shared", if the backend allows it, in which case the local
//     and "on-device" share the same memory allocation.
//
// The Tensor container is lazy in nature: it won't transfer data from local storage to "on device" until needed.
// And if/when it can, it will make it "shared" (generally, when running on CPUs).
// If not "shared", when one (local or on-device) is updated, the others are immediately invalidated.
//
// Transferring tensors to/from local/device areas has a cost, and should be avoided. For example,
// while training weights of an ML model, one generally does not need to transfer those weights to local -- just at
// the end of training to save the model weights. But the Tensor will keep the (local/device) copies cached,
// so they can be used multiple times, and transfer only occurs once.
package tensors

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/pkg/errors"
	"sync"
)

// Tensor represents a multidimensional arrays (from scalar with 0 dimensions, to arbitrarily large dimensions), defined
// by their shape, a data type (dtypes.DType) and its axes' dimensions, and their actual content stored as a flat (1D)
// array of values.
//
// The main use of tensors are to be used as input and output of GoMLX computation graphs.
//
// It is a container for "local" (host CPU) and "on-device" backing of the tensor -- they can be the same ("shared") in
// some cases.
// It is always stored as flat slice of the underlying DType.
//
// Tensor manages caching of Local and Device copies. There is a transferring cost that one needs to be aware when
// using it for large data -- LLM models can have 100s of GB in size... There is a cache
// system to prevent duplicate transfers, but it requires some care from the user (see ConstFlatData and MutableFlatData).
//
// More details in the `tensor` package documentation.
type Tensor struct {
	// shape of the tensor.
	shape shapes.Shape

	// mu protects the local and OnDevices data, but not the shape, which is considered immutable (only changed
	// when Tensor is finalized).
	mu    sync.Mutex
	local *local

	// onDevices maps deviceNum -> on device buffer.
	onDevices map[backends.DeviceNum]*onDevice

	// isShared indicates that the tensor used a shared buffer: it is held "on-device" and the "local" is just
	// a pointer to the "on-device" one.
	//
	// This is allocated, freed and mutated in ondevice.go, by the corresponding onDevice structure that owns
	// the shared buffer.
	isShared   bool
	sharedFlat any // Flat slice, []dtype of the shared memory area.

	// backend to use for on-device tensors.
	backend backends.Backend
}

// newTensor returns a Tensor object initialized only with the shape, but no actual storage (local or on any device)
// The returned tensor is invalid, and some data (local or on device) must be associated to it still.
func newTensor(shape shapes.Shape) *Tensor {
	return &Tensor{
		shape:     shape,
		onDevices: make(map[backends.DeviceNum]*onDevice),
	}
}

// Shape of Local, includes DType.
func (t *Tensor) Shape() shapes.Shape { return t.shape }

// DType returns the DType of the tensor's shape.
// It is a shortcut to `Tensor.Shape().DType`.
func (t *Tensor) DType() dtypes.DType {
	return t.shape.DType
}

// Rank returns the rank of the tensor's shape.
// It is a shortcut to `Tensor.Shape().Rank()`.
func (t *Tensor) Rank() int { return t.shape.Rank() }

// IsScalar returns whether the tensor represents a scalar value.
// It is a shortcut to `Tensor.Shape().IsScalar()`.
func (t *Tensor) IsScalar() bool { return t.shape.IsScalar() }

// Size returns the number of elements in the tensor.
// It is a shortcut to `Tensor.Shape().IsScalar()`.
func (t *Tensor) Size() int { return t.shape.Size() }

// Memory returns the number of bytes used to store the tensor. An alias to Tensor.Shape().Memory().
func (t *Tensor) Memory() uintptr { return t.shape.Memory() }

// Ok returns whether the Tensor is in a valid state: it is not nil, and it hasn't been finalized.
func (t *Tensor) Ok() bool {
	return t != nil && t.shape.Ok() &&
		(!t.local.IsFinalized() || len(t.onDevices) > 0)
}

// IsShared returns whether the underlying tensor storage is shared with the backend engine.
//
// In most cases, an end-user doesn't need to use this.
//
// If true, one shouldn't access it (ConstFlatData or MutableFlatData) during the execution of
// a computation graph that uses it.
//
// The Tensor implementation will try to use shared tensors where possible, since they save an
// extra copy.
func (t *Tensor) IsShared() bool {
	return t.isShared
}

// AssertValid panics if local is nil, or if its shape is invalid.
func (t *Tensor) AssertValid() {
	if t == nil {
		panic(errors.New("Tensor is nil"))
	}
	if !t.shape.Ok() {
		panic(errors.New("Tensor shape is invalid"))
	}
	if t.local.IsFinalized() && len(t.onDevices) == 0 {
		panic(errors.New("Tensor has no local or on-device representation"))
	}
}

// HasClient accepts anything that can return a xla.Client. That includes xla.Client itself and
// graph.Backend.
type HasClient interface {
	Client() *pjrt.Client
}

// FinalizeAll immediately frees all associated data and leave Tensor in an invalid state. Shape is cleared also.
//
// It's the caller responsibility to ensure the tensor buffers are not being used elsewhere (like in the middle of an execution).
func (t *Tensor) FinalizeAll() {
	// Get the list of local and device tensors to finalize.
	t.mu.Lock()
	defer t.mu.Unlock()
	t.lockedFinalizeAll()
}

// lockedFinalizeAll is FinalizeAll but must be called with the tensor already locked.
func (t *Tensor) lockedFinalizeAll() {
	if t == nil {
		t.mu.Unlock()
		return
	}
	if t.local != nil {
		t.local.Finalize()
		t.local = nil
	}
	for _, device := range t.onDevices {
		device.Finalize()
	}
	t.onDevices = nil
	t.shape = shapes.Invalid()
	t.isShared = false
}
