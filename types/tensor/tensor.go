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

// Package tensor implements a `Tensor`, a representation of a multi-dimensional array.
//
// Tensors are multidimensional arrays (from scalar with 0 dimensions, to arbitrarily large dimensions), defined
// by their shape (a data type and its axes dimensions) and their actual content. As a special case, a Tensor can
// also be a tuple of multiple tensors.
//
// The main use of tensors are to be used as input and output of GoMLX computation graph.
//
// There are various ways to construct a Tensor from local data:
//
//   - `FromShape(shape shapes.Shape)`: creates a tensor with the given shape, and uninitialized values.
//   - `FromScalarAndDimensions[T shapes.Supported](value T, dimensions ...int)`: creates a Tensor with the
//     given dimensions, filled with the scalar value given. `T` must be one of the supported types.
//   - `FromFlatDataAndDimensions[T shapes.Supported](data []T, dimensions ...int)`: creates a Tensor with the
//     given dimensions, and set the flattened values with the given data. `T` must be one of the supported types.
//   - `FromValue[S MultiDimensionSlice](value S)`: Generic conversion, works with the scalar supported `DType`s
//     as well as with any arbitrary multidimensional slice of them. Slices of rank > 1 must be regular, that is
//     all the sub-slices must have the same shape. E.g.: `FromValue([][]float{{1,2}, {3, 5}, {7, 11}})`
//   - `FromAnyValue(value any)`: same as `FromValue` but non-generic, it takes an anonymous type `any`. The exception
//     is if `value` is already a tensor, then it is a no-op and it returns the tensor itself.
//
// Behind the scenes Tensor is a container that keeps in sync different materialization's of value:
//
//   - `local`: a copy of the values stored in CPU, as a Go flat array of the underlying dtype.
//   - `onDevices`: a copy of the values stored in the accelerator device(s) (CPU, GPU, TPU, etc.),
//     a wrapper for a "XLA's PJRT buffer" managed by the lower levels (see github.com/gomlx/gopjrt).
//     There can be multiple `Device` backing of a tensor, if there are multiple devices (like a multi-GPU set up).
//
// The Tensor container is lazy in nature: it won't transfer data from local storage to "on device" until needed.
// And if one is updated, the others are immediately invalidated.
//
// Transferring tensors to/from local/device areas has a cost, and should be avoided. For example,
// while training weights of an ML model, one generally does not need to transfer those weights to local -- just at
// the end of training to save the model weights. But the Tensor will keep the (local/device) copies cached,
// so they can be used multiple times, and transfer only occurs once.
package tensor

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/xla"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/pkg/errors"
	"sync"
)

// Tensor represents a multidimensional arrays (from scalar with 0 dimensions, to arbitrarily large dimensions), defined
// by their shape, a data type (dtypes.DType) and its axes' dimensions, and their actual content stored as a flat (1D)
// array of values.
//
// It is a container for Local and Device backing. Local is stored as flat slice of the underlying DType.
// As a special case, a Tensor can also be a tuple of multiple tensors -- only works for the Local representation.
//
// Tensor manages caching of Local and Device copies. There is a transferring cost that one needs to be aware when
// using it for large data -- LLM models can have 100s of GB in size... There is a cache
// system to prevent duplicate transfers, but it requires the user to be careful about informing about changes
// to invalidate the cache (see FlatData and ConstFlatData).
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
	onDevices map[int]*Device
}

// newTensor returns a Tensor object initialized only with the shape, but no actual storage (local or on any device)
// The returned tensor is invalid, and some data (local or on device) must be associated to it still.
func newTensor(shape shapes.Shape) *Tensor {
	return &Tensor{
		shape:     shape,
		onDevices: make(map[int]*Device),
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

// AssertValid panics if local is nil, or if its shape is invalid.
func (t *Tensor) AssertValid() {
	if t == nil {
		panic(errors.New("Tensor is nil"))
	}
	if !t.shape.Ok() {
		panic(errors.New("Tensor shape is invalid"))
	}
	if t.local.IsFinalized() && t.CurrentDevice() == nil {
		panic(errors.New("Tensor has no local or on device representation"))
	}
}

// HasClient accepts anything that can return a xla.Client. That includes xla.Client itself and
// graph.Manager.
type HasClient interface {
	Client() *pjrt.Client
}

// FinalizeAll will finalize (free) all associated data.
func (t *Tensor) FinalizeAll() {
	// Get the list of local and device tensors to finalize.
	t.mu.Lock()
	if t == nil {
		t.mu.Unlock()
		return
	}
	local := t.local
	t.local = nil
	var devices []*Device
	for _, tensorsPerDevice := range t.onDevices {
		for _, d := range tensorsPerDevice {
			if d.cache == t {
				d.cache = nil
			}
			devices = append(devices, d)
		}
	}
	t.onDevices = nil
	t.shape = shapes.Invalid()
	t.mu.Unlock()

	// Cache was cleared and unlocked, now we call Finalize on each tensor separately.
	if local != nil {
		local.Finalize()
	}
	for _, d := range devices {
		d.Finalize()
	}
}

/// ---------------------------------------------------------------------------------------------------------------

// AddDevice to the internal cache.
func (t *Tensor) AddDevice(device *Device) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.lockedAddDevice(device)
}

// lockedAddDevice implements AddDevice, and assumes cache.mu is already locked.
func (t *Tensor) lockedAddDevice(device *Device) {
	device.cache = t
	if t.onDevices == nil {
		t.onDevices = make(map[xla.ClientId]map[int]*Device)
	}
	clientMap, ok := t.onDevices[device.clientId]
	if !ok {
		clientMap = make(map[int]*Device)
		t.onDevices[device.clientId] = clientMap
	}
	clientMap[device.deviceNum] = device
}

// clearDevice from tensor cache -- but doesn't finalize it.
func (t *Tensor) clearDevice(device *Device) {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	if device == nil {
		return
	}
	if device.cache == t {
		device.cache = nil
	} else if device.cache != nil {
		exceptions.Panicf("Tensor.clearDevice given a device that is not managed by the Tensor")
	}
	if t.onDevices == nil {
		return
	}
	clientMap, ok := t.onDevices[device.clientId]
	if !ok {
		return
	}
	if clientMap == nil {
		return
	}
	delete(clientMap, device.deviceNum)
	device.cache = nil
}

// clearLocal cache, and leaves the local cached tensor without a cache.
func (t *Tensor) clearLocal() {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.local.t == t {
		t.local.t = nil
	}
	t.local = nil
}

// localFromDevice converts the tensor to `Local`, if there is not one cached yet.
// And if it needs converting, it uses the `device`, if one is given.
// If `device == nil`, it converts from any associated `Device` tensor.
//
// If `device` is nil, it picks the first Device tensor.
func (t *Tensor) localFromDevice(device *Device) *Local {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.localFromDeviceLocked(device)
}

// localFromDeviceLocked implements localFromDevice, but assumes cache is already locked.
func (t *Tensor) localFromDeviceLocked(device *Device) *Local {
	if t == nil {
		return nil
	}
	if t.local != nil {
		return t.local
	}

	if device == nil {
		if t.onDevices == nil {
			return nil
		}
		// Pick the first on-device tensor and convert it to a local tensor.
	loopAllDevices:
		for _, perDeviceOrdinal := range t.onDevices {
			for _, dev := range perDeviceOrdinal {
				if dev == nil {
					continue
				}
				device = dev
				break loopAllDevices
			}
		}
	}

	literal, err := xla.FromOnDeviceBuffer(device.shapedBuffer)
	if err != nil {
		panic(errors.WithMessage(err, "tensor.Local failed to transfer from Device: %v"))
	}
	return t.lockedAddLocal(&Local{
		shape:   literal.Shape(),
		literal: literal,
	})
}

// Device implements Tensor.Device.
// It returns a `Device` version of the tensor.
// If the underlying tensor is on the given `Device` already, it's a no-op.
// If there is already a cache of a corresponding `Device` tensor on the given device, that is returned.
// Otherwise, the contents of the `Local` tensor (if available), or one of the on-device tensors are
// transferred to a `Device` tensor on the given `deviceNum`.
func (t *Tensor) Device(hasClient HasClient, deviceNum int) *Device {
	t.mu.Lock()
	// TODO: maybe do this more fine-grained: it may make sense to transfer to several devices in parallel (?).
	defer t.mu.Unlock()
	if t == nil {
		return nil
	}

	// Search in cache.
	client := hasClient.Client()
	if t.onDevices != nil {
		if deviceNumMap, found := t.onDevices[client.Id]; found {
			if device, found := deviceNumMap[deviceNum]; found {
				return device
			}
		}
	}

	if t.local == nil {
		// If there is no local, convert from other device tensor to a local tensor,
		// and then to final device.
		// This is expensive, since there is another copy around.
		// TODO: can we transfer directly between different devices !?
		t.local = t.localFromDeviceLocked(nil)
	}
	local := t.local
	cid := client.Id
	device := &Device{
		clientId:  cid,
		deviceNum: deviceNum,
	}
	var err error
	device.shapedBuffer, err = local.literal.ToOnDeviceBuffer(client, deviceNum)
	if err != nil {
		panic(errors.WithMessage(err, "error converting from local tensor to device"))
	}
	device.shape = device.shapedBuffer.Shape()
	t.lockedAddDevice(device)
	return device
}

// CurrentDevice returns the current Device tensor backing this tensor, if there is any.
// If the tensor is Local, this returns nil.
//
// If there is more than one Device tensor, this returns the first one.
func (t *Tensor) CurrentDevice() *Device {
	t.mu.Lock()
	// TODO: maybe do this more fine-grained: it may make sense to transfer to several devices in parallel (?).
	defer t.mu.Unlock()
	if t == nil {
		return nil
	}
	if t.onDevices == nil {
		return nil
	}
	for _, deviceTensors := range t.onDevices {
		for _, deviceT := range deviceTensors {
			return deviceT
		}
	}
	return nil
}
