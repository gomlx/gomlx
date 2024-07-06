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
// It is a container that keeps in sync different materializations of value:
//
//   - `Local`: a copy of the values stored in CPU, as a Go flat array of the underlying dtype.
//   - `Device`: a copy of the values stored in the accelerator (GPU?), a wrapper for a "XLA's PJRT buffer" managed by
//     the lower levels (see github.com/gomlx/gopjrt). There can be multiple `Device` backing of a tensor,
//     if there are multiple devices (like a multi-GPU set up).
//
// The Tensor container is lazy in nature: it won't transfer data from `Device` to `Local` (or vice-versa) until needed.
// And if one is updated, the others are immediately invalidated.
//
// Transferring tensors to/from local/device areas has a cost, and should be avoided. For example,
// while training weights of an ML model, one generally does not need to transfer those weights to local -- just at
// the end of training to save the model weights. But the Tensor will keep the (local/device) copies cached,
// so they can be used multiple times, and transfer only occurs once.
//
// See details on the documentation of Tensor, Device and Local structures.
package tensor

import (
	"github.com/gomlx/gomlx/types/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/xla"
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

	// error that may have occurred during construction/conversion/operation.
	error error

	mu    sync.Mutex
	local *Local

	// Maps ClientId->Map of deviceNum->Device tensor.
	onDevices map[xla.ClientId]map[int]*Device
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
	if local != nil && local.cache == t {
		local.cache = nil
	}
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

	if t.local.cache == t {
		t.local.cache = nil
	}
	t.local = nil
}

// Value returns a multidimensional slice (except if shape is a scalar) containing a
// copy of the tensor values.
// See `Local` and `Local.Value()` for details.
func (t *Tensor) Value() any {
	return t.localFromDevice(nil).Value()
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
