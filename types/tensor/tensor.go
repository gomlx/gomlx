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

// Package tensor provides a `Tensor` interface with 2 different implementations: `Local` and `Device`, they
// differ on where their values are stored: in the local (host) CPU, or on an accelerator device (TPU, GPU, but
// could be also the CPU if no accelerators are available).
//
// The main use of tensors are to be used as input and output of GoMLX computation graph.
//
// Tensors are multidimensional arrays (from scalar with 0 dimensions, to arbitrarily large dimensions), defined
// by their shape (a data type and its axes dimensions) and their actual content. As a special case, a Tensor can
// also be a tuple of multiple tensors.
//
// This implementation uses `gomlx.types.Shape` to represent the shape, and (for now) only explicitly supports dense
// representations of the data. The underlying implementation of both `Local` and `Device` implementation are wrappers
// to similar XLA data representations.
//
// Transferring tensors to/from local/device areas has a cost, and should be avoided. For example,
// while training weights of an ML model, one generally does not need to transfer those weights to local -- just at
// the end of training to save the model weights. Because the tensors are immutable, the transferring is
// cached, so if `Tensor.Local()` or `Tensor.Device()` is called multiple times, the price is paid only once.
//
// To facilitate managing this, this package also implements a cache system whereas a Local or Device tensor caches
// references to their counterparts -- notice there may be multiple Device tensors (one for each device). This is
// all exposed through the Tensor interface.
//
// Concurrency: the cache system is safe from concurrency point of view. But management of the conflicting uses of
// the content of the tensors themselves is left for the users -- TODO: this needs to be improved.
//
// See details on the documentation of Tensor, Device and Local structures.
package tensor

import (
	"fmt"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/xla"
	"github.com/pkg/errors"
	"reflect"
	"sync"
)

// Tensor represents a multidimensional arrays (from scalar with 0 dimensions, to arbitrarily large dimensions), defined
// by their shape (a data type and its axes' dimensions) and their actual content. As a special case, a Tensor can
// also be a tuple of multiple tensors.
//
// Tensor can be implemented by a tensor.Local or tensor.Device, which reflects whether the data is stored in the local
// CPU on or the device actually running the computation: an accelerator like a GPU or the CPU as well.
//
// Local and Device tensors can be converted to each other -- there is a transferring cost to that. There is a cache
// system to prevent duplicate transfers, but it assumes immutability -- call Local.ClearCache after mutating a
// Local tensor. Device tensors are immutable.
//
// More details in the `tensor` package documentation.
type Tensor interface {
	// Local version of the tensor.
	// If the underlying tensor is `Local` already, it's a no-op.
	// If there is already a cache of a corresponding `Local` tensor, that is returned.
	// Otherwise, the contents of one of the on-device tensors (if there is more than one) are
	// transferred locally.
	Local() *Local

	// Device version of the tensor.
	// If the underlying tensor is on the given `Device` already, it's a no-op.
	// If there is already a cache of a corresponding `Device` tensor on the given device, that is returned.
	// Otherwise, the contents of the `Local` tensor (if available), or one of the on-device tensors are
	// transferred to a `Device` tensor on the given `deviceNum`.
	Device(client HasClient, deviceNum int) *Device

	// CurrentDevice returns the current Device tensor backing this tensor, if there is any.
	// If the tensor is Local, this returns nil.
	//
	// If there is more than one Device tensor, this returns the first one.
	CurrentDevice() *Device

	// Shape of the tensor.
	Shape() shapes.Shape

	// DType of the tensor's shape.
	DType() dtypes.DType

	// Rank of the tensor's shape.
	Rank() int

	// String returns a printable version of the tensor. This may lead to a transfer from a Device tensor
	// with the Local().
	String() string

	// Value returns a multidimensional slice (except if shape is a scalar) containing the values.
	// If the underlying tensor is on device (E.g.: GPU), it's transferred locally with Local().
	Value() any

	// FinalizeAll immediately frees the data from all versions of the Tensor -- local or on-device, and makes the
	// tensor invalid.
	// This calls Finalize on the cached Local and all Device tensors.
	FinalizeAll()

	// IsFinalized returns true if the tensor has already been "finalized", and its
	// data freed.
	IsFinalized() bool
}

var (
	// Compile time type assertions.
	localAssert  *Local
	deviceAssert *Device
	_            Tensor = localAssert
	_            Tensor = deviceAssert
)

// HasClient accepts anything that can return a xla.Client. That includes xla.Client itself and
// graph.Manager.
type HasClient interface {
	Client() *xla.Client
}

func shapeForValue(v any) (shape shapes.Shape, err error) {
	err = shapeForValueRecursive(&shape, reflect.ValueOf(v), reflect.TypeOf(v))
	return
}

func shapeForValueRecursive(shape *shapes.Shape, v reflect.Value, t reflect.Type) error {
	if t.Kind() == reflect.Slice {
		// Recurse into inner slices.
		t = t.Elem()
		shape.Dimensions = append(shape.Dimensions, v.Len())
		shapePrefix := shape.Clone()

		// The first element is the reference
		v0 := v.Index(0)
		err := shapeForValueRecursive(shape, v0, t)
		if err != nil {
			return err
		}

		// Test that other elements have the same shape as the first one.
		for ii := 1; ii < v.Len(); ii++ {
			shapeTest := shapePrefix.Clone()
			err = shapeForValueRecursive(&shapeTest, v.Index(ii), t)
			if err != nil {
				return err
			}
			if !shape.Eq(shapeTest) {
				return fmt.Errorf("sub-slices have irregular shapes, found shapes %q, and %q", shape, shapeTest)
			}
		}
	} else if t.Kind() == reflect.Pointer {
		return fmt.Errorf("cannot convert Pointer (%s) to a concrete value for tensors", t)
	} else {
		shape.DType = shapes.FromType(t)
		if shape.DType == shapes.InvalidDType {
			return fmt.Errorf("cannot convert type %s to a value concrete tensor type (maybe type not supported yet?)", t)
		}
	}
	return nil
}

// baseValue will return the underlying of a multi-dimension array/slice. So `baseValue([][]int{})` would return the
// type `int`.
func baseType(valueType reflect.Type) reflect.Type {
	for valueType.Kind() == reflect.Slice || valueType.Kind() == reflect.Array {
		valueType = valueType.Elem()
	}
	return valueType
}

//func depthDTypeAndBaseType(t reflect.Type) (int, dtypes.DType, reflect.Type) {
//	if t.Kind() == reflect.Slice {
//		depth, dtype, baseType := depthDTypeAndBaseType(t.Elem())
//		return depth + 1, dtype, baseType
//	}
//	return 0, shapes.FromType(t), t
//
//}

// cache stores links to materialization of the tensor (local or on device).
type cache struct {
	mu    sync.Mutex
	local *Local

	// Maps ClientId->Map of deviceNum->Device tensor.
	onDevices map[xla.ClientId]map[int]*Device
}

// FinalizeAll will finalize (free) all associated data.
func (c *cache) FinalizeAll() {
	// Get the list of local and device tensors to finalize.
	c.mu.Lock()
	if c == nil {
		c.mu.Unlock()
		return
	}
	local := c.local
	if local != nil && local.cache == c {
		local.cache = nil
	}
	c.local = nil
	var devices []*Device
	for _, tensorsPerDevice := range c.onDevices {
		for _, d := range tensorsPerDevice {
			if d.cache == c {
				d.cache = nil
			}
			devices = append(devices, d)
		}
	}
	c.onDevices = nil
	c.mu.Unlock()

	// Cache was cleared and unlocked, now we call Finalize on each tensor separately.
	if local != nil {
		local.Finalize()
	}
	for _, d := range devices {
		d.Finalize()
	}
}

// AddDevice to the internal cache, and returns itself for convenience.
func (c *cache) AddDevice(device *Device) *Device {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.lockedAddDevice(device)
}

// lockedAddDevice implements AddDevice, and assumes cache.mu is already locked.
func (c *cache) lockedAddDevice(device *Device) *Device {
	device.cache = c
	if c.onDevices == nil {
		c.onDevices = make(map[xla.ClientId]map[int]*Device)
	}
	clientMap, ok := c.onDevices[device.clientId]
	if !ok {
		clientMap = make(map[int]*Device)
		c.onDevices[device.clientId] = clientMap
	}
	clientMap[device.deviceNum] = device
	return device
}

// ClearDevice from cache, and leaves the device tensor passed without a cache.
func (c *cache) ClearDevice(device *Device) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	if device == nil {
		return
	}
	if device.cache == c {
		device.cache = nil
	}
	if c.onDevices == nil {
		return
	}
	clientMap, ok := c.onDevices[device.clientId]
	if !ok {
		return
	}
	if clientMap == nil {
		return
	}
	delete(clientMap, device.deviceNum)
	device.cache = nil
}

// AddLocal to cache and returns the local tensor for convenience.
func (c *cache) AddLocal(local *Local) *Local {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.local = local
	local.cache = c
	return local
}

// lockedAddLocal is the same as AddLocal, but it assumes the cache is already locked.
func (c *cache) lockedAddLocal(local *Local) *Local {
	c.local = local
	local.cache = c
	return local
}

// ClearLocal cache, and leaves the local cached tensor without a cache.
func (c *cache) ClearLocal() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.local.cache == c {
		c.local.cache = nil
	}
	c.local = nil
}

// Value returns a multidimensional slice (except if shape is a scalar) containing a
// copy of the tensor values.
// It is just a shortcut to calling `device.Local().Value()`.
// See `Local` and `Local.Value()` for details.
func (c *cache) Value() any {
	return c.localFromDevice(nil).Value()
}

// localFromDevice converts the tensor to `Local`, if there is not one cached yet.
// And if it needs converting, it uses the `device`, if one is given.
// If `device == nil`, it converts from any associated `Device` tensor.
//
// If `device` is nil, it picks the first Device tensor.
func (c *cache) localFromDevice(device *Device) *Local {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.localFromDeviceLocked(device)
}

// localFromDeviceLocked implements localFromDevice, but assumes cache is already locked.
func (c *cache) localFromDeviceLocked(device *Device) *Local {
	if c == nil {
		return nil
	}
	if c.local != nil {
		return c.local
	}

	if device == nil {
		if c.onDevices == nil {
			return nil
		}
		// Pick the first on-device tensor and convert it to a local tensor.
	loopAllDevices:
		for _, perDeviceOrdinal := range c.onDevices {
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
	return c.lockedAddLocal(&Local{
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
func (c *cache) Device(hasClient HasClient, deviceNum int) *Device {
	c.mu.Lock()
	// TODO: maybe do this more fine-grained: it may make sense to transfer to several devices in parallel (?).
	defer c.mu.Unlock()
	if c == nil {
		return nil
	}

	// Search in cache.
	client := hasClient.Client()
	if c.onDevices != nil {
		if deviceNumMap, found := c.onDevices[client.Id]; found {
			if device, found := deviceNumMap[deviceNum]; found {
				return device
			}
		}
	}

	if c.local == nil {
		// If there is no local, convert from other device tensor to a local tensor,
		// and then to final device.
		// This is expensive, since there is another copy around.
		// TODO: can we transfer directly between different devices !?
		c.local = c.localFromDeviceLocked(nil)
	}
	local := c.local
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
	c.lockedAddDevice(device)
	return device
}

// CurrentDevice returns the current Device tensor backing this tensor, if there is any.
// If the tensor is Local, this returns nil.
//
// If there is more than one Device tensor, this returns the first one.
func (c *cache) CurrentDevice() *Device {
	c.mu.Lock()
	// TODO: maybe do this more fine-grained: it may make sense to transfer to several devices in parallel (?).
	defer c.mu.Unlock()
	if c == nil {
		return nil
	}
	if c.onDevices == nil {
		return nil
	}
	for _, deviceTensors := range c.onDevices {
		for _, deviceT := range deviceTensors {
			return deviceT
		}
	}
	return nil
}
