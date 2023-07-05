package tensor

import (
	"fmt"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/xla"
	"github.com/pkg/errors"
)

// Device represents a tensor (or tuple) stored on a "device": typically an accelerator (GPU, TPU), but it can
// also be the normal memory if using the CPU. The object doesn't offer much functionality, it's simply
// used as input and output of graph execution. A Device is represented by a `ClientId`, typically provided
// by the `graph.Manager` object, and a device number -- in case there are multiple GPUs/TPUs.
//
// To access it, one has to convert it to a `Local` tensor, which some methods (like `Value()`) do
// automatically. But converting to `Local` usually involves copying across devices, so avoid it
// when not needed -- for instance, variables (or weights) of models typically don't need to be copied
// to local, they can simply be left on-device until they need to be saved for instance.
//
// To create it, either create a Local tensor first and then convert it to Device (see `Tensor.Device() method`),
// or use the output of a computation graph execution (see `graph` package) -- they return Device tensors.
//
// It implements the Tensor interface.
//
// When dealing with large tensors, one will want to carefully manage its life cycle.
// `Device` provides a method called `Finalize()` to immediately release its data memory (managed in XLA C++) --
// it is also called automatically during garbage collection.
// And to release all associated versions of the tensor, including the copies on other devices and the Local tensor
// (if any), one can use `FinalizeAll()`.
type Device struct {
	*cache

	shape shapes.Shape

	shapedBuffer *xla.OnDeviceBuffer
	clientId     xla.ClientId
	deviceNum    int
}

// AssertValid panics if device is nil, if its shape is invalid or if it has already been finalized (freed).
func (device *Device) AssertValid() {
	if device == nil {
		panic(errors.New("tensor.Device is nil"))
	}
	if device.IsFinalized() {
		panic(errors.New("tensor.Device has been finalized, or C++ 'ShapedBuffer' storage is nil!?"))
	}
	if !device.shape.Ok() {
		panic(errors.New("tensor.Device shape is invalid"))
	}
}

// IsFinalized returns true if the tensor has already been "finalized", and its
// data freed.
// It implements Tensor.IsFinalized.
func (device *Device) IsFinalized() bool {
	return device.shapedBuffer == nil || device.shapedBuffer.IsNil()
}

// InternalNewDevice creates a Device tensor from XLA's OnDeviceBuffer structure.
//
// Internal implementation, most users shouldn't use this.
// Instead, one creates a `Local` and converts it to a `Device` using the `Tensor.Device()` method.
func InternalNewDevice(buffer *xla.OnDeviceBuffer) (device *Device) {
	if buffer == nil {
		panic(errors.New("cannot create tensor.Device from nil xla.OnDeviceBuffer"))
	}
	device = &Device{
		shapedBuffer: buffer,
		clientId:     buffer.Client().Id,
		shape:        buffer.Shape(),
		deviceNum:    buffer.DeviceOrdinal(),
	}
	cache := &cache{}
	cache.AddDevice(device)
	return
}

// String converts to string, by converting (transferring) the tensor to local and then using Local.String().
// If the tensor is larger than MaxStringSize, it doesn't convert the tensor to local, and instead only prints
// its shape (with no transfer cost).
func (device *Device) String() string {
	device.AssertValid()
	if device.IsTuple() {
		return fmt.Sprintf("Tuple(%d elements)", device.shape.TupleSize())
	}
	if device.shape.Size() < MaxStringSize {
		return device.Local().String()
	}
	return fmt.Sprintf("%s: (... too large, %d values ...)", device.shape, device.shape.Size())
}

// Shape returns the shape of the Device.
func (device *Device) Shape() shapes.Shape {
	device.AssertValid()
	return device.shape
}

// DType returns the DType of the tensor's shape.
func (device *Device) DType() shapes.DType {
	device.AssertValid()
	return device.shape.DType
}

// Rank returns the rank of the tensor's shape.
func (device *Device) Rank() int {
	device.AssertValid()
	return device.shape.Rank()
}

// ShapedBuffer returns the underlying XLA structure. Internal usage only.
func (device *Device) ShapedBuffer() *xla.OnDeviceBuffer {
	device.AssertValid()
	return device.shapedBuffer
}

// IsTuple returns whether Local is a tuple.
func (device *Device) IsTuple() bool {
	device.AssertValid()
	return device.shape.IsTuple()
}

// Finalize releases the memory associated with the Device tensor, it becomes empty.
//
// This is called automatically when garbage-collected.
// But since sometimes device memory is scarce, this allows for finer control of memory usage.
//
// This can be called more than once: after the first time it doesn't do anything, since the data has already
// been released.
func (device *Device) Finalize() {
	if device.shapedBuffer.IsNil() {
		// Already finalized.
		return
	}
	device.ClearCache()
	device.shapedBuffer.Finalize()
	device.shape = shapes.Shape{}
}

// FinalizeAll releases the memory associated with all copies of the tensor, local and on device(s)).
// And then mark them as empty.
func (device *Device) FinalizeAll() {
	device.cache.FinalizeAll()
}

// ClearCache disconnects the device tensor to any corresponding local data. Internal usage only.
func (device *Device) ClearCache() {
	if device.cache != nil {
		device.cache.ClearDevice(device)
	}

	// Create a new cache with itself only.
	cache := &cache{}
	cache.AddDevice(device)
}

// SplitTuple splits a device tensor into its elements.
//
// This makes the current device tensor invalid -- but not any associated Local (or other Device) tensors.
func (device *Device) SplitTuple() []*Device {
	device.AssertValid()
	if !device.IsTuple() {
		panic(errors.Errorf("tensor.Device is not a tuple, instead is shaped %s", device.shape))
	}
	deviceTensors := make([]*Device, 0, device.shape.TupleSize())
	for ii := 0; ii < device.shape.TupleSize(); ii++ {
		subDeviceT, err := device.shapedBuffer.SubTree([]int{ii})
		if err != nil {
			panic(errors.WithMessagef(err, "tensor.Device.SplitTuple failed while generating split %d", ii))
		}
		deviceTensors = append(deviceTensors, InternalNewDevice(subDeviceT))
	}
	device.Finalize()
	return deviceTensors
}

// Local will transfer data from the Device storage to a Local tensor.
// If the tensor has already been converted, return the associated cached copy.
func (device *Device) Local() *Local {
	return device.cache.localFromDevice(device)
}
