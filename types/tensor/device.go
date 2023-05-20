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
// To access it one has to convert it to a `Local` tensor, which some methods (like `Value()`) do
// automatically. But converting to `Local` usually involves copying across devices, so avoid it
// when not needed -- for instance, variables (or weights) of models typically don't need to be copied
// to local, they can simply be left on device until they need to be saved.
//
// To create it, either create a Local tensor first and then convert it to Device, or use the output of
// the execution of a computation graph -- they return Device tensors.
//
// It implements the Tensor interface.
//
// When dealing with large tensors, one will want to carefully manage its life cycle. It provides a method
// called Finalize() to immediately release its data memory (managed in XLA C++) -- it is also called
// automatically during garbage collection. And to release all associated versions of the tensor,
// including the copies on other devices and the local tensor (if any), one can use FinalizeAll().
type Device struct {
	*cache

	shape shapes.Shape

	shapedBuffer *xla.OnDeviceBuffer
	clientId     xla.ClientId
	deviceNum    int

	// error reports back any error during the creation or transfer of this Tensor.
	error error
}

// InternalNewDevice creates a Device tensor from XLA's OnDeviceBuffer structure. Internal implementation,
// most users don't need to use this.
func InternalNewDevice(buffer *xla.OnDeviceBuffer) (deviceT *Device) {
	if buffer == nil {
		deviceT = &Device{error: fmt.Errorf("invalid (nil) underlying OnDeviceBuffer")}
	} else {
		deviceT = &Device{
			shapedBuffer: buffer,
			clientId:     buffer.Client().Id,
			shape:        buffer.Shape(),
			deviceNum:    buffer.DeviceOrdinal(),
		}
	}
	cache := &cache{}
	cache.AddDevice(deviceT)
	return
}

// Empty returns whether Local is holding no data or is in an error state. It's similar
// to a "nil" state for Local.
func (device *Device) Empty() bool {
	return device == nil || device.shapedBuffer.IsNil()
}

// Ok returns whether the shapedBuffer is not empty and has no error.
func (device *Device) Ok() bool {
	return !device.Empty() && device.error == nil
}

// Error returns the message that caused an error state.
func (device *Device) Error() error {
	if device == nil {
		return errors.Errorf("device tensor is nil")
	}
	if device.error != nil {
		return device.error
	}
	if device.Empty() {
		return errors.Errorf("device tensor is empty")
	}
	return device.error
}

// String converts to string, by converting (transferring) the tensor to local and then using Local.String().
func (device *Device) String() string {
	if device.error != nil {
		return fmt.Sprintf("tensor.Device.error=%v", device.error)
	}
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
	if !device.Ok() {
		return shapes.Shape{}
	}
	return device.shape
}

// DType returns the DType of the tensor's shape.
func (device *Device) DType() shapes.DType {
	return device.shape.DType
}

// Rank returns the rank fo the tensor's shape.
func (device *Device) Rank() int {
	return device.shape.Rank()
}

// Value returns a multidimensional slice (except if shape is a scalar) containing a copy of the tensor values.
// It is just a shortcut to calling `device.Local().Value()`. See Local and Local.Value for details.
func (device *Device) Value() any {
	return device.Local().Value()
}

// MakeDeviceWithError creates a device tensor with the given error.
func MakeDeviceWithError(err error) *Device {
	deviceT := &Device{error: err}
	cache := &cache{}
	cache.AddDevice(deviceT)
	return deviceT
}

// ShapedBuffer returns the underlying XLA structure. Internal usage only.
func (device *Device) ShapedBuffer() *xla.OnDeviceBuffer {
	if device.Empty() {
		return nil
	}
	return device.shapedBuffer
}

// IsTuple returns whether Local is a tuple.
func (device *Device) IsTuple() bool { return !device.Empty() && device.shape.IsTuple() }

// Finalize releases the memory associated with the Device tensor, it becomes empty.
//
// This is called automatically when garbage collected. But since sometimes device memory is scarse, this
// allows for finer control of memory usage.
func (device *Device) Finalize() {
	device.ClearCache()
	device.shapedBuffer.Finalize()
	device.shape = shapes.Shape{}
	device.error = nil
}

// FinalizeAll releases the memory associated with all copies of the tensor, local and on device(s)).
// And then mark them as empty.
func (device *Device) FinalizeAll() {
	device.cache.FinalizeAll()
}

// ClearCache disconnects the device tensor to any corresponding local data. Internal usage only.
func (device *Device) ClearCache() {
	device.cache.ClearDevice(device)

	// Create a new cache with itself only.
	cache := &cache{}
	cache.AddDevice(device)
}

// SplitTupleError splits a device tensor into its elements. In case of error, return the error.
//
// This makes the current device tensor invalid.
func (device *Device) SplitTupleError() ([]*Device, error) {
	if !device.Ok() {
		return nil, fmt.Errorf("cannot split device tensor that has error: %w", device.error)
	}
	if !device.IsTuple() {
		return nil, fmt.Errorf("cannot split device tensor that is not a tuple")
	}
	return device.splitTupleImpl(true)
}

// SplitTuple splits a device tensor into its elements. In case of error, returns nil.
//
// This makes the current device tensor invalid.
func (device *Device) SplitTuple() []*Device {
	if !device.Ok() {
		return nil
	}
	if !device.IsTuple() {
		return nil
	}
	parts, _ := device.splitTupleImpl(false)
	return parts
}

func (device *Device) splitTupleImpl(returnError bool) ([]*Device, error) {
	deviceTensors := make([]*Device, 0, device.shape.TupleSize())
	for ii := 0; ii < device.shape.TupleSize(); ii++ {
		subDeviceT, err := device.shapedBuffer.SubTree([]int{ii})
		if err == nil {
			deviceTensors = append(deviceTensors, InternalNewDevice(subDeviceT))
		} else {
			if returnError {
				return nil, fmt.Errorf("failed generating split %d: %w", ii, err)
			}
			deviceTensors = append(deviceTensors,
				MakeDeviceWithError(errors.Wrapf(err, "failed extracting split %d", ii)))
		}
	}
	device.FinalizeAll()
	return deviceTensors, nil
}
