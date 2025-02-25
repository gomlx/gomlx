package xla

import (
	"fmt"
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"reflect"
	"runtime"
	"unsafe"
)

func panicf(format string, args ...any) {
	err := errors.Errorf(format, args...)
	panic(err)
}

// Backend implements the XLA/PJRT backends.Backend for GoMLX.
type Backend struct {
	plugin           *pjrt.Plugin
	client           *pjrt.Client
	pluginName       string
	hasSharedBuffers bool
}

// AssertValid will panic if the backend is not valid: if it's nil or has already been finalized.
func (backend *Backend) AssertValid() {
	if backend == nil {
		exceptions.Panicf("%q backend is nil", BackendName)
	}
	if backend.plugin == nil {
		exceptions.Panicf("%q backend's plugin is nil, has it already been finalized?", BackendName)
	}
}

// Name returns the short name of the backend. E.g.: "xla" for the Xla/PJRT plugin.
func (backend *Backend) Name() string {
	return BackendName
}

// Description is a longer description of the Backend that can be used to pretty-print.
func (backend *Backend) Description() string {
	backend.AssertValid()
	return fmt.Sprintf("%s:%s - %s", BackendName, backend.pluginName, backend.plugin)
}

// NumDevices return the number of devices available for this Backend.
func (backend *Backend) NumDevices() backends.DeviceNum {
	backend.AssertValid()
	return backends.DeviceNum(len(backend.client.AddressableDevices()))
}

// Finalize releases all the associated resources immediately, and makes the backend invalid.
func (backend *Backend) Finalize() {
	if backend.plugin == nil {
		return
	}
	if backend.client != nil {
		err := backend.client.Destroy()
		if err != nil {
			klog.Warningf("Failure while destroying PJRT client: %+v", err)
		}
		backend.client = nil
	}
	backend.plugin = nil
	return
}

// castToPJRT casts the buffer to pjrt.Buffer and panics if not possible.
func castToPJRT(buffer backends.Buffer) *pjrt.Buffer {
	pb, ok := buffer.(*pjrt.Buffer)
	if !ok {
		exceptions.Panicf("buffer given is not a %q backend (pjrt) buffer", BackendName)
	}
	return pb
}

// BufferFinalize allows client to inform backend that buffer is no longer needed and associated resources can be
// freed immediately.
func (backend *Backend) BufferFinalize(buffer backends.Buffer) {
	backend.AssertValid()
	buf := castToPJRT(buffer)
	err := buf.Destroy()
	if err != nil {
		panic(errors.WithMessagef(err, "backend %q: BufferFinalize", BackendName))
	}
}

// BufferShape returns the shape for the buffer.
func (backend *Backend) BufferShape(buffer backends.Buffer) shapes.Shape {
	backend.AssertValid()
	pBuffer := castToPJRT(buffer)
	dtype, err := pBuffer.DType()
	if err != nil {
		panic(errors.WithMessagef(err, "backend %q", BackendName))
	}
	dims, err := pBuffer.Dimensions()
	if err != nil {
		panic(errors.WithMessagef(err, "backend %q", BackendName))
	}
	return shapes.Make(dtype, dims...)
}

// BufferDeviceNum returns the deviceNum for the buffer.
func (backend *Backend) BufferDeviceNum(buffer backends.Buffer) backends.DeviceNum {
	backend.AssertValid()
	pBuffer := castToPJRT(buffer)
	device, err := pBuffer.Device()
	if err != nil {
		panic(errors.WithMessagef(err, "backend %q", BackendName))
	}
	num := pBuffer.Client().NumForDevice(device)
	if num == -1 {
		exceptions.Panicf("backend %q: pjrt buffer stored on an unknown device!?", BackendName)
	}
	return backends.DeviceNum(num)
}

// BufferToFlatData transfers the flat values of buffer to the Go flat array.
// The slice flat must have the exact number of elements required to store the Buffer shape.
//
// See also FlatDataToBuffer, BufferShape, and shapes.Shape.Size.
func (backend *Backend) BufferToFlatData(buffer backends.Buffer, flat any) {
	backend.AssertValid()
	shape := backend.BufferShape(buffer)

	flatV := reflect.ValueOf(flat)
	if flatV.Kind() != reflect.Slice {
		exceptions.Panicf("backend %q: BufferToFlatData, but flat is not a slice, instead it is %T", BackendName, flat)
	}
	flatDType := dtypes.FromGoType(flatV.Type().Elem())
	if flatDType != shape.DType {
		exceptions.Panicf("backend %q: BufferToFlatData with buffer of shape %s, but flat with incompatible dtype, it is %T", BackendName, shape, flat)
	}

	pBuffer := castToPJRT(buffer)
	element0 := flatV.Index(0)
	flatValuesPtr := element0.Addr().UnsafePointer()
	sizeBytes := uintptr(flatV.Len()) * element0.Type().Size()

	var pinner runtime.Pinner
	pinner.Pin(pBuffer)
	pinner.Pin(flatValuesPtr)
	defer pinner.Unpin()
	dst := unsafe.Slice((*byte)(flatValuesPtr), sizeBytes)
	err := pBuffer.ToHost(dst)
	if err != nil {
		panic(errors.WithMessagef(err, "backend %q: BuffferToFlatData", BackendName))
	}
}

// BufferFromFlatData transfers data from Go given as a flat slice (of the type corresponding to the shape DType)
// to the deviceNum, and returns the corresponding Buffer.
func (backend *Backend) BufferFromFlatData(deviceNum backends.DeviceNum, flat any, shape shapes.Shape) backends.Buffer {
	flatV := reflect.ValueOf(flat)
	if flatV.Kind() != reflect.Slice {
		exceptions.Panicf("backend %q: BuffferFromFlatData, but flat is not a slice, instead it is %T", BackendName, flat)
	}
	flatDType := dtypes.FromGoType(flatV.Type().Elem())
	if flatDType != shape.DType {
		exceptions.Panicf("backend %q: BuffferFromFlatData with shape %s, but flat with incompatible dtype, it is %T", BackendName, shape, flat)
	}

	buffer, err := backend.client.BufferFromHost().
		FromFlatDataWithDimensions(flat, shape.Dimensions).
		ToDeviceNum(int(deviceNum)).
		Done()
	if err != nil {
		panic(errors.WithMessagef(err, "backend %q: BuffferFromFlatData", BackendName))
	}
	return buffer
}

// HasSharedBuffers returns whether this PJRT plugin supports "shared buffers".
// In PJRT that means supporting pjrt.Client.CreateViewOfDeviceBuffer.
func (backend *Backend) HasSharedBuffers() bool {
	return backend.hasSharedBuffers
}

// NewSharedBuffer implements backends.Backend interface.
//
// For XLA this means allocating the aligned memory and calling pjrt.Client.CreateViewOfDeviceBuffer
// to create a buffer that shares the memory.
func (backend *Backend) NewSharedBuffer(deviceNum backends.DeviceNum, shape shapes.Shape) (buffer backends.Buffer, flat any) {
	devices := backend.client.AddressableDevices()
	if deviceNum < 0 || int(deviceNum) >= len(devices) {
		panicf("deviceNum=%d not available for backend, only %d devices are available", deviceNum, len(devices))
	}
	device := devices[deviceNum]
	var err error
	buffer, flat, err = backend.client.NewSharedBuffer(shape.DType, shape.Dimensions, device)
	if err != nil {
		panic(err)
	}
	return
}

// BufferData implements backends.Backend interface.
//
// For XLA this means allocating the aligned memory and calling pjrt.Client.CreateViewOfDeviceBuffer
// to create a buffer that shares the memory.
func (backend *Backend) BufferData(buffer backends.Buffer) (flat any) {
	buf := buffer.(*pjrt.Buffer)
	var err error
	flat, err = buf.Data()
	if err != nil {
		panic(errors.WithMessagef(err, "failed to access buffer data directly, maybe not supported by backend?"))
	}
	return
}
