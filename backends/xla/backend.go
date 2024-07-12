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

// Backend implements the XLA/PJRT backends.Backend for GoMLX.
type Backend struct {
	plugin     *pjrt.Plugin
	client     *pjrt.Client
	pluginName string
}

// AssertValid will panic if the backend is not valid: if it's nil or has already been finalized.
func (b *Backend) AssertValid() {
	if b == nil {
		exceptions.Panicf("%q backend is nil", BackendName)
	}
	if b.plugin == nil {
		exceptions.Panicf("%q backend's plugin is nil, has it already been finalized?", BackendName)
	}
}

// Name returns the short name of the backend. E.g.: "xla" for the Xla/PJRT plugin.
func (b *Backend) Name() string {
	return BackendName
}

// Description is a longer description of the Backend that can be used to pretty-print.
func (b *Backend) Description() string {
	b.AssertValid()
	return fmt.Sprintf("%s:%s - %s", BackendName, b.pluginName, b.plugin)
}

// NumDevices return the number of devices available for this Backend.
func (b *Backend) NumDevices() backends.DeviceNum {
	b.AssertValid()
	return backends.DeviceNum(len(b.client.AddressableDevices()))
}

// Builder creates a new builder used to define a new computation.
func (b *Backend) Builder() backends.Builder {
	b.AssertValid()
	return nil
}

// Finalize releases all the associated resources immediately, and makes the backend invalid.
func (b *Backend) Finalize() {
	if b.plugin == nil {
		return
	}
	if b.client != nil {
		err := b.client.Destroy()
		if err != nil {
			klog.Warningf("Failure while destroying PJRT client: %+v", err)
		}
		b.client = nil
	}
	b.plugin = nil
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
func (b *Backend) BufferFinalize(buffer backends.Buffer) {
	b.AssertValid()
	buf := castToPJRT(buffer)
	err := buf.Destroy()
	if err != nil {
		panic(errors.WithMessagef(err, "backend %q: BufferFinalize", BackendName))
	}
}

// BufferShape returns the shape for the buffer.
func (b *Backend) BufferShape(buffer backends.Buffer) shapes.Shape {
	b.AssertValid()
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
func (b *Backend) BufferDeviceNum(buffer backends.Buffer) backends.DeviceNum {
	b.AssertValid()
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

// BufferToFlatData transfers the flat values of buffer to the Go flat array. The slice flat must have
// the exact number of elements required to store the Buffer shape. See BufferShape, and shapes.Shape.Size.
func (b *Backend) BufferToFlatData(buffer backends.Buffer, flat any) {
	b.AssertValid()
	shape := b.BufferShape(buffer)

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
func (b *Backend) BufferFromFlatData(deviceNum backends.DeviceNum, flat any, shape shapes.Shape) backends.Buffer {
	flatV := reflect.ValueOf(flat)
	if flatV.Kind() != reflect.Slice {
		exceptions.Panicf("backend %q: BuffferFromFlatData, but flat is not a slice, instead it is %T", BackendName, flat)
	}
	flatDType := dtypes.FromGoType(flatV.Type().Elem())
	if flatDType != shape.DType {
		exceptions.Panicf("backend %q: BuffferFromFlatData with shape %s, but flat with incompatible dtype, it is %T", BackendName, shape, flat)
	}

	buffer, err := b.client.BufferFromHost().
		FromFlatDataWithDimensions(flat, shape.Dimensions).
		ToDeviceNum(int(deviceNum)).
		Done()
	if err != nil {
		panic(errors.WithMessagef(err, "backend %q: BuffferFromFlatData", BackendName))
	}
	return buffer
}
