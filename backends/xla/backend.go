package xla

import (
	"fmt"
	"reflect"
	"runtime"
	"unsafe"

	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// Backend implements the XLA/PJRT backends.Backend for GoMLX.
type Backend struct {
	plugin           *pjrt.Plugin
	client           *pjrt.Client
	pluginName       string
	hasSharedBuffers bool
	capabilities     backends.Capabilities
}

// Compile-time check:
var (
	_ backends.DataInterface = (*Backend)(nil)
	_ backends.Backend       = (*Backend)(nil)
)

// CheckValid returns an error if the backend is not valid: if it's nil or has already been finalized.
func (backend *Backend) CheckValid() error {
	if backend == nil {
		return errors.Errorf("%q backend is nil", BackendName)
	}
	if backend.plugin == nil {
		return errors.Errorf("backend %q has already been finalized", BackendName)
	}
	return nil
}

// Name returns the short name of the backend. E.g.: "xla" for the Xla/PJRT plugin.
func (backend *Backend) Name() string {
	return BackendName
}

// String returns Name().
func (backend *Backend) String() string {
	return backend.Name()
}

// Description is a longer description of the Backend that can be used to pretty-print.
func (backend *Backend) Description() string {
	if backend.CheckValid() != nil {
		return fmt.Sprintf("%s: in an invalid state!", BackendName)
	}
	return fmt.Sprintf("%s:%s - %s", BackendName, backend.pluginName, backend.plugin)
}

// NumDevices return the number of devices available for this Backend.
func (backend *Backend) NumDevices() backends.DeviceNum {
	if backend.CheckValid() != nil {
		return 0
	}
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

// IsFinalized returns true if the backend is in an invalid state.
func (backend *Backend) IsFinalized() bool {
	return backend == nil || backend.plugin == nil
}

// castToPJRT casts the buffer to pjrt.Buffer and panics if not possible.
func castToPJRT(buffer backends.Buffer) *pjrt.Buffer {
	pb, ok := buffer.(*pjrt.Buffer)
	if !ok {
		exceptions.Panicf("buffer given is not a %q backend (pjrt) buffer", BackendName)
	}
	return pb
}

// BufferFinalize implements backends.DataInterface.
func (backend *Backend) BufferFinalize(buffer backends.Buffer) error {
	if err := backend.CheckValid(); err != nil {
		return errors.WithMessagef(err, "backend %q is invalid", BackendName)
	}
	buf := castToPJRT(buffer)
	err := buf.Destroy()
	if err != nil {
		return errors.WithMessagef(err, "backend %q: BufferFinalize", BackendName)
	}
	return nil
}

// BufferShape returns the shape for the buffer.
func (backend *Backend) BufferShape(buffer backends.Buffer) (shapes.Shape, error) {
	var noShape shapes.Shape
	if err := backend.CheckValid(); err != nil {
		return noShape, err
	}
	pBuffer := castToPJRT(buffer)
	dtype, err := pBuffer.DType()
	if err != nil {
		return noShape, errors.WithMessagef(err, "backend %q", BackendName)
	}
	dims, err := pBuffer.Dimensions()
	if err != nil {
		return noShape, errors.WithMessagef(err, "backend %q", BackendName)
	}
	return shapes.Make(dtype, dims...), nil
}

// BufferDeviceNum returns the deviceNum for the buffer.
func (backend *Backend) BufferDeviceNum(buffer backends.Buffer) (backends.DeviceNum, error) {
	if err := backend.CheckValid(); err != nil {
		return 0, err
	}
	pBuffer := castToPJRT(buffer)
	device, err := pBuffer.Device()
	if err != nil {
		return 0, errors.WithMessagef(err, "backend %q", BackendName)
	}
	num := pBuffer.Client().NumForDevice(device)
	if num == -1 {
		return 0, errors.Errorf("backend %q: pjrt buffer stored on an unknown device!?", BackendName)
	}
	return backends.DeviceNum(num), nil
}

// BufferToFlatData transfers the flat values of buffer to the Go flat array.
// The slice flat must have the exact number of elements required to store the Buffer shape.
//
// See also FlatDataToBuffer, BufferShape, and shapes.Shape.Size.
func (backend *Backend) BufferToFlatData(buffer backends.Buffer, flat any) error {
	if err := backend.CheckValid(); err != nil {
		return err
	}
	shape, err := backend.BufferShape(buffer)
	if err != nil {
		return err
	}
	if shape.IsZeroSize() {
		// No data to transfer.
		return nil
	}

	flatV := reflect.ValueOf(flat)
	if flatV.Kind() != reflect.Slice {
		return errors.Errorf("backend %q: BufferToFlatData, but flat is not a slice, instead it is %T", BackendName, flat)
	}
	flatDType := dtypes.FromGoType(flatV.Type().Elem())
	if flatDType != shape.DType {
		return errors.Errorf("backend %q: BufferToFlatData with buffer of shape %s, but flat with incompatible dtype, it is %T", BackendName, shape, flat)
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
	err = pBuffer.ToHost(dst)
	if err != nil {
		return errors.WithMessagef(err, "backend %q: BuffferToFlatData", BackendName)
	}
	return nil
}

// BufferFromFlatData transfers data from Go given as a flat slice (of the type corresponding to the shape DType)
// to the deviceNum, and returns the corresponding Buffer.
func (backend *Backend) BufferFromFlatData(deviceNum backends.DeviceNum, flat any, shape shapes.Shape) (backends.Buffer, error) {
	flatV := reflect.ValueOf(flat)
	if flatV.Kind() != reflect.Slice {
		return nil, errors.Errorf("backend %q: BuffferFromFlatData, but flat is not a slice, instead it is %T", BackendName, flat)
	}
	flatDType := dtypes.FromGoType(flatV.Type().Elem())
	if flatDType != shape.DType {
		return nil, errors.Errorf("backend %q: BuffferFromFlatData with shape %s, but flat with incompatible dtype, it is %T", BackendName, shape, flat)
	}

	buffer, err := backend.client.BufferFromHost().
		FromFlatDataWithDimensions(flat, shape.Dimensions).
		ToDeviceNum(int(deviceNum)).
		Done()
	if err != nil {
		return nil, errors.WithMessagef(err, "backend %q: BuffferFromFlatData", BackendName)
	}
	return buffer, nil
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
func (backend *Backend) NewSharedBuffer(deviceNum backends.DeviceNum, shape shapes.Shape) (buffer backends.Buffer, flat any, err error) {
	if err = backend.CheckValid(); err != nil {
		return
	}
	devices := backend.client.AddressableDevices()
	if deviceNum < 0 || int(deviceNum) >= len(devices) {
		err = errors.Errorf("deviceNum=%d not available for backend, only %d devices are available", deviceNum, len(devices))
		return
	}
	device := devices[deviceNum]
	buffer, flat, err = backend.client.NewSharedBuffer(shape.DType, shape.Dimensions, device)
	if err != nil {
		err = errors.WithMessagef(err, "backend %q NewSharedBuffer", BackendName)
	}
	return
}

// BufferData implements backends.Backend interface.
//
// For XLA this means allocating the aligned memory and calling pjrt.Client.CreateViewOfDeviceBuffer
// to create a buffer that shares the memory.
func (backend *Backend) BufferData(buffer backends.Buffer) (flat any, err error) {
	if err := backend.CheckValid(); err != nil {
		return nil, err
	}
	buf := buffer.(*pjrt.Buffer)
	flat, err = buf.Data()
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to access buffer data directly, maybe not supported by backend?")
	}
	return
}

// Capabilities returns information about what is supported by this backend.
func (backend *Backend) Capabilities() backends.Capabilities {
	return backend.capabilities
}
