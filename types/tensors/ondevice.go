package tensors

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"maps"
	"reflect"
)

// device holds internal information about on-device storage of a Tensor.
type onDevice struct {
	t         *Tensor
	buffer    backends.Buffer
	deviceNum backends.DeviceNum
}

// FromBuffer creates a Tensor from a backend's buffer. It requires the deviceNum information as well.
func FromBuffer(backend backends.Backend, buffer backends.Buffer) (t *Tensor) {
	// Create tensor.
	t = newTensor(backend.BufferShape(buffer))
	t.backend = backend
	deviceNum := backend.BufferDeviceNum(buffer)
	t.onDevices[deviceNum] = &onDevice{
		t:         t,
		buffer:    buffer,
		deviceNum: deviceNum,
	}
	return
}

// Buffer returns the backend buffer for the tensor.
// It triggers the transfer from local to the device, if the tensor is not already store on device.
//
// If you use the returned buffer in a "donatable" fashion (the accelerator may rewrite the buffer), remember to
// finalize and invalidate the tensor -- it doesn't happen automatically.
//
// The deviceNum is optional. But only one can be given. The default value is 0.
func (t *Tensor) Buffer(backend backends.Backend, deviceNum ...backends.DeviceNum) backends.Buffer {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.AssertValid()
	if len(deviceNum) > 1 {
		exceptions.Panicf("Tensor.Buffer takes at most one deviceNum, %v given", deviceNum)
	}
	if len(deviceNum) == 0 {
		deviceNum = defaultDeviceNums
	}
	t.lockedMaterializeOnDevices(backend, deviceNum...)
	return t.onDevices[deviceNum[0]].buffer
}

// IsFinalized returns true if the tensor has already been "finalized", and its
// data freed.
// It implements Tensor.IsFinalized.
func (d *onDevice) IsFinalized() bool {
	return d == nil || d.buffer == nil
}

// Finalize releases the associated buffer in the PJRT client.
// It doesn't clear the pointer to this Device in the Tensor object.
func (d *onDevice) Finalize() {
	if d.IsFinalized() {
		return
	}
	d.t.backend.BufferFinalize(d.buffer)
	d.buffer = nil
	d.t = nil
}

// MaterializeOnDevices will transfer a Tensor from local storage to the given devices, if needed.
// Generally the user doesn't need to call this function, it's called by the libraries executing GoMLX computations
// automatically when needed.
//
// - If an updated copy of the Tensor is already on the device(s), this is a no-op.
// - If the Tensor has already been used with a different client, this panics: one cannot mix clients on the same Tensor.
// - If no deviceNum is given, 0 is assumed, the default device for the client.
//
// TODO: For now this only transfers from local storage to on-device. Implement cross-device copy in gopjrt.
func (t *Tensor) MaterializeOnDevices(backend backends.Backend, deviceNums ...backends.DeviceNum) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.AssertValid()
	t.lockedMaterializeOnDevices(backend, deviceNums...)
}

// defaultDeviceNums is used whenever `deviceNums` is not provided.
var defaultDeviceNums = []backends.DeviceNum{0}

// lockedMaterializeOnDevices implements Tensor.MaterializeOnDevices
func (t *Tensor) lockedMaterializeOnDevices(backend backends.Backend, deviceNums ...backends.DeviceNum) {
	if t.backend == nil {
		t.backend = backend
	} else if t.backend != backend {
		exceptions.Panicf("Tensor(shape=%s).MaterilizeOnDevices: cannot have a Tensor be stored by different "+
			"backend instances (current=%q, provided=%q), use separate Tensors for this",
			t.shape, t.backend.Name(), backend.Description())
	}

	if t.local == nil {
		// Materialize locally from any other device.
		t.lockedMaterializeLocal()
		if t.local == nil {
			exceptions.Panicf("Tensor(shape=%s).MaterilizeOnDevices: cannot materialize a local copy first to then transfer to devices!?", t.shape)
		}
	}
	if len(deviceNums) == 0 {
		deviceNums = defaultDeviceNums
	}
	for _, deviceNum := range deviceNums {
		d, found := t.onDevices[deviceNum]
		if found && !d.IsFinalized() {
			// Nothing to do.
			continue
		}
		buffer := t.backend.BufferFromFlatData(deviceNum, t.local.flat, t.shape)
		d = &onDevice{
			t:         t,
			buffer:    buffer,
			deviceNum: deviceNum,
		}
		t.onDevices[deviceNum] = d
	}
}

// lockedInvalidateOnDevice destroys all on-device copies of the Tensor. Automatically called when the Tensor is mutated (e.g.: Tensor.MutableFlatData).
// If there is no local copy of the Tensor, this will invalidate the tensor.
//
// Usually, this is called automatically. Mostly for internal use.
func (t *Tensor) lockedInvalidateOnDevice() {
	t.AssertValid()
	for _, d := range t.onDevices {
		d.Finalize()
	}
	maps.DeleteFunc(t.onDevices, func(_ backends.DeviceNum, _ *onDevice) bool {
		return true
	})
}

// lockedMaterializeLocal will transfer a Tensor stored on-device to local storage, that can be accessed by Go.
//
// Usually, this is called automatically by all methods that provide Go access to the data (e.g.: Tensor.ConstFlatData).
func (t *Tensor) lockedMaterializeLocal() {
	if t.local != nil && !t.local.IsFinalized() {
		return
	}
	if t.backend == nil {
		exceptions.Panicf("Tensor(shape=%s) is not associated to any backend, likely with no on-device storage either", t.shape)
	}

	// Get on-device version: try default (deviceNum==0) first.
	deviceNum := backends.DeviceNum(0)
	d, found := t.onDevices[deviceNum]
	if !found {
		for deviceNum, d = range t.onDevices {
			break
		}
	}
	if d.IsFinalized() {
		exceptions.Panicf("Tensor(shape=%s).MaterializeLocal() failed because on-device tensor (deviceNum=%d) is invalid",
			t.shape, deviceNum)
	}

	// Create flat slice.
	flatV := reflect.MakeSlice(reflect.SliceOf(t.shape.DType.GoType()), t.Size(), t.Size())
	t.local = &local{
		t:    t,
		flat: flatV.Interface(),
	}
	t.backend.BufferToFlatData(d.buffer, t.local.flat)
}
