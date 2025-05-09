package tensors

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/pkg/errors"
	"reflect"
)

// device holds internal information about on-device storage of a Tensor.
type onDevice struct {
	t         *Tensor
	buffer    backends.Buffer
	deviceNum backends.DeviceNum
}

// FromBuffer creates a Tensor from a backend's buffer. It requires the deviceNum information as well.
// The ownership of the buffer is transferred to the new Tensor.
//
// This doesn't work for shared buffers, it assumes the buffer is not shared.
func FromBuffer(backend backends.Backend, buffer backends.Buffer) (t *Tensor) {
	// Create tensor.
	shape, err := backend.BufferShape(buffer)
	if err != nil {
		panic(err)
	}
	t = newTensor(shape)
	t.backend = backend
	deviceNum, err := backend.BufferDeviceNum(buffer)
	if err != nil {
		panic(err)
	}
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
// The deviceNum is optional. But only one can be given. The default value is 0.
//
// Careful not to finalize the buffer while the buffer is in use.
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
	t.lockedMaterializeOnDevices(backend, true, deviceNum...)
	return t.onDevices[deviceNum[0]].buffer
}

// DonateBuffer returns the backend buffer for the tensor, and transfers the ownership of the buffer to the caller.
// This may invalidate the tensor, if there is no other on-device storage or local storage -- in particular this
// is true if using "shared buffers" (generally true for CPU based plugins).
//
// Mostly used internally -- by graph.Graph.Run and graph.Exec when the value in the buffer is no longer needed
// after execution.
//
// It will panic if the buffer is shared (see Tensor.IsShared): shared buffers cannot be donated.
//
// It triggers the transfer from local to the device, if the tensor is not already store on device.
//
// It doesn't finalize(release) the local tensor value.
//
// The deviceNum is optional. But only one can be given. The default value is 0.
func (t *Tensor) DonateBuffer(backend backends.Backend, deviceNum ...backends.DeviceNum) backends.Buffer {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.AssertValid()
	if len(deviceNum) > 1 {
		exceptions.Panicf("Tensor.Buffer takes at most one deviceNum, %v given", deviceNum)
	}
	if len(deviceNum) == 0 {
		deviceNum = defaultDeviceNums
	}
	t.lockedMaterializeOnDevices(backend, false, deviceNum...)
	buf := t.onDevices[deviceNum[0]].buffer
	delete(t.onDevices, deviceNum[0])
	if t.local == nil && len(t.onDevices) == 0 {
		t.lockedFinalizeAll()
	}
	return buf
}

// IsFinalized returns true if the tensor has already been "finalized", and its
// data freed.
// It implements Tensor.IsFinalized.
func (d *onDevice) IsFinalized() bool {
	return d == nil || d.buffer == nil
}

// Finalize releases the associated buffer in the PJRT client.
// It's the caller responsibility to ensure this buffer is not being used elsewhere (like in the middle of an execution).
//
// It doesn't clear the pointer to this Device in the Tensor object.
func (d *onDevice) Finalize() {
	if d.IsFinalized() {
		return
	}
	if err := d.t.backend.BufferFinalize(d.buffer); err != nil {
		panic(errors.WithMessagef(err, "Tensor.OnDevice.Finalize: failed to finalize buffer on-device"))
	}
	d.buffer = nil
	d.t = nil
}

// MaterializeOnDevices will transfer a Tensor from local storage to the given devices, if needed.
// Generally the user doesn't need to call this function, it's called by the libraries executing GoMLX computations
// automatically when needed.
//
// If share is true, and if the backend allows for shared buffer, this will create a shared buffer,
// which is more economic.
//
// - If an updated copy of the Tensor is already on the device(s), this is a no-op.
// - If the Tensor has already been used with a different client, this panics: one cannot mix clients on the same Tensor.
// - If no deviceNum is given, 0 is assumed, the default device for the client.
//
// TODO: For now this only transfers from local storage to on-device. Implement cross-device copy in gopjrt.
func (t *Tensor) MaterializeOnDevices(backend backends.Backend, share bool, deviceNums ...backends.DeviceNum) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.AssertValid()
	t.lockedMaterializeOnDevices(backend, share, deviceNums...)
}

// defaultDeviceNums is used whenever `deviceNums` is not provided.
var defaultDeviceNums = []backends.DeviceNum{0}

// lockedMaterializeOnDevices implements Tensor.MaterializeOnDevices
//
// If share is true, it will attempt to materialize to a shared buffer if available. In this case it frees
// the local tensor storage, and starts using the shared data instead.
func (t *Tensor) lockedMaterializeOnDevices(backend backends.Backend, share bool, deviceNums ...backends.DeviceNum) {
	if t.backend == nil {
		t.backend = backend
	} else if t.backend != backend {
		exceptions.Panicf("Tensor(shape=%s).MaterilizeOnDevices: cannot have a Tensor be stored by different "+
			"backend instances (current=%q, provided=%q), use separate Tensors for this",
			t.shape, t.backend.Name(), backend.Description())
	}
	if t.backend == nil {
		exceptions.Panicf("cannote MaterializeOnDevice with a nil backend")
	}

	if len(deviceNums) == 0 {
		deviceNums = defaultDeviceNums
	}
	if len(deviceNums) > 1 {
		exceptions.Panicf("Tensor.MaterializeOnDevices: tensor synchronization across multiple devices not supported yet -- you can create one tensor per device though")
	}
	deviceNum := deviceNums[0]
	d, found := t.onDevices[deviceNum]
	if found && !d.IsFinalized() {
		// Nothing to do.
		return
	}
	if !found && len(t.onDevices) > 0 {
		exceptions.Panicf("Tensor.MaterializeOnDevices: tensor synchronization across multiple devices not supported yet -- you can create one tensor per device though")
	}

	// We need to materialize the onDevice from local:
	if t.local == nil {
		exceptions.Panicf("Tensor(shape=%s) has invalid local and on-device data!?", t.shape)
	}

	var buffer backends.Buffer
	var err error
	if share && backend.HasSharedBuffers() {
		buffer, t.sharedFlat, err = backend.NewSharedBuffer(deviceNum, t.shape)
		if err != nil {
			panic(errors.WithMessagef(err, "Tensor.MaterializeOnDevices: failed to create shared buffer"))
		}
		reflect.Copy(reflect.ValueOf(t.sharedFlat), reflect.ValueOf(t.local.flat))
		t.local = nil // Free local storage.
		t.isShared = true
	} else {
		buffer, err = t.backend.BufferFromFlatData(deviceNum, t.local.flat, t.shape)
		if err != nil {
			panic(errors.WithMessagef(err, "Tensor.MaterializeOnDevices: failed to create buffer"))
		}
	}
	d = &onDevice{
		t:         t,
		buffer:    buffer,
		deviceNum: deviceNum,
	}
	t.onDevices[deviceNum] = d
}

// InvalidateOnDevice destroys all on-device copies of the Tensor, so the local copy becomes the source of truth.
// It does nothing is tensor is shared, see Tensor.IsShared.
//
// It's the caller responsibility to ensure this buffer is not being used elsewhere (like in the middle of an execution).
//
// This is automatically called if the Tensor is mutated (e.g.: Tensor.MutableFlatData) or when the on-device value
// is donated to the execution of a graph.
//
// If there is no local copy of the Tensor, this will invalidate the whole tensor.
//
// Usually, this is called automatically. Mostly for internal use.
func (t *Tensor) InvalidateOnDevice() {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.isShared {
		return
	}
	t.lockedInvalidateOnDevice()
}

// lockedInvalidateOnDevice destroys all on-device copies of the Tensor.
// It does nothing if the tensor is shared.
//
// If there is no local copy of the Tensor, this will invalidate the tensor.
//
// Usually, this is called automatically. Mostly for internal use.
func (t *Tensor) lockedInvalidateOnDevice() {
	if t.isShared {
		return
	}
	t.AssertValid()
	for deviceNum, d := range t.onDevices {
		d.Finalize()
		delete(t.onDevices, deviceNum)
	}
}

// OnDeviceClone creates a clone of the tensor t that has backend storage.
func (t *Tensor) OnDeviceClone(backend backends.Backend, deviceNums ...backends.DeviceNum) *Tensor {
	if len(deviceNums) == 0 {
		deviceNums = defaultDeviceNums
	}
	if len(deviceNums) > 1 {
		exceptions.Panicf("Tensor.OnDeviceClone: tensor synchronization across multiple devices not supported yet -- you can create one tensor per device though")
	}
	deviceNum := deviceNums[0]

	newT := newTensor(t.Shape())
	newT.backend = backend
	var buffer backends.Buffer
	var err error
	if backend.HasSharedBuffers() {
		buffer, newT.sharedFlat, err = backend.NewSharedBuffer(deviceNum, t.shape)
		if err != nil {
			panic(errors.WithMessagef(err, "Tensor.OnDeviceClone: failed to create shared buffer"))
		}
		t.ConstFlatData(func(flat any) {
			reflect.Copy(reflect.ValueOf(newT.sharedFlat), reflect.ValueOf(flat))
		})
	} else {
		t.ConstFlatData(func(flat any) {
			buffer, err = newT.backend.BufferFromFlatData(deviceNum, flat, newT.shape)
			if err != nil {
				panic(errors.WithMessagef(err, "Tensor.OnDeviceClone: failed to create buffer"))
			}
		})
	}
	d := &onDevice{
		t:         newT,
		buffer:    buffer,
		deviceNum: deviceNum,
	}
	newT.onDevices[deviceNum] = d
	return newT
}

// Clone creates a clone of the Tensor value with shared backing with the backend -- if it supports --
// or it falls back to LocalClone.
func (t *Tensor) Clone() *Tensor {
	if t.backend == nil || !t.backend.HasSharedBuffers() {
		return t.LocalClone()
	}
	return t.OnDeviceClone(t.backend)
}

// MaterializeLocal will make sure there is a local storage of the tensor.
// If there isn't already a local copy, this triggers a transfer from an on-device storage to a local copy.
//
// It's a No-op if using shared buffers with the accelerator. If you want to force a local copy,
// use LocalClone instead.
func (t *Tensor) MaterializeLocal() {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.isShared {
		return
	}
	t.lockedMaterializeLocal()
}

// lockedMaterializeLocal will make sure there is a local storage copy of the tensor.
// If there isn't already a local copy, this triggers a transfer from an on-device storage to a local copy.
//
// Usually, this is called automatically by all methods that provide Go access to the data (e.g.: Tensor.ConstFlatData).
//
// Notice this is not aware of shared buffers: shared buffer handling -- in which case likely there is no
// need to materialize a local copy -- has to be done by the caller.
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
	if err := t.backend.BufferToFlatData(d.buffer, t.local.flat); err != nil {
		panic(errors.WithMessagef(err, "Tensor(shape=%s).MaterializeLocal() failed to copy from on-device buffer", t.shape))
	}
}

// CopyFrom will copy the contents from tFrom. t and tFrom must have the same shape.
//
// This is efficient if tFrom is on-device only, in which case the device values are materialized
// locally into t, the receiving tensor.
func (t *Tensor) CopyFrom(tFrom *Tensor) {
	if !t.Shape().Equal(tFrom.Shape()) {
		exceptions.Panicf("Tensor.CopyFrom() among different shaped tensors: receiver has shape %s, and tFrom has shape %s",
			t.Shape(), tFrom.Shape())
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	// Make sure t has a receiving local data.
	var tFlat any // a slice of []dtype
	if t.IsShared() {
		tFlat = t.sharedFlat
	} else {
		if !t.IsLocal() {
			// Bring to local.
			t.lockedMaterializeLocal()
			t.lockedInvalidateOnDevice()
		}
		tFlat = t.local.flat
	}

	// Lock tFrom.
	tFrom.mu.Lock()
	defer tFrom.mu.Unlock()

	if tFrom.IsShared() {
		// Copy from shared buffer.
		reflect.Copy(reflect.ValueOf(tFlat), reflect.ValueOf(tFrom.sharedFlat))
		return

	}
	if tFrom.IsLocal() {
		// Copy from local.
		reflect.Copy(reflect.ValueOf(t.local.flat), reflect.ValueOf(tFrom.local.flat))
		return
	}

	// Materialize tFrom onDevice directly to tFrom.
	// Get on-device version: try default (deviceNum==0) first.
	deviceNum := backends.DeviceNum(0)
	d, found := tFrom.onDevices[deviceNum]
	if !found {
		for deviceNum, d = range tFrom.onDevices {
			break
		}
	}
	if d.IsFinalized() {
		exceptions.Panicf("Tensor(shape=%s).CopyFrom(tFrom) failed because tFrom on-device tensor (deviceNum=%d) is invalid",
			t.shape, deviceNum)
	}
	if err := tFrom.backend.BufferToFlatData(d.buffer, tFlat); err != nil {
		panic(errors.WithMessagef(err, "Tensor(shape=%s).CopyFrom(tFrom) failed to copy from on-device buffer", t.shape))
	}
}

// IsOnDevice checks whether the Tensor has an on-device copy on the given deviceNum.
//
// See MaterializeOnDevices to trigger a transfer/copy to the given device.
func (t *Tensor) IsOnDevice(deviceNum backends.DeviceNum) bool {
	t.AssertValid()
	_, ok := t.onDevices[deviceNum]
	return ok
}
