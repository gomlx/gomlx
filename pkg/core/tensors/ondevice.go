package tensors

import (
	"reflect"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/pkg/errors"
)

// onDevice holds internal information about on-device storage of a Tensor.
type onDevice struct {
	t         *Tensor
	buffer    backends.Buffer
	deviceNum backends.DeviceNum
}

// FromBuffer creates a Tensor from a backend's buffer. It requires the deviceNum information as well.
// The ownership of the buffer is transferred to the new Tensor.
//
// This doesn't work for shared buffers, so it assumes the buffer is not shared.
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
	t.onDevice = &onDevice{
		t:         t,
		buffer:    buffer,
		deviceNum: deviceNum,
	}
	return
}

// Buffer returns the backend buffer for the tensor.
// It triggers the transfer from local to the backend device if the tensor is not already stored on the device.
//
// The deviceNum is optional. But only one can be given. The default value is 0.
//
// Careful not to finalize the tensor while the buffer is in use -- e.g.: during the execution that uses the buffer
// as input.
func (t *Tensor) Buffer(backend backends.Backend, deviceNum ...backends.DeviceNum) (backends.Buffer, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	err := t.CheckValid()
	if err != nil {
		return nil, err
	}
	if len(deviceNum) > 1 {
		return nil, errors.Errorf("Tensor.Buffer takes at most one deviceNum, %v given", deviceNum)
	}
	if len(deviceNum) == 0 {
		deviceNum = defaultDeviceNums
	}
	err = t.lockedMaterializeOnDevice(backend, true, deviceNum...)
	if err != nil {
		return nil, err
	}
	return t.onDevice.buffer, nil
}

// DonateBuffer returns the backend buffer for the tensor and transfers the ownership of the buffer to the caller.
// This may invalidate the tensor if there is no other on-device storage or local storage -- in particular this
// is true if using "shared buffers" (generally true for CPU-based plugins).
//
// Mostly used internally -- by graph.Graph.Run and graph.Exec when the value in the buffer is no longer needed
// after execution.
//
// It will panic if the buffer is shared (see Tensor.IsShared): shared buffers cannot be donated.
//
// It triggers the transfer from local to the backend device if the tensor is not already stored on the device.
//
// It doesn't finalize(release) the local tensor value.
func (t *Tensor) DonateBuffer(backend backends.Backend, deviceNum backends.DeviceNum) backends.Buffer {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.AssertValid()
	t.mustLockedMaterializeOnDevice(backend, false, deviceNum)
	buf := t.onDevice.buffer
	t.onDevice = nil
	if t.local == nil && t.onDevice == nil {
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

// MustFinalize releases the associated buffer in the PJRT client.
// It's the caller's responsibility to ensure this buffer is not being used elsewhere (like in the middle of an execution).
//
// It doesn't clear the pointer to this Device in the Tensor object.
//
// It panics on a backend error.
func (d *onDevice) MustFinalize() {
	err := d.Finalize()
	if err != nil {
		panic(err)
	}
}

// Finalize releases the associated buffer in the PJRT client.
// It's the caller's responsibility to ensure this buffer is not being used elsewhere (like in the middle of an execution).
//
// It doesn't clear the pointer to this Device in the Tensor object.
func (d *onDevice) Finalize() error {
	if d.IsFinalized() {
		return nil
	}
	if !d.t.backend.IsFinalized() {
		// We finalize only if the backend hasn't been finalized yet -- otherwise, we assume all buffers
		// have been freed/finalized/invalidated by the backend already.
		if err := d.t.backend.BufferFinalize(d.buffer); err != nil {
			return errors.WithMessagef(err, "Tensor.OnDevice.MustFinalize: failed to finalize buffer on-device")
		}
	}
	d.buffer = nil
	d.t = nil
	return nil
}

// MustMaterializeOnDevice will transfer a Tensor from local storage to the given device, if needed, or transfer from
// one device to another (freeing the source device copy).
//
// Generally the user doesn't need to call this function, it is called by the libraries executing GoMLX computations
// automatically when needed (e.g.: graph.Exec or context.Exec).
//
// If share is true, and if the backend allows for shared buffers, this will create a shared buffer,
// which can be more economic.
//
// - If an updated copy of the Tensor is already on the device, this is a no-op.
// - If the Tensor has already been used with a different client, this panics: one cannot mix clients on the same Tensor.
//
// It panics on a backend error.
func (t *Tensor) MustMaterializeOnDevice(backend backends.Backend, share bool, deviceNum backends.DeviceNum) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.AssertValid()
	t.mustLockedMaterializeOnDevice(backend, share, deviceNum)
}

// MaterializeOnDevice will transfer a Tensor from local storage to the given device, if needed, or transfer from
// one device to another (freeing the source device copy).
//
// Generally the user doesn't need to call this function, it is called by the libraries executing GoMLX computations
// automatically when needed (e.g.: graph.Exec or context.Exec).
//
// If share is true, and if the backend allows for shared buffers, this will create a shared buffer,
// which can be more economic.
//
// - If an updated copy of the Tensor is already on the device, this is a no-op.
// - If the Tensor has already been used with a different client, this panics: one cannot mix clients on the same Tensor.
func (t *Tensor) MaterializeOnDevice(backend backends.Backend, share bool, deviceNum backends.DeviceNum) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	err := t.CheckValid()
	if err != nil {
		return err
	}
	return t.lockedMaterializeOnDevice(backend, share, deviceNum)
}

// defaultDeviceNums is used whenever `deviceNums` is not provided.
var defaultDeviceNums = []backends.DeviceNum{0}

// mustLockedMaterializeOnDevice implements Tensor.MustMaterializeOnDevice
//
// If share is true, it will attempt to materialize to a shared buffer if available.
// In this case, it frees the local tensor storage and starts using the shared data instead.
func (t *Tensor) mustLockedMaterializeOnDevice(backend backends.Backend, share bool, deviceNum backends.DeviceNum) {
	err := t.lockedMaterializeOnDevice(backend, share, deviceNum)
	if err != nil {
		panic(err)
	}
}

func (t *Tensor) lockedMaterializeOnDevice(backend backends.Backend, share bool, deviceNum backends.DeviceNum) error {
	if t.backend == nil {
		t.backend = backend
	} else if t.backend != backend {
		return errors.Errorf(
			"while attempting to Tensor(shape=%s).MaterilizeOnDevice(backend=%s): cannot use the same Tensor "+
				"across different backend instances, even if they are the same type of backend; see more on the Tensor "+
				"documentation, but a couple of quick solutions: (a) call Tensor.ToLocal() to make sure the "+
				"tensor is moved to local storage and detached from the backend; (b) clone the tensors "+
				"and keep a tensor copy for each desired backend, with tensor.LocalClone() or "+
				"tensor.OnDeviceClone(newBackend)",
			t.shape, backend.Name())
	}
	if t.backend == nil || t.backend.IsFinalized() {
		return errors.New("cannot MustMaterializeOnDevice with a nil or finalized backend")
	}
	if !t.onDevice.IsFinalized() {
		// Tensor already on-device:
		if t.onDevice.deviceNum == deviceNum {
			// Nothing to do.
			return nil
		}
		// Attempt to transfer the buffer to the new device.
		newBuffer, err := backend.BufferCopyToDevice(t.onDevice.buffer, deviceNum)
		if err != nil {
			return errors.WithMessagef(err, "failed to transfer Tensor's buffer from device %d to device %d",
				t.onDevice.deviceNum, deviceNum)
		}
		err := t.onDevice.Finalize()
		if err != nil {
			return errors.WithMessagef(err, "failed to finalize Tensor's on-device buffer on device %d", t.onDevice.deviceNum)
		}
		t.onDevice = &onDevice{
			t:         t,
			buffer:    newBuffer,
			deviceNum: deviceNum,
		}
		return nil
	}

	// We need to materialize the onDevice from local:
	if t.local == nil {
		return errors.Errorf("Tensor(shape=%s) has invalid local and on-device data!?", t.shape)
	}

	var buffer backends.Buffer
	var err error
	if share && backend.HasSharedBuffers() {
		buffer, t.sharedFlat, err = backend.NewSharedBuffer(deviceNum, t.shape)
		if err != nil {
			return errors.WithMessagef(err, "Tensor.MustMaterializeOnDevice: failed to create a shared buffer")
		}
		reflect.Copy(reflect.ValueOf(t.sharedFlat), reflect.ValueOf(t.local.flat))
		t.local = nil // Free local storage.
		t.isShared = true
	} else {
		buffer, err = t.backend.BufferFromFlatData(deviceNum, t.local.flat, t.shape)
		if err != nil {
			return errors.WithMessagef(err, "Tensor.MustMaterializeOnDevice: failed to create a new buffer")
		}
	}
	t.onDevice = &onDevice{
		t:         t,
		buffer:    buffer,
		deviceNum: deviceNum,
	}
	return nil
}

// InvalidateOnDevice destroys all on-device copies of the Tensor, so the local copy becomes the source of truth.
// It does nothing if the tensor is shared -- see the Tensor.IsShared method.
//
// It's the caller's responsibility to ensure this buffer is not being used elsewhere (like in the middle of an execution).
//
// This is automatically called if the Tensor is mutated (e.g.: Tensor.MutableFlatData) or when the on-device value
// is donated to the execution of a graph.
//
// If there is no local copy of the Tensor, this will invalidate the whole tensor.
//
// Usually, this is called automatically. Mostly for internal use.
func (t *Tensor) InvalidateOnDevice() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.isShared {
		return nil
	}
	return t.lockedInvalidateOnDevice()
}

// lockedInvalidateOnDevice destroys all on-device copies of the Tensor.
// It does nothing if the tensor is shared.
//
// If there is no local copy of the Tensor, this will invalidate the tensor.
//
// Usually, this is called automatically. Mostly for internal use.
func (t *Tensor) lockedInvalidateOnDevice() error {
	if t.isShared {
		return nil
	}
	err := t.CheckValid()
	if err != nil {
		return err
	}
	if t.onDevice != nil {
		err = t.onDevice.Finalize()
		if err != nil {
			return errors.WithMessagef(err, "Tensor.InvalidateOnDevice: failed to finalize on-device buffer")
		}
		t.onDevice = nil
	}
	t.backend = nil
	return nil
}

// OnDeviceClone creates a clone of the tensor t that has backend storage.
// It also works to copy tensors to a different backend.
func (t *Tensor) OnDeviceClone(backend backends.Backend, deviceNums ...backends.DeviceNum) *Tensor {
	t.AssertValid()
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
	newT.onDevice = &onDevice{
		t:         newT,
		buffer:    buffer,
		deviceNum: deviceNum,
	}
	return newT
}

// Clone creates a clone of the Tensor value with shared backing with the backend -- if it supports --
// or it falls back to LocalClone.
//
// If you are trying to clone a tensor to use on a different backend, use OnDeviceClone or even LocalClone instead.
func (t *Tensor) Clone() *Tensor {
	t.AssertValid()
	if t.backend == nil || !t.backend.HasSharedBuffers() {
		return t.LocalClone()
	}
	return t.OnDeviceClone(t.backend)
}

// MaterializeLocal will make sure there is a local storage of the tensor.
// If there isn't already a local copy, this triggers a transfer from an on-device storage to a local copy.
//
// It's a "no-op" if using shared buffers with the accelerator. If you want to force a local copy,
// use LocalClone instead.
func (t *Tensor) MaterializeLocal() {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.isShared {
		if t.backend == nil || t.backend.IsFinalized() {
			exceptions.Panicf("attempting to access a Tensor(shape=%s) with shared buffer with a backend that has been finalized already.", t.shape)
		}
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

	// Get the on-device id.
	d := t.onDevice
	if d.IsFinalized() {
		exceptions.Panicf("Tensor(shape=%s).MaterializeLocal() failed because on-device tensor (deviceNum=%d) is invalid",
			t.shape, d.deviceNum)
	}

	// Create a flat slice.
	flatV := reflect.MakeSlice(reflect.SliceOf(t.shape.DType.GoType()), t.Size(), t.Size())
	t.local = &local{
		t:    t,
		flat: flatV.Interface(),
	}
	if err := t.backend.BufferToFlatData(d.buffer, t.local.flat); err != nil {
		panic(errors.WithMessagef(err, "Tensor(shape=%s).MaterializeLocal() failed to copy from on-device buffer", t.shape))
	}
}

// ToLocal forces the tensor to move its data to local (host CPU) storage and detaches itself
// from the backend.
// It returns itself to allow for cascading calls.
//
// This is useful if using tensors across multiple backends.
//
// If the tensor already has a local storage, there is no copy involved.
// Any on-device storage is freed.
func (t *Tensor) ToLocal() *Tensor {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.AssertValid()
	if !t.isShared {
		t.lockedMaterializeLocal()
		t.lockedInvalidateOnDevice()

	} else {
		// Copy shared data.
		flatV := reflect.MakeSlice(reflect.SliceOf(t.shape.DType.GoType()), t.Size(), t.Size())
		t.local = &local{
			t:    t,
			flat: flatV.Interface(),
		}
		reflect.Copy(reflect.ValueOf(t.local.flat), reflect.ValueOf(t.sharedFlat))
		t.isShared = false
		t.lockedInvalidateOnDevice()
	}
	return t
}

// CopyFrom will copy the contents from tFrom. The tensors t and tFrom must have the same shape.
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

	// Make sure the tensor t has a receiving local flat buffer.
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
	// Get the on-device version.
	d := tFrom.onDevice
	if d.IsFinalized() {
		exceptions.Panicf("Tensor(shape=%s).CopyFrom(tFrom) failed because tFrom on-device tensor (deviceNum=%d) is invalid",
			t.shape, d.deviceNum)
	}
	if err := tFrom.backend.BufferToFlatData(d.buffer, tFlat); err != nil {
		panic(errors.WithMessagef(err, "Tensor(shape=%s).CopyFrom(tFrom) failed to copy from on-device buffer", t.shape))
	}
}

// IsOnDevice checks whether the Tensor has an on-device copy on the given deviceNum.
//
// See MustMaterializeOnDevice to trigger a transfer/copy to the given device.
func (t *Tensor) IsOnDevice(deviceNum backends.DeviceNum) bool {
	t.AssertValid()
	return t.onDevice != nil && !t.onDevice.IsFinalized() && t.onDevice.deviceNum == deviceNum
}

// Backend returns the backend of the on-device copy of the Tensor, if any.
//
// It returns an error is the tensor is stored locally.
func (t *Tensor) Backend() (backends.Backend, error) {
	err := t.CheckValid()
	if err != nil {
		return nil, err
	}
	if t.onDevice.IsFinalized() {
		return nil, errors.Errorf("Tensor.Device() called on a local only tensor")
	}
	if t.backend == nil {
		return nil, errors.Errorf(
			"Tensor(shape=%s) has and invalid backend -- please report, this should never happen", t.shape)
	}
	return t.backend, nil
}

// Device returns the deviceNum of the on-device copy of the Tensor, if any.
//
// It returns an error is the tensor is stored locally.
func (t *Tensor) Device() (backends.DeviceNum, error) {
	err := t.CheckValid()
	if err != nil {
		return 0, err
	}
	if t.onDevice.IsFinalized() {
		return 0, errors.Errorf("Tensor.Device() called on a local only tensor")
	}
	return t.onDevice.deviceNum, nil
}
