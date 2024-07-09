package tensor

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/pkg/errors"
	"maps"
)

// device holds internal information about on-device storage of a Tensor.
type onDevice struct {
	t         *Tensor
	buffer    *pjrt.Buffer
	deviceNum int
}

// FromPJRT creates a Tensor from a PJRT buffer. It requires the deviceNum information as well.
func FromPJRT(buffer *pjrt.Buffer) (t *Tensor) {
	dtype, err := buffer.DType()
	if err != nil {
		panic(errors.WithMessage(err, "failed to read DType from PJRT buffer"))
	}
	dims, err := buffer.Dimensions()
	if err != nil {
		panic(errors.WithMessage(err, "failed to read dimensions from PJRT buffer"))
	}
	device, err := buffer.Device()
	if err != nil {
		panic(errors.WithMessage(err, "failed to read device from PJRT buffer"))
	}
	deviceNum := buffer.Client().NumForDevice(device)
	if deviceNum == -1 {
		exceptions.Panicf("cannot find deviceNum for device used by PJRT buffer!?")
	}

	// Create tensor.
	t = newTensor(shapes.Make(dtype, dims...))
	t.onDevices[deviceNum] = &onDevice{
		t:         t,
		buffer:    buffer,
		deviceNum: deviceNum,
	}
	return
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
	err := d.buffer.Destroy()
	if err != nil {
		panic(errors.WithMessagef(err, "while finalizing Tensor (shape=%s) on-device buffer", d.t.shape))
	}
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
func (t *Tensor) MaterializeOnDevices(client *pjrt.Client, deviceNums ...int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.AssertValid()
	t.lockedMaterializeOnDevices(client, deviceNums...)
}

// defaultDeviceNums is used whenever `deviceNums` is not provided.
var defaultDeviceNums = []int{0}

// lockedMaterializeOnDevices implements Tensor.MaterializeOnDevices
func (t *Tensor) lockedMaterializeOnDevices(client *pjrt.Client, deviceNums ...int) {
	if t.client == nil {
		t.client = client
	} else if t.client != client {
		exceptions.Panicf("Tensor(shape=%s).MaterilizeOnDevices: cannot have a Tensor be stored by different PJRT clients, use separate Tensors for this", t.shape)
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
		buffer, err := client.BufferFromHost().
			FromFlatDataWithDimensions(t.local.flat, t.shape.Dimensions).
			ToDeviceNum(deviceNum).
			Done()
		if err != nil {
			panic(errors.WithMessagef(err, "Tensor(shape=%s).MaterializeOnDevices for deviceNum=%d failed", t.shape, deviceNum))
		}
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
	maps.DeleteFunc(t.onDevices, func(_ int, _ *onDevice) bool {
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

	// Get on-device version: try default (deviceNum==0) first.
	deviceNum := 0
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
	flat, dims, err := d.buffer.ToFlatDataAndDimensions()
	if err != nil {
		panic(errors.WithMessagef(err, "Tensor(shape=%s).MaterializeLocal() failed to transfer on-device tensor (deviceNum=%d) to host",
			t.shape, deviceNum))
	}
	t.shape.AssertDims(dims...)
	t.local = &local{
		t:    t,
		flat: flat,
	}
}
