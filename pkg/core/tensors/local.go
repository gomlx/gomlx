package tensors

import (
	"encoding/gob"
	"fmt"
	"os"
	"reflect"
	"strconv"
	"unsafe"

	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"k8s.io/klog/v2"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
)

// local storage for a Tensor.
type local struct {
	// t is the container tensor owner of this Local.
	// t holds the shape of the tensor.
	t *Tensor

	// flat holds the array with actual data. It's owned by Local.
	flat any // Slice of the type for the dtype of the given shape.
}

// FromShape returns a Tensor with the given shape, with the data initialized with zeros.
//
// It panics if you provide an invalid shape.
func FromShape(shape shapes.Shape) *Tensor {
	if !shape.Ok() {
		panic(errors.New("invalid shape"))
	}
	t := newEmptyTensor(shape)
	flatV := reflect.MakeSlice(reflect.SliceOf(t.shape.DType.GoType()), t.Size(), t.Size())
	t.local = &local{
		t:    t,
		flat: flatV.Interface(),
	}
	return t
}

// LocalClone creates a clone of the Tensor value with local backing.
// It will trigger a transfer from on-device data to local if the value is not present in local memory yet.
func (t *Tensor) LocalClone() (*Tensor, error) {
	var clone *Tensor
	err := t.ConstFlatData(func(flat any) {
		clone = newEmptyTensor(t.shape)
		flatV := reflect.ValueOf(flat)
		size := flatV.Len()
		cloneFlatV := reflect.MakeSlice(flatV.Type(), size, size)
		reflect.Copy(cloneFlatV, flatV)
		clone.local = &local{
			t:    clone,
			flat: cloneFlatV.Interface(),
		}
	})
	if err != nil {
		return nil, err
	}
	return clone, nil
}

// IsFinalized returns true if the tensor has already been "finalized", and its
// data freed.
// It implements Tensor.IsFinalized.
func (l *local) IsFinalized() bool {
	return l == nil || l.flat == nil
}

// Finalize releases the memory associated with the local tensor.
func (l *local) Finalize() {
	if l == nil || l.flat == nil {
		return
	}
	l.flat = nil
	l.t = nil
}

// HasLocal returns whether there is an up-to-date copy of the Tensor on local storage.
// If false, any access to the data (e.g.: Tensor.ConstFlatData) will require a transfer (Tensor.MaterializeToLocal).
func (t *Tensor) HasLocal() bool {
	return !t.local.IsFinalized()
}

// MustConstFlatData calls accessFn with the flattened data as a slice of the Go type corresponding to the DType type.
// Even scalar values have a flattened data representation of one element.
// It locks the Tensor until accessFn returns.
//
// It triggers a synchronous transfer from device to local if the tensor is only on-device.
//
// This provides accessFn with the actual Tensor data (not a copy), and it's owned by the Tensor, but it should not be
// changed -- the contents of the corresponding "on device" tensors would go out-of-sync.
// See Tensor.MutableFlatData to access a mutable version of the flat data.
//
// See Tensor.Size for the number of elements, and Tensor.LayoutStrides to calculate the offset of individual
// positions, given the indices at each axis.
//
// Even scalar values have a flattened data representation of one element.
//
// It panics if the tensor is in an invalid state (if it was finalized), or if it is a tuple.
func (t *Tensor) MustConstFlatData(accessFn func(flat any)) {
	t.mu.Lock()
	defer t.mu.Unlock()
	must(t.lockedConstFlatData(accessFn))
}

// ConstFlatData calls accessFn with the flattened data as a slice of the Go type corresponding to the DType type.
// Even scalar values have a flattened data representation of one element.
// It locks the Tensor until accessFn returns.
//
// It triggers a synchronous transfer from device to local if the tensor is only on-device.
//
// This provides accessFn with the actual Tensor data (not a copy), and it's owned by the Tensor, but it should not be
// changed -- the contents of the corresponding "on device" tensors would go out-of-sync.
// See Tensor.MutableFlatData to access a mutable version of the flat data.
//
// See Tensor.Size for the number of elements, and Tensor.LayoutStrides to calculate the offset of individual
// positions, given the indices at each axis.
//
// Even scalar values have a flattened data representation of one element.
func (t *Tensor) ConstFlatData(accessFn func(flat any)) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.lockedConstFlatData(accessFn)
}

// lockedConstFlatData implements Tensor.ConstFlatData.
func (t *Tensor) lockedConstFlatData(accessFn func(flat any)) error {
	if err := t.CheckValid(); err != nil {
		return err
	}
	if t.isShared {
		// Directly access the share data.
		accessFn(t.sharedFlat)
		return nil
	}
	if t.local == nil && t.backend.HasSharedBuffers() {
		// If local is nil, that means there is an on-device tensor instead;
		// we take a view (the data) of the first one.
		flat, err := t.backend.BufferData(t.onDevice.buffer)
		if err != nil {
			return err
		}
		accessFn(flat)
		return nil
	}
	err := t.lockedMaterializeLocal()
	if err != nil {
		return err
	}
	accessFn(t.local.flat)
	return nil
}

// ConstFlatData calls accessFn with the flattened data as a slice of the Go type corresponding to the DType type.
// Even scalar values have a flattened data representation of one element.
// It locks the Tensor until accessFn returns.
//
// It is the "generics" version of Tensor.ConstFlatData(),
//
// This provides accessFn with the actual Tensor data (not a copy), and it's owned by the Tensor, but it should not be
// changed -- the contents of the corresponding "on device" tensors could go out-of-sync.
// See Tensor.MutableFlatData to access a mutable version of the flat data.
//
// See Tensor.Size for the number of elements, and Tensor.LayoutStrides to calculate the offset of individual
// positions, given the indices at each axis.
func ConstFlatData[T dtypes.Supported](t *Tensor, accessFn func(flat []T)) error {
	if t.shape.DType != dtypes.FromGenericsType[T]() {
		var v T
		return errors.Errorf("MustConstFlatData[%T] is incompatible with Tensor's dtype %s -- expected dtype %s",
			v, t.shape.DType, dtypes.FromGenericsType[T]())
	}
	return t.ConstFlatData(func(anyFlat any) {
		flat := anyFlat.([]T)
		accessFn(flat)
	})
}

// MustConstFlatData calls accessFn with the flattened data as a slice of the Go type corresponding to the DType type.
// Even scalar values have a flattened data representation of one element.
// It locks the Tensor until accessFn returns.
//
// It is the "generics" version of Tensor.ConstFlatData(),
//
// This provides accessFn with the actual Tensor data (not a copy), and it's owned by the Tensor, but it should not be
// changed -- the contents of the corresponding "on device" tensors could go out-of-sync.
// See Tensor.MutableFlatData to access a mutable version of the flat data.
//
// See Tensor.Size for the number of elements, and Tensor.LayoutStrides to calculate the offset of individual
// positions, given the indices at each axis.
//
// It panics if the tensor is in an invalid state (if it was finalized), or if it is a tuple.
func MustConstFlatData[T dtypes.Supported](t *Tensor, accessFn func(flat []T)) {
	if t.shape.DType != dtypes.FromGenericsType[T]() {
		var v T
		exceptions.Panicf("MustConstFlatData[%T] is incompatible with Tensor's dtype %s -- expected dtype %s",
			v, t.shape.DType, dtypes.FromGenericsType[T]())
	}
	t.MustConstFlatData(func(anyFlat any) {
		flat := anyFlat.([]T)
		accessFn(flat)
	})
}

// ConstBytes calls accessFn with the data as a bytes slice.
// Even scalar values have a bytes data representation of one element.
// It locks the Tensor until accessFn returns.
//
// This provides accessFn with the actual Tensor data (not a copy), and it's owned by the Tensor, and it should not be
// changed -- the contents of the corresponding "on device" tensors would go out-of-sync.
// See Tensor.MutableBytes to access a mutable version of the data as bytes.
//
// It panics if the tensor is in an invalid state (if it was finalized), or if it is a tuple.
func (t *Tensor) ConstBytes(accessFn func(data []byte)) error {
	return t.ConstFlatData(func(flat any) {
		flatV := reflect.ValueOf(flat)
		element0 := flatV.Index(0)
		flatValuesPtr := element0.Addr().UnsafePointer()
		sizeBytes := uintptr(flatV.Len()) * element0.Type().Size()
		data := unsafe.Slice((*byte)(flatValuesPtr), sizeBytes)
		accessFn(data)
	})
}

// MustMutableFlatData calls accessFn with a flat slice pointing to the Tensor data.
// The type of the slice corresponds to the DType of the tensor.
// The contents of the slice itself can be changed until accessFn returns.
// During this time the Tensor is locked.
//
// If the data is not shared with the backend (usually, only available for CPU), it invalidates and frees any copy
// of the data on-device (e.g., GPU). It also triggers a synchronous transfer from device to local if the tensor is
// only on-device and not shared.
//
// Even scalar values have a flattened data representation of one element.
//
// This returns the actual Tensor data (not a copy), and the slice is owned by the Tensor -- but its contents can
// be changed while inside accessFn.
//
// See Tensor.ConstFlatData to access a mutable version of the flat data.
//
// See Tensor.Size for the number of elements, and Tensor.LayoutStrides to calculate the offset of individual positions,
// given the indices at each axis.
//
// It panics if the tensor is in an invalid state (if it was finalized), or if it is a tuple.
func (t *Tensor) MustMutableFlatData(accessFn func(flat any)) {
	must(t.MutableFlatData(accessFn))
}

// MutableFlatData calls accessFn with a flat slice pointing to the Tensor data.
// The type of the slice corresponds to the DType of the tensor.
// The contents of the slice itself can be changed until accessFn returns.
// During this time the Tensor is locked.
//
// If the data is not shared with the backend (usually, only available for CPU), it invalidates and frees any copy
// of the data on-device (e.g., GPU). It also triggers a synchronous transfer from device to local if the tensor is
// only on-device and not shared.
//
// Even scalar values have a flattened data representation of one element.
//
// This returns the actual Tensor data (not a copy), and the slice is owned by the Tensor -- but its contents can
// be changed while inside accessFn.
//
// See Tensor.ConstFlatData to access a mutable version of the flat data.
//
// See Tensor.Size for the number of elements, and Tensor.LayoutStrides to calculate the offset of individual positions,
// given the indices at each axis.
//
// It returns an error if the tensor is in an invalid state (if it was finalized), or if it is a tuple.
func (t *Tensor) MutableFlatData(accessFn func(flat any)) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if err := t.CheckValid(); err != nil {
		return err
	}
	if t.isShared {
		accessFn(t.sharedFlat)
		return nil
	}
	err := t.lockedMaterializeLocal()
	if err != nil {
		return err
	}
	err = t.lockedInvalidateOnDevice()
	if err != nil {
		return err
	}
	accessFn(t.local.flat)
	return nil
}

// MutableBytes gives mutable access to the local storage of the values for the tensor.
// It's similar to MutableFlatData but provides a bytes view to the same data.
//
// If the tensor is not shared, it triggers a synchronous transfer from device to local if the tensor is only on-device,
// and it invalidates the device storage, since it's assumed they will be out-of-date.
//
// This returns the actual Tensor data (not a copy), and the bytes slice is owned by the Tensor -- but its contents can
// be changed while inside accessFn.
//
// See Tensor.ConstBytes for constant access to the data as bytes -- that doesn't invalidate the device storage.
func (t *Tensor) MutableBytes(accessFn func(data []byte)) error {
	return t.MutableFlatData(func(flat any) {
		flatV := reflect.ValueOf(flat)
		element0 := flatV.Index(0)
		flatValuesPtr := element0.Addr().UnsafePointer()
		sizeBytes := uintptr(flatV.Len()) * element0.Type().Size()
		data := unsafe.Slice((*byte)(flatValuesPtr), sizeBytes)
		accessFn(data)
	})
}

// MustMutableFlatData calls accessFn with a flat slice pointing to the Tensor data.
// The type of the slice corresponds to the DType of the tensor.
// The contents of the slice itself can be changed until accessFn returns.
// During this time the Tensor is locked.
//
// It is the "generics" version of Tensor.MutableFlatData(), see its description for more details.
//
// This returns the actual Tensor data (not a copy), and the data owned by the Tensor, and should only be changed
// inside accessFn.
//
// It panics if the tensor is in an invalid state (if it was finalized), or if it is a tuple.
func MustMutableFlatData[T dtypes.Supported](t *Tensor, accessFn func(flat []T)) {
	must(MutableFlatData(t, accessFn))
}

// MutableFlatData calls accessFn with a flat slice pointing to the Tensor data.
// The type of the slice corresponds to the DType of the tensor.
// The contents of the slice itself can be changed until accessFn returns.
// During this time the Tensor is locked.
//
// It is the "generics" version of Tensor.MutableFlatData(), see its description for more details.
//
// This returns the actual Tensor data (not a copy), and the data owned by the Tensor, and should only be changed
// inside accessFn.
func MutableFlatData[T dtypes.Supported](t *Tensor, accessFn func(flat []T)) error {
	if t.shape.DType != dtypes.FromGenericsType[T]() {
		var v T
		return errors.Errorf("MustMutableFlatData[%T] is incompatible with Tensor's dtype %s",
			v, t.shape.DType)
	}
	var conversionErr error
	accessErr := t.MutableFlatData(func(anyFlat any) {
		flat, ok := anyFlat.([]T)
		if !ok {
			var v T
			conversionErr = errors.Errorf(
				"MustMutableFlatData[%T] failed to cast internal data type (dtype=%s) to %T",
				v, t.shape.DType, v)
			return
		}
		accessFn(flat)
	})
	if accessErr != nil {
		return accessErr
	}
	return conversionErr
}

// AssignFlatData will copy over the values in fromFlat to the storage used by toTensor.
// If the dtypes are not compatible or if the size is wrong, it will panic.
func AssignFlatData[T dtypes.Supported](toTensor *Tensor, fromFlat []T) error {
	var lenErr error
	accessErr := MutableFlatData(toTensor, func(toFlat []T) {
		if len(toFlat) != len(fromFlat) {
			var v T
			lenErr = errors.Errorf(
				"AssignFlatData[%T] is trying to store %d values into shape %s, which requires %d values",
				v,
				len(fromFlat),
				toTensor.Shape(),
				toTensor.Shape().Size(),
			)
			return
		}
		copy(toFlat, fromFlat)
	})
	if accessErr != nil {
		return accessErr
	}
	return lenErr
}

// ToScalar returns the scalar value of the Tensor.
//
// It triggers a synchronous transfer from device to local if the tensor is only on-device.
//
// It will panic if the given generic type doesn't match the DType of the tensor.
func ToScalar[T dtypes.Supported](t *Tensor) T {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.AssertValid()
	t.mustLockedMaterializeLocal()
	if t.shape.DType != dtypes.FromGenericsType[T]() {
		var v T
		exceptions.Panicf("ToScalar[%T] is incompatible with Tensor's dtype %s",
			v, t.shape.DType)
	}
	if !t.shape.IsScalar() {
		var v T
		exceptions.Panicf("ToScalar[%T] requires scalar Tensor, got shape %s instead", v, t.shape)
	}
	return t.local.flat.([]T)[0]
}

// MustCopyFlatData returns a copy of the flat data of the Tensor.
//
// It triggers a synchronous transfer from device to local if the tensor is only on-device.
//
// It will panic if the given generic type doesn't match the DType of the tensor.
func MustCopyFlatData[T dtypes.Supported](t *Tensor) []T {
	var flatCopy []T
	MustConstFlatData(t, func(flat []T) {
		flatCopy = xslices.Copy(flat)
	})
	return flatCopy
}

// MultiDimensionSlice lists the Go types a Tensor can be converted to/from. There are no recursions in
// generics' constraint definitions, so we list up to 7 levels of slices. Feel free to add
// more if needed, the implementation will work with any arbitrary number.
//
// Generated by `github.com/gomlx/gomlx/cmd/constraints_generator`.
type MultiDimensionSlice interface {
	bool | float32 | float64 | int | int32 | int64 | uint8 | uint32 | uint64 | complex64 | complex128 |
		[]bool | []float32 | []float64 | []int | []int32 | []int64 | []uint8 | []uint32 | []uint64 | []complex64 | []complex128 |
		[][]bool | [][]float32 | [][]float64 | [][]int | [][]int32 | [][]int64 | [][]uint8 | [][]uint32 | [][]uint64 | [][]complex64 | [][]complex128 |
		[][][]bool | [][][]float32 | [][][]float64 | [][][]int | [][][]int32 | [][][]int64 | [][][]uint8 | [][][]uint32 | [][][]uint64 | [][][]complex64 | [][][]complex128 |
		[][][][]bool | [][][][]float32 | [][][][]float64 | [][][][]int | [][][][]int32 | [][][][]int64 | [][][][]uint8 | [][][][]uint32 | [][][][]uint64 | [][][][]complex64 | [][][][]complex128 |
		[][][][][]bool | [][][][][]float32 | [][][][][]float64 | [][][][][]int | [][][][][]int32 | [][][][][]int64 | [][][][][]uint8 | [][][][][]uint32 | [][][][][]uint64 | [][][][][]complex64 | [][][][][]complex128 |
		[][][][][][]bool | [][][][][][]float32 | [][][][][][]float64 | [][][][][][]int | [][][][][][]int32 | [][][][][][]int64 | [][][][][][]uint8 | [][][][][][]uint32 | [][][][][][]uint64 | [][][][][][]complex64 | [][][][][][]complex128
}

// LayoutStrides return the strides for each axis. This can be handy when manipulating the flat data.
func (t *Tensor) LayoutStrides() (strides []int) {
	return t.shape.Strides()
}

// Value returns a multidimensional slice (except if the shape is a scalar) containing a copy of the values stored
// in the tensor.
// This is expensive and usually only used for smaller tensors in tests and to print results.
//
// If the local tensor is empty, it panics with the corresponding error.
func (t *Tensor) Value() any {
	v, err := t.ValueSafe()
	must(err)
	return v
}

// ValueSafe returns a multidimensional slice (except if the shape is a scalar) containing a copy of the values stored
// in the tensor.
// This is expensive and usually only used for smaller tensors in tests and to print results.
func (t *Tensor) ValueSafe() (any, error) {
	var mdSlice any
	err := t.ConstFlatData(func(flat any) {
		if t.shape.IsScalar() {
			// Avoid creating yet another slice:
			srcV := reflect.ValueOf(flat)
			mdSlice = srcV.Index(0).Interface()
			return
		}

		// Create a copy of the flat slice with all data.
		flatCopyV := reflect.MakeSlice(reflect.SliceOf(t.shape.DType.GoType()), t.Size(), t.Size())
		reflect.Copy(flatCopyV, reflect.ValueOf(flat))
		if t.shape.Rank() == 1 {
			mdSlice = flatCopyV.Interface()
			return
		}

		// If multi-dimensional slice, returns slice pointing to the flatCopy.
		mdSlice = convertDataToSlices(flatCopyV, t.shape.Dimensions...).Interface()
	})
	if err != nil {
		return nil, err
	}
	return mdSlice, nil
}

// GobSerialize Tensor in binary format.
//
// It triggers a synchronous transfer from device to local if the tensor is only on-device.
//
// It returns an error for I/O errors.
// It panics for invalid tensors.
func (t *Tensor) GobSerialize(encoder *gob.Encoder) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if err := t.CheckValid(); err != nil {
		return err
	}
	err := t.shape.GobSerialize(encoder)
	if err != nil {
		return err
	}
	accessErr := t.lockedConstFlatData(func(flat any) {
		err = encoder.Encode(flat)
		if err != nil {
			err = errors.Wrapf(err, "failed to write tensor.Local data")
		}
	})
	if accessErr != nil {
		return accessErr
	}
	return err
}

// GobDeserialize a Tensor from the reader.
//
// If the tensor is only going to be directly consumed by the execution of a graph (an ML model),
// use GoDeserializeOnDevice instead; it works faster for some backends.
func GobDeserialize(decoder *gob.Decoder) (*Tensor, error) {
	shape, err := shapes.GobDeserialize(decoder)
	if err != nil {
		err = errors.Wrapf(err, "failed to deserialize Tensor shape data")
		return nil, err
	}
	flatPtrV := reflect.New(reflect.SliceOf(shape.DType.GoType()))
	err = decoder.Decode(flatPtrV.Interface())
	if err != nil {
		err = errors.Wrapf(err, "failed to deserialize Tensor data")
		return nil, err
	}
	// Build the new tensor from scratch, using the data returned by the decoder (to avoid a copy).
	t := newEmptyTensor(shape)
	t.local = &local{
		t:    t,
		flat: flatPtrV.Elem().Interface(),
	}
	return t, nil
}

// GobDeserializeToDevice deserialize a Tensor from the reader directly to on-device memory.
// If the tensor is expected to be consumed by graph execution, this may work faster by
// avoiding an unnecessary copy.
//
// Returns new Tensor (with shared or onDevice storage or an error).
func GobDeserializeToDevice(
	decoder *gob.Decoder,
	backend backends.Backend,
	deviceNum backends.DeviceNum,
) (*Tensor, error) {
	if !backend.HasSharedBuffers() {
		// Load locally, and then materialize on-device.
		t, err := GobDeserialize(decoder)
		if err != nil {
			return nil, err
		}
		err = t.MaterializeOnDevice(backend, false, deviceNum)
		if err != nil {
			return nil, err
		}
		t.FinalizeLocal() // Free local memory.
		return t, nil
	}

	// Deserialize shape first.
	shape, err := shapes.GobDeserialize(decoder)
	if err != nil {
		err = errors.Wrapf(err, "failed to deserialize Tensor shape data")
		return nil, err
	}

	// Create a shared buffer.
	var buffer backends.Buffer
	var flatAny any
	buffer, flatAny, err = backend.NewSharedBuffer(deviceNum, shape)
	if err != nil {
		return nil, err
	}

	// Deserialize tensor contents.
	flatV := reflect.ValueOf(flatAny)
	// Since flatV.Addr() doesn't work,  we create a pointer to a new Slice and assign flatAny to it.
	flatPtrV := reflect.New(flatV.Type())
	flatPtrV.Elem().Set(flatV)
	err = decoder.Decode(flatPtrV.Interface())
	if err != nil {
		err = errors.Wrapf(err, "failed to deserialize Tensor data")
		// Destroy the buffer since it's not going to be used.
		err2 := backend.BufferFinalize(buffer)
		if err2 != nil {
			klog.Warningf("failed to destroy buffer for backend %q: %v", backend.Name(), err2)
		}
		return nil, err
	}

	// Build tensor with the deserialized buffer.
	t := newEmptyTensor(shape)
	t.isShared = true
	t.sharedFlat = flatAny
	t.sharedDevice = deviceNum
	t.backend = backend
	t.onDevice = &onDevice{
		t:         t,
		buffer:    buffer,
		deviceNum: deviceNum,
	}
	return t, nil
}

// Save the Local tensor to the given file path.
func (t *Tensor) Save(filePath string) error {
	if err := t.CheckValid(); err != nil {
		return err
	}
	var f *os.File
	f, err := os.Create(filePath)
	if err != nil {
		return errors.Wrapf(err, "creating %q to save tensor", filePath)
	}
	enc := gob.NewEncoder(f)
	err = t.GobSerialize(enc)
	if err != nil {
		return errors.WithMessagef(err, "saving Tensor to %q", filePath)
	}
	err = f.Close()
	if err != nil {
		return errors.Wrapf(err, "close file %q, where tensor was saved", filePath)
	}
	return nil
}

// Load a Local tensor from the file path given.
func Load(filePath string) (*Tensor, error) {
	f, err := os.Open(filePath)
	if os.IsNotExist(err) {
		return nil, errors.Wrapf(err, "failed to open tensor file %q", filePath)
	}
	if err != nil {
		return nil, errors.Wrapf(err, "opening %q to load Tensor", filePath)
	}
	dec := gob.NewDecoder(f)
	t, err := GobDeserialize(dec)
	if err != nil {
		return nil, errors.WithMessagef(err, "loading Tensor from %q", filePath)
	}
	_ = f.Close()
	return t, nil
}

// TensorStringDefaultPrecision used by Tensor.String.
const TensorStringDefaultPrecision = 4

// String converts to string, if not too large. It uses t.Summary(precision=4).
func (t *Tensor) String() string {
	return t.Summary(TensorStringDefaultPrecision)
}

// FromScalar creates a local tensor with the given scalar.
// The `DType` is inferred from the value.
func FromScalar[T dtypes.Supported](value T) (t *Tensor) {
	return FromScalarAndDimensions(value)
}

// FromScalarAndDimensions creates a local tensor with the given dimensions, filled with the
// given scalar value replicated everywhere.
// The `DType` is inferred from the value.
func FromScalarAndDimensions[T dtypes.Supported](value T, dimensions ...int) *Tensor {
	dtype := dtypes.FromGenericsType[T]()
	shape := shapes.Make(dtype, dimensions...)
	t := FromShape(shape)
	MustMutableFlatData(t, func(flat []T) {
		xslices.FillSlice(flat, value)
	})
	return t
}

// FromFlatDataAndDimensions creates a tensor with the given dimensions, filled with the flattened values given in `data`.
// The data is copied to the Tensor.
// The `DType` is inferred from the `data` type.
//
// It panics if the size of data is wrong for the shape.
func FromFlatDataAndDimensions[T dtypes.Supported](data []T, dimensions ...int) *Tensor {
	dtype := dtypes.FromGenericsType[T]()
	shape := shapes.Make(dtype, dimensions...)
	if len(data) != shape.Size() {
		exceptions.Panicf(
			"FromFlatDataAndDimensions(%s): data size is %d, but dimensions size is %d",
			shape,
			len(data),
			shape.Size(),
		)
	}
	t := FromShape(shape)
	var dummy T
	switch any(dummy).(type) {
	case int:
		// The underlying tensor data could be int32 or int64 depending on the type int for the platform.
		// In this case we just copy the bytes.
		err := t.MutableBytes(func(tensorData []byte) {
			dataAsBytes := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(data))), uintptr(len(data))*unsafe.Sizeof(dummy))
			if len(dataAsBytes) != len(tensorData) {
				exceptions.Panicf(
					"failed to convert FromFlatDataAndDimensions for type int: data has %d bytes (%d elements), "+
						"but corresponding tensor will have %d bytes -- pls report, this shouldn't happen",
					len(dataAsBytes), len(data), len(tensorData))
			}
			copy(tensorData, dataAsBytes)
		})
		if err != nil {
			panic(err)
		}
	default:
		MustMutableFlatData(t, func(flat []T) {
			copy(flat, data)
		})
	}
	return t
}

// FromValue returns a `Local` tensor constructed from the given multi-dimension slice (or scalar).
// If the rank of the `value` is larger than 1, the shape of all sub-slices must be the same.
//
// It panics if the shape is not regular.
//
// Notice that FromFlatDataAndDimensions is much faster if speed here is a concern.
func FromValue[S MultiDimensionSlice](value S) *Tensor {
	return FromAnyValue(value)
}

// FromAnyValue is a non-generic version of FromValue that returns a *tensors.Tensor (not specified if local or on device).
// The input is expected to be either a scalar or a slice of slices with homogeneous dimensions.
// If the input is a tensor already (Local or Device), it is simply returned.
// If value is anything but a Device tensor, it will return a Local tensor.
//
// It panics with an error if the value type is unsupported or the shape is not regular.
func FromAnyValue(value any) *Tensor {
	if valueT, ok := value.(*Tensor); ok {
		// Input is already a Tensor.
		return valueT
	}
	shape, err := shapes.FromAnyValue(value)
	if err != nil {
		panic(errors.Wrapf(err, "cannot create shape from %T", value))
	}
	t := FromShape(shape)
	t.MustMutableFlatData(func(flatAny any) {
		if baseType(reflect.TypeOf(value)) == reflect.TypeOf(int(0)) {
			// Go `int` type can be either an int32 or int64 depending on the architecture (anything else would panic
			// already). For the copy operation to work, we have to cast flatRefAny (either a []int64 or []int32) as an []int.
			// This is not pretty (using unsafe), but it avoids individually converting values, which is important for large tensors.
			if strconv.IntSize == 64 { //nolint:mnd // Standard value.
				flatRef := flatAny.([]int64)
				flatAny = unsafe.Slice((*int)(unsafe.Pointer(unsafe.SliceData(flatRef))), len(flatRef))
			} else if strconv.IntSize == 32 { //nolint:mnd // Standard value.
				flatRef := flatAny.([]int32)
				flatAny = unsafe.Slice((*int)(unsafe.Pointer(unsafe.SliceData(flatRef))), len(flatRef))
			} else {
				exceptions.Panicf("cannot use `int` of %d bits with GoMLX -- try using int32 or int64", strconv.IntSize)
			}
		}
		flatV := reflect.ValueOf(flatAny)
		if shape.IsScalar() {
			elem := flatV.Index(0)
			elem.Set(reflect.ValueOf(value))
			return
		}
		// Copy over multi-dimensional slice recursively.
		copySlicesRecursively(flatV, reflect.ValueOf(value), t.LayoutStrides())
	})
	return t
}

// copySlicesRecursively copy values on a multi-dimension slice to a flat data slice
// assuming the strides for each dimension.
func copySlicesRecursively(data reflect.Value, mdSlice reflect.Value, strides []int) {
	if len(strides) == 1 {
		// Last level of slice, just copy over the slice.
		reflect.Copy(data, mdSlice)
		return
	}

	numElements := mdSlice.Len()
	subStrides := strides[1:]
	for ii := 0; ii < numElements; ii++ {
		start := ii * strides[0]
		end := (ii + 1) * strides[0]
		subData := data.Slice(start, end)
		copySlicesRecursively(subData, mdSlice.Index(ii), subStrides)
	}
}

// convertDataToSlices takes data as a flat slice and creates a multidimensional slice with the given dimensions that
// points to the given data.
func convertDataToSlices(dataV reflect.Value, dimensions ...int) reflect.Value {
	if len(dimensions) <= 1 {
		return dataV
	}
	resultT := dataV.Type().Elem()
	for range dimensions {
		resultT = reflect.SliceOf(resultT)
	}
	strides := make([]int, len(dimensions))
	currentStride := 1
	for dim := len(dimensions) - 1; dim >= 0; dim-- {
		strides[dim] = currentStride
		currentStride *= dimensions[dim]
	}
	return createSlicesRecursively(resultT, dataV, dimensions, strides)
}

// createSlicesRecursively recursively creates slices copy values on a multi-dimension slice to a flat data slice
// assuming the strides for each dimension.
func createSlicesRecursively(resultT reflect.Type, data reflect.Value, dimensions []int, strides []int) reflect.Value {
	if len(strides) == 1 {
		// Last level of slice, just copy over the slice (not the data, just the slice).
		return data
	}

	numElements := dimensions[0]
	slice := reflect.MakeSlice(resultT, numElements, numElements)

	subStrides := strides[1:]
	subDimensions := dimensions[1:]
	subResultT := resultT.Elem()
	for ii := 0; ii < numElements; ii++ {
		start := ii * strides[0]
		end := (ii + 1) * strides[0]
		subData := data.Slice(start, end)
		subSlice := createSlicesRecursively(subResultT, subData, subDimensions, subStrides)
		slice.Index(ii).Set(subSlice)
	}
	return slice
}

func shapeForValue(v any) (shapes.Shape, error) {
	var shape shapes.Shape
	err := shapeForValueRecursive(&shape, reflect.ValueOf(v), reflect.TypeOf(v))
	return shape, err
}

func shapeForValueRecursive(shape *shapes.Shape, v reflect.Value, t reflect.Type) error {
	switch t.Kind() {
	case reflect.Slice:
		// Recurse into inner slices.
		t = t.Elem()
		shape.Dimensions = append(shape.Dimensions, v.Len())
		shapePrefix := shape.Clone()

		// The first element is the reference
		if v.Len() == 0 {
			exceptions.Panicf(
				"value with empty slice not valid for Tensor conversion: %T: %v -- notice it's impossible to represent tensors with zero-dimensions generically using Go slices - try shapes.Make maybe ?",
				v.Interface(),
				v,
			)
		}
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
			if !shape.Equal(shapeTest) {
				return fmt.Errorf("sub-slices have irregular shapes, found shapes %q, and %q", shape, shapeTest)
			}
		}

	case reflect.Pointer:
		return fmt.Errorf("cannot convert Pointer (%s) to a concrete value for tensors", t)

	default:
		shape.DType = dtypes.FromGoType(t)
		if shape.DType == dtypes.InvalidDType {
			return fmt.Errorf(
				"cannot convert type %s to a value concrete tensor type (maybe type not supported yet?)",
				t,
			)
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

// Equal checks weather t == otherTensor.
// If they are the same pointer, they are considered equal.
// If the shapes are different, it returns false.
// If either side is invalid (nil), it panics.
//
// Slow implementation: fine for small tensors, but write something specialized for the DType if speed is desired.
func (t *Tensor) Equal(otherTensor *Tensor) bool {
	t.AssertValid()
	otherTensor.AssertValid()

	if t == otherTensor {
		return true
	}
	if !t.shape.Equal(otherTensor.shape) {
		return false
	}
	equal := true // Set to false at the first difference.
	t.MustConstFlatData(func(flat0 any) {
		otherTensor.MustConstFlatData(func(flat1 any) {
			t0V := reflect.ValueOf(flat0)
			t1V := reflect.ValueOf(flat1)
			if t0V.Len() != t1V.Len() {
				equal = false
				return
			}
			for ii := range t0V.Len() {
				if !t0V.Index(ii).Equal(t1V.Index(ii)) {
					equal = false
					return
				}
			}
		})
	})
	return equal
}

// InDelta checks weather Abs(t - otherTensor) < delta for every element.
// If they are the same pointer, they are considered equal.
// If the shapes are different, it returns false.
// If either is invalid (nil), it panics. If the DType is not a float or complex, it also panics.
//
// Slow implementation: fine for small tensors, but write something specialized for the DType if speed is desired.
func (t *Tensor) InDelta(otherTensor *Tensor, delta float64) bool {
	t.AssertValid()
	otherTensor.AssertValid()

	if t == otherTensor {
		return true
	}
	if !t.shape.Equal(otherTensor.shape) {
		return false
	}
	if t.shape.IsZeroSize() {
		// If any of the axes is zero-dimensional, there is no data to compare.
		return true
	}

	inDelta := true // Set to false at the first difference.
	t.MustConstFlatData(func(flat0 any) {
		otherTensor.MustConstFlatData(func(flat1 any) {
			inDelta = xslices.SlicesInDelta(flat0, flat1, delta)
		})
	})
	return inDelta
}

// IsLocal returns true if there is a local storage copy of the tensor.
// If the tensor is shared (see Tensor.IsShared), it also returns true.
//
// See MaterializeLocal to trigger a transfer/copy to the local storage.
func (t *Tensor) IsLocal() bool {
	t.AssertValid()
	if t.isShared {
		return true
	}
	return t.local != nil && !t.local.IsFinalized()
}

// FinalizeLocal immediately frees the local storage copy of the tensor.
// If there are no on-device copies of the tensor, it becomes invalid.
//
// If the storage is shared (see Tensor.IsShared), this is a no-op.
//
// If the tensor is invalid (already finalized), this is a no-op.
func (t *Tensor) FinalizeLocal() {
	if t.CheckValid() != nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.local != nil {
		t.local.flat = nil
		t.local.t = nil
		t.local = nil
	}
}
