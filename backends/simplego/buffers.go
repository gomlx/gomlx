package simplego

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
	"reflect"
	"strings"
	"sync"
	"unsafe"
)

// Compile-time check:
var _ backends.DataInterface = (*Backend)(nil)

// Buffer for SimpleGo backend holds a shape and a reference to the flat data.
//
// The flat data may be shared -- for temporary buffers from compiled graphs they are
// taken from larger blobs of bytes allocated in one Go -- or owned by the buffer.
type Buffer struct {
	shape shapes.Shape
	valid bool

	// flat is always a slice of the underlying data type (shape.DType).
	flat any
}

type bufferPoolKey struct {
	dtype  dtypes.DType
	length int
}

// getBufferPool for given dtype/length.
func (b *Backend) getBufferPool(dtype dtypes.DType, length int) *sync.Pool {
	key := bufferPoolKey{dtype: dtype, length: length}
	poolInterface, ok := b.bufferPools.Load(key)
	if !ok {
		poolInterface, _ = b.bufferPools.LoadOrStore(key, &sync.Pool{
			New: func() interface{} {
				return &Buffer{
					flat:  reflect.MakeSlice(reflect.SliceOf(dtype.GoType()), length, length).Interface(),
					shape: shapes.Make(dtype, length),
				}
			},
		})
	}
	return poolInterface.(*sync.Pool)
}

// getBuffer from backend pool of buffers.
func (b *Backend) getBuffer(dtype dtypes.DType, length int) *Buffer {
	pool := b.getBufferPool(dtype, length)
	buf := pool.Get().(*Buffer)
	buf.valid = true
	//fmt.Printf("> Buffer.Get(%p) - dtype=%s, len=%d\n", buf, dtype, length)
	return buf
}

// putBuffer back into the backend pool of buffers.
// After this any references to buffer should be dropped.
func (b *Backend) putBuffer(buffer *Buffer) {
	if buffer == nil || !buffer.shape.Ok() {
		return
	}
	buffer.valid = false
	pool := b.getBufferPool(buffer.shape.DType, buffer.shape.Size())
	pool.Put(buffer)
}

// copyFlat assumes both flat slices are of the same underlying type.
func copyFlat(flatDst, flatSrc any) {
	reflect.Copy(reflect.ValueOf(flatDst), reflect.ValueOf(flatSrc))
}

// mutableBytes returns the slice of the bytes used by the flat given -- it works with any of the supported data types for buffers.
func (b *Buffer) mutableBytes() []byte {
	return dispatchMutableBytes.Dispatch(b.shape.DType, b.flat).([]byte)
}

var dispatchMutableBytes = NewDTypeDispatcher("MutableBytes")

// mutableBytesGeneric is the generic implementation of mutableBytes.
func mutableBytesGeneric[T SupportedTypesConstraints](params ...any) any {
	flat := params[0].([]T)
	bytePointer := (*byte)(unsafe.Pointer(&flat[0]))
	var t T
	return unsafe.Slice(bytePointer, len(flat)*int(unsafe.Sizeof(t)))
}

// cloneBuffer using the pool to allocate a new one.
func (b *Backend) cloneBuffer(buffer *Buffer) *Buffer {
	if buffer == nil || buffer.flat == nil || !buffer.shape.Ok() || !buffer.valid {
		// the buffer is already empty.
		var issues []string
		if buffer != nil {
			if buffer.flat == nil {
				issues = append(issues, "buffer.flat was nil")
			}
			if !buffer.shape.Ok() {
				issues = append(issues, "buffer.shape was invalid")
			}
			if !buffer.valid {
				issues = append(issues, "buffer was marked as invalid")
			}
		} else {
			issues = append(issues, "buffer was nil")
		}
		exceptions.Panicf("cloneBuffer(%p): %s -- buffer was already finalized!?\n", buffer, strings.Join(issues, ", "))
		return nil
	}
	newBuffer := b.getBuffer(buffer.shape.DType, buffer.shape.Size())
	newBuffer.shape = buffer.shape.Clone()
	copyFlat(newBuffer.flat, buffer.flat)
	return newBuffer
}

// NewBuffer creates the buffer with a newly allocated flat space.
func (b *Backend) NewBuffer(shape shapes.Shape) *Buffer {
	buffer := b.getBuffer(shape.DType, shape.Size())
	buffer.shape = shape.Clone()
	return buffer
}

// BufferFinalize allows the client to inform backend that buffer is no longer needed and associated resources can be
// freed immediately.
//
// A finalized buffer should never be used again. Preferably, the caller should set its references to it to nil.
func (b *Backend) BufferFinalize(backendBuffer backends.Buffer) error {
	buffer := backendBuffer.(*Buffer)
	if buffer == nil || buffer.flat == nil || !buffer.shape.Ok() || !buffer.valid {
		// The buffer is already empty.
		var issues []string
		if buffer != nil {
			if buffer.flat == nil {
				issues = append(issues, "buffer.flat was nil")
			}
			if !buffer.shape.Ok() {
				issues = append(issues, "buffer.shape was invalid")
			}
			if !buffer.valid {
				issues = append(issues, "buffer was marked as invalid")
			}
		} else {
			issues = append(issues, "buffer was nil")
		}
		return errors.Errorf("BufferFinalize(%p): %s -- buffer was already finalized!?\n", buffer, strings.Join(issues, ", "))
	}
	//fmt.Printf("> BufferFinalize(%p): shape=%s\n", buffer, buffer.shape)
	//fmt.Printf("\tStack trace:\n%s\n", debug.Stack())
	b.putBuffer(buffer)
	return nil
}

// BufferShape returns the shape for the buffer.
func (b *Backend) BufferShape(buffer backends.Buffer) (shapes.Shape, error) {
	buf, ok := buffer.(*Buffer)
	if !ok {
		return shapes.Invalid(), errors.Errorf("buffer is not a %q backend buffer", BackendName)
	}
	return buf.shape, nil
}

// BufferDeviceNum returns the deviceNum for the buffer.
func (b *Backend) BufferDeviceNum(buffer backends.Buffer) (backends.DeviceNum, error) {
	_, ok := buffer.(*Buffer)
	if !ok {
		return 0, errors.Errorf("buffer is not a %q backend buffer", BackendName)
	}
	return 0, nil
}

// BufferToFlatData transfers the flat values of the buffer to the Go flat array.
// The slice flat must have the exact number of elements required to store the backends.Buffer shape.
//
// See also FlatDataToBuffer, BufferShape, and shapes.Shape.Size.
func (b *Backend) BufferToFlatData(backendBuffer backends.Buffer, flat any) error {
	buf, ok := backendBuffer.(*Buffer)
	if !ok {
		return errors.Errorf("buffer is not a %q backend buffer", BackendName)
	}
	copyFlat(flat, buf.flat)
	return nil
}

// BufferFromFlatData transfers data from Go given as a flat slice (of the type corresponding to the shape DType)
// to the deviceNum, and returns the corresponding backends.Buffer.
func (b *Backend) BufferFromFlatData(deviceNum backends.DeviceNum, flat any, shape shapes.Shape) (backends.Buffer, error) {
	if deviceNum != 0 {
		return nil, errors.Errorf("backend (%s) only supports deviceNum 0, cannot create buffer on deviceNum %d (shape=%s)",
			b.Name(), deviceNum, shape)
	}
	if dtypes.FromGoType(reflect.TypeOf(flat).Elem()) != shape.DType {
		return nil, errors.Errorf("flat data type (%s) does not match shape DType (%s)",
			reflect.TypeOf(flat).Elem(), shape.DType)
	}
	buffer := b.NewBuffer(shape)
	copyFlat(buffer.flat, flat)
	return buffer, nil
}

// HasSharedBuffers returns whether the backend supports "shared buffers": these are buffers
// that can be used directly by the engine and has a local address that can be read or mutated
// directly by the client.
func (b *Backend) HasSharedBuffers() bool {
	return true
}

// NewSharedBuffer returns a "shared buffer" that can be both used as input for execution of
// computations and directly read or mutated by the clients.
//
// It panics if the backend doesn't support shared buffers -- see HasSharedBuffer.
//
// The shared buffer should not be mutated while it is used by an execution.
// Also, the shared buffer cannot be "donated" during execution.
//
// When done, to release the memory, call BufferFinalized on the returned buffer.
//
// It returns a handle to the buffer and a slice of the corresponding data type pointing
// to the shared data.
func (b *Backend) NewSharedBuffer(deviceNum backends.DeviceNum, shape shapes.Shape) (buffer backends.Buffer, flat any, err error) {
	if deviceNum != 0 {
		return nil, nil, errors.Errorf("backend (%s) only supports deviceNum 0, cannot create buffer on deviceNum %d (shape=%s)",
			b.Name(), deviceNum, shape)
	}
	goBuffer := b.NewBuffer(shape)
	return goBuffer, goBuffer.flat, nil
}

// BufferData returns a slice pointing to the buffer storage memory directly.
//
// This only works if HasSharedBuffer is true, that is, if the backend engine runs on CPU, or
// shares CPU memory.
//
// The returned slice becomes invalid after the buffer is destroyed.
func (b *Backend) BufferData(buffer backends.Buffer) (flat any, err error) {
	buf, ok := buffer.(*Buffer)
	if !ok {
		return nil, errors.Errorf("buffer is not a %q backend buffer", BackendName)
	}
	return buf.flat, nil
}
