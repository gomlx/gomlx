package simplego

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"reflect"
	"sync"
)

// Buffer for SimpleGo backend holds a shape and a reference to the flat data.
//
// The flat data may be shared -- for temporary buffers from compiled graphs they are
// taken from larger blobs of bytes allocated in one Go -- or owned by the buffer.
type Buffer struct {
	shape shapes.Shape

	// flat is always a slice of the underlying data type (shape.DType).
	flat any
}

type bufferPoolKey struct {
	dtype  dtypes.DType
	length int
}

var (
	bufferPools sync.Map // map[bufferPoolKey]*sync.Pool
)

func getBuffer(dtype dtypes.DType, length int) *Buffer {
	key := bufferPoolKey{dtype: dtype, length: length}
	poolInterface, _ := bufferPools.LoadOrStore(key, &sync.Pool{
		New: func() interface{} {
			return &Buffer{
				flat: reflect.MakeSlice(reflect.SliceOf(dtype.GoType()), length, length).Interface(),
			}
		},
	})
	pool := poolInterface.(*sync.Pool)
	return pool.Get().(*Buffer)
}

func putBuffer(buffer *Buffer) {
	if buffer == nil {
		return
	}
	key := bufferPoolKey{
		dtype:  buffer.shape.DType,
		length: buffer.shape.Size(),
	}
	if pool, ok := bufferPools.Load(key); ok {
		pool.(*sync.Pool).Put(buffer)
	}
}

// NewBuffer creates the buffer with a newly allocated flat space.
func NewBuffer(shape shapes.Shape) *Buffer {
	buffer := getBuffer(shape.DType, shape.Size())
	buffer.shape = shape.Clone()
	return buffer
}

// BufferFinalize allows client to inform backend that buffer is no longer needed and associated resources can be
// freed immediately.
func (b *Backend) BufferFinalize(buffer backends.Buffer) {
	goBuffer := buffer.(*Buffer)
	goBuffer.shape = shapes.Invalid()
	putBuffer(goBuffer)
}

// BufferShape returns the shape for the buffer.
func (b *Backend) BufferShape(buffer backends.Buffer) shapes.Shape {
	return buffer.(*Buffer).shape
}

// BufferDeviceNum returns the deviceNum for the buffer.
func (b *Backend) BufferDeviceNum(buffer backends.Buffer) backends.DeviceNum {
	return 0
}

// BufferToFlatData transfers the flat values of buffer to the Go flat array.
// The slice flat must have the exact number of elements required to store the backends.Buffer shape.
//
// See also FlatDataToBuffer, BufferShape, and shapes.Shape.Size.
func (b *Backend) BufferToFlatData(buffer backends.Buffer, flat any) {
	goBuffer := flat.(*Buffer)
	reflect.Copy(reflect.ValueOf(flat), reflect.ValueOf(goBuffer.flat))
}

// BufferFromFlatData transfers data from Go given as a flat slice (of the type corresponding to the shape DType)
// to the deviceNum, and returns the corresponding backends.Buffer.
func (b *Backend) BufferFromFlatData(deviceNum backends.DeviceNum, flat any, shape shapes.Shape) backends.Buffer {
	goBuffer := NewBuffer(shape)
	reflect.Copy(reflect.ValueOf(goBuffer.flat), reflect.ValueOf(flat))
	return goBuffer
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
func (b *Backend) NewSharedBuffer(deviceNum backends.DeviceNum, shape shapes.Shape) (buffer backends.Buffer, flat any) {
	if deviceNum != 0 {
		exceptions.Panicf("backend (%s) only supports deviceNum 0, cannot create buffer on deviceNum %d (shape=%s)",
			b.Name, deviceNum, shape)
	}
	goBuffer := NewBuffer(shape)
	return goBuffer, goBuffer.flat
}

// BufferData returns a slice pointing to the buffer storage memory directly.
//
// This only works if HasSharedBuffer is true, that is, if the backend engine runs on CPU, or
// shares CPU memory.
//
// The returned slice becomes invalid after the buffer is destroyed.
func (b *Backend) BufferData(buffer backends.Buffer) (flat any) {
	return buffer.(*Buffer).flat
}
