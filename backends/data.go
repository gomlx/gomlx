package backends

import "github.com/gomlx/gomlx/types/shapes"

// Buffer represents actual data (a tensor) stored in the accelerator that is actually going to execute the graph.
// It's used as input/output of computation execution.
// A Buffer is always associated to a DeviceNum, even if there is only one.
//
// It is opaque from GoMLX perspective, but one of the backend methods take this value as input, and needs
type Buffer any

// DataInterface is the Backend's subinterface that defines the API to transfer Buffer to/from accelerators for the backend.
type DataInterface interface {
	// BufferFinalize allows the client to inform backend that buffer is no longer needed and associated resources can be
	// freed immediately -- as opposed to waiting for a GC.
	//
	// A finalized buffer should never be used again. Preferably, the caller should set its references to it to nil.
	BufferFinalize(buffer Buffer) error

	// BufferShape returns the shape for the buffer.
	BufferShape(buffer Buffer) (shapes.Shape, error)

	// BufferDeviceNum returns the deviceNum for the buffer.
	BufferDeviceNum(buffer Buffer) (DeviceNum, error)

	// BufferToFlatData transfers the flat values of buffer to the Go flat array.
	// The slice flat must have the exact number of elements required to store the Buffer shape.
	//
	// See also FlatDataToBuffer, BufferShape, and shapes.Shape.Size.
	BufferToFlatData(buffer Buffer, flat any) error

	// BufferFromFlatData transfers data from Go given as a flat slice (of the type corresponding to the shape DType)
	// to the deviceNum, and returns the corresponding Buffer.
	BufferFromFlatData(deviceNum DeviceNum, flat any, shape shapes.Shape) (Buffer, error)

	// HasSharedBuffers returns whether the backend supports "shared buffers": these are buffers
	// that can be used directly by the engine and has a local address that can be read or mutated
	// directly by the client.
	HasSharedBuffers() bool

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
	NewSharedBuffer(deviceNum DeviceNum, shape shapes.Shape) (buffer Buffer, flat any, err error)

	// BufferData returns a slice pointing to the buffer storage memory directly.
	//
	// This only works if HasSharedBuffer is true, that is, if the backend engine runs on CPU, or
	// shares CPU memory.
	//
	// The returned slice becomes invalid after the buffer is destroyed.
	BufferData(buffer Buffer) (flat any, err error)
}
